#!/usr/bin/env python3
"""Convert Qwen3-0.6B to CoreML .mlpackage for ANE draft inference.

Usage:
    source .venv/bin/activate
    python scripts/convert_draft_coreml.py

Produces: models/qwen3-0.6b-draft.mlpackage

Two models are produced:
1. Decode model: input_ids[1,1], pos[1], kv_cache → logits[1,vocab], kv_cache_out
   - Single-token decode with explicit KV cache I/O
   - KV cache: [n_layers*2, n_kv_heads, max_seq, head_dim] (K and V interleaved)
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_PATH = OUTPUT_DIR / "qwen3-0.6b-draft.mlpackage"
MAX_SEQ = 512

# ─── Load model ─────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID}...")
t0 = time.time()
config = AutoConfig.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
hf_model.eval()

N_LAYERS = config.num_hidden_layers  # 28
N_HEADS = config.num_attention_heads  # 16
N_KV_HEADS = config.num_key_value_heads  # 8
HEAD_DIM = config.head_dim  # 128
HIDDEN = config.hidden_size  # 1024
INTER = config.intermediate_size  # 3072
VOCAB = config.vocab_size  # 151936
ROPE_THETA = config.to_dict().get("rope_theta", 1000000.0)
RMS_EPS = config.rms_norm_eps  # 1e-6

print(f"Loaded in {time.time()-t0:.1f}s — L={N_LAYERS} H={HIDDEN} heads={N_HEADS} kv={N_KV_HEADS}")


# ─── Self-contained decode model ────────────────────────────────────────
# Build from extracted weights to avoid HF internal API dependencies.

class QwenDecoder(nn.Module):
    """Minimal Qwen3 single-token decoder with explicit KV cache."""

    def __init__(self, hf_model, config):
        super().__init__()
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.max_seq = MAX_SEQ
        self.rms_eps = config.rms_norm_eps
        self.rope_theta = config.to_dict().get("rope_theta", 1000000.0)
        self.vocab_size = config.vocab_size

        # Extract weights as plain parameters (no HF module dependencies)
        m = hf_model.model

        self.embed_weight = nn.Parameter(m.embed_tokens.weight.data.clone())
        self.final_norm_weight = nn.Parameter(m.norm.weight.data.clone())

        # Per-layer weights
        self.ln1_w = nn.ParameterList()
        self.ln2_w = nn.ParameterList()
        self.q_proj_w = nn.ParameterList()
        self.q_proj_b = nn.ParameterList()
        self.k_proj_w = nn.ParameterList()
        self.k_proj_b = nn.ParameterList()
        self.v_proj_w = nn.ParameterList()
        self.v_proj_b = nn.ParameterList()
        self.o_proj_w = nn.ParameterList()
        self.q_norm_w = nn.ParameterList()
        self.k_norm_w = nn.ParameterList()
        self.gate_w = nn.ParameterList()
        self.up_w = nn.ParameterList()
        self.down_w = nn.ParameterList()

        for layer in m.layers:
            sa = layer.self_attn
            mlp = layer.mlp

            self.ln1_w.append(nn.Parameter(layer.input_layernorm.weight.data.clone()))
            self.ln2_w.append(nn.Parameter(layer.post_attention_layernorm.weight.data.clone()))

            self.q_proj_w.append(nn.Parameter(sa.q_proj.weight.data.clone()))
            self.q_proj_b.append(nn.Parameter(sa.q_proj.bias.data.clone() if sa.q_proj.bias is not None else torch.zeros(self.n_heads * self.head_dim)))
            self.k_proj_w.append(nn.Parameter(sa.k_proj.weight.data.clone()))
            self.k_proj_b.append(nn.Parameter(sa.k_proj.bias.data.clone() if sa.k_proj.bias is not None else torch.zeros(self.n_kv_heads * self.head_dim)))
            self.v_proj_w.append(nn.Parameter(sa.v_proj.weight.data.clone()))
            self.v_proj_b.append(nn.Parameter(sa.v_proj.bias.data.clone() if sa.v_proj.bias is not None else torch.zeros(self.n_kv_heads * self.head_dim)))
            self.o_proj_w.append(nn.Parameter(sa.o_proj.weight.data.clone()))

            # QK norm (Qwen3 specific)
            self.q_norm_w.append(nn.Parameter(sa.q_norm.weight.data.clone()))
            self.k_norm_w.append(nn.Parameter(sa.k_norm.weight.data.clone()))

            self.gate_w.append(nn.Parameter(mlp.gate_proj.weight.data.clone()))
            self.up_w.append(nn.Parameter(mlp.up_proj.weight.data.clone()))
            self.down_w.append(nn.Parameter(mlp.down_proj.weight.data.clone()))

        # Precompute RoPE inv_freq
        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rms_norm(self, x, weight):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_eps)
        return (x * weight).to(x.dtype)

    def _rope(self, x, pos):
        """Apply RoPE to tensor x at position pos."""
        # x: [1, heads, 1, head_dim], pos: scalar
        dim = self.head_dim
        # Compute sin/cos for this position
        freqs = pos.float() * self.inv_freq  # [dim/2]
        cos_val = torch.cos(freqs).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,dim/2]
        sin_val = torch.sin(freqs).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        x1 = x[..., :dim//2]
        x2 = x[..., dim//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * torch.cat([cos_val, cos_val], dim=-1) + rotated * torch.cat([sin_val, sin_val], dim=-1)

    def forward(self, input_ids, position, kv_cache):
        """
        input_ids: [1, 1] int64
        position:  [1] int64 — current position (0-indexed)
        kv_cache:  [n_layers*2, n_kv_heads, max_seq, head_dim] float32
                   Layout: [k0, v0, k1, v1, ..., k27, v27]

        Returns: (logits [1, vocab], kv_cache_out [same shape])
        """
        pos = position[0]

        # Embed
        hidden = F.embedding(input_ids, self.embed_weight)  # [1, 1, hidden]

        kv_out = kv_cache.clone()

        for i in range(self.n_layers):
            residual = hidden

            # Pre-attention RMSNorm
            hidden = self._rms_norm(hidden, self.ln1_w[i])

            # QKV projections
            q = F.linear(hidden, self.q_proj_w[i], self.q_proj_b[i])  # [1, 1, n_heads*head_dim]
            k = F.linear(hidden, self.k_proj_w[i], self.k_proj_b[i])  # [1, 1, n_kv*head_dim]
            v = F.linear(hidden, self.v_proj_w[i], self.v_proj_b[i])

            q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)   # [1, n_heads, 1, head_dim]
            k = k.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [1, n_kv, 1, head_dim]
            v = v.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # QK RMSNorm (Qwen3)
            q = self._rms_norm(q, self.q_norm_w[i])
            k = self._rms_norm(k, self.k_norm_w[i])

            # RoPE
            q = self._rope(q, pos)
            k = self._rope(k, pos)

            # Write new KV to cache at position using scatter (trace-safe)
            ki = 2 * i
            vi = 2 * i + 1
            pos_idx = pos.long().view(1, 1, 1).expand(self.n_kv_heads, 1, self.head_dim)
            kv_k = kv_out[ki].scatter(1, pos_idx, k[0])
            kv_v = kv_out[vi].scatter(1, pos_idx, v[0])
            # Rebuild kv_out without in-place ops (trace-safe)
            slices = []
            for j in range(self.n_layers * 2):
                if j == ki:
                    slices.append(kv_k)
                elif j == vi:
                    slices.append(kv_v)
                else:
                    slices.append(kv_out[j])
            kv_out = torch.stack(slices)

            # Read full K,V cache
            k_full = kv_k.unsqueeze(0)  # [1, n_kv, max_seq, head_dim]
            v_full = kv_v.unsqueeze(0)

            # GQA: repeat KV heads
            if self.n_kv_heads < self.n_heads:
                reps = self.n_heads // self.n_kv_heads
                k_full = k_full.repeat_interleave(reps, dim=1)
                v_full = v_full.repeat_interleave(reps, dim=1)

            # Attention: Q @ K^T / sqrt(d)
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # [1, n_heads, 1, max_seq]

            # Causal mask: only attend to positions <= current
            positions = torch.arange(self.max_seq, device=scores.device)
            mask = torch.where(positions <= pos, 0.0, float("-inf"))
            scores = scores + mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_full)  # [1, n_heads, 1, head_dim]

            # Merge heads + output projection
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, self.n_heads * self.head_dim)
            attn_out = F.linear(attn_out, self.o_proj_w[i])

            hidden = residual + attn_out

            # MLP
            residual = hidden
            hidden = self._rms_norm(hidden, self.ln2_w[i])
            gate = F.linear(hidden, self.gate_w[i])
            up = F.linear(hidden, self.up_w[i])
            hidden = F.silu(gate) * up
            hidden = F.linear(hidden, self.down_w[i])
            hidden = residual + hidden

        # Final norm + lm_head (tied weights)
        hidden = self._rms_norm(hidden, self.final_norm_weight)
        logits = F.linear(hidden, self.embed_weight)  # [1, 1, vocab]
        logits = logits.squeeze(1)  # [1, vocab]

        return logits, kv_out


# ─── Verify against HuggingFace ─────────────────────────────────────────

print("Building self-contained decoder...")
decoder = QwenDecoder(hf_model, config)
decoder.eval()

KV_SHAPE = (N_LAYERS * 2, N_KV_HEADS, MAX_SEQ, HEAD_DIM)

print("Verifying against HuggingFace model...")
test_ids = torch.tensor([[42]], dtype=torch.long)
test_pos = torch.tensor([0], dtype=torch.long)
test_kv = torch.zeros(KV_SHAPE, dtype=torch.float32)

with torch.no_grad():
    our_logits, our_kv = decoder(test_ids, test_pos, test_kv)
    hf_out = hf_model(input_ids=test_ids, use_cache=False)
    hf_logits = hf_out.logits[:, -1, :]  # [1, vocab]

our_top1 = our_logits.argmax(-1).item()
hf_top1 = hf_logits.argmax(-1).item()
cos_sim = F.cosine_similarity(our_logits.float(), hf_logits.float()).item()
print(f"  Our top-1: {our_top1}, HF top-1: {hf_top1}, match: {our_top1 == hf_top1}")
print(f"  Cosine similarity: {cos_sim:.6f}")

if our_top1 != hf_top1:
    print("  WARNING: Top-1 mismatch! Checking if cos_sim is close enough...")
    if cos_sim < 0.99:
        print("  ERROR: Logits diverge too much. Aborting.")
        sys.exit(1)
    print("  Close enough, continuing...")

# Multi-token verification: check "Hello" decoding
print("  Multi-token check (3-step decode)...")
tokens = [9707]  # "Hello"
kv = torch.zeros(KV_SHAPE, dtype=torch.float32)
our_generated = []
with torch.no_grad():
    for step, tok in enumerate(tokens):
        logits, kv = decoder(torch.tensor([[tok]], dtype=torch.long),
                             torch.tensor([step], dtype=torch.long), kv)
    # Generate 3 tokens
    for step in range(len(tokens), len(tokens) + 3):
        next_tok = logits.argmax(-1).item()
        our_generated.append(next_tok)
        logits, kv = decoder(torch.tensor([[next_tok]], dtype=torch.long),
                             torch.tensor([step], dtype=torch.long), kv)

# Compare with HF
hf_input = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    hf_generated = hf_model.generate(hf_input, max_new_tokens=3, do_sample=False,
                                      temperature=None, top_p=None)
hf_gen = hf_generated[0, len(tokens):].tolist()
print(f"  Our generated: {our_generated}")
print(f"  HF  generated: {hf_gen}")
match = our_generated == hf_gen
print(f"  Match: {match}")
if not match:
    print("  WARNING: Generation mismatch. Will proceed but may need debugging.")

# ─── Trace ───────────────────────────────────────────────────────────────

print("\nTracing model...")
dummy_ids = torch.tensor([[1]], dtype=torch.long)
dummy_pos = torch.tensor([0], dtype=torch.long)
dummy_kv = torch.zeros(KV_SHAPE, dtype=torch.float32)

with torch.no_grad():
    traced = torch.jit.trace(decoder, (dummy_ids, dummy_pos, dummy_kv))
print("  Trace successful")

# Verify trace
with torch.no_grad():
    t_logits, t_kv = traced(test_ids, test_pos, test_kv)
    diff = (our_logits - t_logits).abs().max().item()
    print(f"  Traced vs eager max diff: {diff:.2e}")

# ─── Convert to CoreML ──────────────────────────────────────────────────

print("\nConverting to CoreML (this may take a few minutes)...")
t0 = time.time()

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="kv_cache", shape=KV_SHAPE, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="kv_cache_out", dtype=np.float16),
    ],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.macOS15,
)

elapsed = time.time() - t0
print(f"  Conversion done in {elapsed:.1f}s")

# ─── Save ────────────────────────────────────────────────────────────────

import numpy as np

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving to {OUTPUT_PATH}...")
mlmodel.save(str(OUTPUT_PATH))
size_mb = sum(f.stat().st_size for f in OUTPUT_PATH.rglob('*') if f.is_file()) / 1e6
print(f"  Saved! Size: {size_mb:.1f} MB")

# ─── Verify CoreML ──────────────────────────────────────────────────────

print("\nVerifying CoreML predictions...")
loaded = ct.models.MLModel(str(OUTPUT_PATH))

cm_kv = np.zeros(KV_SHAPE, dtype=np.float16)
pred = loaded.predict({
    "input_ids": np.array([[42]], dtype=np.int32),
    "position": np.array([0], dtype=np.int32),
    "kv_cache": cm_kv,
})
cm_logits = np.array(pred["logits"]).flatten()

pt_logits = our_logits.numpy().flatten().astype(np.float16).astype(np.float32)
cm_logits_f32 = cm_logits.astype(np.float32)

pt_top = np.argmax(pt_logits)
cm_top = np.argmax(cm_logits_f32)
print(f"  PyTorch top-1: {pt_top}, CoreML top-1: {cm_top}, match: {pt_top == cm_top}")

dot = np.dot(pt_logits, cm_logits_f32)
na = np.linalg.norm(pt_logits)
nb = np.linalg.norm(cm_logits_f32)
print(f"  Cosine similarity: {dot/(na*nb+1e-8):.6f}")

# ─── Speed test ──────────────────────────────────────────────────────────

print("\nSpeed test (20 decode steps)...")
kv = np.zeros(KV_SHAPE, dtype=np.float16)

# Warmup
for _ in range(3):
    pred = loaded.predict({
        "input_ids": np.array([[42]], dtype=np.int32),
        "position": np.array([0], dtype=np.int32),
        "kv_cache": kv,
    })

# Timed run
t0 = time.time()
generated = []
for pos in range(20):
    token = generated[-1] if generated else 42
    pred = loaded.predict({
        "input_ids": np.array([[token]], dtype=np.int32),
        "position": np.array([pos], dtype=np.int32),
        "kv_cache": kv,
    })
    kv = np.array(pred["kv_cache_out"])
    logits = np.array(pred["logits"]).flatten()
    next_tok = int(np.argmax(logits))
    generated.append(next_tok)

elapsed = time.time() - t0
print(f"  20 tokens in {elapsed:.3f}s = {20/elapsed:.1f} tok/s")
print(f"  Generated tokens: {generated[:10]}...")

print("\nDone! Model saved to:", OUTPUT_PATH)
