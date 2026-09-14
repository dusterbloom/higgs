import numpy as np
import mlx.core as mx

rng = np.random.default_rng(20260914)
q = mx.array(rng.normal(size=(1, 16, 1024, 256)).astype(np.float32))
k = mx.array(rng.normal(size=(1, 2, 16384, 256)).astype(np.float32))
v = mx.array(rng.normal(size=(1, 2, 16384, 256)).astype(np.float32))
mask = mx.array(np.arange(16384)[None, :] <= np.arange(1024)[:, None] + 15360)
mx.eval(q, k, v, mask)
for _ in range(2):
    parts = []
    for i in range(0, 1024, 128):
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i:i + 128], k, v, scale=1 / 16, mask=mask[i:i + 128]
        )
        mx.eval(out)
        parts.append(out)
    mx.eval(mx.concatenate(parts, axis=2))
print("dense_profile_complete", flush=True)
