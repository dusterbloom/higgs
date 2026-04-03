# SpecPrefill TDD Implementation Plan

## Overview

**Goal**: Implement full sparse prefill using `rope_dynamic()` to achieve **400+ tok/s prefill** (currently 152 tok/s).

**Approach**: Test-Driven Development - write failing tests first (RED), then implement to pass (GREEN), then refactor.

## Phase 1: Token Selection & Gathing (Week 1)

### 1.1 Token Selection Tests

**Test 1: select_tokens_basic**
```rust
#[test]
fn test_select_tokens_basic() {
    let tokens = Array::from_slice(&[1u32, 2, 3, 4, 5], &[5]);
    let indices = vec![0, 2, 4];
    let selected = select_tokens(&tokens, &indices).unwrap();
    assert_eq!(selected.shape(), &[1, 3]);
    let selected_vec: Vec<u32> = selected.to_vec().unwrap();
    assert_eq!(selected_vec, vec![1, 3, 5]);
}
```

**RED**: Write test, watch fail (function doesn't exist)
**GREEN**: Implement `select_tokens()` using `take_along_axis`
**REFACTOR**: Optimize, add error handling

---

**Test 2: select_tokens_with_batch**
```rust
#[test]
fn test_select_tokens_with_batch() {
    let tokens = Array::from_iter(vec![1u32, 2, 3, 4, 5, 6], &[2, 3]);
    let indices = vec![0, 2];
    let selected = select_tokens(&tokens, &indices).unwrap();
    assert_eq!(selected.shape(), &[2, 2]);
}
```

**RED**: Write test
**GREEN**: Handle batch dimension
**REFACTOR**: Generalize to N-dimensional

---

### 1.2 Position Array Creation

**Test 3: create_position_array**
```rust
#[test]
fn test_create_position_array() {
    let indices = vec![0, 10, 25, 100];
    let positions = create_position_array(&indices, 0);
    assert_eq!(positions.shape(), &[4]);
    let pos_vec: Vec<i32> = positions.to_vec().unwrap();
    assert_eq!(pos_vec, vec![0, 10, 25, 100]);
}
```

**RED**: Write test
**GREEN**: Implement `create_position_array()`
**REFACTOR**: Add offset support

---

**Test 4: create_position_array_with_offset**
```rust
#[test]
fn test_create_position_array_with_offset() {
    let indices = vec![0, 10, 25];
    let positions = create_position_array(&indices, 50);
    let pos_vec: Vec<i32> = positions.to_vec().unwrap();
    assert_eq!(pos_vec, vec![50, 60, 75]);
}
```

**RED**: Write test
**GREEN**: Handle offset parameter
**REFACTOR**: ---

---

## Phase 2: Custom RoPE Application (Week 2)

### 2.1 Single-Layer RoPE Tests

**Test 5: apply_rope_at_positions_basic**
```rust
#[test]
fn test_apply_rope_at_positions_basic() {
    let rope = RopeBuilder::new(64).build().unwrap();
    let queries = Array::from_iter(vec![1.0f32; 128], &[1, 2, 1, 64]);
    let keys = Array::from_iter(vec![1.0f32; 128], &[1, 2, 1, 64]);
    let positions = Array::from_slice(&[0i32, 100], &[2]);
    
    let (q_with_rope, k_with_rope) = apply_rope_at_positions(
        &queries, &keys, &positions, &rope
    ).unwrap();
    
    // Shape should be preserved
    assert_eq!(q_with_rope.shape(), &[1, 2, 2, 64]);
    assert_eq!(k_with_rope.shape(), &[1, 2, 2, 64]);
    
    // Values should be different from input (RoPE applied)
    assert_ne!(q_with_rope.to_vec::<f32>().unwrap(), queries.to_vec::<f32>().unwrap());
}
```

**RED**: Write test
**GREEN**: Use `Qwen3NextAttention::apply_rope_at_positions()`
**REFACTOR**: Add batch support

---

**Test 6: apply_rope_at_positions_contiguous_matches_standard**
```rust
#[test]
fn test_apply_rope_at_positions_contiguous_matches_standard() {
    // RoPE at [0, 1, 2] should match standard RoPE with offset=0
    let rope = RopeBuilder::new(64).build().unwrap();
    let queries = Array::from_iter(vec![1.0f32; 384], &[1, 2, 3, 64]);
    let keys = Array::from_iter(vec![1.0f32; 384], &[1, 2, 3, 64]);
    let positions = Array::from_slice(&[0i32, 1, 2], &[3]);
    
    let (q_custom, k_custom) = apply_rope_at_positions(
        &queries, &keys, &positions, &rope
    ).unwrap();
    
    let q_standard = rope.forward(&queries).unwrap();
    let k_standard = rope.forward(&keys).unwrap();
    
    // Should be approximately equal
    let q_diff = (q_custom - q_standard).abs().sum(None, false).unwrap();
    assert!(q_diff.item::<f32>().unwrap() < 1e-5);
}
```

**RED**: Write test
**GREEN**: Verify rope_dynamic matches standard RoPE
**REFACTOR**: ---

---

### 2.2 Multi-Layer RoPE Tests

**Test 7: apply_rope_all_layers**
```rust
#[test]
fn test_apply_rope_all_layers() {
    // Test RoPE application across all attention layers
    let model = create_test_model(); // Small Qwen3Next model
    let queries = create_test_queries();
    let keys = create_test_keys();
    let positions = Array::from_slice(&[0i32, 10, 25], &[3]);
    
    let result = apply_rope_all_layers(&model, &queries, &keys, &positions).unwrap();
    
    assert_eq!(result.len(), model.model.layers.len());
    // Each layer should have RoPE applied
}
```

**RED**: Write test
**GREEN**: Iterate through layers, apply `apply_rope_at_positions()`
**REFACTOR**: Parallelize layer processing

---

## Phase 3: Sparse Attention Forward (Week 3)

### 3.1 Single-Layer Sparse Attention

**Test 8: sparse_attention_single_layer**
```rust
#[test]
fn test_sparse_attention_single_layer() {
    let layer = create_test_attention_layer();
    let x = Array::from_iter(vec![1.0f32; 384], &[1, 3, 128]);
    let positions = Array::from_slice(&[0i32, 10, 25], &[3]);
    
    let output = sparse_attention_forward(&layer, &x, &positions, None).unwrap();
    
    assert_eq!(output.shape(), &[1, 3, 128]);
    // Output should be valid (not NaN/Inf)
    assert!(output.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}
```

**RED**: Write test
**GREEN**: Implement single-layer sparse attention with custom RoPE
**REFACTOR**: Extract common code

---

**Test 9: sparse_attention_matches_dense**
```rust
#[test]
fn test_sparse_attention_matches_dense() {
    // Sparse attention at [0, 1, 2] should match dense attention
    let layer = create_test_attention_layer();
    let x = Array::from_iter(vec![1.0f32; 384], &[1, 3, 128]);
    let positions = Array::from_slice(&[0i32, 1, 2], &[3]);
    
    let sparse_output = sparse_attention_forward(&layer, &x, &positions, None).unwrap();
    let dense_output = layer.forward(&x, None, &mut create_test_cache()).unwrap();
    
    // Should be approximately equal
    let diff = (sparse_output - dense_output).abs().sum(None, false).unwrap();
    assert!(diff.item::<f32>().unwrap() < 1e-4);
}
```

**RED**: Write test
**GREEN**: Verify sparse matches dense for contiguous positions
**REFACTOR**: ---

---

### 3.2 Full Model Sparse Forward

**Test 10: sparse_model_forward_basic**
```rust
#[test]
fn test_sparse_model_forward_basic() {
    let model = create_test_model();
    let tokens = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let indices = vec![0, 1, 2];
    
    let (logits, state) = sparse_model_forward(&model, &tokens, &indices, 0).unwrap();
    
    assert_eq!(logits.dim(1), 1); // Single token output
    assert_eq!(state.selected_len, 3);
}
```

**RED**: Write test
**GREEN**: Implement full sparse forward pass
**REFACTOR**: Extract position array creation

---

**Test 11: sparse_model_forward_matches_dense**
```rust
#[test]
fn test_sparse_model_forward_matches_dense() {
    let model = create_test_model();
    let tokens = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let indices = vec![0, 1, 2];
    
    let (sparse_logits, _) = sparse_model_forward(&model, &tokens, &indices, 0).unwrap();
    let dense_logits = model.forward(&tokens, None, &mut create_test_cache()).unwrap();
    
    // Last token logits should match
    let sparse_last = sparse_logits.index((.., -1, ..));
    let dense_last = dense_logits.index((.., -1, ..));
    
    let diff = (sparse_last - dense_last).abs().sum(None, false).unwrap();
    assert!(diff.item::<f32>().unwrap() < 1e-3);
}
```

**RED**: Write test
**GREEN**: Verify sparse matches dense for contiguous selection
**REFACTOR**: ---

---

## Phase 4: Engine Integration (Week 4)

### 4.1 SpecPrefill Decision Logic

**Test 12: should_use_spec_prefill**
```rust
#[test]
fn test_should_use_spec_prefill() {
    let config = SpecPrefillConfig::default();
    
    assert!(!should_use_spec_prefill(&config, 1000));
    assert!(!should_use_spec_prefill(&config, 8191));
    assert!(should_use_spec_prefill(&config, 8192));
    assert!(should_use_spec_prefill(&config, 16384));
    assert!(!should_use_spec_prefill(&config, 65537));
}
```

**RED**: Write test
**GREEN**: Implement decision logic
**REFACTOR**: ---

---

**Test 13: compute_keep_rate**
```rust
#[test]
fn test_compute_keep_rate() {
    assert_eq!(compute_keep_rate(1000), 1.0);
    assert_eq!(compute_keep_rate(8191), 1.0);
    assert!((compute_keep_rate(8192) - 0.30).abs() < 1e-6);
    assert!((compute_keep_rate(32768) - 0.20).abs() < 1e-6);
}
```

**RED**: Write test
**GREEN**: Implement keep rate computation
**REFACTOR**: ---

---

### 4.2 Integration Tests

**Test 14: spec_prefill_end_to_end**
```rust
#[test]
fn test_spec_prefill_end_to_end() {
    let engine = create_test_engine();
    let tokens = vec![1u32; 8192]; // 8k tokens
    
    let result = engine.generate_with_spec_prefill(&tokens, 100).unwrap();
    
    assert!(result.text.len() > 0);
    assert!(result.completion_tokens <= 100);
}
```

**RED**: Write test
**GREEN**: Full integration with engine
**REFACTOR**: ---

---

**Test 15: spec_prefill_performance**
```rust
#[test]
fn test_spec_prefill_performance() {
    let engine = create_test_engine();
    let tokens = vec![1u32; 8192];
    
    let start = std::time::Instant::now();
    engine.generate_with_spec_prefill(&tokens, 10).unwrap();
    let elapsed = start.elapsed();
    
    // Should achieve 400+ tok/s prefill
    let prefill_speed = 8192.0 / elapsed.as_secs_f32();
    assert!(prefill_speed > 400.0, "Prefill speed {} tok/s < 400 tok/s", prefill_speed);
}
```

**RED**: Write test (will fail initially)
**GREEN**: Optimize to pass
**REFACTOR**: Profile and optimize bottlenecks

---

## Phase 5: Benchmarking & Optimization (Week 5)

### 5.1 Performance Benchmarks

**Benchmark 1: prefill_8k_tokens**
```rust
#[bench]
fn bench_prefill_8k(b: &mut Bencher) {
    let model = create_benchmark_model();
    let tokens = vec![1u32; 8192];
    
    b.iter(|| {
        model.forward(&tokens, None, &mut cache)
    });
}
```

**Target**: 400+ tok/s

---

**Benchmark 2: prefill_16k_tokens**
```rust
#[bench]
fn bench_prefill_16k(b: &mut Bencher) {
    let model = create_benchmark_model();
    let tokens = vec![1u32; 16384];
    
    b.iter(|| {
        model.forward(&tokens, None, &mut cache)
    });
}
```

**Target**: 400+ tok/s

---

**Benchmark 3: prefill_32k_tokens**
```rust
#[bench]
fn bench_prefill_32k(b: &mut Bencher) {
    let model = create_benchmark_model();
    let tokens = vec![1u32; 32768];
    
    b.iter(|| {
        model.forward(&tokens, None, &mut cache)
    });
}
```

**Target**: 400+ tok/s

---

## Success Criteria

| Phase | Tests | Performance | Status |
|-------|-------|-------------|--------|
| 1. Token Selection | 4 tests | N/A | ⏳ TODO |
| 2. Custom RoPE | 3 tests | N/A | ⏳ TODO |
| 3. Sparse Attention | 4 tests | N/A | ⏳ TODO |
| 4. Engine Integration | 4 tests | N/A | ⏳ TODO |
| 5. Benchmarks | 3 benchmarks | 400+ tok/s | ⏳ TODO |

**Total**: 18 tests + 3 benchmarks

---

## Implementation Notes

### Key APIs to Use

1. **`fast::rope_dynamic()`** - Apply RoPE at arbitrary positions
   ```rust
   fast::rope_dynamic(
       x,              // [B, n_heads, L, head_dim]
       dimensions,     // RoPE dimensions
       traditional,    // false
       Some(base),     // RoPE base
       scale,          // RoPE scale
       positions,      // [L] position array
       None,           // freqs
   )
   ```

2. **`Qwen3NextAttention::apply_rope_at_positions()`** - Already implemented
   ```rust
   attention.apply_rope_at_positions(&queries, &keys, &positions)
   ```

3. **`Array::take_along_axis()`** - Select tokens by indices
   ```rust
   selected = inputs.take_along_axis(&indices, axis)
   ```

### Common Patterns

**Token Selection:**
```rust
fn select_tokens(inputs: &Array, indices: &[usize]) -> Result<Array, Exception> {
    let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    Ok(inputs.index((.., indices_i32.as_slice())))
}
```

**Position Array:**
```rust
fn create_position_array(indices: &[usize], offset: i32) -> Array {
    let positions: Vec<i32> = indices.iter()
        .map(|&i| (i as i32) + offset)
        .collect();
    Array::from_slice(&positions, &[positions.len() as i32])
}
```

**Sparse Forward:**
```rust
fn sparse_model_forward(
    model: &mut Qwen3NextCausalLM,
    inputs: &Array,
    indices: &[usize],
    offset: i32,
) -> Result<(Array, SparsePrefillState), Exception> {
    // 1. Select tokens
    let selected = select_tokens(inputs, indices)?;
    
    // 2. Create position array
    let positions = create_position_array(indices, offset);
    
    // 3. Forward pass with custom RoPE
    let logits = model.forward_with_custom_rope(&selected, &positions)?;
    
    // 4. Return state
    Ok((logits, SparsePrefillState { ... }))
}
```

---

## Timeline

| Week | Phase | Tests | Deliverable |
|------|-------|-------|-------------|
| 1 | Token Selection | 4 | `select_tokens()`, `create_position_array()` |
| 2 | Custom RoPE | 3 | `apply_rope_at_positions()` verified |
| 3 | Sparse Attention | 4 | `sparse_model_forward()` working |
| 4 | Engine Integration | 4 | Full SpecPrefill in engine |
| 5 | Optimization | 3 benchmarks | **400+ tok/s prefill** |

**Total**: 5 weeks for full implementation

---

## Next Actions

1. **Start with Phase 1, Test 1** - Write `test_select_tokens_basic`
2. **Run test, watch it fail** (RED)
3. **Implement `select_tokens()`** (GREEN)
4. **Refactor** if needed
5. **Move to next test**

**Let's begin!**
