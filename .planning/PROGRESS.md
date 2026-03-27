# Overnight Session Progress — CoreML ANE Draft Model

## Status: IN PROGRESS

## Phase 1: Model Conversion (Python) -- DONE
- [x] Python venv created (3.12) and deps installed (coremltools 9.0, torch 2.11)
- [x] Conversion script: `scripts/convert_draft_coreml.py`
- [x] Model converts to .mlpackage (1193 MB fp16, 77s conversion)
- [x] KV cache as explicit I/O (not MLState — simpler, works)
- [x] Predictions verified: top-1 match, cosine_sim=0.999987
- [x] Multi-token decode matches HF exactly
- [x] Speed from Python: 6.8 tok/s (already > 4.4 tok/s CPU baseline)

## Phase 2: Rust CoreML Bridge
- [ ] Obj-C bridge API for CoreML (load, predict, state management)
- [ ] Rust FFI bindings
- [ ] CoreMlDraftModel implements DraftModel trait
- [ ] Unit tests pass

## Phase 3: Integration & Benchmarks
- [ ] All 88 higgs tests pass
- [ ] All 201 engine tests pass (minus pre-existing failures)
- [ ] Correctness: temp=0 matches normal decode
- [ ] Benchmark: tok/s measured
- [ ] No new clippy errors

## Phase 4: Bonus Improvements
- [ ] Identified 3 candidates
- [ ] Implemented most impactful one

---

## Log

### Session Start
- Branch: `feat/ane-spec-decode` at `06cda57`
- Starting Phase 1: Python model conversion
