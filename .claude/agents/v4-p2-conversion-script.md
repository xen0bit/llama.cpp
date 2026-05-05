# v4-p2-conversion-script: `DeepseekV4Model` in `convert_hf_to_gguf.py`

**Phase:** 2 — V4 architecture skeleton (loadable, even if forward pass is stubbed)
**Complexity:** M
**Branch:** `feat/v4-p2-conversion-script`
**Dependencies:** `v4-p2-tensor-ids`

## Scope

Port the `DeepseekV4Model` class from the *external* antirez fork (https://github.com/antirez/llama.cpp-deepseek-v4-flash, `convert_hf_to_gguf.py` `DeepseekV4Model`): register `DeepseekV4ForCausalLM`, copy the FP4-dequant table (`_fp4_table`) and tensor renaming, write all the new hparams. **Pre-dequant FP4 → F32 at conversion time** (no native FP4 GGUF). **Smoke test input:** convert a single-layer slice (`--deepseek4-max-layers 1`) of the official HF DeepSeek V4 release once available (https://huggingface.co/unsloth/DeepSeek-V4-Flash), or a synthetic mock HF checkpoint built from `test-llama-archs` fixtures. `convert_hf_to_gguf.py` consumes HF safetensors, not pre-converted GGUFs — the antirez 80GB GGUF on huggingface.co/antirez is *not* a valid input. **Out:** quantization beyond what V3.2 supports.

## Acceptance criteria

- The change is contained to the file paths cited in the scope above. Do not edit unrelated subsystems.
- Any new code path is covered by at least one test in `tests/` (extend `test-backend-ops.cpp`, `test-llama-archs.cpp`, or add a dedicated test file as appropriate for the task).
- The build passes on the developer's M3 Ultra: `cmake --build build -j` exits 0.
- The relevant test command from `dev-team.json` `test_commands` passes locally.
- For Metal kernel tasks: confirm via `GGML_METAL=1` environment that the new path executes and matches CPU output to within `1e-3` RMS on the corresponding `test-backend-ops` case.
- For arch / loader / converter tasks: ensure the change does not regress any existing arch in `tests/test-llama-archs.cpp`.
- Commit messages reference the parent roadmap (`docs/plans/v4-roadmap.md`) and the task id.

## Out of scope

Anything explicitly listed under "**Out:**" in the scope paragraph above is **not** part of this task. If you find yourself doing it, stop and split out a follow-up task rather than expanding scope.

## How this task was generated

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p2-conversion-script` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
