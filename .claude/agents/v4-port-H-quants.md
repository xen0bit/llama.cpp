# v4-port-H-quants: produce Q4_K_M GGUF for DeepSeek V4 Flash

## Goal
Add `DeepseekV4ForCausalLM` (and `DeepseekV4FlashForCausalLM` if HF uses that name) to `convert_hf_to_gguf.py`, produce an f16 GGUF from the cloned base safetensors at `~/models/DeepSeek-V4-Flash/`, then quantize to Q4_K_M. Validate against the existing V4 gate suite. Commit the converter; **do not commit the GGUF artifacts** (they live in `~/models/`, not in git).

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-05-v4-port-H-quants.md`. Follow it exactly — it's already broken into 8 bite-sized tasks with full code snippets.**

The plan was pre-written during brainstorming. Per dev-team convention, the architect should adopt this plan as-is (set state directly to `plan-review` with `plan_path` pointing at the file above) rather than rewriting it. If the architect finds issues during plan-review, codex will catch them.

## Gate (must pass before code-review)
After conversion + quantization, run:
```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh
```
Expected: `ALL GATES PASS` (loader, coherence×2, speed, tools, server-chat). Note: `gate-server-chat-q8.sh` is added by Task G later — it is NOT part of this task's gate.

## Preconditions (architect must verify before any work)
- `~/models/DeepSeek-V4-Flash/` exists and clone is complete (no `.lock` files, `config.json` present)
- ≥800 GiB free on the volume holding `~/models/`
- `cmake --build build -j` produces a clean `./build/bin/llama-quantize`

## What this task explicitly does NOT do
- Other quant levels (Q6_K, Q8_0, IQ4_XS). Those become trivial follow-ups once the f16 GGUF exists.
- Imatrix calibration.
- Terminal-bench end-to-end validation under Q4_K_M.
- Any V4-runtime code changes — converter only.

## Iteration budget
- Up to 8 builder fix rounds for converter bugs
- Up to 8 codex review rounds for code-review (high reasoning effort, fall back to default on stalls per spec)

## Branch
- `feat/v4-port-H-quants` off `feat/v4-port`
- Worktree isolation required

## Definition of done
- `DeepseekV4Model` class registered in `convert_hf_to_gguf.py`
- `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists (≈150 GiB)
- `run-all-gates.sh` PASSES against the new Q4_K_M
- Decode tok/s recorded for the new quant
- `docs/plans/v4-port-quants-completion.md` committed with recipe + sha256s + perf
- Codex code-review APPROVED
- Pushed to `mine`

## Cross-cutting ground rules
- Never push to `origin` (= ggml-org/llama.cpp upstream)
- Never run `gh pr create`
- Don't amend existing commits
- Don't skip git hooks
- 8+8 exhaustion → mark `needs-human`
