# v4-roadmap: audit V3.2/DSA and plan V4 (Flash) support

## Goal
Audit the existing DeepSeek V3.2 / DeepSeek Sparse Attention (DSA) implementation on the
`deepseek-dsa` branch and produce a phased implementation plan for adding DeepSeek V4 (Flash)
support to llama.cpp, with Metal backend coverage for both V3.2 and V4.

## Background
- **Current branch:** `deepseek-dsa` (fairydreaming's PR #21149)
  - Adds DeepSeek V3.2 + DeepSeek Sparse Attention (DSA) with lightning indexer
  - CPU + CUDA backends only (no Metal yet)
  - Still draft upstream
- **Target hardware:** Apple M3 Ultra — Metal kernels are required for any new GGML ops
- **Reference implementations:**
  - Upstream V3.2 PR (this branch): https://github.com/ggml-org/llama.cpp/pull/21149
  - antirez's V4-Flash fork (rejected by maintainers, but architecturally informative): https://github.com/antirez/llama.cpp-deepseek-v4-flash
  - Upstream WIP V4 PR (reference-only): https://github.com/ggml-org/llama.cpp/pull/22378
  - DeepSeek V4 Flash model card: https://huggingface.co/unsloth/DeepSeek-V4-Flash
  - antirez's V4 GGUF (~80 GB IQ2XXS mixed quant): https://huggingface.co/antirez/deepseek-v4-gguf

## Out of scope for this task
- Writing any C++ / Metal / CUDA code
- Cloning model weights
- Running inference
This task produces planning artifacts only.

## Deliverables

### 1. Roadmap document at `docs/plans/v4-roadmap.md`

The architect must produce a roadmap covering:

#### 1a. V3.2 inventory
List every file changed by this branch vs. `origin/master`, grouped by component. For each
file, give a one-line description of what it adds. Use:
```bash
git diff --stat origin/master...HEAD
```
to get the file list, then read the changes for each.

Group into:
- **Architecture / model registry** (e.g. `src/llama-arch.cpp`, `src/llama-model.h`)
- **Conversion script** (`convert_hf_to_gguf.py`)
- **Forward pass / model graph**
- **New GGML ops** (e.g. `GGML_OP_SCATTER`, `GGML_OP_HADAMARD` and friends — confirm against PR description)
- **CPU backend implementations** of those new ops
- **CUDA backend implementations** of those new ops
- **KV cache changes** (DSA introduces `llama_kv_cache_dsa`)
- **Tests / examples**

#### 1b. Metal gap analysis
For each new GGML op identified in 1a, determine:
- Does a Metal kernel exist anywhere in `ggml/src/ggml-metal/` for this op? (No is the expected answer for the new ops, but verify.)
- What existing Metal kernel is the closest analog we can model the new kernel on?
- What's the rough complexity (S/M/L) of porting it?

#### 1c. V4 delta over V3.2
Compare V4 (Flash) against V3.2 architecturally. Use the Hugging Face model card and
antirez's fork as references, but flag every claim as needing verification against the
official DeepSeek V4 release. Specifically identify:
- Native FP4+FP8 mixed precision for experts — what does this mean for our quant pipeline?
- Any V4-specific routing or attention changes beyond V3.2's DSA
- The custom encoding format vs. Jinja chat template — does Mistral-style template work or do we need custom handling?
- Tool-calling / function-calling format

#### 1d. Phased task breakdown
Sequenced list of dev-team tasks, each independently shippable. Use this naming:
- `v4-p1-*` — Phase 1: Metal kernels for V3.2's new ops (gets us a Metal-capable V3.2 first)
- `v4-p2-*` — Phase 2: V4 architecture skeleton (arch enum, conversion script, model loader)
- `v4-p3-*` — Phase 3: V4 forward pass + any new V4-specific ops (with CPU + Metal backends)
- `v4-p4-*` — Phase 4: V4 chat template / tool calling
- `v4-p5-*` — Phase 5: validation against reference outputs

For each task, give:
- `id` (e.g. `v4-p1-metal-scatter`)
- `title` (one line)
- `scope` (one paragraph — what's in, what's out)
- `dependencies` (other task ids it blocks on)
- `complexity` (S/M/L)

#### 1e. Risks and unknowns
- What could derail the work (e.g. Metal backend doesn't expose the primitives we need)
- What we'd need to verify with reference outputs (and how to get them — DeepSeek API? a known-good GGUF?)
- Hardware/tooling we don't have (no NVIDIA box, no CI)
- Where we should expect to be wrong

### 2. Task scaffolding from the roadmap
After the roadmap is approved, the builder phase converts §1d into actual files:
- For each phased task, create `.claude/agents/<task-id>.md` (one per task) with the full spec
- For each phased task, create `tasks/active/<task-id>.json` initialized with state `roadmap`
- The roadmap doc itself remains at `docs/plans/v4-roadmap.md`

Example task JSON skeleton (the builder must write one of these per task):
```json
{
  "id": "v4-p1-metal-scatter",
  "title": "Implement GGML_OP_SCATTER on Metal backend",
  "state": "roadmap",
  "branch": "feat/v4-p1-metal-scatter",
  "spec_path": ".claude/agents/v4-p1-metal-scatter.md",
  "plan_path": null,
  "review_rounds": 0,
  "fix_attempts": 0,
  "history": [],
  "test_report": null,
  "errors": [],
  "in_progress": false
}
```

## Definition of done
- `docs/plans/v4-roadmap.md` exists, was reviewed by codex, and codex returned APPROVE
- `tasks/active/v4-p*.json` populated (one per phased task)
- `.claude/agents/v4-p*.md` populated (one per phased task)
- The build still passes: `cmake --build build -j` exits 0
- The work was committed on `feat/v4-roadmap` and pushed to remote `mine`
- No PR was opened (push only — see ground rules)

## Ground rules specific to this task
- **Push to `mine` only — DO NOT run `gh pr create`.** Per `dev-team.json` push_policy, we are not opening PRs against ggml-org/llama.cpp upstream. Push the feature branch to `mine` and stop. The user will decide when to upstream.
- **No code edits.** This task is purely planning + scaffolding. If you find yourself editing a `.cpp` or `.metal` file, you're doing the wrong task.
- **Cite file paths and line numbers** in the roadmap so a future agent can find what you found.
