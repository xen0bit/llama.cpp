# DeepSeek-V4 REAP — CPU SSD-streaming optimization

A set of fork-only optimizations for running `DeepSeek-V4-Flash-REAP` models
(52.6 GB / 144 routed experts) **CPU-only** by SSD-streaming the MoE expert
weights instead of requiring the whole model in RAM.

The model math: 6 experts/tok × 43 layers → ~1.74 GB read per token if
nothing is cached, ~0 if fully resident. The non-routed weights (attention,
shared expert, embeddings, LM head) are ~9 GB and must always be resident.
Routed experts dominate the size (~42 GB) and are the streaming target.

---

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench
```

Confirm AVX2/FMA were enabled (GGML_NATIVE is ON by default):
```bash
grep -iE "GGML_NATIVE|MARCH|znver|AVX" build/CMakeCache.txt
```

---

## Flags and workflow (all fork-only, off by default)

### `--ssd-stream` (Stage 2a — the foundation)

Single flag that does everything needed to convert from "load the whole model
into RAM" to genuine demand-streaming:

- **Disables `MAP_POPULATE`** so the kernel reads the model file lazily rather
  than pulling the full 49 GiB into RAM at load.
- **Switches `posix_fadvise` / `madvise` to `RANDOM`** (was `SEQUENTIAL`) so the
  kernel doesn't waste bandwidth prefetching pages around each faulted expert.
- **`mlock`s non-routed weights** (everything except `*_exps` tensors) so expert
  demand-paging can never evict attention, shared expert, embeddings, or LM
  head.
- **Implies `--no-warmup`** to skip the all-expert warmup pass (which would
  fault every expert tensor immediately).

```bash
llama-cli --model model.gguf --ssd-stream      \
  --threads <physical-cores> -n 100 -st ...
  #  --no-repack is automatically honoured via the mmap path;
  #     explicitly pass --no-repack if you see repack-related crashes
```

**Before/after (cold 10 tokens):** disk reads 18.2 → 5.8 GiB,
peak RSS 18.8 → 5.9 GiB (warm page cache: 2.85 GiB disk, 0.8 t/s).

### `--ssd-stream-hotlist <path>` (Stage 2b — expert pinning)

Pins the hottest routed-expert slices resident via `mlock` at load time.
The hotlist is a profiler output file (one `layer expert` or `layer expert count`
pair per line, sorted descending by frequency):

```bash
# After one profiling pass (see below), the hotlist is ready:
llama-cli --model model.gguf --ssd-stream \
  --ssd-stream-hotlist ./model.hotlist ...
```

**Auto-discovery:** when `--ssd-stream` is set without `--ssd-stream-hotlist`,
the loader checks for `<model_path>.hotlist` automatically. So copying the
profiler output to `model.gguf.hotlist` makes subsequent runs pick it up:

```bash
cp /tmp/myprofile.txt /path/to/DeepSeek-V4-....gguf.hotlist
# Next --ssd-stream run uses it automatically
```

### `LLAMA_EXPERT_PROFILE=<path>` (profiler + cache simulator)

Env var, not a flag. Records every expert selection from the MoE routers and
writes a detailed report at exit:

- **Per-(layer,expert) selection counts**, sorted descending.
- **Frequency coverage curve** — the upper bound for a hotness cache
  ("pin top X% of units → Y% hit").
- **LRU hit-rate curve** — what the kernel page cache achieves at several cache
  sizes on the real temporal access order (from an online LRU simulator).
- **A `.hotlist` companion file** at `<path>.hotlist`, ready for
  `--ssd-stream-hotlist`.

```bash
LLAMA_EXPERT_PROFILE=/tmp/ds4_profile.txt \
  llama-cli --model model.gguf --ssd-stream -n 200 -st -p "your prompt"
```

The LRU-vs-frequency gap tells you how much an explicit hotness-aware cache
would buy over the kernel's page cache LRU at each cache size.

---

## Workflows

### 1. Quick benchmark (cold + warm)

```bash
MODEL=/path/to/model.gguf BIN=./build/bin \
  ./perf/ds4-stream-bench.sh warm 96          # warm page cache
```

```bash
MODEL=/path/to/model.gguf BIN=./build/bin \
  sudo -E ./perf/ds4-stream-bench.sh cold 96  # drop caches first --ssd-stream-hotlist
```

Reports decode TPS, prompt TPS, bytes/token from disk, peak RSS.

### 2. One-shot profiling → hotlist → inference

```bash
# Step 1: run a profiling pass (200+ tokens for a representative sample)
LLAMA_EXPERT_PROFILE=/tmp/ds4_profile.txt \
  llama-cli --model model.gguf --ssd-stream -n 200 -st -p "your workload prompt"

# Step 2: install the generated hotlist for auto-discovery
cp /tmp/ds4_profile.txt.hotlist model.gguf.hotlist

# Step 3: all future --ssd-stream runs use it automatically
llama-cli --model model.gguf --ssd-stream -n 100 -st -p "What is..."
```

### 3. MEMMAX sweep (hit-rate vs RAM budget)

On a systemd cgroup v2 machine, sweep the RAM cap to map the continuous
speed-RAM curve:

```bash
MEMMAX=8G  ./perf/ds4-stream-bench.sh cold 96
MEMMAX=16G ./perf/ds4-stream-bench.sh cold 96
MEMMAX=24G ./perf/ds4-stream-bench.sh cold 96
# etc. — each report's bytes/token tells you the effective hit rate at that RAM
```

### 4. A/B: baseline vs hotlist

```bash
# Cold run WITHOUT hotlist (baseline)
python3 -c "import os; os.posix_fadvise(os.open('model.gguf',0),0,0,os.POSIX_FADV_DONTNEED)"
llama-cli --model model.gguf --ssd-stream ...

# Cold run WITH hotlist
cp /tmp/profile.txt model.gguf.hotlist
python3 -c "import os; os.posix_fadvise(os.open('model.gguf',0),0,0,os.POSIX_FADV_DONTNEED)"
llama-cli --model model.gguf --ssd-stream ...
```

---

## Implementation notes

### Thread safety
All env-var-gated features are single-thread read-once in the constructor or
`ith==0` in the compute path. The profiler uses a `std::mutex` on its access
counters.

### MEMORY.md index
Project memories in `~/.claude/projects/.../memory/` capture the non-obvious
findings: why Q2_K repack is gated behind AVX-512, the `atexit`-in-ctor static
lifetime bug in the profiler, and the two root causes of full-model materialization
(MAP_POPULATE + warmup all-expert routing).

### Key realizations (hard-won)
- **`--no-repack` is mandatory for streaming.** Repack copies tensors into
  malloc'd buffers, defeating mmap and forcing a full ~49 GiB read + ~37 GiB RSS
  even with `--no-warmup`. The streaming path uses the mmap directly.
- **mmap is already zero-copy** for expert data — the page cache holds each
  expert page once and the matmul reads from it directly. An explicit userspace
  expert cache (O_DIRECT arena) avoids page-cache double-buffering and gives
  hotness-aware eviction, but does not save memory over mmap alone.
- **The kernel page cache approximates LRU well** for typical generations. The
  hotlist pinning wins most at low RAM budgets (~2× hit rate at 1–2 GiB), where
  the page cache would drop hot experts under pressure.

### File layout
```
ggml/src/ggml-cpu/expert-profile.h   — profiler C ABI
ggml/src/ggml-cpu/expert-profile.cpp — profiler + LRU simulator + hotlist writer
ggml/src/ggml-cpu/ggml-cpu.c         — ith==0 profiler hook in mul_mat_id
src/llama-mmap.cpp                    — RANDOM advice for streaming
src/llama-model.cpp                   — init_mappings prefetch control + hotlist call
src/llama-model-loader.cpp            — ssd_stream flag + mlock + pin_hot_experts
src/llama-model-loader.h              — ssd_stream/ssd_model_path + pin_hot_experts
include/llama.h                       — ssd_stream + ssd_stream_hotlist params
common/common.h / common.cpp          — plumbing to CLI
common/arg.cpp                        — --ssd-stream / --ssd-stream-hotlist flags
perf/ds4-stream-bench.sh              — benchmark harness
perf/TIER2-DESIGN.md                  — full Stage 2b design doc
```