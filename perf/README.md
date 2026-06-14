# DeepSeek-V4 REAP — CPU SSD-streaming perf investigation

Goal: faster **CPU-only** decode of the 52.6 GB `DeepSeek-V4-Flash-...-Q2-REAP-ds4.gguf`
by treating it as an SSD-streaming problem (only ~1.83 GB of routed-expert
weights are read per token; the rest of the 44 GB of experts sit idle).

## The number that decides the plan

Decode is **I/O-bound below ~20 TPS** for this model on CPU. The single
measurement that matters is **bytes streamed from the SSD per generated token**:

- `~1.83 GB/token` → nothing useful is cached; you are paying full SSD traffic.
- `~0 GB/token`    → model is effectively resident in RAM (page cache); now
  you're compute-bound (~20–30 TPS ceiling on AVX2).
- anything in between → the page cache is holding some hot experts; bytes/token
  scales with how much RAM the cache has. This is the continuous speed knob.

`ds4-stream-bench.sh` measures this directly from `/proc/<pid>/io read_bytes`
(real block-device reads — captures mmap page-ins, unlike `rchar`).

## Build (target machine)

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-bench
```

Confirm AVX2/FMA were compiled in (GGML_NATIVE is ON by default):
```
./build/bin/llama-cli --version            # then check the build log / cpu features
grep -iE "GGML_NATIVE|MARCH|znver|AVX" build/CMakeCache.txt
```

## Run the measurement

```
MODEL=/path/to/DeepSeek-V4-Flash-...-Q2-REAP-ds4.gguf \
BIN=$PWD/build/bin \
./perf/ds4-stream-bench.sh warm 96          # compute + cached-I/O ceiling

MODEL=... BIN=$PWD/build/bin \
sudo -E ./perf/ds4-stream-bench.sh cold 96  # true SSD streaming (drops cache)
```

Args: `[cold|warm] [n_gen] [threads]`. Threads defaults to physical cores.

### Report back these numbers (cold and warm)
- decode TPS (tg) and prompt TPS (pp)
- steady **bytes/token** (the headline)
- peak RSS and the host's total RAM
- SSD model + bus (Gen3/4/5 NVMe? RAID?), from `lsblk -d -o NAME,ROTA,MODEL`

That tells us the real SSD bandwidth ceiling and how much the existing page
cache already helps — which sizes the Tier-2 expert cache design.

## Tier-1 config (no code; what to actually run for inference)

```
./build/bin/llama-cli \
  --model "$MODEL" --jinja --reasoning off \
  --ctx-size 4096 \
  --threads <PHYSICAL_CORES> \              # not logical/SMT count
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  -p "..."
```

Changes from the original command and why:
- **Dropped `--no-repack`.** Q2_K/IQ2_XXS get no AVX2 repack anyway (the Q2_K
  `q2_K_8x8` kernel is gated behind AVX-512, `ggml/src/ggml-cpu/repack.cpp`),
  but the Q8_0 attention/shared tensors *do* have an AVX2 repack path, so
  leaving repack on speeds the resident part. (If a run crashes or regresses
  with repack on, that's itself a useful finding — re-add `--no-repack` and note
  it.)
- **`--threads` = physical cores.** GEMM rarely benefits from SMT; on the
  8c/16t test box use 8.
- Keep **mmap on** (default). `src/llama-mmap.cpp` already issues
  `MADV_RANDOM`, which is the correct hint for random expert paging. Do **not**
  add `--no-mmap` (forces a full 52 GB read at load) and do **not** `--mlock`
  unless the whole model fits in RAM.

### Note on `-ot` / "keep non-routed weights pinned"
In CPU-only mode there is no separate disk-backed buffer type, so `-ot` cannot
split "resident vs streamed" the way it splits CPU vs GPU. Today the page cache
+ `MADV_RANDOM` is the de-facto expert cache. Giving non-routed weights
*guaranteed* residency (so hot attention/shared/embeds never evict under
expert-paging pressure) is exactly what the Tier-2 cache adds.

## What's next (Tier 2, code)
A dedicated routed-expert streaming cache in the CPU loader: non-routed weights
locked resident; routed experts held in a fixed-RAM-budget, hotness-aware cache;
misses loaded via large aligned `O_DIRECT` `pread`s (no page-cache
double-buffering → ~2× more experts per GB); the 6 experts of a layer fetched
async and overlapped with router/gate compute. Borrowed from antirez/ds4's
`--ssd-streaming` (Metal-only there). Sized to the measured bytes/token + SSD
bandwidth from the step above.
