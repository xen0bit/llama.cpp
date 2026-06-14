# Tier-2: routed-expert SSD-streaming cache (CPU)

> **Status update (2026-06-14):** Stage 2a (streaming foundation) and Stage 2b
> (hotlist pinning) are shipped behind `--ssd-stream` / `--ssd-stream-hotlist`.
> The O_DIRECT arena, clock eviction, and async prefetch described below remain
> forward-looking design. See `perf/README.md` for shipped features and workflow.

Design for streaming the routed MoE experts of `DeepSeek-V4-Flash-...-Q2-REAP-ds4`
from a fast SSD instead of requiring the whole 52.6 GB model in RAM, so the
machine's RAM becomes a continuous speed knob rather than a hard cutoff.

Borrows the idea set from antirez/ds4 `--ssd-streaming` (non-routed weights
resident; routed experts in a fixed-budget cache; load on miss; hotlist-biased),
but implemented in the llama.cpp fork's CPU path so we keep the working REAP
graph and ggml's AVX2 kernels.

---

## 1. What we're caching, and the numbers

Per-expert weights (one routed expert, one layer):

| tensor          | shape            | quant     | bytes/expert |
|-----------------|------------------|-----------|--------------|
| `ffn_gate_exps` | [4096, 2048, E]  | IQ2_XXS   | ~2.06 MB     |
| `ffn_up_exps`   | [4096, 2048, E]  | IQ2_XXS   | ~2.06 MB     |
| `ffn_down_exps` | [2048, 4096, E]  | Q2_K      | ~2.62 MB     |
| **per expert**  |                  |           | **~6.74 MB** |

- 144 experts × 43 layers = 6192 expert-units → **~41.7 GB** routed (cacheable pool).
- 6 selected/token × 43 layers = 258 expert-units/token → **~1.74 GB/token** if every
  access misses.
- Non-routed (attention, shared expert, router gate, embeddings, lm_head, norms)
  ≈ **~9 GB** → always resident.

Decode throughput (single stream):
```
t_token ≈ miss_bytes / ssd_bw  +  compute_time
miss_bytes = (1 - hit_rate) * 1.74 GB
```
`compute_time ≈ 30–50 ms` (IQ2/Q2_K AVX2, 8 cores) → compute ceiling ~20–30 TPS.
So every point of hit_rate buys time until I/O drops below compute. **Hit rate is
the whole game.** The benchmark (`ds4-stream-bench.sh`) gives the starting hit
rate (page-cache today) and the SSD bandwidth to size the cache against.

---

## 2. Where this hooks into the code (idiomatic, no core-ggml patch)

The CPU backend already lets an "extra buffer type" intercept an op before normal
dispatch:

- `ggml/src/ggml-cpu/ggml-cpu.c:1706` — `if (ggml_cpu_extra_compute_forward(params, tensor)) return;`
- `ggml/src/ggml-cpu/traits.cpp:12` — iterates `ggml_backend_cpu_get_extra_buffer_types()`,
  finds the tensor's `tensor_traits` (via `tensor->extra`), calls `compute_forward`.
- `ggml/src/ggml-cpu/traits.h` — the `extra_buffer_type` / `tensor_traits` vtable.
- Registration list: `ggml/src/ggml-cpu/ggml-cpu.cpp:42` `ggml_backend_cpu_get_extra_buffer_types()`
  (currently AMX + repack). Add `ggml_backend_cpu_ssd_expert_buffer_type()` here,
  gated so it's only present when streaming is requested.

This is the same mechanism repack uses (`repack.cpp:4727` sets `tensor->extra`).
We add a sibling: **`ggml/src/ggml-cpu/ssd-experts.cpp`** implementing an
`extra_buffer_type` whose `supports_op` claims `GGML_OP_MUL_MAT_ID` (and `MUL_MAT`
for the shared expert if streamed) when `src0` is a streamed expert tensor, and
whose `tensor_traits::compute_forward` does cache-resolve + matmul.

### Why MUL_MAT_ID is the right interception point
`ggml_compute_forward_mul_mat_id` (`ggml-cpu.c:1521`) already touches **only the
selected experts**: the per-expert loop (`:1634`) does `if (cne1 == 0) continue;`
and reads `src0_cur = src0->data + cur_a*nb02` (`:1641`). We fork that function
into the traits and replace that one line with a cache lookup:

```c
src0_cur = ssd_cache_resolve(cache, src0, cur_a);   // resident slot ptr
```

Everything downstream (the chunked `vec_dot` over `src0_cur + ir0*nb01`, the
src1→q8_K quantization in `wdata`, threading) is unchanged. `work_size` mirrors
the stock `mul_mat_id` scratch plus a small `[n_as]` slot-pointer array.

---

## 3. Two stages (de-risked)

### Stage 2a — managed mmap residency (low risk, ships first)
Keep the expert tensors **mmap-backed** (so `tensor->data` is the real 52.6 GB
mapping and the stock matmul Just Works). The traits/compute path stays the
normal one; we only manage *which pages are resident*:

1. **Pin non-routed weights**: `mlock` the ~9 GB of non-expert tensors so expert
   paging can never evict attention/shared/embeddings. (Per-tensor mlock at load,
   driven by tensor-name classification — there's no CLI for this today.)
2. **Pinned hot set**: an `mlock`'d RAM budget holding the top-K hottest
   expert-units (from a hotlist, §5). Cold experts stay demand-paged by the
   kernel (page cache, `MADV_RANDOM` already set at `llama-mmap.cpp:469`).
3. **Predictive prefetch thread**: at token start, `madvise(MADV_WILLNEED)` each
   layer's *previously selected* experts (temporal locality), overlapping with
   attention compute.

Pros: tiny diff, no loader changes, validates the hit-rate / hotlist model and
the ~1.74 GB/token figure on real hardware. Cons: page cache still double-uses
RAM with the pinned set; no O_DIRECT; eviction of the unpinned tail is kernel-LRU,
not hotness-aware.

### Stage 2b — explicit O_DIRECT expert cache (the real thing)
Replace the page cache for expert data with our own arena:

1. **Loader change** (`src/llama-model-loader.cpp`): for `ffn_*_exps` tensors when
   streaming is enabled, **don't allocate/copy the full tensor**. Allocate the
   tensor descriptor only and record `{fd, file_offset, nb02 (per-expert stride),
   per-expert byte size, type}` in the streaming buffer. (The buft's
   `get_alloc_size` returns just the fixed cache-arena size, not `nbytes`.)
2. **Cache arena**: a fixed `--ssd-expert-cache <bytes>` region of
   `posix_memalign(4096)` slots, each holding one expert-unit (gate|up|down
   contiguous, or three sub-slots). LRU + hotness eviction (§5).
3. **Resolve on miss**: `pread`/`io_uring` with `O_DIRECT`, reads aligned to 4 KB
   (round `file_offset` down, slot is 4 KB-aligned), into a free/evicted slot;
   return slot ptr. No page-cache copy → RAM budget is exactly arena + non-routed,
   deterministic on any machine.
4. **Async + parallel**: issue the layer's ≤6 misses concurrently (io_uring or a
   small reader-thread pool); `compute_forward`'s single-thread pre-pass
   (ith==0, then `ggml_barrier`) waits only for the slots this op needs, then the
   parallel chunk loop runs against resolved pointers.

Pros: minimal, deterministic RAM; no double buffering (~2× more experts per GB);
hotness-aware eviction; true low-RAM operation. Cons: loader hook + alignment
handling + an async I/O layer.

---

## 4. Data structures (Stage 2b)

```c
// one cacheable unit = (layer, expert); 3 quantized sub-tensors
typedef struct {
    uint64_t file_off[3];   // gate, up, down  (GGUF data offsets)
    uint32_t nbytes[3];
    int32_t  slot;          // -1 if not resident
    uint32_t hits;          // hotness counter (decayed)
    uint16_t layer, expert;
} expert_unit;

typedef struct {
    int      fd;            // model file, opened O_DIRECT
    void    *arena;         // posix_memalign(4096), N_slots * slot_stride
    int32_t *slot_owner;    // slot -> expert_unit index  (LRU clock or list)
    expert_unit *units;     // [n_layer * n_expert]
    // LRU/clock state, in-flight read table, io_uring ring, locks
} ssd_expert_cache;
```

`ssd_cache_resolve(cache, layer, expert)`:
1. unit resident → bump hits, return slot ptr.
2. miss → pick victim (clock/LRU skipping pinned hot set), submit aligned
   O_DIRECT read of the 3 sub-tensors into the victim slot, wait (or return a
   future the pre-pass joins on), mark resident, return ptr.

Tensor → (layer, expert) mapping: parse from the tensor name (`blk.%d.ffn_*_exps`)
+ `cur_a` (the expert index inside the op). Stored on `tensor->extra`.

---

## 5. Hotlist / hotness (shared by both stages)
- **Profiler**: log per-(layer,expert) selection counts from the router `ids`
  each layer (mirror ds4's `ds4_expert_profile`). Write a `*.hotlist` after a run.
- **Use**: at startup, seed the pinned/hot set with the top expert-units so the
  first tokens aren't cold; bias eviction to keep high-hit units.
- **Online**: decayed hit counter so the hot set tracks the current prompt's
  routing, not just the global prior.
Expert routing has real skew + strong temporal locality across adjacent tokens,
so even a cache holding a fraction of the 41.7 GB pool should give a large hit
rate — that's the lever that turns 1.74 GB/token into a small fraction.

---

## 6. Control surface
- `--ssd-expert-cache <bytes|GiB>` : arena size (Stage 2b) / pinned hot-set size
  (Stage 2a). Default = auto from free RAM minus non-routed minus KV/scratch
  (port `ds4_ssd_auto_cache_plan`).
- `--ssd-expert-hotlist <path>`    : load/save the hotlist.
- `--ssd-expert-profile`           : write selection counts for hotlist building.
- Streaming off by default; absent flags ⇒ today's behavior.

---

## 7. Risks / open questions (resolve against benchmark data)
- **Hit rate vs cache size** — the core unknown. Measure expert reuse from the
  profiler before committing arena size. If skew is low, Stage 2b's ceiling is
  set by SSD bandwidth at ~1.74 GB/token.
- **O_DIRECT alignment**: GGUF tensor offsets are 32 B-aligned, not 4 KB. Read
  `[floor(off,4096), ceil(off+size,4096))` into an aligned slot; keep the intra-slot
  offset. Verify `general.alignment` for this GGUF.
- **Barrier/threading**: confirm the pre-pass + `ggml_barrier(params->threadpool)`
  pattern inside a traits `compute_forward` (stock `mul_mat_id` already barriers
  for `wdata` quantization).
- **Prefill**: long prompts touch many experts; cache may thrash. Prefill is
  compute-bound and less latency-sensitive — consider bypassing the cache (stream
  straight through) during prefill, caching only during decode.
- **KV cache**: DS4's compressed KV is small at ctx 4096; treat as resident for
  now. antirez's "KV as disk citizen" is a later, separate lever.
- **NUMA / multi-SSD**: arena + reader threads should respect NUMA; RAID-0 /多
  NVMe multiplies the bandwidth ceiling linearly — relevant if a single SSD caps us.

---

## 8. Implementation order
1. ~~(2a) tensor-name classifier + per-tensor `mlock` of non-routed~~ ✅ `--ssd-stream`
2. ~~(2a) router-`ids` profiler → hotlist file; pinned hot set~~ ✅ `LLAMA_EXPERT_PROFILE` + `--ssd-stream-hotlist`
   `WILLNEED` prefetch thread. Re-measure.
3. (2b) streaming buft + loader hook (record offsets, don't load exps) behind
   `--ssd-expert-cache`; `compute_forward` forked from `mul_mat_id` with
   `ssd_cache_resolve`; synchronous `pread` first.
4. (2b) O_DIRECT + io_uring async + parallel layer fetch + overlap.
5. Auto cache-budget planner; tune slot layout, eviction, decay against measured
   hit-rate curves.
