# V4 Port — Overnight Run Completion Report

**Generated:** 2026-05-05T07:34:23Z
**Branch:** feat/v4-port @ a6231aec9
**Pushed:** https://github.com/cchuter/llama.cpp/tree/feat/v4-port

## Streams shipped

| Stream | State | Codex review | Notes |
|---|---|---|---|
| E (validation harness) | done | APPROVED r4 | 6 gate scripts + fixture in tests/v4-port/ |
| A (V4 loader + forward pass class) | done | APPROVED r3 | Structural port to per-arch class pattern |
| B (long-context fix + CPU validation) | done | APPROVED r2 | Cherry-pick 188df615c clean; CPU coherence PASS |
| C (Metal kernels + speed) | done | APPROVED r1 | 25.91 tok/s on M3 Ultra (antirez: 21.1 on M3 Max) |
| D (chat template + tool calling) | done | APPROVED r2 | Anthropic mixed text+tool_result rendering fixed |
| F (this report) | done | — | — |

## Antirez commits ported (in order applied)

1. `06c504247` Add DeepSeek V4 Flash inference support — translated from monolithic-switch style to per-arch class
2. `188df615c` Fix V4 long-context graph metadata — cherry-pick clean
3. `b67f5db5c` Optimize V4 Metal HC decode — cherry-pick clean
4. `57c4283b5` Remove stale V4 quantize tool entry — empty (already in upstream)
5. `2f2d44052` Speed up V4 prompt replay — cherry-pick clean
6. `3ba61fbb4` Add V4 tool-call chat template — cherry-pick + Jinja template fix for Anthropic format

## Integration gate results

```
PASS: loader recognizes V4 GGUF
PASS: coherence (NGL=0, gen=' Paris. The capital of Italy is Rome. The capital of Portugal is Lisbon. The capital of Spain is Madrid. The capital of Norway is Oslo.')
PASS: coherence (NGL=999, gen=' Paris. The capital of Italy is Rome. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital of Spain is')
PASS: speed (NGL=999, 25.19 tok/s)
PASS: tool calling (5/5 with tool_calls)
ALL GATES PASS
```

Decode rate this run: 25.19 tok/s (Metal, NGL=999, M3 Ultra). Matches prior C-stream measurement of 25.91 tok/s within run-to-run noise; well above the 10 tok/s floor.

CPU coherence (NGL=0): 29 tokens decoded, 100% printable.
Metal coherence (NGL=999): 29 tokens decoded, 100% printable.
Tool calling: all 5 requests returned HTTP 200 with exactly 1 `tool_calls` entry each.

## What's working tomorrow

```bash
cd /Users/cchuter/work/llama.cpp
./build/bin/llama-server \
  -m /Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --jinja --port 8080 -ngl 999

# Another terminal:
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d @tests/v4-port/tool-call-fixture.json
```

Expected: HTTP 200 with `tool_calls` array. Decode rate >25 tok/s on Metal.

## What needs human attention tomorrow

Nothing — all gates green.

## Known caveats (per design spec)

- **DSA Metal kernels not implemented** — V3.2 sparse-attention models won't run on Metal in this build. V4 does because antirez wrote V4-specific Metal kernels.
- **CUDA not built** — CPU + Metal only, per overnight scope.
- **Quantization scheme is "trust antirez"** — re-quantizing from official safetensors is a separate task.
- **Validation is loose-parity** — gates check coherence (>80% printable, no degenerate decode) and speed/HTTP-status, not token-level parity with antirez's binary.

## Repo state

- Branch: `feat/v4-port` on `mine` (cchuter/llama.cpp)
- Commits since base: 26
- master: yes — fast-forwarded `mine/master` to `feat/v4-port` after all gates passed
