#!/usr/bin/env bash
#
# ds4-stream-bench.sh — measure whether DeepSeek-V4 REAP decode is I/O-bound
# (SSD-streaming) or compute-bound on a CPU-only host.
#
# It runs llama-cli and, while it runs, samples /proc/<pid>/io "read_bytes"
# (bytes actually pulled from the block device — this captures mmap page-ins
# that never go through read()/rchar). The steady-state slope of read_bytes
# during generation, divided by the decode token rate, is the real number we
# care about: BYTES STREAMED FROM SSD PER GENERATED TOKEN.
#
# Model math for reference (DeepSeek-V4-Flash REAP, 6 experts/tok, 43 layers,
# ~2.25 bpw experts): ~1.83 GB/token if nothing is cached, ~0 if fully resident.
#
# Usage:
#   MODEL=/path/to/DeepSeek-...-Q2-REAP-ds4.gguf BIN=/path/to/build/bin \
#     ./perf/ds4-stream-bench.sh [cold|warm] [n_gen] [threads]
#
#   cold  : drop the page cache first (needs root or sudo) -> true streaming
#   warm  : run once to warm cache, then measure -> compute + cached-I/O ceiling
#
# Reports: decode TPS, prompt TPS, total disk read_bytes, steady-state
# bytes/token, peak RSS.

set -u

MODE="${1:-warm}"
NGEN="${2:-96}"
THREADS="${3:-$(nproc --all 2>/dev/null | awk '{print int($1/2)}')}"   # default: physical cores
: "${MODEL:?set MODEL=/path/to/model.gguf}"
: "${BIN:?set BIN=/path/to/build/bin (dir containing llama-cli)}"
PROMPT="${PROMPT:-What is the capital of France? Explain the history of the city in detail.}"
CTX="${CTX:-4096}"
# Extra llama-cli flags. MUST keep --no-repack for the streaming use case:
# repacking copies tensors into malloc'd RAM buffers, which defeats mmap demand-
# paging and forces a full ~49 GB read + ~37 GB resident at load (measured).
# Drop --no-repack ONLY when deliberately measuring a fully-resident run.
# --ignore-eos forces exactly NGEN decode tokens (chat models otherwise stop at
# end-of-turn before -n, giving inconsistent token counts across runs).
EXTRA="${EXTRA:---jinja --reasoning off --no-repack --ignore-eos --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0}"

CLI="$BIN/llama-cli"
[ -x "$CLI" ] || { echo "no executable llama-cli at $CLI"; exit 1; }
[ -f "$MODEL" ] || { echo "no model at $MODEL"; exit 1; }

drop_caches() {
  sync
  if [ -w /proc/sys/vm/drop_caches ]; then
    echo 3 > /proc/sys/vm/drop_caches && return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && return 0
  fi
  echo "WARN: could not drop caches (need root); 'cold' result will be partly warm" >&2
  return 1
}

run_once() {
  local tag="$1"; shift
  local logf iolog
  logf="$(mktemp)"; iolog="$(mktemp)"

  # Optional: cap RAM (page cache + RSS) with a cgroup to force the streaming
  # regime on a high-RAM box. MEMMAX=16G ./ds4-stream-bench.sh ...
  # Sweep MEMMAX to map hit-rate vs cache size. Needs systemd cgroup v2.
  local wrap=()
  if [ -n "${MEMMAX:-}" ]; then
    if command -v systemd-run >/dev/null 2>&1; then
      wrap=(systemd-run --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0)
      command -v sudo >/dev/null 2>&1 && [ "$(id -u)" != 0 ] && wrap=(sudo "${wrap[@]}")
      tag="$tag  [MemoryMax=$MEMMAX]"
    else
      echo "WARN: MEMMAX set but systemd-run missing; ignoring" >&2
    fi
  fi

  "${wrap[@]}" "$CLI" --model "$MODEL" --ctx-size "$CTX" --threads "$THREADS" \
         -n "$NGEN" -st $EXTRA -p "$PROMPT" >"$logf" 2>&1 &
  local pid=$!
  # Under a systemd scope the cli is a grandchild; find the real llama-cli pid.
  if [ -n "${MEMMAX:-}" ]; then
    for _ in $(seq 1 20); do
      local real; real=$(pgrep -n -f "llama-cli .*--model $MODEL" 2>/dev/null)
      [ -n "$real" ] && { pid="$real"; break; }
      sleep 0.2
    done
  fi

  # Sample disk reads + RSS every 0.5s: "t_ms read_bytes rss_kb"
  local t0 now rb rss
  t0=$(date +%s%3N)
  printf 'ms read_bytes rss_kb\n' > "$iolog"
  while kill -0 "$pid" 2>/dev/null; do
    rb=$(awk '/^read_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null)
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    now=$(date +%s%3N)
    [ -n "${rb:-}" ] && printf '%s %s %s\n' "$((now-t0))" "$rb" "${rss:-0}" >> "$iolog"
    sleep 0.5
  done
  wait "$pid"

  local tg pp peak_rss total_rb
  tg=$(grep -E '^[^:]*: *eval time' "$logf" | grep -oE '[0-9.]+ tokens per second' | tail -1 | grep -oE '^[0-9.]+')
  pp=$(grep -E 'prompt eval time' "$logf" | grep -oE '[0-9.]+ tokens per second' | tail -1 | grep -oE '^[0-9.]+')
  peak_rss=$(awk 'NR>1{if($3>m)m=$3} END{printf "%.2f", m/1048576}' "$iolog")  # GiB
  total_rb=$(awk 'NR>1{v=$2} END{printf "%.0f", v}' "$iolog")

  # Steady-state decode read slope: fit read_bytes over the LAST 60% of samples
  # (skips the load/prefill spike). bytes/token = slope_bytes_per_sec / tg.
  local bpt
  bpt=$(awk -v tg="$tg" 'NR>1{ms[n]=$1; rb[n]=$2; n++}
    END{
      if(n<4||tg==""||tg+0==0){print "n/a"; exit}
      s=int(n*0.4); if(s<1)s=1;
      dt=(ms[n-1]-ms[s])/1000.0; db=rb[n-1]-rb[s];
      if(dt<=0){print "n/a"; exit}
      bps=db/dt; printf "%.3f", (bps/tg)/1e9 }' "$iolog")

  echo "---- $tag ----"
  printf "  decode TPS (tg)        : %s\n" "${tg:-?}"
  printf "  prompt TPS (pp)        : %s\n" "${pp:-?}"
  printf "  peak RSS               : %s GiB\n" "${peak_rss:-?}"
  printf "  total disk read_bytes  : %.2f GiB\n" "$(awk -v v="$total_rb" 'BEGIN{print v/1073741824}')"
  printf "  steady bytes/token     : %s GB   <-- compare to ~1.83 (uncached) / ~0 (resident)\n" "${bpt:-?}"
  printf "  io samples log         : %s\n" "$iolog"
  rm -f "$logf"
}

echo "model   : $MODEL"
echo "bin     : $CLI"
echo "threads : $THREADS   n_gen: $NGEN   mode: $MODE"
echo "extra   : $EXTRA"
echo

if [ "$MODE" = "cold" ]; then
  drop_caches
  run_once "COLD (page cache dropped — true SSD streaming)"
else
  echo "warming page cache (one short run)..." >&2
  "$CLI" --model "$MODEL" --ctx-size "$CTX" --threads "$THREADS" -n 8 -st $EXTRA \
         -p "$PROMPT" >/dev/null 2>&1
  run_once "WARM (cache primed — compute + cached-I/O ceiling)"
fi
