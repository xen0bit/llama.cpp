#include "expert-profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct profiler {
    bool                                  enabled = false;
    std::string                           out_path = "/tmp/ds4_expert_profile.txt";
    std::mutex                            mtx;
    std::unordered_map<uint64_t, uint64_t> counts;   // key = (layer<<32)|expert -> selections
    uint64_t                             total_selections = 0;
    uint64_t                             gate_calls       = 0; // ~= tokens * moe_layers
    int                                  n_expert         = 0;

    profiler() {
        const char * e = getenv("LLAMA_EXPERT_PROFILE");
        if (!e || !*e) {
            return;
        }
        enabled = true;
        if (strchr(e, '/')) {
            out_path = e; // treat value as an output path if it looks like one
        }
    }

    // Dump from the destructor (runs at program exit with the object still
    // alive). Using atexit() from the ctor would be LIFO-ordered *before* this
    // object's own destruction, reading a destroyed map.
    ~profiler() { dump(); }

    static profiler & get() {
        static profiler p;
        return p;
    }

    void dump() {
        std::lock_guard<std::mutex> lk(mtx);
        if (counts.empty()) {
            return;
        }
        // Flatten and sort expert-units by selection count (descending).
        std::vector<std::pair<uint64_t, uint64_t>> v(counts.begin(), counts.end());
        std::sort(v.begin(), v.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        FILE * f = fopen(out_path.c_str(), "w");
        if (!f) {
            return;
        }
        fprintf(f, "# ds4 expert profile\n");
        fprintf(f, "# n_expert=%d total_selections=%llu gate_calls=%llu active_units=%zu\n",
                n_expert, (unsigned long long) total_selections,
                (unsigned long long) gate_calls, v.size());

        // Skew curve: what fraction of all selections is covered by the hottest
        // X% of expert-units. This is the cache hit-rate-vs-size curve: caching
        // the top X% of units yields ~that hit rate. Steep => caching wins big.
        fprintf(f, "# --- coverage curve (cache top X%% of units -> hit rate) ---\n");
        const double frac[] = {0.05, 0.10, 0.25, 0.50, 0.75, 1.00};
        for (double fr : frac) {
            size_t   k = (size_t) (fr * v.size());
            if (k == 0) k = 1;
            uint64_t cum = 0;
            for (size_t i = 0; i < k && i < v.size(); ++i) {
                cum += v[i].second;
            }
            const double hit = total_selections ? (double) cum / total_selections : 0.0;
            fprintf(f, "#   top %5.1f%% units (%4zu) -> %5.1f%% of selections\n",
                    fr * 100.0, k, hit * 100.0);
        }

        fprintf(f, "# --- per-unit counts: layer expert selections ---\n");
        for (const auto & kv : v) {
            const uint32_t layer  = (uint32_t) (kv.first >> 32);
            const uint32_t expert = (uint32_t) (kv.first & 0xffffffffu);
            fprintf(f, "%u %u %llu\n", layer, expert, (unsigned long long) kv.second);
        }
        fclose(f);
        fprintf(stderr, "ggml: wrote expert profile to %s (%llu selections over %zu units)\n",
                out_path.c_str(), (unsigned long long) total_selections, v.size());
    }
};

// parse "blk.<N>." -> N, or -1 if not found
int parse_layer(const char * name) {
    const char * p = strstr(name, "blk.");
    if (!p) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    return atoi(p);
}

} // namespace

void ggml_expert_profile_record(const char * tensor_name, int n_expert,
                                const void * ids_data,
                                int64_t ne0, int64_t ne1,
                                size_t nb0, size_t nb1) {
    profiler & p = profiler::get();
    if (!p.enabled || !tensor_name || !ids_data) {
        return;
    }
    // Count once per layer per call on the down projection: it is always a
    // separate mul_mat_id and shares the same routing as gate/up.
    if (!strstr(tensor_name, "ffn_down_exps")) {
        return;
    }
    const int layer = parse_layer(tensor_name);
    if (layer < 0) {
        return;
    }

    std::lock_guard<std::mutex> lk(p.mtx);
    p.n_expert = n_expert;
    p.gate_calls += (uint64_t) ne1; // one "token route" per token in this call
    const char * base = (const char *) ids_data;
    for (int64_t t = 0; t < ne1; ++t) {
        for (int64_t i = 0; i < ne0; ++i) {
            const int32_t e = *(const int32_t *) (base + t * nb1 + i * nb0);
            if (e < 0 || e >= n_expert) {
                continue;
            }
            const uint64_t key = ((uint64_t) (uint32_t) layer << 32) | (uint32_t) e;
            p.counts[key]++;
            p.total_selections++;
        }
    }
}
