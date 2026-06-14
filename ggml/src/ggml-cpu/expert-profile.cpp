#include "expert-profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Online LRU cache simulator over (layer,expert) units. Replaying the real
// access order tells us the hit rate a global expert cache of a given capacity
// would actually get (this is what the kernel page cache approximates).
struct lru_sim {
    size_t                                                cap;
    std::list<uint64_t>                                   order;   // front = MRU
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> pos;
    uint64_t                                              hits     = 0;
    uint64_t                                              accesses = 0;

    explicit lru_sim(size_t c) : cap(c) {}

    void access(uint64_t key) {
        accesses++;
        auto it = pos.find(key);
        if (it != pos.end()) {
            hits++;
            order.erase(it->second);
            order.push_front(key);
            it->second = order.begin();
            return;
        }
        if (pos.size() >= cap) {
            uint64_t victim = order.back();
            order.pop_back();
            pos.erase(victim);
        }
        order.push_front(key);
        pos[key] = order.begin();
    }
};

struct profiler {
    bool                                   enabled = false;
    std::string                            out_path = "/tmp/ds4_expert_profile.txt";
    std::mutex                             mtx;
    std::unordered_map<uint64_t, uint64_t> counts;   // (layer<<32)|expert -> selections
    std::vector<lru_sim>                   sims;      // LRU hit rate at several capacities
    uint64_t                               total_selections = 0;
    uint64_t                               gate_calls       = 0;
    int                                    n_expert         = 0;

    // expert-unit size in MiB (gate IQ2_XXS + up IQ2_XXS + down Q2_K ~= 6.74)
    static constexpr double UNIT_MIB = 6.74;

    profiler() {
        const char * e = getenv("LLAMA_EXPERT_PROFILE");
        if (!e || !*e) {
            return;
        }
        enabled = true;
        if (strchr(e, '/')) {
            out_path = e;
        }
        // simulated global expert-cache capacities (in units); maps to GiB below
        for (size_t c : {128u, 256u, 512u, 1024u, 2048u, 4096u}) {
            sims.emplace_back(c);
        }
    }

    // Dump from the destructor (runs at exit with the object still alive);
    // atexit() from the ctor would be ordered before this object's destruction.
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
        std::vector<std::pair<uint64_t, uint64_t>> v(counts.begin(), counts.end());
        std::sort(v.begin(), v.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        FILE * f = fopen(out_path.c_str(), "w");
        if (!f) {
            return;
        }
        fprintf(f, "# ds4 expert profile\n");
        fprintf(f, "# n_expert=%d total_selections=%llu gate_calls=%llu active_units=%zu unit=%.2fMiB\n",
                n_expert, (unsigned long long) total_selections,
                (unsigned long long) gate_calls, v.size(), UNIT_MIB);

        // (A) Frequency coverage = UPPER BOUND for a hotness/LFU cache: if we
        // pinned the hottest K units, what fraction of selections would hit.
        fprintf(f, "# --- frequency coverage (upper bound: pin hottest units) ---\n");
        for (double fr : {0.05, 0.10, 0.25, 0.50, 0.75, 1.00}) {
            size_t   k = (size_t) (fr * v.size());
            if (k == 0) k = 1;
            uint64_t cum = 0;
            for (size_t i = 0; i < k && i < v.size(); ++i) cum += v[i].second;
            const double hit = total_selections ? (double) cum / total_selections : 0.0;
            fprintf(f, "#   pin top %5.1f%% units (%4zu, %6.1f GiB) -> %5.1f%% hit\n",
                    fr * 100.0, k, k * UNIT_MIB / 1024.0, hit * 100.0);
        }

        // (B) LRU hit rate = what the kernel page cache realistically achieves at
        // a fixed cache size, on the real temporal access order. The GAP between
        // (A) and (B) at the same GiB is the upside of an explicit hotness cache.
        fprintf(f, "# --- LRU hit rate (realistic page-cache-equivalent) ---\n");
        for (const auto & s : sims) {
            const double hit = s.accesses ? (double) s.hits / s.accesses : 0.0;
            fprintf(f, "#   cache %5zu units (%6.1f GiB) -> %5.1f%% hit (LRU)\n",
                    s.cap, s.cap * UNIT_MIB / 1024.0, hit * 100.0);
        }

        fprintf(f, "# --- per-unit counts: layer expert selections ---\n");
        for (const auto & kv : v) {
            fprintf(f, "%u %u %llu\n", (uint32_t) (kv.first >> 32),
                    (uint32_t) (kv.first & 0xffffffffu), (unsigned long long) kv.second);
        }
        fclose(f);
        fprintf(stderr, "ggml: wrote expert profile to %s (%llu selections, %zu units)\n",
                out_path.c_str(), (unsigned long long) total_selections, v.size());

        // Write companion .hotlist (sorted "layer expert" lines, no comments) for
        // auto-discovery with --ssd-stream. The loader looks for <model>.hotlist.
        const std::string hotlist_path = out_path + ".hotlist";
        FILE * fh = fopen(hotlist_path.c_str(), "w");
        if (fh) {
            for (const auto & kv : v) {
                fprintf(fh, "%u %u\n", (uint32_t) (kv.first >> 32),
                        (uint32_t) (kv.first & 0xffffffffu));
            }
            fclose(fh);
            fprintf(stderr, "ggml: wrote hotlist companion to %s (%zu units)\n",
                    hotlist_path.c_str(), v.size());
        }
    }
};

int parse_layer(const char * name) {
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4;
    if (*p < '0' || *p > '9') return -1;
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
    // separate mul_mat_id and shares the routing of gate/up.
    if (!strstr(tensor_name, "ffn_down_exps")) {
        return;
    }
    const int layer = parse_layer(tensor_name);
    if (layer < 0) {
        return;
    }

    std::lock_guard<std::mutex> lk(p.mtx);
    p.n_expert = n_expert;
    p.gate_calls += (uint64_t) ne1;
    const char * base = (const char *) ids_data;
    // token-major replay so the LRU sim sees a realistic access order
    for (int64_t t = 0; t < ne1; ++t) {
        for (int64_t i = 0; i < ne0; ++i) {
            const int32_t e = *(const int32_t *) (base + t * nb1 + i * nb0);
            if (e < 0 || e >= n_expert) {
                continue;
            }
            const uint64_t key = ((uint64_t) (uint32_t) layer << 32) | (uint32_t) e;
            p.counts[key]++;
            p.total_selections++;
            for (auto & s : p.sims) {
                s.access(key);
            }
        }
    }
}
