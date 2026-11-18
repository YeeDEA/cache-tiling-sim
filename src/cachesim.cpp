// cachesim.cpp - set-associative cache simulator with 3C miss classification
// and matrix-multiply access-pattern workloads (naive / ikj / tiled).
//
// Build: make        Run: ./cachesim --help
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using u64 = uint64_t;

// Unlike assert(), CHECK is never compiled out (-DNDEBUG), so --selftest cannot pass vacuously.
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "selftest FAILED: %s (line %d)\n", #cond, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct Stats {
    u64 reads = 0, writes = 0, hits = 0, misses = 0;
    u64 compulsory = 0, capacity = 0, conflict = 0, writebacks = 0;
};

// Fully-associative LRU cache with the same number of blocks as the real cache.
// A miss in the real cache that HITS here is a conflict miss; a miss in both
// (on a block that was seen before) is a capacity miss.
class FullyAssocLRU {
    size_t cap_;
    std::list<u64> lru_;
    std::unordered_map<u64, std::list<u64>::iterator> pos_;

public:
    explicit FullyAssocLRU(size_t cap) : cap_(cap) {}
    bool access(u64 blk) {
        auto it = pos_.find(blk);
        if (it != pos_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            return true;
        }
        lru_.push_front(blk);
        pos_[blk] = lru_.begin();
        if (lru_.size() > cap_) {
            pos_.erase(lru_.back());
            lru_.pop_back();
        }
        return false;
    }
};

static bool is_pow2(u64 x) { return x && !(x & (x - 1)); }
static unsigned log2u(u64 x) {
    unsigned r = 0;
    while (x >>= 1) ++r;
    return r;
}

enum class Policy { LRU, FIFO, RANDOM, PLRU };

static bool parse_policy(const std::string& s, Policy& p) {
    if (s == "lru") p = Policy::LRU;
    else if (s == "fifo") p = Policy::FIFO;
    else if (s == "random") p = Policy::RANDOM;
    else if (s == "plru") p = Policy::PLRU;
    else return false;
    return true;
}

// Write-back, write-allocate set-associative cache with selectable replacement
// (true LRU, FIFO, random with a fixed seed, or tree pseudo-LRU).
// The 3C classifier always uses a fully-associative *LRU* shadow, so for
// non-LRU policies "conflict" means "a FA-LRU cache of equal size would have hit".
class Cache {
    struct Line {
        bool valid = false, dirty = false;
        u64 tag = 0, lru = 0, ins = 0;  // last-use tick, insertion tick
    };
    size_t sets_, assoc_;
    unsigned off_bits_, idx_bits_, way_bits_;
    Policy policy_;
    std::vector<Line> lines_;
    std::vector<uint8_t> plru_;  // per set: assoc-1 tree bits (heap order, node 1 = root)
    u64 tick_ = 0, rng_ = 0x9E3779B97F4A7C15ULL;
    FullyAssocLRU shadow_;
    std::unordered_set<u64> seen_;
    Stats st_;

    u64 next_rand() {  // xorshift64: deterministic, so results are reproducible
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 7;
        rng_ ^= rng_ << 17;
        return rng_;
    }
    // Make every tree node on the path to way w point away from w.
    void plru_touch(u64 set, size_t w) {
        uint8_t* t = &plru_[set * assoc_];
        size_t node = 1;
        for (unsigned l = 0; l < way_bits_; ++l) {
            const size_t bit = (w >> (way_bits_ - 1 - l)) & 1;
            t[node] = uint8_t(!bit);
            node = 2 * node + bit;
        }
    }
    size_t plru_victim(u64 set) const {
        const uint8_t* t = &plru_[set * assoc_];
        size_t node = 1, w = 0;
        for (unsigned l = 0; l < way_bits_; ++l) {
            w = (w << 1) | t[node];
            node = 2 * node + t[node];
        }
        return w;
    }

public:
    Cache(size_t size, size_t assoc, size_t block, Policy policy = Policy::LRU)
        : sets_(size / (assoc * block)),
          assoc_(assoc),
          off_bits_(log2u(block)),
          idx_bits_(log2u(size / (assoc * block))),
          way_bits_(log2u(assoc)),
          policy_(policy),
          lines_((size / (assoc * block)) * assoc),
          plru_((size / (assoc * block)) * assoc, 0),
          shadow_(size / block) {
        if (!is_pow2(size) || !is_pow2(assoc) || !is_pow2(block) || size < assoc * block) {
            std::fprintf(stderr, "size/assoc/block must be powers of two and size >= assoc*block\n");
            std::exit(2);
        }
    }

    void access(u64 addr, bool write) {
        (write ? st_.writes : st_.reads)++;
        const u64 blk = addr >> off_bits_;
        const u64 set = blk & (sets_ - 1);
        const u64 tag = blk >> idx_bits_;
        const bool shadow_hit = shadow_.access(blk);  // keep shadow in sync on every access
        Line* base = &lines_[set * assoc_];
        ++tick_;

        for (size_t w = 0; w < assoc_; ++w) {
            if (base[w].valid && base[w].tag == tag) {
                ++st_.hits;
                base[w].lru = tick_;
                if (policy_ == Policy::PLRU) plru_touch(set, w);
                if (write) base[w].dirty = true;
                return;
            }
        }

        ++st_.misses;
        if (!seen_.count(blk))
            ++st_.compulsory;
        else if (shadow_hit)
            ++st_.conflict;
        else
            ++st_.capacity;
        seen_.insert(blk);

        size_t victim = 0;
        bool found_invalid = false;
        for (size_t w = 0; w < assoc_; ++w) {
            if (!base[w].valid) {
                victim = w;
                found_invalid = true;
                break;
            }
        }
        if (!found_invalid) {
            switch (policy_) {
            case Policy::LRU:
                for (size_t w = 1; w < assoc_; ++w)
                    if (base[w].lru < base[victim].lru) victim = w;
                break;
            case Policy::FIFO:
                for (size_t w = 1; w < assoc_; ++w)
                    if (base[w].ins < base[victim].ins) victim = w;
                break;
            case Policy::RANDOM:
                victim = size_t(next_rand() % assoc_);
                break;
            case Policy::PLRU:
                victim = plru_victim(set);
                break;
            }
            if (base[victim].dirty) ++st_.writebacks;
        }
        base[victim] = {true, write, tag, tick_, tick_};
        if (policy_ == Policy::PLRU) plru_touch(set, victim);
    }

    void read(u64 a) { access(a, false); }
    void write(u64 a) { access(a, true); }
    const Stats& stats() const { return st_; }
};

struct Config {
    size_t size = 32768, assoc = 8, block = 64;
    size_t n = 256, tile = 32, pad = 0, elem = 8, bits = 64;
    std::string workload = "naive", trace;
    Policy policy = Policy::LRU;
    double hit_time = 1.0, penalty = 100.0;
    bool selftest = false, csv = false;
};

// Matrices are row-major with element width --bits bits (default 64 = double;
// supports 64/32/16/8/4 for fp64..packed int4) and leading dimension ld = n + pad.
static void run_matmul(Cache& c, const Config& cfg) {
    const u64 n = cfg.n, ld = cfg.n + cfg.pad, B_bits = cfg.bits;
    const u64 matrix_bytes = (n * ld * B_bits + 7) / 8;
    const u64 A = 0x10000000ULL, B = A + matrix_bytes, C = B + matrix_bytes;
    auto a = [&](u64 i, u64 k) { return A + (((i * ld + k) * B_bits) >> 3); };
    auto b = [&](u64 k, u64 j) { return B + (((k * ld + j) * B_bits) >> 3); };
    auto cc = [&](u64 i, u64 j) { return C + (((i * ld + j) * B_bits) >> 3); };

    if (cfg.workload == "naive") {
        // ijk, accumulator kept in a register, C stored once per (i,j)
        for (u64 i = 0; i < n; ++i)
            for (u64 j = 0; j < n; ++j) {
                for (u64 k = 0; k < n; ++k) {
                    c.read(a(i, k));
                    c.read(b(k, j));
                }
                c.write(cc(i, j));
            }
    } else if (cfg.workload == "ikj") {
        // loop interchange: inner loop walks rows of B and C (unit stride)
        for (u64 i = 0; i < n; ++i)
            for (u64 k = 0; k < n; ++k) {
                c.read(a(i, k));
                for (u64 j = 0; j < n; ++j) {
                    c.read(b(k, j));
                    c.read(cc(i, j));
                    c.write(cc(i, j));
                }
            }
    } else if (cfg.workload == "tiled") {
        // blocked ijk: partial sums of C are loaded/stored once per (i,j,kk-tile)
        const u64 T = cfg.tile;
        for (u64 ii = 0; ii < n; ii += T)
            for (u64 jj = 0; jj < n; jj += T)
                for (u64 kk = 0; kk < n; kk += T)
                    for (u64 i = ii; i < std::min(ii + T, n); ++i)
                        for (u64 j = jj; j < std::min(jj + T, n); ++j) {
                            c.read(cc(i, j));
                            for (u64 k = kk; k < std::min(kk + T, n); ++k) {
                                c.read(a(i, k));
                                c.read(b(k, j));
                            }
                            c.write(cc(i, j));
                        }
    } else {
        std::fprintf(stderr, "unknown workload '%s' (naive|ikj|tiled)\n", cfg.workload.c_str());
        std::exit(2);
    }
}

// Trace format: one access per line, "R <addr>" or "W <addr>" (addr decimal or 0x-hex).
static void run_trace(Cache& c, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open trace '%s'\n", path.c_str());
        std::exit(2);
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string op, addr;
        if (!(ss >> op >> addr)) continue;
        u64 a = std::stoull(addr, nullptr, 0);
        c.access(a, op[0] == 'W' || op[0] == 'w');
    }
}

static int selftest() {
    {  // direct-mapped, 4 sets of 64 B: addresses 0 and 256 collide in set 0
        Cache c(256, 1, 64);
        c.read(0);    // miss (compulsory)
        c.read(0);    // hit
        c.read(256);  // miss (compulsory)
        c.read(0);    // miss: evicted by 256, but a 4-block FA cache would hold both -> conflict
        const Stats& s = c.stats();
        CHECK(s.hits == 1 && s.misses == 3);
        CHECK(s.compulsory == 2 && s.conflict == 1 && s.capacity == 0);
    }
    {  // fully associative, 2 blocks: 0,64,128 then 0 again was evicted by LRU -> capacity
        Cache c(128, 2, 64);
        c.read(0);
        c.read(64);
        c.read(128);
        c.read(0);
        const Stats& s = c.stats();
        CHECK(s.misses == 4 && s.compulsory == 3 && s.capacity == 1 && s.conflict == 0);
    }
    {  // write-back: dirty line evicted -> one writeback
        Cache c(64, 1, 64);  // a single block
        c.write(0);
        c.read(64);
        CHECK(c.stats().writebacks == 1);
    }
    {  // padding breaks power-of-two stride conflict misses
        Config c0, c8;
        c0.size = 16384; c0.assoc = 8; c0.block = 64; c0.n = 128; c0.pad = 0; c0.bits = 64;
        c8 = c0; c8.pad = 8;
        Cache ca(c0.size, c0.assoc, c0.block), cb(c8.size, c8.assoc, c8.block);
        run_matmul(ca, c0);
        run_matmul(cb, c8);
        CHECK(ca.stats().conflict > 0);
        CHECK(cb.stats().conflict == 0);
        CHECK(ca.stats().writes == 128ULL * 128ULL);
    }
    {  // packed 4-bit (INT4): 128 elements fit in one 64 B cache line
        Config cfg;
        cfg.size = 1024; cfg.assoc = 4; cfg.block = 64; cfg.n = 16; cfg.pad = 0; cfg.bits = 4;
        Cache c(cfg.size, cfg.assoc, cfg.block);
        const u64 base = 0x10000000ULL;
        for (u64 k = 0; k < 128; ++k) c.read(base + ((k * cfg.bits) >> 3));
        CHECK(c.stats().compulsory == 1 && c.stats().hits == 127);
        c.read(base + ((128ULL * cfg.bits) >> 3));
        CHECK(c.stats().compulsory == 2 && c.stats().misses == 2);
    }
    {  // partial-tile boundary: N=65 with T=32 executes 3 kk-tiles cleanly without out-of-bounds
        Config cfg;
        cfg.workload = "tiled"; cfg.n = 65; cfg.tile = 32; cfg.pad = 8; cfg.bits = 32;
        Cache c(cfg.size, cfg.assoc, cfg.block);
        run_matmul(c, cfg);
        CHECK(c.stats().writes == 3ULL * 65ULL * 65ULL);
    }
    {  // one 2-way set: 0,64,0,128,0. LRU evicts 64 (0 was reused) -> 0 hits;
       // FIFO evicts 0 (inserted first) -> 0 misses. 2-way tree-PLRU == LRU.
        const u64 seq[] = {0, 64, 0, 128, 0};
        Cache lru(128, 2, 64, Policy::LRU), fifo(128, 2, 64, Policy::FIFO), plru(128, 2, 64, Policy::PLRU);
        for (u64 a : seq) { lru.read(a); fifo.read(a); plru.read(a); }
        CHECK(lru.stats().misses == 3 && fifo.stats().misses == 4 && plru.stats().misses == 3);
    }
    {  // 4-way tree-PLRU differs from true LRU. Fill ways 0..3 with 0,64,128,192, then hit 0.
       // True LRU victim = way 1 (64). PLRU: touching way 0 points the root at ways 2/3, and
       // node 3 still points at way 2 (way 3 was touched last) -> victim = way 2 (128).
       // So the following access to 128 hits under LRU but misses under PLRU.
        Cache p(256, 4, 64, Policy::PLRU), l(256, 4, 64, Policy::LRU);
        const u64 seq[] = {0, 64, 128, 192, 0, 256, 128};
        for (u64 a : seq) { p.read(a); l.read(a); }
        CHECK(l.stats().misses == 5 && p.stats().misses == 6);
    }
    {  // cyclic access to assoc+1 blocks in one set: LRU and FIFO miss every time,
       // random replacement keeps some blocks resident (fixed seed -> deterministic).
        Cache l(256, 4, 64, Policy::LRU), f(256, 4, 64, Policy::FIFO), r(256, 4, 64, Policy::RANDOM);
        for (int rep = 0; rep < 20; ++rep)
            for (u64 b = 0; b < 5; ++b) { l.read(b * 256); f.read(b * 256); r.read(b * 256); }
        CHECK(l.stats().misses == 100 && f.stats().misses == 100);
        CHECK(r.stats().misses < 100 && r.stats().misses >= 5);
    }
    std::puts("selftest passed");
    return 0;
}

static void usage() {
    std::puts(
        "usage: cachesim [options]\n"
        "  --workload naive|ikj|tiled   matmul access pattern (default naive)\n"
        "  --n N --tile T --pad P       matrix size, tile size, row padding in elements\n"
        "  --elem B                     bytes per element (8=fp64, 4=fp32, 2=fp16, 1=int8)\n"
        "  --bits B                     bits per element (64, 32, 16, 8, or 4=packed int4)\n"
        "  --size B --assoc W --block B cache bytes, ways, block bytes (32768 / 8 / 64)\n"
        "  --hit H --penalty P          hit time and miss penalty in cycles (1 / 100)\n"
        "  --policy lru|fifo|random|plru replacement policy (default lru; random uses a fixed seed)\n"
        "  --trace FILE                 replay a trace (\"R 0x..\" / \"W 0x..\") instead of a workload\n"
        "  --csv                        one-line CSV output\n"
        "  --selftest                   run built-in sanity tests");
}

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--workload") cfg.workload = next();
        else if (a == "--n") cfg.n = std::stoull(next());
        else if (a == "--tile") cfg.tile = std::stoull(next());
        else if (a == "--pad") cfg.pad = std::stoull(next());
        else if (a == "--elem") {
            cfg.elem = std::stoull(next());
            cfg.bits = cfg.elem * 8;
        } else if (a == "--bits") {
            cfg.bits = std::stoull(next());
        }
        else if (a == "--size") cfg.size = std::stoull(next());
        else if (a == "--assoc") cfg.assoc = std::stoull(next());
        else if (a == "--block") cfg.block = std::stoull(next());
        else if (a == "--hit") cfg.hit_time = std::stod(next());
        else if (a == "--penalty") cfg.penalty = std::stod(next());
        else if (a == "--trace") cfg.trace = next();
        else if (a == "--policy") {
            const std::string p = next();
            if (!parse_policy(p, cfg.policy)) {
                std::fprintf(stderr, "unknown policy '%s' (lru|fifo|random|plru)\n", p.c_str());
                return 2;
            }
        }
        else if (a == "--csv") cfg.csv = true;
        else if (a == "--selftest") cfg.selftest = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (cfg.selftest) return selftest();
    if (cfg.tile == 0 || cfg.n == 0 || (cfg.bits != 4 && cfg.bits != 8 && cfg.bits != 16 && cfg.bits != 32 && cfg.bits != 64)) {
        std::fprintf(stderr, "n and tile must be > 0, and bits must be 64/32/16/8/4 (or elem 8/4/2/1)\n");
        return 2;
    }

    Cache cache(cfg.size, cfg.assoc, cfg.block, cfg.policy);
    if (!cfg.trace.empty()) {
        run_trace(cache, cfg.trace);
        cfg.workload = "trace";
    } else {
        run_matmul(cache, cfg);
    }

    const Stats& s = cache.stats();
    const u64 total = s.reads + s.writes;
    const double miss_rate = total ? double(s.misses) / double(total) : 0.0;
    const double amat = cfg.hit_time + miss_rate * cfg.penalty;

    if (cfg.csv) {
        std::printf("%s,%zu,%zu,%zu,%llu,%.4f,%llu,%llu,%llu,%llu,%.3f\n", cfg.workload.c_str(), cfg.n,
                    cfg.tile, cfg.pad, (unsigned long long)s.misses, 100.0 * miss_rate,
                    (unsigned long long)s.compulsory, (unsigned long long)s.capacity,
                    (unsigned long long)s.conflict, (unsigned long long)s.writebacks, amat);
        return 0;
    }
    std::printf("config: %zu B, %zu-way, %zu B blocks (%zu sets)\n", cfg.size, cfg.assoc, cfg.block,
                cfg.size / (cfg.assoc * cfg.block));
    std::printf("accesses: %llu (R %llu / W %llu)\n", (unsigned long long)total,
                (unsigned long long)s.reads, (unsigned long long)s.writes);
    std::printf("hits: %llu  misses: %llu  miss rate: %.3f%%\n", (unsigned long long)s.hits,
                (unsigned long long)s.misses, 100.0 * miss_rate);
    std::printf("  compulsory: %llu  capacity: %llu  conflict: %llu\n", (unsigned long long)s.compulsory,
                (unsigned long long)s.capacity, (unsigned long long)s.conflict);
    std::printf("writebacks: %llu\n", (unsigned long long)s.writebacks);
    std::printf("AMAT: %.3f cycles (hit %.1f + %.4f x %.0f)\n", amat, cfg.hit_time, miss_rate, cfg.penalty);
    return 0;
}
