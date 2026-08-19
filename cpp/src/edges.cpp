#include "denovo/edges.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace denovo {
namespace {

// edges.py AA_MASSES.  I and L are equal, so 19 distinct quantised values.
constexpr std::array<double, 20> kAA = {
    71.03711, 156.10111, 114.04293, 115.02694, 103.00919, 129.04259,
    128.05858,  57.02146, 137.05891, 113.08406, 113.08406, 128.09496,
    131.04049, 147.06841,  97.05276,  87.03203, 101.04768, 186.07931,
    163.06333,  99.06841};

constexpr int kPrecision = 3;               // 0.001 Da
constexpr double kScale  = 1000.0;

struct Bitset {
    std::vector<uint64_t> w;
    explicit Bitset(size_t bits) : w((bits + 63) / 64, 0ULL) {}
    void set(size_t i)        { w[i >> 6] |= 1ULL << (i & 63); }
    bool test(size_t i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
    void or_with(const Bitset& o) {
        for (size_t k = 0; k < w.size(); ++k) w[k] |= o.w[k];
    }
    // this |= (src << shift)
    void or_shifted(const Bitset& src, size_t shift) {
        const size_t word = shift >> 6, bit = shift & 63;
        const size_t n = w.size();
        if (bit == 0) {
            for (size_t k = word; k < n; ++k) w[k] |= src.w[k - word];
        } else {
            for (size_t k = word; k < n; ++k) {
                uint64_t v = src.w[k - word] << bit;
                if (k > word) v |= src.w[k - word - 1] >> (64 - bit);
                w[k] |= v;
            }
        }
    }
};

}  // namespace

std::vector<double> build_edges(int max_n) {
    if (max_n < 1) throw std::invalid_argument("edges: max_n must be >= 1");

    // Quantise, then dedup (I/L collapse) -- edges.py's `list1` is a set.
    std::vector<size_t> aa;
    for (double m : kAA) {
        size_t q = static_cast<size_t>(std::llround(m * kScale));
        bool seen = false;
        for (size_t v : aa) if (v == q) { seen = true; break; }
        if (!seen) aa.push_back(q);
    }
    size_t max_aa = 0;
    for (size_t q : aa) max_aa = std::max(max_aa, q);

    const size_t bits = max_aa * static_cast<size_t>(max_n) + 1;
    Bitset cur(bits), cumulative(bits);
    for (size_t q : aa) cur.set(q);
    cumulative.or_with(cur);

    // List_n = List_{n-1} (+) List_1, the Minkowski sum edges.py takes.
    for (int n = 2; n <= max_n; ++n) {
        Bitset next(bits);
        for (size_t q : aa) next.or_shifted(cur, q);
        cur = std::move(next);
        cumulative.or_with(cur);
    }

    std::vector<double> out;
    out.reserve(1u << 21);
    for (size_t i = 0; i < bits; ++i)
        if (cumulative.test(i)) out.push_back(static_cast<double>(i) / kScale);
    return out;
}

std::vector<double> load_edges_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open edges file: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();

    const std::string key = "\"all_masses\"";
    size_t p = s.find(key);
    if (p == std::string::npos)
        throw std::runtime_error("no \"all_masses\" array in " + path);
    p = s.find('[', p);
    size_t end = s.find(']', p);
    if (p == std::string::npos || end == std::string::npos)
        throw std::runtime_error("malformed \"all_masses\" array in " + path);

    std::vector<double> out;
    out.reserve(1u << 21);
    const char* cur = s.data() + p + 1;
    const char* stop = s.data() + end;
    while (cur < stop) {
        char* next = nullptr;
        double v = std::strtod(cur, &next);
        if (next == cur) { ++cur; continue; }
        out.push_back(v);
        cur = next;
        while (cur < stop && (*cur == ',' || *cur == ' ' || *cur == '\n')) ++cur;
    }
    return out;
}

}  // namespace denovo
