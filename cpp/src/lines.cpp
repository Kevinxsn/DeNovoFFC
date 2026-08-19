#include "denovo/lines.hpp"
#include "denovo/constants.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <unordered_set>

namespace denovo {
namespace {

// greedy_line._enumerate_charges: i = 1..z-1, j = 1..z-i.  Both (i,j) and
// (j,i) appear when i != j, which is what makes the pooling orientation-aware
// before the canonicalisation collapses them again.
std::vector<std::pair<int,int>> enumerate_charges(int z, bool parental_only) {
    std::vector<std::pair<int,int>> out;
    for (int i = 1; i < z; ++i)
        for (int j = 1; j <= z - i; ++j)
            if (!parental_only || i + j == z) out.emplace_back(i, j);
    return out;
}

// _sort_and_split_oa: single-linkage gap clustering.  Returns clusters as
// runs over `order` (ascending mass), so clusters arrive lowest-mass first --
// which is what makes the "ties go to the lowest-mass cluster" rule fall out.
struct Clustering {
    std::vector<size_t> order;        // positions, ascending by value
    std::vector<size_t> starts;       // cluster k == order[starts[k] .. starts[k+1])
};

Clustering sort_and_split(const std::vector<double>& v, double delta,
                          bool ppm) {
    Clustering c;
    const size_t n = v.size();
    if (n == 0) return c;
    c.order.resize(n);
    std::iota(c.order.begin(), c.order.end(), size_t{0});
    std::stable_sort(c.order.begin(), c.order.end(),
                     [&](size_t a, size_t b) { return v[a] < v[b]; });
    c.starts.push_back(0);
    for (size_t k = 0; k + 1 < n; ++k) {
        const double lo  = v[c.order[k]];
        const double gap = v[c.order[k + 1]] - lo;
        const double thr = ppm ? std::fabs(lo) * delta * 1e-6 : delta;
        if (gap > thr) c.starts.push_back(k + 1);
    }
    c.starts.push_back(n);
    return c;
}

struct Pool {
    std::vector<double> mass;
    std::vector<int>    lo, hi;       // canonical charge (min, max)
    std::vector<size_t> row;          // index into `alive`
};

Pool build_pool(const FfcTable& t, const std::vector<size_t>& alive,
                const std::vector<std::pair<int,int>>& pairs) {
    Pool p;
    const size_t n = alive.size();
    p.mass.reserve(n * pairs.size());
    p.lo.reserve(n * pairs.size());
    p.hi.reserve(n * pairs.size());
    p.row.reserve(n * pairs.size());
    for (const auto& ij : pairs) {
        const int i = ij.first, j = ij.second;
        const int lo = std::min(i, j), hi = std::max(i, j);
        for (size_t k = 0; k < n; ++k) {
            const size_t r = alive[k];
            p.mass.push_back(i * t.mz_a[r] + j * t.mz_b[r]);
            p.lo.push_back(lo);
            p.hi.push_back(hi);
            p.row.push_back(k);
        }
    }
    return p;
}

// _most_informative_line_oa.  Returns false when no cluster reaches
// min_ffc_number distinct FFCs.
bool most_informative_line(const FfcTable& t, const std::vector<size_t>& alive,
                           int parent_charge, const DetectParams& prm,
                           bool parental_only, Line& out) {
    if (alive.empty()) return false;
    const auto pairs = enumerate_charges(parent_charge, parental_only);
    if (pairs.empty()) return false;

    const Pool pool = build_pool(t, alive, pairs);
    const bool ppm = (prm.delta_unit == "ppm");
    const Clustering cl = sort_and_split(pool.mass, prm.delta, ppm);
    if (cl.starts.size() < 2) return false;

    long   best_distinct = -1;
    double best_spread   = 0.0;
    size_t best_cluster  = 0;
    int    best_lo = 0, best_hi = 0;
    bool   have_best = false;

    std::map<std::pair<int,int>, int> counts;
    std::unordered_set<size_t> seen;

    for (size_t c = 0; c + 1 < cl.starts.size(); ++c) {
        const size_t b = cl.starts[c], e = cl.starts[c + 1];

        // Dominant CANONICAL charge, by (count, i+j, -i).  Ties on that whole
        // key can only happen between identical keys, so this is
        // order-independent despite Python's max() taking the first maximum.
        counts.clear();
        for (size_t k = b; k < e; ++k)
            ++counts[{pool.lo[cl.order[k]], pool.hi[cl.order[k]]}];
        std::pair<int,int> dom{0, 0};
        long dom_count = -1, dom_sum = 0, dom_neg_lo = 0;
        for (const auto& kv : counts) {
            const long cnt = kv.second;
            const long sum = kv.first.first + kv.first.second;
            const long nlo = -kv.first.first;
            if (cnt > dom_count ||
                (cnt == dom_count && (sum > dom_sum ||
                 (sum == dom_sum && nlo > dom_neg_lo)))) {
                dom = kv.first; dom_count = cnt; dom_sum = sum; dom_neg_lo = nlo;
            }
        }

        seen.clear();
        double mn = 0.0, mx = 0.0;
        bool any = false;
        for (size_t k = b; k < e; ++k) {
            const size_t p = cl.order[k];
            if (pool.lo[p] != dom.first || pool.hi[p] != dom.second) continue;
            seen.insert(pool.row[p]);
            if (!any) { mn = mx = pool.mass[p]; any = true; }
            else { mn = std::min(mn, pool.mass[p]); mx = std::max(mx, pool.mass[p]); }
        }
        const long   distinct = static_cast<long>(seen.size());
        const double spread   = any ? (mx - mn) : 0.0;

        // key = (distinct, -spread), strictly greater -> lowest-mass cluster
        // wins a tie, because clusters arrive in ascending mass order.
        if (!have_best || distinct > best_distinct ||
            (distinct == best_distinct && -spread > -best_spread)) {
            have_best = true;
            best_distinct = distinct;
            best_spread   = spread;
            best_cluster  = c;
            best_lo = dom.first; best_hi = dom.second;
        }
    }

    if (!have_best || best_distinct < prm.min_ffc_number) return false;

    const size_t b = cl.starts[best_cluster], e = cl.starts[best_cluster + 1];
    std::vector<size_t> member_rows;
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t k = b; k < e; ++k) {
        const size_t p = cl.order[k];
        if (pool.lo[p] != best_lo || pool.hi[p] != best_hi) continue;
        member_rows.push_back(pool.row[p]);
        sum += pool.mass[p];                 // mean includes BOTH orientations
        ++cnt;                               // of a doubly-present FFC
    }
    std::sort(member_rows.begin(), member_rows.end());
    member_rows.erase(std::unique(member_rows.begin(), member_rows.end()),
                      member_rows.end());

    out = Line{};
    out.i = best_lo;
    out.j = best_hi;
    out.mass = sum / static_cast<double>(cnt);
    out.cluster_size = static_cast<int>(e - b);
    out.n_on_line = static_cast<int>(best_distinct);
    out.members.reserve(member_rows.size());
    for (size_t k : member_rows) out.members.push_back(alive[k]);
    return true;
}

// _remove_lines_oa: drop every FFC lying on any of `lines` in EITHER
// orientation, plus the recorded member indices.
std::vector<size_t> remove_lines(const FfcTable& t,
                                 const std::vector<size_t>& alive,
                                 const std::vector<Line>& lines,
                                 double tol, bool ppm) {
    std::unordered_set<size_t> drop;
    for (const Line& ln : lines) {
        const double s = ppm ? std::fabs(ln.mass) * tol * 1e-6 : tol;
        for (size_t r : alive) {
            const double v1 = ln.i * t.mz_a[r] + ln.j * t.mz_b[r];
            const double v2 = ln.j * t.mz_a[r] + ln.i * t.mz_b[r];
            if (std::fabs(v1 - ln.mass) <= s || std::fabs(v2 - ln.mass) <= s)
                drop.insert(r);
        }
        for (size_t m : ln.members) drop.insert(m);
    }
    std::vector<size_t> out;
    out.reserve(alive.size());
    for (size_t r : alive) if (!drop.count(r)) out.push_back(r);
    return out;
}

// greedy_line.satellite_lines: same (i, j), mass offset only.  These carry no
// members -- they exist purely so the parental ladder is cleared together with
// its isotope / neutral echoes, which is what stops greedy peeling from
// splitting the ladder's evidence across them.
std::vector<Line> satellite_lines(const std::vector<Line>& parental) {
    static const double losses[3] = {SAT_H2O, SAT_NH3, SAT_CO};
    std::vector<Line> sats;
    for (const Line& p : parental) {
        for (int k = 1; k <= 3; ++k) {
            Line s; s.i = p.i; s.j = p.j;
            s.mass = p.mass + k * ISOTOPE_SPACING;
            s.kind = "isotopic";
            sats.push_back(s);
        }
        for (double d : losses) {
            Line s; s.i = p.i; s.j = p.j;
            s.mass = p.mass - d;
            s.kind = "neutral";
            sats.push_back(s);
        }
    }
    return sats;
}

}  // namespace

DetectResult detect_lines_oa(const FfcTable& t, int parent_charge,
                             const DetectParams& p) {
    const double line_tol = (p.line_tol < 0.0) ? p.delta : p.line_tol;
    const bool   ppm      = (p.delta_unit == "ppm");

    std::vector<size_t> all(t.size());
    std::iota(all.begin(), all.end(), size_t{0});

    DetectResult res;

    // ── Pass 1: the parental (charge-sum) lines, cleared first ─────────────
    std::vector<size_t> work = all;
    for (int r = 0; r < p.max_lines; ++r) {
        Line ln;
        if (!most_informative_line(t, work, parent_charge, p, true, ln)) break;
        ln.kind = "parental";
        res.master.push_back(ln);
        auto next = remove_lines(t, work, {ln}, line_tol, ppm);
        if (next.size() == work.size()) break;
        work.swap(next);
    }

    // ── Pass 2: greedy peeling of the residual, from the FULL table with the
    //           parental lines AND their satellites removed ─────────────────
    std::vector<Line> to_clear = res.master;
    for (const Line& s : satellite_lines(res.master)) to_clear.push_back(s);
    work = remove_lines(t, all, to_clear, line_tol, ppm);

    for (int r = 0; r < p.max_lines; ++r) {
        Line ln;
        if (!most_informative_line(t, work, parent_charge, p, false, ln)) break;
        res.residual.push_back(ln);
        auto next = remove_lines(t, work, {ln}, line_tol, ppm);
        if (next.size() == work.size()) break;
        work.swap(next);
    }

    return res;
}

}  // namespace denovo
