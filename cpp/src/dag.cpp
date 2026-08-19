#include "denovo/dag.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>

namespace denovo {
namespace {

constexpr double NEG = -std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-9;

// Nearest allowed compound mass to `gap` within `tol`, or -1.
// ⚠ The reference checks idx BEFORE idx-1, so when both are inside the window
// the one at or above `gap` wins even if idx-1 is closer.  Kept verbatim.
inline double allowed_mass(const std::vector<double>& a, double gap, double tol) {
    if (gap < a.front() - tol || gap > a.back() + tol) return -1.0;
    const size_t idx = static_cast<size_t>(
        std::lower_bound(a.begin(), a.end(), gap) - a.begin());
    if (idx < a.size() && std::fabs(a[idx] - gap) <= tol) return a[idx];
    if (idx >= 1 && std::fabs(a[idx - 1] - gap) <= tol) return a[idx - 1];
    return -1.0;
}

}  // namespace

Dag build_dag(const std::vector<double>& coords,
              const std::vector<double>& sigmas,
              const std::vector<double>& allowed,
              double total, double source_x1, double source_x2,
              double precursor_sigma) {
    // sig = dict(sigma); then setdefault(source, 0.0) -- so a coordinate that
    // happens to equal a source component keeps its own sigma.
    std::map<double, double> sig;
    for (size_t k = 0; k < coords.size(); ++k) sig[coords[k]] = sigmas[k];
    sig.emplace(source_x1, 0.0);
    sig.emplace(source_x2, 0.0);   // the terminus is a definition, not a
                                   // measurement
    Dag g;
    g.coords.reserve(sig.size());
    for (const auto& kv : sig) g.coords.push_back(kv.first);
    const int C = static_cast<int>(g.coords.size());
    g.n_coords = C;
    g.adj.assign(static_cast<size_t>(C) * C, {});

    auto index_of = [&](double v) {
        return static_cast<int>(std::lower_bound(g.coords.begin(), g.coords.end(), v)
                                - g.coords.begin());
    };
    const int s1 = index_of(source_x1), s2 = index_of(source_x2);
    g.source_node = g.node_id(s1, s2);

    if (allowed.empty()) return g;

    std::vector<double> sv(C);
    for (int k = 0; k < C; ++k) sv[k] = sig[g.coords[k]];

    const double min_edge = allowed.front();
    const double max_coord = g.coords.back();
    double max_tol = 0.0;
    for (double s : sv) max_tol = std::max(max_tol, s);
    max_tol *= 2.0;

    auto first_above = [&](double val) {
        return static_cast<int>(
            std::lower_bound(g.coords.begin(), g.coords.end(),
                             val + min_edge - max_tol) - g.coords.begin());
    };

    for (int a = 0; a < C; ++a) {
        for (int b = 0; b < C; ++b) {
            const int node = g.node_id(a, b);
            if (a == b && node != g.source_node) continue;   // not a node
            const double x1 = g.coords[a], x2 = g.coords[b];
            auto& out = g.adj[node];
            const double max_current = std::max(x1, x2);

            // Sink: whatever residue run is left between the two ladders.
            const double remainder = total - x1 - x2;
            if (remainder > 0) {
                const double m = allowed_mass(allowed, remainder,
                                              sv[a] + sv[b] + precursor_sigma);
                if (m >= 0) out.push_back({-1, m, 0});
            }

            for (int t = first_above(x1); t < C; ++t) {
                const double gap = g.coords[t] - x1;
                if (gap > max_coord - x1 + max_tol) break;
                const double m = allowed_mass(allowed, gap, sv[t] + sv[a]);
                if (m < 0) continue;
                if ((t == b) && g.node_id(t, b) != g.source_node) continue;
                if (max_current < std::max(g.coords[t], x2))
                    out.push_back({g.node_id(t, b), m, 1});
            }
            for (int t = first_above(x2); t < C; ++t) {
                const double gap = g.coords[t] - x2;
                if (gap > max_coord - x2 + max_tol) break;
                const double m = allowed_mass(allowed, gap, sv[t] + sv[b]);
                if (m < 0) continue;
                if ((a == t) && g.node_id(a, t) != g.source_node) continue;
                if (max_current < std::max(x1, g.coords[t]))
                    out.push_back({g.node_id(a, t), m, 2});
            }
        }
    }
    return g;
}

// ── Topological order ────────────────────────────────────────────────────
// Every move strictly increases max(x1, x2) and the Sink is terminal, so the
// graph is acyclic and processing nodes in DECREASING max(x1, x2) visits every
// successor before its predecessor.  Nodes tied on max have no edge between
// them, so their relative order is irrelevant.
namespace {
std::vector<int> topo_order(const Dag& g) {
    std::vector<int> order;
    order.reserve(g.adj.size());
    for (size_t n = 0; n < g.adj.size(); ++n) {
        int a, b; g.decode(static_cast<int>(n), a, b);
        if (a == b && static_cast<int>(n) != g.source_node) continue;
        order.push_back(static_cast<int>(n));
    }
    std::sort(order.begin(), order.end(), [&](int p, int q) {
        int pa, pb, qa, qb;
        g.decode(p, pa, pb); g.decode(q, qa, qb);
        return std::max(g.coords[pa], g.coords[pb])
             > std::max(g.coords[qa], g.coords[qb]);
    });
    return order;
}
}  // namespace

// ── Score-only DP ────────────────────────────────────────────────────────
// max_paths never affects the best weight, so the goodness test -- which runs
// hundreds of times per peptide -- never materialises a path.
double heaviest_weight(const Dag& g, const std::vector<double>& w) {
    if (g.source_node < 0) return NEG;
    std::vector<double> memo(g.adj.size(), NEG);
    for (int node : topo_order(g)) {
        double best = NEG;
        for (const auto& e : g.adj[node]) {
            if (e.target < 0) {                       // Sink: dfs == 0, edge 1
                best = std::max(best, 1.0);
                continue;
            }
            const double nb = memo[e.target];
            if (nb == NEG) continue;
            int ta, tb; g.decode(e.target, ta, tb);
            best = std::max(best, nb + w[(e.kind == 1) ? ta : tb]);
        }
        memo[node] = best;
    }
    return memo[g.source_node];
}

// ── Full path enumeration ────────────────────────────────────────────────
namespace {
// `[node] + n_seq` in the reference copies the list, but the contents are
// shared, so a persistent cons-list gives identical results without the
// quadratic memory.
struct Cons {
    int node;
    std::shared_ptr<const Cons> next;
};
using ConsPtr = std::shared_ptr<const Cons>;
}  // namespace

double heaviest_paths(const Dag& g, const std::vector<double>& w,
                      int max_paths, std::vector<Path>& out) {
    out.clear();
    if (g.source_node < 0) return NEG;
    std::vector<double> best_of(g.adj.size(), NEG);
    std::vector<std::vector<ConsPtr>> paths_of(g.adj.size());
    const ConsPtr sink_path = std::make_shared<const Cons>(Cons{-1, nullptr});

    for (int node : topo_order(g)) {
        double best = NEG;
        std::vector<ConsPtr> best_paths;
        for (const auto& e : g.adj[node]) {
            double nb_score, add;
            const std::vector<ConsPtr>* nb_paths;
            std::vector<ConsPtr> sink_holder;
            if (e.target < 0) {
                nb_score = 0.0;
                add = 1.0;
                sink_holder.push_back(sink_path);
                nb_paths = &sink_holder;
            } else {
                if (best_of[e.target] == NEG) continue;
                nb_score = best_of[e.target];
                nb_paths = &paths_of[e.target];
                int ta, tb; g.decode(e.target, ta, tb);
                add = w[(e.kind == 1) ? ta : tb];
            }
            const double cur = nb_score + add;
            if (cur > best + EPS) { best = cur; best_paths.clear(); }
            else if (cur < best - EPS ||
                     static_cast<int>(best_paths.size()) >= max_paths) continue;
            for (const ConsPtr& p : *nb_paths) {
                best_paths.push_back(std::make_shared<const Cons>(Cons{node, p}));
                if (static_cast<int>(best_paths.size()) >= max_paths) break;
            }
        }
        best_of[node] = best;
        if (best != NEG) paths_of[node] = std::move(best_paths);
    }

    for (const ConsPtr& p : paths_of[g.source_node]) {
        Path path;
        for (const Cons* c = p.get(); c && c->node >= 0; c = c->next.get()) {
            int a, b; g.decode(c->node, a, b);
            path.nodes.emplace_back(g.coords[a], g.coords[b]);
        }
        out.push_back(std::move(path));
    }
    return best_of[g.source_node];
}

// ── Scorer ───────────────────────────────────────────────────────────────
Scorer::Scorer(const std::vector<double>& allowed, double total,
               double source_x1, double source_x2, double precursor_sigma,
               int max_paths)
    : allowed_(allowed), total_(total), sx1_(source_x1), sx2_(source_x2),
      precursor_sigma_(precursor_sigma), max_paths_(max_paths) {}

Dag Scorer::graph_of(const View& v) {
    ++n_builds_;
    return build_dag(v.coords, v.sigmas, allowed_, total_, sx1_, sx2_,
                     precursor_sigma_);
}

namespace {
// The reference keys its cache on (coords@4dp, sigmas@5dp, weights@3dp).
std::string cache_key(const View& v) {
    std::string k;
    k.reserve(v.coords.size() * 30);
    char buf[64];
    for (double c : v.coords) { snprintf(buf, sizeof buf, "%.4f|", c); k += buf; }
    k += '#';
    for (double s : v.sigmas) { snprintf(buf, sizeof buf, "%.5f|", s); k += buf; }
    k += '#';
    for (double w : v.weights) { snprintf(buf, sizeof buf, "%.3f|", w); k += buf; }
    return k;
}

// weights.get(coord, 1.0), lifted onto the DAG's coordinate index.
std::vector<double> weights_on(const Dag& g, const View& v) {
    std::map<double, double> m;
    for (size_t k = 0; k < v.coords.size(); ++k) m[v.coords[k]] = v.weights[k];
    std::vector<double> w(g.coords.size(), 1.0);
    for (size_t k = 0; k < g.coords.size(); ++k) {
        auto it = m.find(g.coords[k]);
        if (it != m.end()) w[k] = it->second;
    }
    return w;
}
}  // namespace

double Scorer::score(const View& v) {
    const std::string key = cache_key(v);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    const Dag g = graph_of(v);
    const double best = heaviest_weight(g, weights_on(g, v));
    const double val = (best == NEG) ? -1.0 : best;
    cache_.emplace(key, val);
    return val;
}

double Scorer::paths(const View& v, std::vector<Path>& out) {
    const Dag g = graph_of(v);
    return heaviest_paths(g, weights_on(g, v), max_paths_, out);
}

}  // namespace denovo
