#include "denovo/pipeline.hpp"
#include "denovo/dag.hpp"
#include "denovo/optimal_paths.hpp"
#include "denovo/rbdp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace denovo {
namespace {

std::string fmt(const char* f, double v) {
    char buf[64];
    snprintf(buf, sizeof buf, f, v);
    return buf;
}

// The TFFC's two possible placements ("trust Y", then "trust X"), each with
// its own sigma.  One FFC alone cannot say which side is the N-terminal
// fragment; on a parental line the two coincide, since x + y == parent_pair.
struct Images { double hc, hs, vc, vs; };
inline Images images_of(const Tffc& t, const Basis& b, double tol_ppm,
                        double precursor_ppm) {
    Images im;
    coord_from_side(t.y, b, tol_ppm, precursor_ppm, im.hc, im.hs);
    coord_from_side(t.x, b, tol_ppm, precursor_ppm, im.vc, im.vs);
    return im;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// master_reconstruction.classify_line -- from the alignment shift alone.
void classify_line(double a, const std::vector<double>& allowed,
                   std::string& category, std::string& detail,
                   double parent_tol = 0.5, double loss_tol = 0.05,
                   double edge_tol = 0.05) {
    if (std::fabs(a) < parent_tol) { category = "parental"; detail = "parent"; return; }
    const std::pair<const char*, double> losses[3] = {
        {"H2O", LOSS_H2O}, {"NH3", LOSS_NH3}, {"CO", LOSS_CO}};
    for (const auto& L : losses)
        if (std::fabs(a - L.second) <= loss_tol) {
            category = "loss"; detail = L.first; return;
        }
    if (a >= MIN_AA_MASS - edge_tol) {
        const size_t idx = static_cast<size_t>(
            std::lower_bound(allowed.begin(), allowed.end(), a) - allowed.begin());
        for (long k : {static_cast<long>(idx) - 1, static_cast<long>(idx)}) {
            if (k >= 0 && k < static_cast<long>(allowed.size()) &&
                std::fabs(allowed[k] - a) <= edge_tol) {
                category = "internal"; detail = fmt("%.3f", a); return;
            }
        }
        // A residue run heavier than the pre-computed table still counts.
        if (a > allowed.back()) {
            category = "internal"; detail = fmt("%.3f", a) + " (beyond edge table)";
            return;
        }
        category = "other"; detail = fmt("%.3f", a) + " not a residue-run mass";
        return;
    }
    category = "other"; detail = fmt("%.3f", a);
}

// master_reconstruction.mark_isotope_satellites.
//
// If a fragment on a line is picked up at its +1 isotope peak, the whole line
// reappears at A - 1.00335: same TFFCs, every coordinate 1 Da off -- close
// enough to look plausible, far enough to be wrong.  A line is a satellite
// when its A sits k*1.00335 BELOW that of a line with at least as many FFCs;
// the direction is what makes the test specific (no pair of residue-run masses
// differs by 1.003 Da).
void mark_isotope_satellites(std::vector<LineInfo>& lines, double tol = 0.02,
                             int max_k = 2) {
    std::vector<size_t> order(lines.size());
    for (size_t k = 0; k < lines.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t p, size_t q) {
        if (lines[p].n_ffc() != lines[q].n_ffc())
            return lines[p].n_ffc() > lines[q].n_ffc();
        return lines[p].key < lines[q].key;
    });
    for (size_t r = 0; r < lines.size(); ++r) {
        if (lines[r].category != "internal") continue;
        for (size_t o : order) {
            if (o == r || lines[o].n_ffc() < lines[r].n_ffc()) continue;
            bool hit = false;
            for (int k = 1; k <= max_k && !hit; ++k)
                if (std::fabs(lines[o].a_shift - lines[r].a_shift
                              - k * ISOTOPE_OFFSET) <= tol) {
                    lines[r].category = "other";
                    lines[r].detail = "+" + std::to_string(k) +
                        " isotope satellite of " + lines[o].key +
                        " (A=" + fmt("%.3f", lines[o].a_shift) + ")";
                    hit = true;
                }
            if (hit) break;
        }
    }
}

// k if A ~= k*1.00335 for a small non-zero k, else 0.  Such a line is the
// PARENTAL ladder re-appearing with one side on its +-k-th isotope peak (the
// deconvolution does not always pick the monoisotopic peak).  classify_line
// files it under `other`, so without this it contributes nothing despite
// being the ladder.
//
// ⚠ Admitted as an EXTENSION candidate, never as a seed.  On a satellite a
// TFFC's two images are u and u+d, 1.003 Da apart, exactly one real; seeding
// would bank the false one too.  Routed through the ordinary two-image
// goodness test, the false image is hundreds of sigma from any bond, so it
// can only lose the test or surface as a wrong coordinate.
int isotope_k(double a, double tol, int max_k = 2) {
    for (int k = 1; k <= max_k; ++k)
        for (int sign : {1, -1})
            if (std::fabs(a - sign * k * ISOTOPE_OFFSET) <= tol) return sign * k;
    return 0;
}

// Uncertainty of a line's alignment shift A.  A = median(parent_pair-(x+y)) is
// a DIFFERENCE OF LARGE MASSES, so its absolute error is set by `total`, not
// by A itself -- which is exactly why the annotation module needed a separate,
// looser mock_tol_ppm knob.  Here it is derived: two measured sides at tol_ppm
// each, then a median over n FFCs, capped so a sparse line cannot claim an
// unusably wide window.
double mock_sigma(double total, int n_ffc, double tol_ppm, double cap) {
    const double per_ffc = total * tol_ppm * 1e-6 * std::sqrt(2.0);
    return std::min(per_ffc / std::sqrt(static_cast<double>(std::max(n_ffc, 1))),
                    cap);
}

// May this line's mock be banked?  (The SECONDARY-PAIR test.)
//
// An internal line's mock claims the mass A was lost from an END of the
// peptide, so the cleavage that released it is a breakpoint at fold(A).  But
// an internal line arises a second way: a SECONDARY pair is two ordinary
// terminal ions with an INTERIOR run between them, and then fold(A) is not a
// cleavage at all -- it fabricates a breakpoint near a terminus.
//
// The two separate without ground truth: a secondary pair's two sides are both
// genuine terminal fragments, so BOTH images are already known breakpoints; a
// true internal pair has at most one side that is.
bool mock_ok(const LineInfo& li, double gate) {
    if (li.tffcs.empty()) return false;
    return li.n_both_known <= gate * static_cast<double>(li.tffcs.size());
}

// Member FFCs of a line, with each one's orientation resolved.  Detection is
// orientation-agnostic: the line stores a canonical (i, j) and each member's
// true orientation is whichever of i*mzA+j*mzB / j*mzA+i*mzB lands closer to
// the line mass.  Once the per-side charge is known,
// i*mz - (i-1)*MASS_H is that side's [M+H]+ mass.
std::vector<Tffc> extract_tffcs(const FfcTable& t, const Line& ln) {
    std::vector<Tffc> out;
    out.reserve(ln.members.size());
    for (size_t r : ln.members) {
        const double a = t.mz_a[r], b = t.mz_b[r];
        const double d1 = std::fabs(ln.i * a + ln.j * b - ln.mass);
        const double d2 = std::fabs(ln.j * a + ln.i * b - ln.mass);
        const int ch_a = (d1 <= d2) ? ln.i : ln.j;
        const int ch_b = (d1 <= d2) ? ln.j : ln.i;
        Tffc f;
        f.ch_x = ch_a; f.ch_y = ch_b;
        f.x = ch_a * a - (ch_a - 1) * MASS_H;
        f.y = ch_b * b - (ch_b - 1) * MASS_H;
        f.ranking = t.ranking[r];
        f.row = t.row.empty() ? static_cast<long>(r) : t.row[r];
        out.push_back(f);
    }
    return out;
}

// FNV-1a over a canonical text of everything discovery produced.  Used only
// to prove that the reconstruction method and the final-graph sigma scale do
// not reach back into discovery; it is not a cryptographic hash.
std::string discovery_digest(const Result& r) {
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](const std::string& s) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
        h ^= 0xff; h *= 1099511628211ULL;
    };
    char buf[64];
    auto num = [&](double v) { std::snprintf(buf, sizeof buf, "%.17g", v); feed(buf); };
    for (const LineInfo& li : r.lines) {
        feed(li.key); feed(li.category); num(li.a_shift); num(li.mass);
        feed(std::to_string(li.n_ffc()) + "/" + std::to_string(li.n_banked) + "/" +
             std::to_string(li.eligible));
    }
    std::vector<const Entry*> es;
    for (const Entry& e : r.master) es.push_back(&e);
    std::sort(es.begin(), es.end(),
              [](const Entry* a, const Entry* b) { return a->coord < b->coord; });
    for (const Entry* e : es) { num(e->coord); num(e->sigma); feed(std::to_string(e->n)); }
    feed(std::to_string(r.rounds));
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Final graph, search, counts, rendering.  Everything here works on the
// frozen view; `p.recon_sigma_scale` narrows the windows of THIS graph only.
void finalize(Result& res, const View& v, const Params& p,
              const std::vector<double>& allowed, long discovery_builds) {
    const double scale = p.recon_sigma_scale;
    if (!(scale > 0.0) || !std::isfinite(scale))
        throw std::invalid_argument("recon_sigma_scale must be a positive finite number");

    double merge_tol = 0.0;                         // rendering: unscaled
    for (double s : v.sigmas) merge_tol = std::max(merge_tol, s);
    res.merge_tol = merge_tol * 2.0;
    res.graph_precursor_sigma = res.precursor_sigma * scale;

    View gv = v;
    if (scale != 1.0)
        for (double& s : gv.sigmas) s *= scale;

    // nodes, in sorted-view order (== node_id)
    std::map<double, std::pair<int, double>> lookup;     // coord -> (id, discovery sigma)
    for (size_t k = 0; k < v.coords.size(); ++k) {
        FrozenNode n;
        n.node_id = static_cast<int>(k);
        n.coord = v.coords[k];
        n.discovery_sigma = v.sigmas[k];
        n.graph_sigma = gv.sigmas[k];
        n.weight = v.weights[k];
        for (size_t e = 0; e < res.master.size(); ++e)
            if (res.master[e].coord == n.coord) { n.entry = e; break; }
        res.nodes.push_back(n);
        lookup.emplace(n.coord, std::make_pair(n.node_id, n.discovery_sigma));
    }

    const Dag dag = build_dag(gv.coords, gv.sigmas, allowed, res.total, 0.0, res.term,
                              res.graph_precursor_sigma);
    res.dag_builds = discovery_builds + 1;
    const std::vector<double> w = weights_on(dag, gv);

    // The unit control.  `heaviest_paths` is the search the parity suite
    // guards; the tie DAG re-derives its optimum and gives the exact count.
    std::vector<Path> raw;
    const double best = heaviest_paths(dag, w, p.max_paths, raw);
    const TieDag control = tie_dag(dag, Coloring::uniform(w), MaskMode::Hard, 1e-9);
    const bool feasible = (best != -std::numeric_limits<double>::infinity());
    if (feasible != control.feasible() ||
        (feasible && std::fabs(best - control.optimum()) > 1e-9)) {
        res.status = "invalid_parity";
        res.rbdp.notes.push_back("tie-DAG optimum disagrees with heaviest_paths");
    }
    res.best_weight = feasible ? best : -1.0;
    BigCount exact = control.n_optimal();
    std::vector<std::vector<std::string>> hyps(raw.size());

    std::vector<int> dag_to_node(dag.coords.size(), -1);
    for (size_t k = 0; k < dag.coords.size(); ++k) {
        auto it = lookup.find(dag.coords[k]);
        if (it != lookup.end()) dag_to_node[k] = it->second.first;
    }

    if (p.reconstruction_method != "unit") {
        RbdpOutcome o = run_rbdp(res, dag, w, control, p, dag_to_node);
        if (o.replace_paths) {
            raw.clear();
            for (const auto& ids : o.paths) raw.push_back(to_path(dag, ids));
            hyps = o.hypotheses;
            exact = o.exact;
            res.best_weight = o.optimum;
            res.score_method = o.score_method;
        } else if (o.withhold_paths) {
            raw.clear();
            hyps.clear();
            exact = BigCount();
            res.best_weight = -1.0;
            res.score_method = o.score_method;
        }
    }

    // Render, then order for presentation.
    std::vector<StructuredPath> sp;
    std::vector<std::string> texts;
    std::vector<GappedStats> stats;
    const NodeLookup look = [&](double c, int& id, double& s) {
        auto it = lookup.find(c);
        if (it == lookup.end()) { id = -1; s = 0.0; return; }
        id = it->second.first; s = it->second.second;
    };
    for (const Path& pth : raw) {
        sp.push_back(structure_path(pth, res.total, res.term, res.merge_tol,
                                    p.label_tol, res.precursor_sigma, look));
        texts.push_back(sp.back().text);
        stats.push_back(sp.back().stats);
    }
    const std::vector<size_t> order = presentation_order(raw, texts, stats,
                                                         p.presentation_order);
    res.reconstructions.clear();
    res.paths_raw.clear();
    for (size_t k : order) {
        res.reconstructions.push_back(texts[k]);
        res.paths_raw.push_back(raw[k]);
        res.structured.push_back(sp[k]);
        res.path_hypotheses.push_back(k < hyps.size() ? hyps[k]
                                                      : std::vector<std::string>{});
        std::vector<std::pair<int,int>> ids;
        for (const auto& n : raw[k].nodes) {
            int a = -1, b = -1; double s;
            look(n.first, a, s);
            look(n.second, b, s);
            ids.emplace_back(a, b);
        }
        res.path_node_ids.push_back(ids);
    }
    res.n_paths = static_cast<int>(res.reconstructions.size());
    res.exact_optimal_path_count = (res.best_weight < 0 && raw.empty()) ? "0" : exact.str();
    res.enumeration_truncated = BigCount(static_cast<uint64_t>(res.n_paths)) < exact;
    if (res.status == "ok" && raw.empty()) res.status = "no_path";
}

}  // namespace

int LineInfo::best_rank() const {
    int best = -1;
    for (const Tffc& t : tffcs)
        if (t.ranking >= 0 && (best < 0 || t.ranking < best)) best = t.ranking;
    return best;
}

Result run_denovo(const FfcTable& t, int parent_charge, double parent_mass,
                  const std::string& terminus, const std::string& fragmentation,
                  const Params& p, const std::vector<double>& allowed) {
    const Basis basis = make_basis(fragmentation, parent_mass, terminus);
    const double total = basis.coord_total();
    const double term  = basis.coord_term();
    const double parent_pair = basis.parent_pair();
    const double precursor_sigma = basis.total * p.precursor_ppm * 1e-6;

    Result res;
    res.n_ffc = t.size();
    res.basis = basis;
    res.total = total;
    res.term  = term;

    // ── 1. Line detection (already fully de novo) ─────────────────────────
    DetectParams dp;
    dp.delta = p.delta;
    dp.delta_unit = p.delta_unit;
    dp.min_ffc_number = p.min_ffc_number;
    dp.line_tol = p.line_tol;
    const int line_charge = basis.line_charge(parent_charge);
    res.line_charge = line_charge;
    const DetectResult det = detect_lines_oa(t, line_charge, dp);

    // ── 2. Turn each line into TFFCs and an alignment shift ───────────────
    // A comes from the FFC masses, NOT from the detection shift, which carries
    // an extra (i+j-z)*proton: a charge sweep is a charge ASSUMPTION, and A has
    // no proton term so it needs no guess.
    std::vector<Line> all_lines = det.master;
    all_lines.insert(all_lines.end(), det.residual.begin(), det.residual.end());
    for (const Line& ln : all_lines) {
        std::vector<Tffc> tf = extract_tffcs(t, ln);
        if (tf.empty()) continue;
        std::vector<double> shifts;
        shifts.reserve(tf.size());
        for (const Tffc& f : tf) shifts.push_back(parent_pair - (f.x + f.y));
        LineInfo li;
        li.i = ln.i; li.j = ln.j; li.mass = ln.mass;
        li.a_shift = median_of(shifts);
        li.key = "(" + std::to_string(ln.i) + "," + std::to_string(ln.j) +
                 ") A=" + fmt("%+.3f", li.a_shift);
        li.tffcs = std::move(tf);
        classify_line(li.a_shift, allowed, li.category, li.detail);
        res.lines.push_back(std::move(li));
    }

    mark_isotope_satellites(res.lines);

    // Parental isotope satellites.  Done AFTER mark_isotope_satellites, which
    // guards a different case entirely and only ever touches `internal` lines,
    // so the two cannot interfere.
    if (p.include_isotope_lines) {
        for (LineInfo& li : res.lines) {
            if (li.category != "other") continue;
            const int k = isotope_k(li.a_shift, p.isotope_tol_da);
            if (k) {
                li.category = "isotope";
                li.detail = "parental " + std::string(k > 0 ? "+" : "") +
                            std::to_string(k) + " isotope satellite";
            }
        }
    }

    // ── 3. Seed the master set from the parental lines ────────────────────
    // A parental TFFC's two sides are two independent measurements of the SAME
    // cleavage (x + y == parent_pair), so both are added and MasterSet merges
    // them; the lighter side, measured more precisely in absolute terms,
    // dominates the inverse-variance mean.
    const double rank_scale = (p.rank_scale > 0) ? p.rank_scale
                                                 : std::max<double>(t.size(), 1.0);
    MasterSet master(total, p.sigma_floor_ppm, p.mock_weight, p.evidence_weight,
                     p.evidence_cap, p.rank_floor, rank_scale);
    for (size_t q = 0; q < res.lines.size(); ++q) {
        const LineInfo& li = res.lines[q];
        if (li.category != "parental") continue;
        const int lid = static_cast<int>(q);
        for (const Tffc& f : li.tffcs) {
            const Images im = images_of(f, basis, p.tol_ppm, p.precursor_ppm);
            master.add(im.hc, im.hs, "parental", li.key, f.ranking, lid);
            master.add(im.vc, im.vs, "parental", li.key, f.ranking, lid);
        }
    }

    Scorer scorer(allowed, total, 0.0, term, precursor_sigma, p.max_paths);
    auto score_with = [&](const MasterSet& ms,
                          const std::vector<Candidate>& extra = {}) {
        return scorer.score(ms.view(extra));
    };

    // ── 4. Extend, in batch rounds, until the master set stops growing ─────
    // Every TFFC in a round is judged against the set as it stood at the START
    // of the round and the winners are banked together at the end -- that is
    // what keeps a round order-independent.  Only unambiguous TFFCs are banked
    // during the iteration; ambiguous ones are held back and re-tested, since
    // banking a forced placement is exactly what can disambiguate them.
    std::unordered_map<std::string, size_t> by_key;      // last wins, as Python's
    for (size_t k = 0; k < res.lines.size(); ++k) by_key[res.lines[k].key] = k;

    // Mock: the line is missing a fragment of mass A, so the cleavage that
    // produced it is a breakpoint at fold(A).  Only `internal` lines get one --
    // a neutral loss is not a cleavage.
    if (p.use_mocks) {
        for (LineInfo& li : res.lines) {
            if (li.category == "internal" && li.n_ffc() >= p.mock_min_ffc) {
                li.has_mock = true;
                li.mock_coord = fold(li.a_shift, total);
                li.mock_sigma = mock_sigma(basis.total, li.n_ffc(), p.tol_ppm,
                                           p.mock_sigma_cap_da);
            }
        }
    }

    std::vector<std::string> pending_keys;               // insertion order
    std::unordered_map<std::string, std::vector<Tffc>> pending;
    for (const LineInfo& li : res.lines) {
        if (li.category != "internal" && li.category != "loss" &&
            li.category != "isotope") continue;
        if (!pending.count(li.key)) pending_keys.push_back(li.key);
        pending[li.key] = li.tffcs;
    }

    // Most populated line first: the best-corroborated line's forced
    // placements are the safest to bank.  The key is stable, so this is
    // deterministic.
    std::vector<std::string> order = pending_keys;
    std::sort(order.begin(), order.end(), [&](const std::string& a,
                                              const std::string& b) {
        const int na = res.lines[by_key[a]].n_ffc();
        const int nb = res.lines[by_key[b]].n_ffc();
        if (na != nb) return na > nb;
        return a < b;
    });

    MasterSet work = master;
    std::unordered_set<std::string> mocks_added;
    int rounds_used = 0;

    struct Pick { std::string key; size_t idx; double c, s; int rank; };

    for (int r = 0; r < p.max_rounds; ++r) {
        rounds_used = r + 1;
        const double s0 = score_with(work);

        // Eligibility is re-derived every round: a line with no inactive TFFC
        // in round 1 can acquire one later and only then qualify.
        std::vector<std::string> round_eligible;
        for (const std::string& k : order) {
            LineInfo& li = res.lines[by_key[k]];
            int n_inactive = 0, n_both = 0;
            for (const Tffc& f : li.tffcs) {
                const Images im = images_of(f, basis, p.tol_ppm, p.precursor_ppm);
                const bool in_h = work.contains(im.hc, im.hs);
                const bool in_v = work.contains(im.vc, im.vs);
                n_inactive += (in_h || in_v);
                n_both     += (in_h && in_v);
            }
            li.n_inactive = n_inactive;
            li.n_both_known = n_both;
            if (n_inactive >= p.master_align_threshold) round_eligible.push_back(k);
        }

        std::vector<Pick> picks;
        std::vector<Pick> new_mocks;
        for (const std::string& k : round_eligible) {
            LineInfo& li = res.lines[by_key[k]];
            li.eligible = true;
            if (li.has_mock && !mocks_added.count(k)) {
                mocks_added.insert(k);
                const double mc = li.mock_coord, ms = li.mock_sigma;
                const bool secondary_ok = mock_ok(li, p.secondary_gate);
                const bool ok = secondary_ok &&
                    (!p.gate_mock_on_goodness || work.contains(mc, ms) ||
                     score_with(work, {{mc, ms, true, -1}}) > s0 + 1e-9);
                if (!secondary_ok) li.mock_refused = "secondary";
                if (ok && !work.contains(mc, ms))
                    new_mocks.push_back({k, 0, mc, ms, li.best_rank()});
            }
            auto& rest = pending[k];
            for (size_t q = 0; q < rest.size(); ++q) {
                const Tffc& f = rest[q];
                const Images im = images_of(f, basis, p.tol_ppm, p.precursor_ppm);
                if (work.contains(im.hc, im.hs) || work.contains(im.vc, im.vs))
                    continue;                       // inactive; settled at the end
                const bool good_h =
                    score_with(work, {{im.hc, im.hs, false, f.ranking}}) > s0 + 1e-9;
                const bool good_v =
                    score_with(work, {{im.vc, im.vs, false, f.ranking}}) > s0 + 1e-9;
                if (good_h == good_v) continue;     // ambiguous / unproductive
                picks.push_back({k, q, good_h ? im.hc : im.vc,
                                 good_h ? im.hs : im.vs, f.ranking});
            }
        }

        if (picks.empty() && new_mocks.empty()) { rounds_used = r; break; }

        for (const Pick& m : new_mocks) {
            work.add(m.c, m.s, "mock", m.key, m.rank, static_cast<int>(by_key[m.key]));
            res.lines[by_key[m.key]].n_banked += 1;
        }
        // Bank first, then erase -- erasing invalidates the recorded indices.
        std::unordered_map<std::string, std::vector<size_t>> erase_at;
        for (const Pick& pk : picks) {
            work.add(pk.c, pk.s, "unambiguous", pk.key, pk.rank,
                     static_cast<int>(by_key[pk.key]));
            LineInfo& li = res.lines[by_key[pk.key]];
            li.n_unambiguous += 1;
            li.n_banked += 1;
            erase_at[pk.key].push_back(pk.idx);
        }
        for (auto& kv : erase_at) {
            auto& v = pending[kv.first];
            std::sort(kv.second.rbegin(), kv.second.rend());
            for (size_t idx : kv.second) v.erase(v.begin() + idx);
        }
    }

    // ── 5. Settle whatever never got banked, against the converged set ────
    {
        const double s0 = score_with(work);
        for (const std::string& k : pending_keys) {
            LineInfo& li = res.lines[by_key[k]];
            for (const Tffc& f : pending[k]) {
                const Images im = images_of(f, basis, p.tol_ppm, p.precursor_ppm);
                if (work.contains(im.hc, im.hs) || work.contains(im.vc, im.vs)) {
                    li.n_unproductive += 1;         // adds nothing new
                    continue;
                }
                const bool good_h =
                    score_with(work, {{im.hc, im.hs, false, f.ranking}}) > s0 + 1e-9;
                const bool good_v =
                    score_with(work, {{im.vc, im.vs, false, f.ranking}}) > s0 + 1e-9;
                if (good_h && good_v) {
                    li.n_ambiguous += 1;
                    // An ambiguous TFFC's evidence is "one of these two", never
                    // "this one", so the pair is atomic: both or neither.
                    if (li.eligible && p.accept_ambiguity) {
                        const int lid = static_cast<int>(by_key[k]);
                        work.add(im.hc, im.hs, "ambiguous", k, f.ranking, lid);
                        work.add(im.vc, im.vs, "ambiguous", k, f.ranking, lid);
                        li.n_banked += 2;
                    }
                } else if (good_h || good_v) {
                    li.n_unambiguous += 1;
                    if (li.eligible) {
                        work.add(good_h ? im.hc : im.vc, good_h ? im.hs : im.vs,
                                 "unambiguous", k, f.ranking,
                                 static_cast<int>(by_key[k]));
                        li.n_banked += 1;
                    }
                } else {
                    li.n_unproductive += 1;
                }
            }
        }
    }

    // ── 6. Freeze discovery, then reconstruct on the final graph ──────────
    // Discovery is finished here: nothing below may add or remove a master
    // entry, so `unit` and `rbdp` (and any sigma scale) compare on one set.
    res.master = work.entries();
    res.rounds = rounds_used;
    res.precursor_sigma = precursor_sigma;
    res.discovery_hash = discovery_digest(res);
    finalize(res, work.view(), p, allowed, scorer.builds());
    return res;
}

}  // namespace denovo
