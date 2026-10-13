// denovo_ffc -- the DeNovoFFC sequencer, command line.
//
//   denovo_ffc --data map.tsv --charge 5 --mass 4001.3781 --terminus NH3 \
//              --top-n 500 --json out.json
//   denovo_ffc combine ...          sequence-free prefix combination
//   denovo_ffc enumerate-fills ...  bounded exact gap dictionaries
//
// The input table must already be MERGED (mirror duplicates collapsed,
// rankings reset).  `--top-n` applies prepare_ffc_data's cut only.  See
// DESIGN.md 4.2.
#include "denovo/cli.hpp"
#include "denovo/edges.hpp"
#include "denovo/json.hpp"
#include "denovo/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace denovo;

namespace {

const char* const ENGINE_VERSION = "2.0.0";
const char* const MASS_CONVENTION_VERSION = "1";

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout, R"(denovo_ffc -- de novo sequencing from an MMS (FFC) map

Subcommands
  denovo_ffc [reconstruction flags]     reconstruct one map (below)
  denovo_ffc combine --help             combine oriented prefix evidence
  denovo_ffc enumerate-fills --help     exact fill dictionaries for gaps

Required
  --data PATH            merged FFC table (TSV/CSV with a header)
  --charge N             precursor charge
  --mass M               precursor NEUTRAL mass (Da)

Chemistry (both fail SILENTLY when wrong -- paths are still found, none right)
  --terminus H2O|NH3     C-terminal group                     [H2O]
  --fragmentation MODE   HCD | ETD | ETHCD   (acquisition)    [HCD]
  --ion-series S         auto | by | cz  (auto: HCD->by, ETD/ETHCD->cz) [auto]

Input
  --top-n N              keep the best-ranked N FFCs (0 = all)  [500]
  --col-a NAME           m/z column A                      [m/z A]
  --col-b NAME           m/z column B                      [m/z B]
  --ranking-col NAME     ranking column                   [Ranking]
  --dataset-id ID        identity used in evidence ids       [data]
  --acquisition-id ID    physical acquisition identity  [= dataset id]

Line detection
  --delta X              Sort-and-Split gap                   [0.015]
  --delta-unit da|ppm                                            [da]
  --min-ffc N            distinct FFCs to keep a line             [3]
  --line-tol X           FFC-on-line tolerance             [= --delta]

Tolerance model
  --tol-ppm X            fragment-mass relative error          [25.0]
  --precursor-ppm X      precursor-mass relative error         [10.0]
  --sigma-floor-ppm X    floor on a merged coordinate's sigma   [3.0]
  --mock-sigma-cap X     cap on a sparse line's mock window     [0.05]

Master set
  --align-threshold N    inactive TFFCs a line needs to join       [1]
  --accept-ambiguity     bank BOTH images of an ambiguous TFFC   [off]
  --no-mocks             internal lines contribute no mock
  --mock-min-ffc N       FFCs a line needs before it may mock       [4]
  --secondary-gate X     refuse the mock of a secondary line    [0.25]
  --no-isotope-lines     ignore parental isotope-satellite lines
  --no-mock-gate         skip the mock goodness test
  --max-rounds N         convergence cap                          [20]

Reconstruction (the final graph; discovery above is never affected)
  --reconstruction-method M   unit | rbdp | rbdp-guarded         [unit]
  --rbdp-evidence V      v2 (loss-aware) | v1 (reviewed rules)     [v2]
  --recon-sigma-scale X  scale final-graph sigmas (and precursor) [1.0]
  --presentation-order P enumeration | informative        [enumeration]
  --max-paths N          rendered tied-path cap; never changes the
                         optimum or the exact count            [1000]
  --label-tol X          residue-naming window                   [0.02]
  --edges-max-n N        residues per compound edge mass           [20]
  --edges-json PATH      read edges from a reference edges.json
                         (parity checking only; generation is faster)

Output
  --json PATH            write the full result as JSON (schema v2)
  --emit-evidence PATH   also write stable FFC/line/node evidence ids
  --quiet                suppress the human-readable summary

Exit codes: 0 analysis completed (including an explicit no-path status),
1 I/O or runtime failure, 2 invalid input or a detected parity failure.
)");
    std::exit(code);
}

const char* arg_after(int argc, char** argv, int& k, const char* flag) {
    if (k + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
    return argv[++k];
}

double num_arg(int argc, char** argv, int& k, const char* flag) {
    const char* s = arg_after(argc, argv, k, flag);
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "%s: not a number: %s\n", flag, s);
        std::exit(2);
    }
    return v;
}

int int_arg(int argc, char** argv, int& k, const char* flag) {
    const char* s = arg_after(argc, argv, k, flag);
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "%s: not an integer: %s\n", flag, s);
        std::exit(2);
    }
    return static_cast<int>(v);
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

std::string line_id(size_t k) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "L%03zu", k);
    return buf;
}

std::string ffc_id(const std::string& dataset, long row) {
    return dataset + ":r" + std::to_string(row);
}

void write_params(JsonWriter& j, const Params& p, int top_n) {
    j.key("effective_parameters").begin_object()
     .field("top_n", top_n)
     .field("delta", p.delta).field("delta_unit", p.delta_unit)
     .field("min_ffc_number", p.min_ffc_number);
    if (p.line_tol < 0) j.field_null("line_tol"); else j.field("line_tol", p.line_tol);
    j.field("tol_ppm", p.tol_ppm).field("precursor_ppm", p.precursor_ppm)
     .field("sigma_floor_ppm", p.sigma_floor_ppm)
     .field("mock_sigma_cap_da", p.mock_sigma_cap_da)
     .field("master_align_threshold", p.master_align_threshold)
     .field("accept_ambiguity", p.accept_ambiguity).field("use_mocks", p.use_mocks)
     .field("include_isotope_lines", p.include_isotope_lines)
     .field("isotope_tol_da", p.isotope_tol_da).field("mock_min_ffc", p.mock_min_ffc)
     .field("mock_weight", p.mock_weight).field("secondary_gate", p.secondary_gate)
     .field("evidence_weight", p.evidence_weight).field("evidence_cap", p.evidence_cap)
     .field("rank_floor", p.rank_floor)
     .field("gate_mock_on_goodness", p.gate_mock_on_goodness)
     .field("max_rounds", p.max_rounds).field("max_paths", p.max_paths)
     .field("label_tol", p.label_tol).field("edges_max_n", p.edges_max_n)
     .field("recon_sigma_scale", p.recon_sigma_scale)
     .field("presentation_order", p.presentation_order)
     .field("reconstruction_method", p.reconstruction_method);
    if (p.reconstruction_method != "unit") {
        j.key("rbdp").begin_object()
         .field("evidence", p.rbdp.evidence)
         .field("required_category", p.rbdp.required_category)
         .field("require_line_eligible", p.rbdp.require_line_eligible)
         .field("min_unique_parental_pairs", p.rbdp.min_unique_parental_pairs)
         .field("max_both_known_fraction", p.rbdp.max_both_known_fraction)
         .field("loss_aware", p.rbdp.loss_aware)
         .field("orientation_evidence_scope", "parental_only")
         .field("include_mock_orientation", false)
         .field("eps", p.rbdp.eps)
         .end_object();
    }
    j.end_object();
}

void write_candidates(JsonWriter& j, const Result& r) {
    // node_id -> origin line ids, for boundary provenance
    std::vector<std::vector<int>> node_lines(r.nodes.size());
    for (const FrozenNode& n : r.nodes) node_lines[n.node_id] = r.master[n.entry].origin_lines;

    j.key("candidates").begin_array();
    for (size_t k = 0; k < r.structured.size(); ++k) {
        const StructuredPath& s = r.structured[k];
        j.begin_object()
         .field("id", "c" + std::to_string(k + 1))
         .field("objective_score", r.best_weight)
         .field("score_method", r.score_method);
        j.key("source_hypotheses").array_of(r.path_hypotheses[k]);
        j.field("display_sequence", s.text);
        j.key("blocks").begin_array();
        for (const Block& b : s.blocks) {
            j.begin_object().field("kind", b.residue ? "residue" : "gap");
            if (b.residue) j.field("symbol", std::string(1, b.symbol));
            j.field("mass_da", b.mass);
            if (b.residue) j.field("residue_mass_da", b.residue_mass);
            j.field("start_prefix_mass", b.start).field("end_prefix_mass", b.end)
             .field("uncertainty_da", b.uncertainty)
             .field("provenance", b.closes_sink ? "graph_inferred" : "measured")
             .field("closes_sink", b.closes_sink);
            std::vector<int> src = s.boundaries[b.start_boundary].node_ids;
            for (int id : s.boundaries[b.end_boundary].node_ids) src.push_back(id);
            j.key("source_node_ids").array_of(src);
            j.end_object();
        }
        j.end_array();
        j.key("prefix_boundaries").begin_array();
        for (const Boundary& bd : s.boundaries) {
            std::set<int> lines;
            for (int id : bd.node_ids)
                for (int l : node_lines[id]) lines.insert(l);
            std::vector<std::string> src;
            for (int l : lines) src.push_back(line_id(static_cast<size_t>(l)));
            j.begin_object().field("mass_da", bd.mass)
             .field("uncertainty_da", bd.uncertainty)
             .field("precursor_dependent", bd.precursor_dependent)
             .field("structural", bd.structural);
            j.key("source_node_ids").array_of(bd.node_ids);
            j.key("sides").array_of(bd.sides);
            j.key("source_ids").array_of(src);
            j.end_object();
        }
        j.end_array();
        j.key("path_node_ids").begin_array();
        for (const auto& pr : r.path_node_ids[k]) {
            j.begin_array();
            if (pr.first < 0) j.null(); else j.value(pr.first);
            if (pr.second < 0) j.null(); else j.value(pr.second);
            j.end_array();
        }
        j.end_array();
        j.field("resolved_residues", s.stats.resolved)
         .field("longest_resolved_run", s.stats.longest)
         .field("gap_count", s.stats.gaps);
        j.end_object();
    }
    j.end_array();
}

void write_rbdp(JsonWriter& j, const RbdpReport& b) {
    j.key("rbdp").begin_object()
     .field("requested_method", b.requested_method)
     .field("effective_method", b.effective_method)
     .field("status", b.status);
    if (b.fallback_reason.empty()) j.field_null("fallback_reason");
    else j.field("fallback_reason", b.fallback_reason);
    if (b.selected_line_id.empty()) j.field_null("selected_line");
    else {
        j.key("selected_line").begin_object()
         .field("line_id", b.selected_line_id).field("display_key", b.selected_line_key)
         .field("alignment_shift", b.selected_a_shift).end_object();
    }
    j.field("parental_seed_size", b.seed_size);
    j.key("anchors").begin_array();
    for (const auto& a : b.anchors) {
        j.begin_object().field("node_id", a.node_id).field("coord", a.coord)
         .field("x_side", a.x_side).field("color_H_X_b", a.color_H_X_b)
         .field("color_H_X_y", a.color_H_X_y);
        j.key("ranks").array_of(a.ranks);
        j.key("evidence_ids").array_of(a.evidence_ids);
        j.end_object();
    }
    j.end_array();
    j.key("control").begin_object().field("feasible", b.control_feasible);
    if (b.control_feasible) j.field("optimum", b.control_optimum); else j.field_null("optimum");
    j.field("exact_optimal_path_count", b.control_exact_count).end_object();
    j.key("branches").begin_array();
    for (const auto& br : b.branches) {
        j.begin_object().field("hypothesis", br.hypothesis).field("feasible", br.feasible);
        if (br.feasible) j.field("optimum", br.optimum); else j.field_null("optimum");
        j.field("exact_optimal_path_count", br.exact_count)
         .field("returned", br.returned).field("truncated", br.truncated).end_object();
    }
    j.end_array();
    j.key("winning").array_of(b.winning);
    if (b.has_global_optimum) j.field("global_optimum", b.global_optimum);
    else j.field_null("global_optimum");
    j.key("union").begin_object().field("unique", b.union_unique)
     .field("intersection", b.union_intersection).field("method", b.union_method)
     .end_object();
    j.key("control_optima_surviving").begin_object()
     .field("H_X_b", b.survivors_b).field("H_X_y", b.survivors_y)
     .field("both", b.survivors_both).field("either", b.survivors_either).end_object();
    j.key("diagnostics").begin_object()
     .field("anchor_votes_split", b.anchor_votes_split)
     .field("both_branches_lose_length", b.both_branches_lose_length);
    if (b.has_best_of_two_loss) j.field("best_of_two_length_loss", b.best_of_two_length_loss);
    else j.field_null("best_of_two_length_loss");
    j.key("anchor_votes").begin_array();
    for (const auto& v : b.votes) {
        j.begin_object().field("node_id", v.node_id).field("coord", v.coord)
         .field("axis_H_X_b", v.axis_H_X_b).field("axis_H_X_y", v.axis_H_X_y);
        if (v.has_b) j.field("best_through_H_X_b", v.best_b); else j.field_null("best_through_H_X_b");
        if (v.has_y) j.field("best_through_H_X_y", v.best_y); else j.field_null("best_through_H_X_y");
        j.field("vote", v.vote);
        j.key("ranks").array_of(v.ranks);
        j.end_object();
    }
    j.end_array().end_object();
    j.key("notes").array_of(b.notes);
    j.key("line_candidates").begin_array();
    for (const auto& c : b.candidates) {
        j.begin_object().field("line_id", c.line_id).field("display_key", c.key)
         .field("category", c.category).field("upstream_eligible", c.upstream_eligible)
         .field("alignment_shift", c.a_shift).field("n_members", c.n_members)
         .field("n_accepted_members", c.n_accepted_members)
         .field("alignment_score", c.alignment_score)
         .field("n_both_known", c.n_both_known_seed);
        if (c.has_fraction) j.field("both_known_fraction", c.both_known_fraction);
        else j.field_null("both_known_fraction");
        if (c.best_accepted_rank >= 0) j.field("best_accepted_rank", c.best_accepted_rank);
        else j.field_null("best_accepted_rank");
        j.key("reason_counts").begin_object();
        for (const auto& kv : c.reason_counts) j.field(kv.first, kv.second);
        j.end_object();
        j.key("conflicts").array_of(c.conflicts);
        j.field("status", c.status).field("selected", c.selected).end_object();
    }
    j.end_array();
    j.end_object();
}

void write_result_json(std::ostream& o, const Result& r, const Params& p,
                       const FfcTable& tab, int top_n, double seconds,
                       const std::string& acq_fragmentation,
                       const std::string& ion_series, int charge, double mass,
                       const std::string& terminus) {
    JsonWriter j(o);
    j.begin_object();
    // ── legacy fields (unchanged names and meaning) ──
    j.field("n_ffc", static_cast<unsigned long>(tab.size()))
     .field("fragmentation", r.basis.mode)
     .field("line_charge", r.line_charge)
     .field("coord_total", r.total)
     .field("coord_term", r.term)
     .field("rounds", r.rounds)
     .field("n_paths", r.n_paths)
     .field("best_weight", r.best_weight)
     .field("seconds", seconds)
     .field("n_lines", static_cast<unsigned long>(r.lines.size()));
    j.key("reconstructions").array_of(r.reconstructions);
    // node_id of each master entry
    std::vector<int> entry_node(r.master.size(), -1);
    for (const FrozenNode& n : r.nodes) entry_node[n.entry] = n.node_id;
    j.key("master_set").begin_array();
    for (size_t k = 0; k < r.master.size(); ++k) {
        const Entry& e = r.master[k];
        j.begin_object().field("coord", e.coord).field("sigma", e.sigma)
         .field("n", e.n).field("best_rank", e.best_rank);
        j.key("kinds").array_of(e.kinds);
        j.key("origins").array_of(e.origins);
        j.field("node_id", entry_node[k]).field("all_mock", e.all_mock);
        std::vector<std::string> ol;
        for (int l : e.origin_lines) ol.push_back(line_id(static_cast<size_t>(l)));
        j.key("origin_line_ids").array_of(ol);
        j.end_object();
    }
    j.end_array();
    j.key("paths").begin_array();
    for (const Path& pth : r.paths_raw) {
        j.begin_array();
        for (const auto& n : pth.nodes) j.array_of(std::vector<double>{n.first, n.second});
        j.end_array();
    }
    j.end_array();
    j.key("lines").begin_array();
    for (size_t k = 0; k < r.lines.size(); ++k) {
        const LineInfo& li = r.lines[k];
        j.begin_object().field("key", li.key).field("line_id", line_id(k))
         .field("i", li.i).field("j", li.j).field("mass", li.mass)
         .field("a_shift", li.a_shift).field("category", li.category)
         .field("detail", li.detail).field("n_ffc", li.n_ffc())
         .field("n_inactive", li.n_inactive).field("eligible", li.eligible)
         .field("n_banked", li.n_banked).field("unproductive", li.n_unproductive)
         .field("unambiguous", li.n_unambiguous).field("ambiguous", li.n_ambiguous)
         .field("mock", li.has_mock).field("mock_refused", li.mock_refused)
         // A near-zero shift on another charge sum (e.g. z-2 next to the c/z
         // z-1 ladder) is NOT primary-ladder evidence; say which it is.
         .field("charge_sum", li.i + li.j)
         .field("primary_charge_sum", li.i + li.j == r.line_charge)
         .end_object();
    }
    j.end_array();

    // ── schema v2 ──
    j.field("schema_version", 2)
     .field("mass_convention_version", MASS_CONVENTION_VERSION)
     .field("kind", "denovo_reconstruction")      // sequence-free by construction
     .field("reference_assisted", false)
     .field("status", r.status);
    j.key("engine").begin_object().field("name", "cpp")
     .field("version", ENGINE_VERSION).end_object();
    j.key("chemistry").begin_object()
     .field("acquisition_fragmentation", acq_fragmentation)
     .field("ion_series", ion_series)
     .field("coordinate_basis", r.basis.mode)
     .field("precursor_charge", charge)
     .field("line_charge", r.line_charge)
     .field("neutral_mass", mass)
     .field("terminus", terminus)
     .field("terminus_mass", r.basis.term)
     .field("coord_total", r.total)
     .field("coord_term", r.term)
     .field("residue_sum", r.total - r.term)
     .field("parent_pair", r.basis.parent_pair())
     .end_object();
    write_params(j, p, top_n);
    j.key("reconstruction").begin_object()
     .field("requested_method", p.reconstruction_method)
     .field("effective_method", r.rbdp.requested ? r.rbdp.effective_method : "unit")
     .field("score_method", r.score_method)
     .field("presentation_order", p.presentation_order)
     .field("recon_sigma_scale", p.recon_sigma_scale)
     .field("precursor_sigma", r.precursor_sigma)
     .field("graph_precursor_sigma", r.graph_precursor_sigma)
     .field("render_merge_tol", r.merge_tol)
     .end_object();
    j.field("discovery_hash", r.discovery_hash);
    if (r.exact_optimal_path_count.empty()) j.field_null("exact_optimal_path_count");
    else j.field("exact_optimal_path_count", r.exact_optimal_path_count);
    j.field("returned_path_count", r.n_paths)
     .field("enumeration_truncated", r.enumeration_truncated);
    std::set<std::string> uniq(r.reconstructions.begin(), r.reconstructions.end());
    j.field("returned_unique_sequence_count", static_cast<unsigned long>(uniq.size()));
    j.key("nodes").begin_array();
    for (const FrozenNode& n : r.nodes) {
        j.begin_object().field("node_id", n.node_id).field("folded_coord", n.coord)
         .field("discovery_sigma", n.discovery_sigma).field("graph_sigma", n.graph_sigma)
         .field("scalar_weight", n.weight)
         .field("all_mock", r.master[n.entry].all_mock)
         .field("n", r.master[n.entry].n).end_object();
    }
    j.end_array();
    write_candidates(j, r);
    if (r.rbdp.requested) write_rbdp(j, r.rbdp);
    j.end_object();
}

void write_evidence_json(std::ostream& o, const Result& r, const FfcTable& tab,
                         const std::string& dataset, const std::string& acquisition) {
    std::map<long, size_t> pos;                    // row -> position after top-N
    for (size_t k = 0; k < tab.size(); ++k) pos[tab.row[k]] = k;
    JsonWriter j(o);
    j.begin_object().field("schema_version", 1)
     .field("dataset_id", dataset).field("acquisition_id", acquisition)
     .field("row_basis", "0-based data row of the engine input table");
    std::set<long> used;
    for (const LineInfo& li : r.lines) for (const Tffc& f : li.tffcs) used.insert(f.row);
    j.key("ffcs").begin_array();
    for (long row : used) {
        auto it = pos.find(row);
        j.begin_object().field("ffc_id", ffc_id(dataset, row))
         .field("dataset_id", dataset).field("acquisition_id", acquisition);
        j.key("original_row_ids").begin_array().value(row).end_array();
        if (it != pos.end()) {
            j.field("mz_a", tab.mz_a[it->second]).field("mz_b", tab.mz_b[it->second])
             .field("original_rank", tab.ranking[it->second])
             .field("effective_rank", static_cast<unsigned long>(it->second + 1));
        }
        j.end_object();
    }
    j.end_array();
    j.key("lines").begin_array();
    for (size_t k = 0; k < r.lines.size(); ++k) {
        const LineInfo& li = r.lines[k];
        j.begin_object().field("line_id", line_id(k)).field("display_key", li.key)
         .field("i", li.i).field("j", li.j).field("mass", li.mass)
         .field("alignment_shift", li.a_shift).field("category", li.category)
         .field("eligible", li.eligible);
        j.key("members").begin_array();
        for (size_t q = 0; q < li.tffcs.size(); ++q) {
            const Tffc& f = li.tffcs[q];
            j.begin_object().field("ordinal", static_cast<unsigned long>(q))
             .field("ffc_id", ffc_id(dataset, f.row))
             .field("x_mh", f.x).field("y_mh", f.y)
             .field("charge_x", f.ch_x).field("charge_y", f.ch_y)
             .field("charge_source", "line").field("ranking", f.ranking).end_object();
        }
        j.end_array().end_object();
    }
    j.end_array();
    j.key("nodes").begin_array();
    for (const FrozenNode& n : r.nodes) {
        const Entry& e = r.master[n.entry];
        std::vector<std::string> ol;
        for (int l : e.origin_lines) ol.push_back(line_id(static_cast<size_t>(l)));
        j.begin_object().field("node_id", n.node_id).field("folded_coord", n.coord)
         .field("discovery_sigma", n.discovery_sigma).field("graph_sigma", n.graph_sigma)
         .field("scalar_weight", n.weight).field("all_mock", e.all_mock);
        j.key("kinds").array_of(e.kinds);
        j.key("evidence_line_ids").array_of(ol);
        j.end_object();
    }
    j.end_array();
    j.key("projection").begin_array();
    for (const ProjectionEvidence& e : r.rbdp.evidence) {
        j.begin_object().field("evidence_id", e.evidence_id).field("line_id", e.line_id)
         .field("ffc_id", ffc_id(dataset, e.ffc_row)).field("ranking", e.ranking)
         .field("x_mh", e.x_mh).field("y_mh", e.y_mh)
         .field("both_known", e.both_known).field("terminal_side", e.terminal_side);
        if (e.has_xy) j.field("ordered_X", e.X).field("ordered_Y", e.Y);
        else j.field_null("ordered_X").field_null("ordered_Y");
        if (e.seed_id >= 0) j.field("anchor_seed_id", e.seed_id); else j.field_null("anchor_seed_id");
        if (e.final_node_id >= 0) j.field("node_id", e.final_node_id); else j.field_null("node_id");
        j.field("x_side", e.x_side).field("closure_residual", e.closure_residual)
         .field("closure_tol", e.closure_tol)
         .field("a_seed_matches", e.a_seed_matches).field("b_seed_matches", e.b_seed_matches)
         .field("loss_check_side", e.loss_check_side).field("loss_hits", e.loss_hits)
         .field("rejection_reason", e.reason == "accepted" ? "" : e.reason)
         .field("reason", e.reason).end_object();
    }
    j.end_array();
    j.end_object();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2) {
        const std::string sub = argv[1];
        if (sub == "combine")         return cmd_combine(argc - 1, argv + 1);
        if (sub == "enumerate-fills") return cmd_enumerate_fills(argc - 1, argv + 1);
    }

    std::string data, terminus = "H2O", fragmentation = "HCD", ion_series = "auto";
    std::string col_a = "m/z A", col_b = "m/z B", ranking_col = "Ranking";
    std::string json_path, edges_json, evidence_path, acquisition_id;
    int charge = 0, top_n = 500;
    double mass = 0.0;
    bool quiet = false, have_mass = false;
    Params p;

    if (argc == 1) usage(1);
    for (int k = 1; k < argc; ++k) {
        const std::string a = argv[k];
        if      (a == "-h" || a == "--help") usage(0);
        else if (a == "--data")        data = arg_after(argc, argv, k, "--data");
        else if (a == "--charge")      charge = int_arg(argc, argv, k, "--charge");
        else if (a == "--mass")      { mass = num_arg(argc, argv, k, "--mass"); have_mass = true; }
        else if (a == "--terminus")    terminus = arg_after(argc, argv, k, "--terminus");
        else if (a == "--fragmentation") fragmentation = arg_after(argc, argv, k, "--fragmentation");
        else if (a == "--ion-series")  ion_series = arg_after(argc, argv, k, "--ion-series");
        else if (a == "--top-n")       top_n = int_arg(argc, argv, k, "--top-n");
        else if (a == "--col-a")       col_a = arg_after(argc, argv, k, "--col-a");
        else if (a == "--col-b")       col_b = arg_after(argc, argv, k, "--col-b");
        else if (a == "--ranking-col") ranking_col = arg_after(argc, argv, k, "--ranking-col");
        else if (a == "--dataset-id")  p.dataset_id = arg_after(argc, argv, k, "--dataset-id");
        else if (a == "--acquisition-id") acquisition_id = arg_after(argc, argv, k, "--acquisition-id");
        else if (a == "--delta")       p.delta = num_arg(argc, argv, k, "--delta");
        else if (a == "--delta-unit")  p.delta_unit = arg_after(argc, argv, k, "--delta-unit");
        else if (a == "--min-ffc")     p.min_ffc_number = int_arg(argc, argv, k, "--min-ffc");
        else if (a == "--line-tol")    p.line_tol = num_arg(argc, argv, k, "--line-tol");
        else if (a == "--tol-ppm")     p.tol_ppm = num_arg(argc, argv, k, "--tol-ppm");
        else if (a == "--precursor-ppm") p.precursor_ppm = num_arg(argc, argv, k, "--precursor-ppm");
        else if (a == "--sigma-floor-ppm") p.sigma_floor_ppm = num_arg(argc, argv, k, "--sigma-floor-ppm");
        else if (a == "--mock-sigma-cap") p.mock_sigma_cap_da = num_arg(argc, argv, k, "--mock-sigma-cap");
        else if (a == "--align-threshold") p.master_align_threshold = int_arg(argc, argv, k, "--align-threshold");
        else if (a == "--accept-ambiguity") p.accept_ambiguity = true;
        else if (a == "--no-mocks")    p.use_mocks = false;
        else if (a == "--mock-min-ffc") p.mock_min_ffc = int_arg(argc, argv, k, "--mock-min-ffc");
        else if (a == "--secondary-gate") p.secondary_gate = num_arg(argc, argv, k, "--secondary-gate");
        else if (a == "--no-isotope-lines") p.include_isotope_lines = false;
        else if (a == "--isotope-tol") p.isotope_tol_da = num_arg(argc, argv, k, "--isotope-tol");
        else if (a == "--no-mock-gate") p.gate_mock_on_goodness = false;
        else if (a == "--max-rounds")  p.max_rounds = int_arg(argc, argv, k, "--max-rounds");
        else if (a == "--max-paths")   p.max_paths = int_arg(argc, argv, k, "--max-paths");
        else if (a == "--label-tol")   p.label_tol = num_arg(argc, argv, k, "--label-tol");
        else if (a == "--edges-max-n") p.edges_max_n = int_arg(argc, argv, k, "--edges-max-n");
        else if (a == "--edges-json")  edges_json = arg_after(argc, argv, k, "--edges-json");
        else if (a == "--recon-sigma-scale") p.recon_sigma_scale = num_arg(argc, argv, k, "--recon-sigma-scale");
        else if (a == "--presentation-order") p.presentation_order = arg_after(argc, argv, k, "--presentation-order");
        else if (a == "--reconstruction-method") p.reconstruction_method = arg_after(argc, argv, k, "--reconstruction-method");
        else if (a == "--rbdp-evidence") p.rbdp.evidence = arg_after(argc, argv, k, "--rbdp-evidence");
        else if (a == "--json")        json_path = arg_after(argc, argv, k, "--json");
        else if (a == "--emit-evidence") evidence_path = arg_after(argc, argv, k, "--emit-evidence");
        else if (a == "--quiet")       quiet = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(2); }
    }
    if (data.empty() || charge <= 0 || !have_mass) {
        std::fprintf(stderr, "--data, --charge and --mass are required\n");
        usage(2);
    }

    // ── validate everything before any work (exit 2 on invalid input) ──
    auto invalid = [](const std::string& msg) {
        std::fprintf(stderr, "error: %s\n", msg.c_str());
        return 2;
    };
    const std::string frag = upper(fragmentation);
    if (frag != "HCD" && frag != "ETD" && frag != "ETHCD")
        return invalid("--fragmentation must be HCD, ETD or ETHCD, got " + fragmentation);
    if (ion_series == "auto") ion_series = (frag == "HCD") ? "by" : "cz";
    if (ion_series != "by" && ion_series != "cz")
        return invalid("--ion-series must be auto, by or cz, got " + ion_series);
    if (frag == "HCD" && ion_series == "cz")
        return invalid("an HCD acquisition has no c/z ladder; use --ion-series by");
    const std::string basis_mode = (ion_series == "cz") ? "ETD" : "HCD";
    // The two detected fragments of a complementary pair each carry >= 1
    // charge, so the ladder's charge budget must be >= 2: z >= 2 for b/y and,
    // since electron transfer consumes one proton, z >= 3 for c/z.
    if (ion_series == "cz" && charge < 3)
        return invalid("a c/z pair carries z-1 charges; precursor charge must be >= 3, "
                       "got " + std::to_string(charge));
    if (ion_series == "by" && charge < 2)
        return invalid("a b/y pair needs a precursor charge >= 2, got " + std::to_string(charge));
    if (!(mass > 0.0) || !std::isfinite(mass)) return invalid("--mass must be positive and finite");
    if (terminus != "H2O" && terminus != "OH" && terminus != "NH3" && terminus != "NH2")
        return invalid("--terminus must be H2O/OH or NH3/NH2, got " + terminus);
    if (!(p.recon_sigma_scale > 0.0) || !std::isfinite(p.recon_sigma_scale))
        return invalid("--recon-sigma-scale must be positive and finite");
    if (p.presentation_order != "enumeration" && p.presentation_order != "informative")
        return invalid("--presentation-order must be enumeration or informative");
    if (p.reconstruction_method != "unit" && p.reconstruction_method != "rbdp" &&
        p.reconstruction_method != "rbdp-guarded")
        return invalid("--reconstruction-method must be unit, rbdp or rbdp-guarded");
    if (p.reconstruction_method != "unit" && ion_series != "by")
        return invalid("RBDP is validated for single-series HCD b/y only; it is not "
                       "available for the c/z series (integration design 8.1)");
    if (p.rbdp.evidence != "v1" && p.rbdp.evidence != "v2")
        return invalid("--rbdp-evidence must be v1 or v2");
    p.rbdp.loss_aware = (p.rbdp.evidence == "v2");
    if (p.max_paths < 1) return invalid("--max-paths must be >= 1");
    if (acquisition_id.empty()) acquisition_id = p.dataset_id;

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto secs = [&](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    try {
        const auto allowed = edges_json.empty() ? build_edges(p.edges_max_n)
                                                : load_edges_json(edges_json);
        const auto t1 = clk::now();

        const FfcTable raw = read_ffc_table(data, col_a, col_b, ranking_col);
        const FfcTable tab = prepare(raw, top_n);
        const auto t2 = clk::now();

        const Result r = run_denovo(tab, charge, mass, terminus, basis_mode, p, allowed);
        const auto t3 = clk::now();

        int n_par = 0, n_int = 0, n_loss = 0, n_iso = 0, n_other = 0;
        for (const auto& li : r.lines) {
            if      (li.category == "parental") ++n_par;
            else if (li.category == "internal") ++n_int;
            else if (li.category == "loss")     ++n_loss;
            else if (li.category == "isotope")  ++n_iso;
            else ++n_other;
        }

        if (!quiet) {
            std::printf("FFCs %zu -> %zu  |  basis %s, ladder charge sum %d, "
                        "coord_total %.4f\n", raw.size(), tab.size(),
                        r.basis.mode.c_str(), r.line_charge, r.total);
            std::printf("lines %zu  (parental %d, internal %d, loss %d, "
                        "isotope %d, other %d)\n",
                        r.lines.size(), n_par, n_int, n_loss, n_iso, n_other);
            std::printf("master TFFC-set %zu breakpoints after %d round(s); "
                        "%d tied path(s) of %s exact, weight %.0f\n",
                        r.master.size(), r.rounds, r.n_paths,
                        r.exact_optimal_path_count.c_str(), r.best_weight);
            if (r.rbdp.requested)
                std::printf("reconstruction %s -> %s (status %s%s%s)\n",
                            r.rbdp.requested_method.c_str(),
                            r.rbdp.effective_method.c_str(), r.rbdp.status.c_str(),
                            r.rbdp.fallback_reason.empty() ? "" : ", fallback: ",
                            r.rbdp.fallback_reason.c_str());
            for (size_t k = 0; k < r.reconstructions.size() && k < 5; ++k)
                std::printf("  [%zu] %s\n", k + 1, r.reconstructions[k].c_str());
            if (r.reconstructions.size() > 5)
                std::printf("  ... %zu more tied path(s)\n",
                            r.reconstructions.size() - 5);
            std::printf("timing: edges %.3fs  read %.3fs  denovo %.3fs  "
                        "total %.3fs  (%ld DAG builds)\n",
                        secs(t0, t1), secs(t1, t2), secs(t2, t3), secs(t0, t3),
                        r.dag_builds);
        }

        // Atomic for regular files (write, then rename); "-" or a device
        // path is written straight through.
        auto emit = [&](const std::string& path, auto&& writer) -> bool {
            if (path == "-") { writer(std::cout); std::cout.flush(); return true; }
            if (path.rfind("/dev/", 0) == 0) {
                std::ofstream o(path);
                if (!o) return false;
                writer(o);
                return static_cast<bool>(o);
            }
            const std::string tmp = path + ".partial";
            {
                std::ofstream o(tmp);
                if (!o) return false;
                writer(o);
                if (!o) return false;
            }
            return std::rename(tmp.c_str(), path.c_str()) == 0;
        };
        if (!json_path.empty() &&
            !emit(json_path, [&](std::ostream& o) {
                write_result_json(o, r, p, tab, top_n, secs(t0, t3), frag, ion_series,
                                  charge, mass, terminus); })) {
            std::fprintf(stderr, "cannot write %s\n", json_path.c_str());
            return 1;
        }
        if (!evidence_path.empty() &&
            !emit(evidence_path, [&](std::ostream& o) {
                write_evidence_json(o, r, tab, p.dataset_id, acquisition_id); })) {
            std::fprintf(stderr, "cannot write %s\n", evidence_path.c_str());
            return 1;
        }
        if (r.status == "invalid_parity") {
            std::fprintf(stderr, "error: internal parity check failed\n");
            return 2;
        }
    } catch (const std::invalid_argument& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
