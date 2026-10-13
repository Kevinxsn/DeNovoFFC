// run_denovo -- the de novo pipeline.
//
//   detect lines
//     -> seed the master TFFC-set from the parental lines
//     -> extend with aligned internal / loss lines, in batch rounds, until it
//        converges
//     -> heaviest path through the 2D-DAG -> sequence
//
// A faithful port of src/de_novo/denovo_core.py::run_denovo.  Nothing here
// ever sees a peptide sequence; scoring against a known one is a separate
// step that lives in Python (src/de_novo/evaluate.py).
#pragma once
#include <string>
#include <vector>
#include <map>
#include "denovo/constants.hpp"
#include "denovo/ffc.hpp"
#include "denovo/lines.hpp"
#include "denovo/masterset.hpp"
#include "denovo/bigcount.hpp"
#include "denovo/dag.hpp"
#include "denovo/reconstruction.hpp"
#include "denovo/rbdp_types.hpp"

namespace denovo {

struct Params {
    // line detection
    double delta = 0.015;
    std::string delta_unit = "da";
    int    min_ffc_number = 3;
    double line_tol = -1.0;
    // tolerance model
    double tol_ppm = 25.0;
    double precursor_ppm = 10.0;
    double sigma_floor_ppm = 3.0;
    double mock_sigma_cap_da = 0.05;
    // master-set construction
    int    master_align_threshold = 1;
    bool   accept_ambiguity = false;
    bool   use_mocks = true;
    bool   include_isotope_lines = true;
    double isotope_tol_da = 0.03;
    int    mock_min_ffc = 4;
    double mock_weight = 1.0;
    double secondary_gate = 0.25;
    double evidence_weight = 0.0;
    double evidence_cap = 1.0;
    double rank_floor = 1.0;
    double rank_scale = -1.0;          // <0 -> len(table)
    bool   gate_mock_on_goodness = true;
    int    max_rounds = 20;
    // reconstruction
    int    max_paths = 1000;
    double label_tol = 0.02;
    int    edges_max_n = 20;
    //: Multiplies every node sigma AND the precursor sigma of the FINAL graph
    //: only -- discovery (the rounds) and rendering (merge_tol) are untouched.
    //: 1.0 is the public default; the research RBDP benchmark used 0.4.
    double recon_sigma_scale = 1.0;
    //: "enumeration" (DFS order, the public default) | "informative".
    std::string presentation_order = "enumeration";
    //: unit | rbdp | rbdp-guarded  (see rbdp_types.hpp)
    std::string reconstruction_method = "unit";
    RbdpSettings rbdp;
    // identity, carried into evidence ids only
    std::string dataset_id = "data";
};

// One member FFC of a line, as two singly-protonated side masses.
struct Tffc {
    double x = 0, y = 0;
    int ch_x = 1, ch_y = 1;
    int ranking = -1;
    long row = -1;                     // FfcTable::row of the member FFC
};

struct LineInfo {
    std::string key;
    int i = 0, j = 0;
    double mass = 0.0;
    double a_shift = 0.0;              // alignment shift, from the FFC masses
    std::vector<Tffc> tffcs;
    std::string category = "other";    // parental | loss | internal | isotope | other
    std::string detail;
    int  n_inactive = 0, n_both_known = 0;
    bool eligible = false;
    int  n_banked = 0, n_unambiguous = 0, n_ambiguous = 0, n_unproductive = 0;
    bool   has_mock = false;
    double mock_coord = 0.0, mock_sigma = 0.0;
    std::string mock_refused;

    int n_ffc() const { return static_cast<int>(tffcs.size()); }
    int best_rank() const;
};

// One node of the frozen final graph: a master entry in sorted-coordinate
// order.  node_id is its position in that order.
struct FrozenNode {
    int    node_id = 0;
    size_t entry = 0;                  // index into Result::master
    double coord = 0.0;
    double discovery_sigma = 0.0;      // what the rounds used
    double graph_sigma = 0.0;          // discovery_sigma * recon_sigma_scale
    double weight = 1.0;
};

struct Result {
    double total = 0.0, term = 0.0;
    size_t n_ffc = 0;                  // rows after the top-N cut
    Basis  basis;
    int    line_charge = 0;
    std::vector<LineInfo> lines;
    std::vector<Entry>    master;
    std::vector<std::string> reconstructions;
    //: The same paths as node (x1, x2) sequences.  Only the benchmark scorer
    //: needs these -- it re-derives each path's implied breakpoints to ask
    //: whether they all sit on a real residue boundary.
    std::vector<Path> paths_raw;
    int    n_paths = 0;
    int    rounds = 0;
    double best_weight = -1.0;
    long   dag_builds = 0;

    // ── frozen final graph and v2 artifact fields ──────────────────────────
    std::vector<FrozenNode> nodes;
    double precursor_sigma = 0.0;          // discovery
    double graph_precursor_sigma = 0.0;    // final search (scaled)
    double merge_tol = 0.0;                // rendering (unscaled)
    //: aligned with `reconstructions` / `paths_raw` (presentation order)
    std::vector<StructuredPath> structured;
    std::vector<std::vector<std::pair<int,int>>> path_node_ids;
    std::vector<std::vector<std::string>> path_hypotheses;
    //: exact number of optimal paths of the REPORTED method, from the tie
    //: DAG -- never from enumeration.  Empty string == unknown (no path).
    std::string exact_optimal_path_count;
    bool   enumeration_truncated = false;
    std::string score_method = "unit";
    //: digest of everything discovery produced (lines, master set).  Must not
    //: depend on the reconstruction method or the final-graph sigma scale.
    std::string discovery_hash;
    std::string status = "ok";             // ok | no_path | invalid_parity | rbdp statuses
    RbdpReport rbdp;
};

Result run_denovo(const FfcTable& t, int parent_charge, double parent_mass,
                  const std::string& terminus, const std::string& fragmentation,
                  const Params& p, const std::vector<double>& allowed);

}  // namespace denovo
