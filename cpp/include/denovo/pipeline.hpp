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
#include "denovo/dag.hpp"

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
};

// One member FFC of a line, as two singly-protonated side masses.
struct Tffc {
    double x = 0, y = 0;
    int ch_x = 1, ch_y = 1;
    int ranking = -1;
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

struct Result {
    double total = 0.0, term = 0.0;
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
};

Result run_denovo(const FfcTable& t, int parent_charge, double parent_mass,
                  const std::string& terminus, const std::string& fragmentation,
                  const Params& p, const std::vector<double>& allowed);

}  // namespace denovo
