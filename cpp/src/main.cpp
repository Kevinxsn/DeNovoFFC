// denovo_mms -- the de novo MMS sequencer, command line.
//
//   denovo_mms --data map.tsv --charge 5 --mass 4001.3781 --terminus NH3 \
//              --top-n 500 --json out.json
//
// The input table must already be MERGED (mirror duplicates collapsed,
// rankings reset).  `--top-n` applies prepare_ffc_data's cut only.  See
// DESIGN.md 4.2.
#include "denovo/pipeline.hpp"
#include "denovo/edges.hpp"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace denovo;

namespace {

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout, R"(denovo_mms -- de novo sequencing from an MMS (FFC) map

Required
  --data PATH            merged FFC table (TSV/CSV with a header)
  --charge N             precursor charge
  --mass M               precursor NEUTRAL mass (Da)

Chemistry (both fail SILENTLY when wrong -- paths are still found, none right)
  --terminus H2O|NH3     C-terminal group                     [H2O]
  --fragmentation MODE   HCD | ETD | ETHCD                    [HCD]

Input
  --top-n N              keep the best-ranked N FFCs (0 = all)  [500]
  --col-a NAME           m/z column A                      [m/z A]
  --col-b NAME           m/z column B                      [m/z B]
  --ranking-col NAME     ranking column                   [Ranking]

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

Reconstruction
  --max-paths N          tied-path cap                          [1000]
  --label-tol X          residue-naming window                   [0.02]
  --edges-max-n N        residues per compound edge mass           [20]
  --edges-json PATH      read edges from a reference edges.json
                         (parity checking only; generation is faster)

Output
  --json PATH            write the full result as JSON
  --quiet                suppress the human-readable summary
)");
    std::exit(code);
}

const char* arg_after(int argc, char** argv, int& k, const char* flag) {
    if (k + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
    return argv[++k];
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data, terminus = "H2O", fragmentation = "HCD";
    std::string col_a = "m/z A", col_b = "m/z B", ranking_col = "Ranking";
    std::string json_path, edges_json;
    int charge = 0, top_n = 500;
    double mass = 0.0;
    bool quiet = false, have_mass = false;
    Params p;

    if (argc == 1) usage(1);
    for (int k = 1; k < argc; ++k) {
        const std::string a = argv[k];
        if      (a == "-h" || a == "--help") usage(0);
        else if (a == "--data")        data = arg_after(argc, argv, k, "--data");
        else if (a == "--charge")      charge = std::atoi(arg_after(argc, argv, k, "--charge"));
        else if (a == "--mass")      { mass = std::atof(arg_after(argc, argv, k, "--mass")); have_mass = true; }
        else if (a == "--terminus")    terminus = arg_after(argc, argv, k, "--terminus");
        else if (a == "--fragmentation") fragmentation = arg_after(argc, argv, k, "--fragmentation");
        else if (a == "--top-n")       top_n = std::atoi(arg_after(argc, argv, k, "--top-n"));
        else if (a == "--col-a")       col_a = arg_after(argc, argv, k, "--col-a");
        else if (a == "--col-b")       col_b = arg_after(argc, argv, k, "--col-b");
        else if (a == "--ranking-col") ranking_col = arg_after(argc, argv, k, "--ranking-col");
        else if (a == "--delta")       p.delta = std::atof(arg_after(argc, argv, k, "--delta"));
        else if (a == "--delta-unit")  p.delta_unit = arg_after(argc, argv, k, "--delta-unit");
        else if (a == "--min-ffc")     p.min_ffc_number = std::atoi(arg_after(argc, argv, k, "--min-ffc"));
        else if (a == "--line-tol")    p.line_tol = std::atof(arg_after(argc, argv, k, "--line-tol"));
        else if (a == "--tol-ppm")     p.tol_ppm = std::atof(arg_after(argc, argv, k, "--tol-ppm"));
        else if (a == "--precursor-ppm") p.precursor_ppm = std::atof(arg_after(argc, argv, k, "--precursor-ppm"));
        else if (a == "--sigma-floor-ppm") p.sigma_floor_ppm = std::atof(arg_after(argc, argv, k, "--sigma-floor-ppm"));
        else if (a == "--mock-sigma-cap") p.mock_sigma_cap_da = std::atof(arg_after(argc, argv, k, "--mock-sigma-cap"));
        else if (a == "--align-threshold") p.master_align_threshold = std::atoi(arg_after(argc, argv, k, "--align-threshold"));
        else if (a == "--accept-ambiguity") p.accept_ambiguity = true;
        else if (a == "--no-mocks")    p.use_mocks = false;
        else if (a == "--mock-min-ffc") p.mock_min_ffc = std::atoi(arg_after(argc, argv, k, "--mock-min-ffc"));
        else if (a == "--secondary-gate") p.secondary_gate = std::atof(arg_after(argc, argv, k, "--secondary-gate"));
        else if (a == "--no-isotope-lines") p.include_isotope_lines = false;
        else if (a == "--isotope-tol") p.isotope_tol_da = std::atof(arg_after(argc, argv, k, "--isotope-tol"));
        else if (a == "--no-mock-gate") p.gate_mock_on_goodness = false;
        else if (a == "--max-rounds")  p.max_rounds = std::atoi(arg_after(argc, argv, k, "--max-rounds"));
        else if (a == "--max-paths")   p.max_paths = std::atoi(arg_after(argc, argv, k, "--max-paths"));
        else if (a == "--label-tol")   p.label_tol = std::atof(arg_after(argc, argv, k, "--label-tol"));
        else if (a == "--edges-max-n") p.edges_max_n = std::atoi(arg_after(argc, argv, k, "--edges-max-n"));
        else if (a == "--edges-json")  edges_json = arg_after(argc, argv, k, "--edges-json");
        else if (a == "--json")        json_path = arg_after(argc, argv, k, "--json");
        else if (a == "--quiet")       quiet = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(2); }
    }
    if (data.empty() || charge <= 0 || !have_mass) {
        std::fprintf(stderr, "--data, --charge and --mass are required\n");
        usage(2);
    }

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

        const Result r = run_denovo(tab, charge, mass, terminus, fragmentation,
                                    p, allowed);
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
                        "%d tied path(s), weight %.0f\n",
                        r.master.size(), r.rounds, r.n_paths, r.best_weight);
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

        if (!json_path.empty()) {
            std::ofstream o(json_path);
            if (!o) { std::fprintf(stderr, "cannot write %s\n", json_path.c_str()); return 1; }
            o.setf(std::ios::fixed);
            o << "{\n  \"n_ffc\": " << tab.size()
              << ",\n  \"fragmentation\": \"" << r.basis.mode << "\""
              << ",\n  \"line_charge\": " << r.line_charge
              << ",\n  \"coord_total\": " << std::setprecision(6) << r.total
              << ",\n  \"coord_term\": " << r.term
              << ",\n  \"rounds\": " << r.rounds
              << ",\n  \"n_paths\": " << r.n_paths
              << ",\n  \"best_weight\": " << std::setprecision(3) << r.best_weight
              << ",\n  \"seconds\": " << secs(t0, t3)
              << ",\n  \"n_lines\": " << r.lines.size()
              << ",\n  \"reconstructions\": [";
            for (size_t k = 0; k < r.reconstructions.size(); ++k)
                o << (k ? ", " : "") << '"' << json_escape(r.reconstructions[k]) << '"';
            o << "],\n  \"master_set\": [";
            for (size_t k = 0; k < r.master.size(); ++k) {
                const Entry& e = r.master[k];
                o << (k ? ",\n    " : "\n    ")
                  << "{\"coord\": " << std::setprecision(6) << e.coord
                  << ", \"sigma\": " << e.sigma
                  << ", \"n\": " << e.n
                  << ", \"best_rank\": " << e.best_rank
                  << ", \"origins\": [";
                for (size_t q = 0; q < e.origins.size(); ++q)
                    o << (q ? ", " : "") << '"' << json_escape(e.origins[q]) << '"';
                o << "]}";
            }
            o << "\n  ],\n  \"lines\": [";
            for (size_t k = 0; k < r.lines.size(); ++k) {
                const LineInfo& li = r.lines[k];
                o << (k ? ",\n    " : "\n    ")
                  << "{\"key\": \"" << json_escape(li.key) << '"'
                  << ", \"i\": " << li.i << ", \"j\": " << li.j
                  << ", \"mass\": " << std::setprecision(4) << li.mass
                  << ", \"a_shift\": " << li.a_shift
                  << ", \"category\": \"" << li.category << '"'
                  << ", \"detail\": \"" << json_escape(li.detail) << '"'
                  << ", \"n_ffc\": " << li.n_ffc()
                  << ", \"n_inactive\": " << li.n_inactive
                  << ", \"eligible\": " << (li.eligible ? "true" : "false")
                  << ", \"n_banked\": " << li.n_banked
                  << ", \"unproductive\": " << li.n_unproductive
                  << ", \"unambiguous\": " << li.n_unambiguous
                  << ", \"ambiguous\": " << li.n_ambiguous
                  << ", \"mock\": " << (li.has_mock ? "true" : "false")
                  << ", \"mock_refused\": \"" << li.mock_refused << "\"}";
            }
            o << "\n  ]\n}\n";
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
