// The 2D-DAG and its heaviest-path search.
//
// Same edge semantics as the reference (path_finding.build_adjacency_graph /
// master_reconstruction.build_dag): Down, Right and Sink moves, and every gap
// must be an allowed compound residue mass.  The one change de novo makes is
// the window:
//
//     annotation:  tol = |heavier endpoint of the gap| * tol_ppm
//     here:        tol = sigma(a) + sigma(b)
//
// A gap is a difference of two measurements, so its uncertainty is the sum of
// theirs.  That is what lets a mock (error ~ total * tol_ppm) and a
// well-measured light fragment (~60x smaller) share one graph without either a
// window too tight to reach the mock or one loose enough to invent edges among
// the light coordinates.
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "denovo/masterset.hpp"

namespace denovo {

struct Dag {
    std::vector<double> coords;           // sorted, unique, incl. the source
    int source_node = -1;                 // node id of (0, term)
    // adjacency[node] = list of (target, gap mass, kind); target < 0 == Sink.
    struct Edge { int target; double mass; unsigned char kind; };  // 0=sink 1=down 2=right
    std::vector<std::vector<Edge>> adj;
    int n_coords = 0;

    int  node_id(int a, int b) const { return a * n_coords + b; }
    void decode(int id, int& a, int& b) const { a = id / n_coords; b = id % n_coords; }
};

Dag build_dag(const std::vector<double>& coords,
              const std::vector<double>& sigmas,
              const std::vector<double>& allowed,
              double total, double source_x1, double source_x2,
              double precursor_sigma);

// One path, N->C: the node sequence with the Sink sentinel already dropped.
struct Path { std::vector<std::pair<double,double>> nodes; };

// Heaviest weight only.  This is what the goodness test calls, hundreds of
// times per peptide, so it never materialises a path.
double heaviest_weight(const Dag& g, const std::vector<double>& weights_by_coord);

// Every maximum-weight source->sink path, capped at max_paths.  Replicates the
// reference's truncation semantics (neighbour order, break-on-cap) so the
// reported tie count matches.
double heaviest_paths(const Dag& g, const std::vector<double>& weights_by_coord,
                      int max_paths, std::vector<Path>& out);

// The evaluator with a cache, keyed on the rounded (coords, sigmas, weights)
// triple exactly as the reference's Scorer is -- different TFFCs frequently
// propose the same breakpoint, so this pays for itself many times over.
class Scorer {
public:
    Scorer(const std::vector<double>& allowed, double total,
           double source_x1, double source_x2, double precursor_sigma,
           int max_paths = 1000);

    double score(const View& v);
    double paths(const View& v, std::vector<Path>& out);
    long   builds() const { return n_builds_; }

private:
    Dag graph_of(const View& v);

    const std::vector<double>& allowed_;
    double total_, sx1_, sx2_, precursor_sigma_;
    int    max_paths_;
    long   n_builds_ = 0;
    std::unordered_map<std::string, double> cache_;
};

}  // namespace denovo
