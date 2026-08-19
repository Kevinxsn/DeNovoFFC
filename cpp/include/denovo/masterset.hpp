// The master TFFC-set: breakpoint coordinates, each carrying its own absolute
// uncertainty.
//
// This is the one real design idea of the de novo module (DESIGN.md 5.2).
// The annotation module snapped every coordinate onto the exact prefix[k] of
// the known sequence; de novo we cannot, so instead:
//
//   * every coordinate's sigma is derived from THE MASS THAT WAS MEASURED,
//     never from the folded coordinate (folding moves a coordinate but not
//     its error), and
//   * two TFFCs landing on one breakpoint are independent measurements of one
//     quantity, so they are merged by inverse-variance weighting -- the
//     coordinate gets closer to exact the more evidence supports it.  That is
//     the de novo stand-in for snapping onto truth.
#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace denovo {

struct Entry {
    double coord = 0.0;
    double sigma = 0.0;
    int    n = 1;                       // independent TFFCs merged in
    bool   all_mock = true;             // every contributing kind was "mock"
    int    best_rank = -1;              // best (lowest) supporting FFC ranking
    std::vector<std::string> origins;   // which line proposed it
    std::vector<std::string> kinds;
};

// (sorted coords, their sigmas, their weights) -- the triple every DAG build
// and every goodness test consumes.  May contain duplicate coords, exactly as
// the reference's MasterSet.view() can.
struct View {
    std::vector<double> coords, sigmas, weights;
};

// A candidate mixed into a view without being committed: how the goodness
// test asks "would adding this breakpoint buy anything?".
struct Candidate {
    double coord, sigma;
    bool   is_mock = false;
    int    rank = -1;
};

class MasterSet {
public:
    MasterSet(double total, double sigma_floor_ppm = 3.0,
              double mock_weight = 1.0, double evidence_weight = 0.0,
              double evidence_cap = 1.0, double rank_floor = 1.0,
              double rank_scale = 1.0);

    // Two coordinates are the same breakpoint when they agree to within the
    // SUM of their uncertainties: a precisely measured light coordinate gets a
    // tight window and a mock a loose one, automatically.
    const Entry* find(double coord, double sigma = 0.0) const;
    bool contains(double coord, double sigma = 0.0) const {
        return find(coord, sigma) != nullptr;
    }
    size_t size() const { return entries_.size(); }
    const std::vector<Entry>& entries() const { return entries_; }

    // Returns true if the breakpoint was genuinely new.
    bool add(double coord, double sigma, const std::string& kind = "observed",
             const std::string& origin = "", int rank = -1);

    View view(const std::vector<Candidate>& extra = {}) const;

    double total() const { return total_; }

private:
    double weight_of(bool all_mock, int n, int rank) const;
    double rank_factor(int rank) const;

    double total_, sigma_floor_ppm_, mock_weight_;
    double evidence_weight_, evidence_cap_, rank_floor_, rank_scale_;
    std::vector<Entry> entries_;
};

}  // namespace denovo
