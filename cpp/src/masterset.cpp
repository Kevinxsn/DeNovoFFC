#include "denovo/masterset.hpp"
#include "denovo/constants.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace denovo {

MasterSet::MasterSet(double total, double sigma_floor_ppm, double mock_weight,
                     double evidence_weight, double evidence_cap,
                     double rank_floor, double rank_scale)
    : total_(total), sigma_floor_ppm_(sigma_floor_ppm),
      mock_weight_(mock_weight), evidence_weight_(evidence_weight),
      evidence_cap_(evidence_cap), rank_floor_(rank_floor),
      rank_scale_(std::max(rank_scale, 1.0)) {}

double MasterSet::rank_factor(int rank) const {
    if (rank_floor_ >= 1.0 || rank < 0) return 1.0;
    const double frac = std::min(std::max(rank / rank_scale_, 0.0), 1.0);
    return 1.0 - (1.0 - rank_floor_) * frac;
}

double MasterSet::weight_of(bool all_mock, int n, int rank) const {
    double base = all_mock ? mock_weight_ : 1.0;
    base *= rank_factor(rank);
    return base + std::min(evidence_weight_ * (n - 1), evidence_cap_);
}

const Entry* MasterSet::find(double coord, double sigma) const {
    const Entry* best = nullptr;
    double best_d = 0.0;
    for (const Entry& e : entries_) {
        const double d = std::fabs(e.coord - coord);
        if (d <= e.sigma + sigma && (best == nullptr || d < best_d)) {
            best = &e; best_d = d;
        }
    }
    return best;
}

bool MasterSet::add(double coord, double sigma, const std::string& kind,
                    const std::string& origin, int rank, int origin_line) {
    const double c = fold(coord, total_);
    if (c <= sigma || c >= total_ - sigma) return false;   // a terminus

    // find() is const; re-locate the mutable entry the same way.
    Entry* hit = nullptr;
    double best_d = 0.0;
    for (Entry& e : entries_) {
        const double d = std::fabs(e.coord - c);
        if (d <= e.sigma + sigma && (hit == nullptr || d < best_d)) {
            hit = &e; best_d = d;
        }
    }
    if (hit != nullptr) {
        // Inverse-variance merge: the textbook estimator for independent
        // measurements of one quantity, which is what two TFFCs on the same
        // cleavage are.  The floor stops sigma shrinking past what the
        // instrument can actually deliver.
        const double w_old = 1.0 / std::pow(std::max(hit->sigma, 1e-9), 2);
        const double w_new = 1.0 / std::pow(std::max(sigma,       1e-9), 2);
        hit->coord = (hit->coord * w_old + c * w_new) / (w_old + w_new);
        hit->sigma = std::max(1.0 / std::sqrt(w_old + w_new),
                              hit->coord * sigma_floor_ppm_ * 1e-6);
        hit->n += 1;
        hit->kinds.push_back(kind);
        hit->all_mock = hit->all_mock && (kind == "mock");
        if (rank >= 0 && (hit->best_rank < 0 || rank < hit->best_rank))
            hit->best_rank = rank;
        if (!origin.empty()) hit->origins.push_back(origin);
        if (origin_line >= 0) hit->origin_lines.push_back(origin_line);
        return false;
    }

    Entry e;
    e.coord = c;
    e.sigma = std::max(sigma, c * sigma_floor_ppm_ * 1e-6);
    e.n = 1;
    e.all_mock = (kind == "mock");
    e.best_rank = rank;
    e.kinds.push_back(kind);
    if (!origin.empty()) e.origins.push_back(origin);
    if (origin_line >= 0) e.origin_lines.push_back(origin_line);
    entries_.push_back(std::move(e));
    return true;
}

View MasterSet::view(const std::vector<Candidate>& extra) const {
    std::vector<std::tuple<double,double,double>> vals;
    vals.reserve(entries_.size() + extra.size());
    for (const Entry& e : entries_)
        vals.emplace_back(e.coord, e.sigma,
                          weight_of(e.all_mock, e.n, e.best_rank));
    for (const Candidate& c : extra) {
        const double fc = fold(c.coord, total_);
        if (c.sigma < fc && fc < total_ - c.sigma)
            vals.emplace_back(fc, c.sigma, weight_of(c.is_mock, 1, c.rank));
    }
    std::sort(vals.begin(), vals.end());        // lexicographic, as Python's

    View v;
    v.coords.reserve(vals.size());
    v.sigmas.reserve(vals.size());
    v.weights.reserve(vals.size());
    for (const auto& t : vals) {
        v.coords.push_back(std::get<0>(t));
        v.sigmas.push_back(std::get<1>(t));
        v.weights.push_back(std::get<2>(t));
    }
    return v;
}

}  // namespace denovo
