// Physical constants and the fragmentation basis.
//
// Every number here is copied from the Python reference
// (src/de_novo/denovo_core.py, src/reconstruct/master_reconstruction.py,
// src/deiso/python_code/greedy_line.py).  Do not "clean them up": several
// differ in the 5th decimal between modules on purpose (MASS_H vs PROTON,
// ISOTOPE_SPACING vs ISOTOPE_OFFSET) and the port must reproduce the
// reference bit-for-bit.
#pragma once
#include <array>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace denovo {

constexpr double PROTON          = 1.00725;    // strips the ionising proton
constexpr double MASS_H          = 1.00784;    // charge deconv: i*mz-(i-1)*H
constexpr double WATER           = 18.010565;
constexpr double AMMONIA         = 17.026549;
constexpr double NH3             = 17.02655;   // c-ion increment over b
constexpr double HYDROGEN        = 1.00783;    // the H ETD delivers
constexpr double ISOTOPE_OFFSET  = 1.00335;    // 13C - 12C (alignment shifts)
constexpr double ISOTOPE_SPACING = 1.00235;    // greedy_line satellite spacing
constexpr double MIN_AA_MASS     = 57.02146;   // glycine

// classify_line's named neutral losses, in the reference's dict order.
constexpr double LOSS_H2O = 18.01056;
constexpr double LOSS_NH3 = 17.02655;
constexpr double LOSS_CO  = 27.99491;

// greedy_line.DEFAULT_NEUTRAL_LOSSES (note: NOT the same literals as above).
constexpr double SAT_H2O = 18.010565;
constexpr double SAT_NH3 = 17.026549;
constexpr double SAT_CO  = 27.994915;

// path_finding.amino_acid_masses_merge -- I and L are merged into J.
// ⚠ The ORDER matters: _format_gapped picks the residue by
// min(|mass - gap|) and Python's min() keeps the FIRST minimum, so ties
// resolve by insertion order.
struct Residue { char letter; double mass; };
inline const std::array<Residue, 19>& residues() {
    static const std::array<Residue, 19> R{{
        {'A',  71.03711}, {'R', 156.10111}, {'N', 114.04293}, {'D', 115.02694},
        {'C', 103.00919}, {'E', 129.04259}, {'Q', 128.05858}, {'G',  57.02146},
        {'H', 137.05891}, {'J', 113.08406}, {'K', 128.09496}, {'M', 131.04049},
        {'F', 147.06841}, {'P',  97.05276}, {'S',  87.03203}, {'T', 101.04768},
        {'W', 186.07931}, {'Y', 163.06333}, {'V',  99.06841},
    }};
    return R;
}

inline double terminus_mass(const std::string& name) {
    if (name == "H2O" || name == "OH")  return WATER;
    if (name == "NH3" || name == "NH2") return AMMONIA;
    throw std::invalid_argument("terminus must be H2O/OH or NH3/NH2, got " + name);
}

// ── The fragmentation basis (denovo_core.Basis) ──────────────────────────
//
//            N-term ion       C-term ion           pair sum    ladder charge
//  HCD b/y   prefix[k]        total - prefix[k]    M + 2p      i + j = z
//  ETD c/z.  b_k + NH3        y_j + H - NH3        M + H + 2p  i + j = z - 1
//
// Everything downstream works on the COORDINATE basis these induce; only
// `parent_pair` and the sigma scales use the real precursor mass.
struct Basis {
    std::string mode = "HCD";
    int    charge_offset = 0;   //  0 for b/y, -1 for c/z (etd.md 2b)
    double side_nh3      = 0.0; //  subtracted from every measured side
    double pair_h        = 0.0; //  extra H in the pair sum
    double total         = 0.0; //  REAL neutral peptide mass
    double term          = 0.0; //  REAL C-terminal group

    // `total` on the coordinate basis (TOTAL_STAR for ETD).
    double coord_total() const { return total + pair_h - 2.0 * side_nh3; }
    // y-axis origin on the coordinate basis.  NEGATIVE for ETD -- harmless,
    // it is only ever an origin, and coord_total - coord_term == prefix[n].
    double coord_term()  const { return term  + pair_h - 2.0 * side_nh3; }
    // Sum of a complementary pair's two [M+H]+ masses.
    double parent_pair() const { return total + pair_h + 2.0 * PROTON; }

    // ⚠ ETD's -1 fails SILENTLY: electron transfer consumes one of the
    // precursor's protons, so a c/z pair from [M+6H]6+ carries five charges.
    // The ladder then sits at the same mass on a different charge sum and the
    // parental pass (i + j == parent_charge) cannot see it.
    //
    // A complementary pair is two DETECTED fragments, each carrying at least
    // one charge, so the budget must be >= 2.  The reference used to clamp an
    // impossible budget up to 2 silently; it is now an input error.
    int line_charge(int parent_charge) const {
        const int z = parent_charge + charge_offset;
        if (z < 2)
            throw std::invalid_argument(
                "precursor charge " + std::to_string(parent_charge) + " leaves a " +
                mode + " ladder charge budget of " + std::to_string(z) +
                "; a complementary pair needs >= 2 (b/y: z >= 2, c/z: z >= 3)");
        return z;
    }
};

inline Basis make_basis(const std::string& fragmentation, double parent_mass,
                        const std::string& terminus) {
    std::string key = fragmentation;
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
    Basis b;
    b.total = parent_mass;
    b.term  = terminus_mass(terminus);
    if (key == "HCD") {
        b.mode = "HCD";
    } else if (key == "ETD" || key == "ETHCD") {
        // EThcD carries both ladders at nearly the same line mass on different
        // charge sums (etd.md 6b), so one detector pass cannot serve both.
        // Run it as ETD to read the c/z ladder (much the stronger of the two).
        b.mode = "ETD";
        b.charge_offset = -1;
        b.side_nh3 = NH3;
        b.pair_h   = HYDROGEN;
    } else {
        throw std::invalid_argument("fragmentation must be HCD, ETD or ETHCD, got "
                                    + fragmentation);
    }
    return b;
}

// Fold a breakpoint coordinate into the lower half (0, total/2].
inline double fold(double r, double total) {
    return (r <= total - r) ? r : total - r;
}

// ([M+H]+ mass of one FFC side) -> (folded coordinate, sigma).
//
// ⚠ Two things happen here and they must not be confused.  The COORDINATE is
// on the fragmentation's basis (strip the proton, strip side_nh3, fold).
// SIGMA is referenced to the mass that was actually MEASURED -- never to the
// folded coordinate and never to the basis-shifted one.  The coordinate may
// sit tens of Da away (c/z) or on the far side of the peptide (a mirrored
// side), but the error travels with the measurement.  Mirroring additionally
// inherits whatever uncertainty the precursor mass carries.
inline void coord_from_side(double ion_mass, const Basis& b, double tol_ppm,
                            double precursor_ppm, double& coord, double& sigma) {
    const double m_meas = ion_mass - PROTON;
    const double u = m_meas - b.side_nh3;
    const double sig = std::fabs(m_meas) * tol_ppm * 1e-6;
    const double ct = b.coord_total();
    if (u <= ct - u) { coord = u; sigma = sig; }
    else { coord = ct - u; sigma = sig + b.total * precursor_ppm * 1e-6; }
}

}  // namespace denovo
