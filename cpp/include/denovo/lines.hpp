// Orientation-agnostic line detection -- a port of `detect_lines_oa` in
// src/line_based_annotation/line_internal_annotation_oa_v2.py.
//
// A "line" is a group of FFCs sharing a reconstructed mass v = i*mzA + j*mzB.
// A line (i, j) and its mirror (j, i) at the same mass are the SAME physical
// line (the deconvolution's A/B ordering is arbitrary), so membership is
// counted on the canonical charge (min(i,j), max(i,j)) pooling both
// orientations, and each member's true orientation is resolved per-FFC later.
//
// Detection reads NO charge information from the table; each side's charge is
// recovered from the line the FFC lands on.  That is the whole point of `_oa`.
#pragma once
#include <vector>
#include <string>
#include "denovo/ffc.hpp"

namespace denovo {

struct Line {
    int    i = 0, j = 0;              // canonical charge (min, max)
    double mass = 0.0;                // mean over the dominant-charge members
    int    cluster_size = 0;
    int    n_on_line = 0;
    std::vector<size_t> members;      // row positions into the FfcTable
    std::string kind = "greedy";      // parental | isotopic | neutral | greedy
};

struct DetectParams {
    double delta = 0.015;             // Sort-and-Split gap
    std::string delta_unit = "da";    // "da" | "ppm"
    int    min_ffc_number = 3;        // distinct FFCs needed to keep a line
    double line_tol = -1.0;           // <0 -> defaults to delta
    int    max_lines = 10000;
};

struct DetectResult {
    std::vector<Line> master;         // parental-charge lines, cleared first
    std::vector<Line> residual;       // everything greedy peeling found after
};

DetectResult detect_lines_oa(const FfcTable& t, int parent_charge,
                             const DetectParams& p);

}  // namespace denovo
