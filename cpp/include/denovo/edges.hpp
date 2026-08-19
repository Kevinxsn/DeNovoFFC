// The compound amino-acid mass table ("edges") -- every mass a run of 1..N
// residues can have, quantised to 0.001 Da.  A DAG edge is admitted only if
// its gap matches one of these.
//
// The Python reference (src/reconstruct/edges.py) precomputes this into a
// 114 MB edges.json and reloads it on every run -- 1.5 s of JSON parsing per
// peptide, the single largest fixed cost of the Python pipeline.  Generating
// it here from a bitset takes ~30 ms, so the file is never needed.
#pragma once
#include <vector>
#include <string>

namespace denovo {

// Sorted, distinct compound masses of 1..max_n residues (Da).
// max_n = 20 reproduces the reference edges.json exactly (2,113,632 masses).
std::vector<double> build_edges(int max_n = 20);

// Read the "all_masses" array out of a reference edges.json.  Only for
// parity-checking against the Python path; build_edges() is faster and
// produces the identical list.
std::vector<double> load_edges_json(const std::string& path);

}  // namespace denovo
