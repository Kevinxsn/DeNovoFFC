// The FFC table: one row is a fragment-fragment correlation, an m/z pair with
// per-side charges and a Ranking.
//
// The table this reads is expected to be ALREADY MERGED (mirror duplicates
// collapsed, rankings reset) -- see DESIGN.md 4.2 for why that stage stays in
// Python.  `--top-n` here only does prepare_ffc_data's part: drop Ranking==-1,
// sort by Ranking, keep the head.
#pragma once
#include <string>
#include <vector>

namespace denovo {

struct FfcTable {
    std::vector<double> mz_a, mz_b;
    std::vector<int>    ranking;      // -1 == unranked
    size_t size() const { return mz_a.size(); }
};

// Read a TSV/CSV with a header naming at least the two m/z columns.
FfcTable read_ffc_table(const std::string& path,
                        const std::string& col_a = "m/z A",
                        const std::string& col_b = "m/z B",
                        const std::string& ranking_col = "Ranking");

// prepare_ffc_data(top_n): drop unranked, sort by Ranking, head(top_n).
// top_n <= 0 keeps everything (but still drops unranked and sorts).
FfcTable prepare(const FfcTable& in, int top_n);

}  // namespace denovo
