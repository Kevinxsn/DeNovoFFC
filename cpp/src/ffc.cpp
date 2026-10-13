#include "denovo/ffc.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace denovo {
namespace {

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(line);
    while (std::getline(ss, cur, sep)) out.push_back(cur);
    if (!line.empty() && line.back() == sep) out.emplace_back();
    return out;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back()  == '\r' || s.back()  == ' ')) s.pop_back();
    size_t b = 0;
    while (b < s.size() && s[b] == ' ') ++b;
    return s.substr(b);
}

char sniff_sep(const std::string& header) {
    if (header.find('\t') != std::string::npos) return '\t';
    if (header.find(',')  != std::string::npos) return ',';
    return '\t';
}

}  // namespace

FfcTable read_ffc_table(const std::string& path, const std::string& col_a,
                        const std::string& col_b, const std::string& ranking_col) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open FFC table: " + path);

    std::string header;
    if (!std::getline(in, header))
        throw std::runtime_error("empty FFC table: " + path);
    const char sep = sniff_sep(header);
    const auto cols = split(header, sep);

    int ia = -1, ib = -1, ir = -1;
    for (size_t k = 0; k < cols.size(); ++k) {
        const std::string c = trim(cols[k]);
        if (c == col_a) ia = static_cast<int>(k);
        if (c == col_b) ib = static_cast<int>(k);
        if (c == ranking_col) ir = static_cast<int>(k);
    }
    if (ia < 0 || ib < 0)
        throw std::runtime_error("FFC table " + path + " has no '" + col_a +
                                 "' / '" + col_b + "' columns");

    FfcTable t;
    std::string line;
    long data_row = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        ++data_row;
        const auto f = split(line, sep);
        if (static_cast<int>(f.size()) <= std::max(ia, ib)) continue;
        const std::string sa = trim(f[ia]), sb = trim(f[ib]);
        if (sa.empty() || sb.empty()) continue;          // dropna()
        const double a = std::strtod(sa.c_str(), nullptr);
        const double b = std::strtod(sb.c_str(), nullptr);
        if (std::isnan(a) || std::isnan(b)) continue;
        int rank = -1;
        if (ir >= 0 && ir < static_cast<int>(f.size())) {
            const std::string sr = trim(f[ir]);
            if (!sr.empty()) rank = static_cast<int>(std::strtod(sr.c_str(), nullptr));
        }
        t.mz_a.push_back(a);
        t.mz_b.push_back(b);
        t.ranking.push_back(rank);
        t.row.push_back(data_row);
    }
    return t;
}

FfcTable prepare(const FfcTable& in, int top_n) {
    std::vector<size_t> order;
    order.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k)
        if (in.ranking[k] != -1) order.push_back(k);
    // pandas sort_values is a stable mergesort by default.
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t p, size_t q) { return in.ranking[p] < in.ranking[q]; });
    if (top_n > 0 && order.size() > static_cast<size_t>(top_n))
        order.resize(static_cast<size_t>(top_n));

    FfcTable out;
    out.mz_a.reserve(order.size());
    out.mz_b.reserve(order.size());
    out.ranking.reserve(order.size());
    out.row.reserve(order.size());
    for (size_t k : order) {
        out.mz_a.push_back(in.mz_a[k]);
        out.mz_b.push_back(in.mz_b[k]);
        out.ranking.push_back(in.ranking[k]);
        out.row.push_back(in.row.empty() ? static_cast<long>(k) : in.row[k]);
    }
    return out;
}

}  // namespace denovo
