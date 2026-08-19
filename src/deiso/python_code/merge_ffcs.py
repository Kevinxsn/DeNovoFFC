"""
Duplicate-FFC merger
====================
After deconvolution the same raw FFC point can appear multiple times
(different charge-state hypotheses, different line passes, etc.).
``merge_duplicate_ffcs`` collapses those duplicates, keeping the single
row that has the best (lowest) Ranking for each unique (m/z A, m/z B) pair.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_MASS_H = 1.00784  # hydrogen mass (Da); matches annotation.py, used to convert observed m/z to neutral mass


def _cluster_dedup_by_mz(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    ranking_col: str,
    tolerance: float,
    symmetric: bool,
    unranked_value: int,
) -> pd.DataFrame:
    """Deduplicate rows whose (col_a, col_b) values are within `tolerance` of
    each other, using proximity clustering rather than fixed-grid rounding.

    Grid rounding (np.round(x / tolerance) * tolerance) can split two values
    that are genuinely within `tolerance` of each other into different bins
    when they straddle a grid boundary (e.g. 896.451551 and 896.451255 are
    only ~0.0003 apart but round to 896.452 vs 896.451 at tolerance=0.001).
    This clusters points via a KD-tree radius query (Chebyshev distance, so
    both m/z A and m/z B must independently be within `tolerance`), unioning
    transitively connected points into one cluster, and keeps the
    best-Ranking row per cluster.
    """
    if df.empty or tolerance <= 0 or len(df) < 2:
        return df.copy()

    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)

    if symmetric:
        key_a, key_b = np.minimum(a, b), np.maximum(a, b)
    else:
        key_a, key_b = a, b

    tree = cKDTree(np.column_stack([key_a, key_b]))
    pairs = tree.query_pairs(r=tolerance, p=np.inf)

    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        union(i, j)

    sort_rank = df[ranking_col].replace(unranked_value, float("inf")).to_numpy(dtype=float)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep_indices = sorted(min(members, key=lambda idx: sort_rank[idx]) for members in clusters.values())
    return df.iloc[keep_indices].reset_index(drop=True)


def merge_duplicate_ffcs(
    df: pd.DataFrame,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    tolerance: float = 0.0,
    symmetric: bool = False,
    unranked_value: int = -1,
    reset_ranking: bool = True,
) -> pd.DataFrame:
    """
    Collapse duplicate FFC rows, keeping the best-ranked row per unique pair.

    Parameters
    ----------
    df : DataFrame
        FFC data that may contain duplicate (col_a, col_b) pairs.
    col_a, col_b : str
        Column names for the two m/z values used to identify duplicates.
    ranking_col : str
        Column used to pick the winner within each duplicate group.
        Lower value = better.  Rows whose ranking equals ``unranked_value``
        are treated as worst and only kept if no ranked row exists for that pair.
    tolerance : float
        If > 0, m/z values are rounded to the nearest multiple of this value
        before grouping, so pairs within ±tolerance/2 of each other collapse.
        If 0 (default), exact float equality is used.
    symmetric : bool
        If True, treat (A, B) and (B, A) as the same FFC pair by
        canonicalising so the smaller value is always in col_a.
    unranked_value : int
        Sentinel value that marks an unranked row (default -1).
    reset_ranking : bool
        If True, reassign rankings to consecutive integers (1, 2, 3, …) based
        on the relative order of rows after merging.  FFC(X, Y) and FFC(Y, X)
        are treated as the same pair for this purpose and receive the same new
        rank number.  Unranked rows (ranking == unranked_value) keep
        ``unranked_value`` and remain at the end.

    Returns
    -------
    DataFrame with one row per unique pair, sorted by ranking_col ascending.
    All original columns are preserved; only the best row's values are kept.
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    # ── Build group keys ──────────────────────────────────────────────────
    a = result[col_a].to_numpy(dtype=float)
    b = result[col_b].to_numpy(dtype=float)

    if tolerance > 0:
        a = np.round(a / tolerance) * tolerance
        b = np.round(b / tolerance) * tolerance

    if symmetric:
        # Canonicalise so the smaller value is always in key_a
        key_a = np.minimum(a, b)
        key_b = np.maximum(a, b)
    else:
        key_a, key_b = a, b

    result["_key_a"] = key_a
    result["_key_b"] = key_b

    # ── Sort so best-ranked rows come first ───────────────────────────────
    # Replace unranked sentinel with inf so it sorts last
    sort_rank = result[ranking_col].replace(unranked_value, float("inf"))
    result["_sort_rank"] = sort_rank

    result = (
        result
        .sort_values("_sort_rank")
        .drop_duplicates(subset=["_key_a", "_key_b"], keep="first")
        .drop(columns=["_key_a", "_key_b", "_sort_rank"])
    )

    # Restore a clean index, sorted by ranking (unranked last)
    ranked = result[result[ranking_col] != unranked_value].sort_values(ranking_col)
    unranked = result[result[ranking_col] == unranked_value]
    result = pd.concat([ranked, unranked], ignore_index=True)

    if reset_ranking:
        # Build canonical (min, max) keys so FFC(X,Y) and FFC(Y,X) land in the
        # same group regardless of the symmetric merge setting.
        a_vals = result[col_a].to_numpy(dtype=float)
        b_vals = result[col_b].to_numpy(dtype=float)
        if tolerance > 0:
            a_vals = np.round(a_vals / tolerance) * tolerance
            b_vals = np.round(b_vals / tolerance) * tolerance

        result["_sym_a"] = np.minimum(a_vals, b_vals)
        result["_sym_b"] = np.maximum(a_vals, b_vals)

        # Representative rank for each canonical group = min of non-unranked values.
        def _group_min(x: pd.Series) -> float:
            valid = x[x != unranked_value]
            return float(valid.min()) if len(valid) > 0 else float("inf")

        result["_group_rank"] = result.groupby(["_sym_a", "_sym_b"])[ranking_col].transform(_group_min)

        # Assign sequential integers to ranked groups; preserve unranked_value for unranked groups.
        unique_groups = (
            result[["_sym_a", "_sym_b", "_group_rank"]]
            .drop_duplicates(subset=["_sym_a", "_sym_b"])
            .sort_values("_group_rank")
        )
        ranked_groups = unique_groups[unique_groups["_group_rank"] != float("inf")].copy()
        unranked_groups = unique_groups[unique_groups["_group_rank"] == float("inf")].copy()
        ranked_groups["_new_rank"] = range(1, len(ranked_groups) + 1)
        unranked_groups["_new_rank"] = unranked_value

        group_map = pd.concat([ranked_groups, unranked_groups])[["_sym_a", "_sym_b", "_new_rank"]]
        result = result.merge(group_map, on=["_sym_a", "_sym_b"], how="left")
        result[ranking_col] = result["_new_rank"].astype(result[ranking_col].dtype)
        result = result.drop(columns=["_sym_a", "_sym_b", "_group_rank", "_new_rank"])

    return result


def merge_duplicate_ffcs_charge(
    df: pd.DataFrame,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    charge_col_a: str = "charge_A",
    charge_col_b: str = "charge_B",
    ranking_col: str = "Ranking",
    tolerance: float = 0.001,
    symmetric: bool = False,
    unranked_value: int = -1,
    reset_ranking: bool = True,
    default_charge: float = 1.0,
) -> pd.DataFrame:
    """
    Charge-aware variant of merge_duplicate_ffcs.

    Groups FFC rows by their Transformed FFC (TFFC) key — the charge-independent
    neutral mass derived from each observed m/z and its assigned charge:

        mass_i(X) = i * X - (i - 1) * MASS_H
        mass_j(Y) = j * Y - (j - 1) * MASS_H

    Two rows are treated as duplicates if their (mass_i(X), mass_j(Y)) pair
    matches (within ``tolerance``), regardless of which (charge_A, charge_B)
    assignments produced those masses.

    Rows with a missing (NaN) charge on either side are treated as though
    that charge were ``default_charge`` (1, by default) purely for computing
    their TFFC key. With charge 1 the mass transform is the identity
    (mass = m/z), so this simply groups those rows by their raw m/z — which
    means FFC(X, Y) / FFC(Y, X) duplicates are still recognised and merged
    even when neither copy has a charge assignment (when ``symmetric=True``),
    and these rows participate in the same best-ranking-wins dedup and
    reset_ranking pass as charge-valid rows, instead of being appended
    afterwards, unranked.

    Before the TFFC pass, rows are first deduplicated by raw (m/z A, m/z B)
    proximity alone (ignoring charge), via merge_duplicate_ffcs(). This
    matters because the same physical peak pair is often reported multiple
    times by different deconvolution methods with inconsistent charge
    completeness -- e.g. one method fully resolves both charges while
    another leaves one side NaN. Assuming default_charge for that NaN side
    would compute a TFFC key based on the *wrong* charge (mismatching the
    other method's correctly-resolved key), so the two copies of the same
    peak would otherwise survive as two separate "unique" TFFCs instead of
    one. Raw-m/z dedup catches this regardless of charge completeness; the
    subsequent TFFC pass then still catches genuinely different raw m/z
    that represent the same underlying neutral fragment at a different
    real charge state.

    Parameters
    ----------
    df : DataFrame
        FFC data with m/z and charge columns.
    col_a, col_b : str
        Column names for the two observed m/z values.
    charge_col_a, charge_col_b : str
        Column names carrying the deconvolution charge assignments.
    ranking_col : str
        Column used to pick the winner within each duplicate group.
        Lower value = better.
    tolerance : float
        If > 0, transformed masses are rounded to this grid before grouping
        (Stage 2). The raw-m/z pre-pass (Stage 1) uses the same tolerance as
        a proximity-clustering radius instead (see _cluster_dedup_by_mz).
    symmetric : bool
        If True, treat (mass_i(X), mass_j(Y)) and (mass_j(Y), mass_i(X)) as
        the same TFFC by canonicalising so the smaller mass is first. Also
        applied to the raw-m/z pre-pass.
    unranked_value : int
        Sentinel value that marks an unranked row (default -1).
    reset_ranking : bool
        If True, reassign rankings to consecutive integers (1, 2, 3, …) based
        on relative order after merging.  TFFC(X*, Y*) and TFFC(Y*, X*) share
        the same new rank.
    default_charge : float
        Charge assumed for rows with a NaN charge_col_a/charge_col_b value,
        used only to compute their TFFC key (default 1.0).

    Returns
    -------
    DataFrame: one row per unique TFFC, sorted by ranking_col ascending.
    """
    if df.empty:
        return df.copy()

    # ── Stage 1: dedup by raw m/z proximity, regardless of charge ─────────────
    # Collapses redundant detections of the same physical peak pair across
    # deconv methods before charge-based TFFC grouping is attempted. Uses
    # proximity clustering (not grid rounding) so genuinely-close values
    # straddling a rounding-grid boundary aren't missed.
    result = _cluster_dedup_by_mz(
        df, col_a=col_a, col_b=col_b, ranking_col=ranking_col,
        tolerance=tolerance, symmetric=symmetric, unranked_value=unranked_value,
    )

    # ── Stage 2: dedup by TFFC (charge-transformed neutral mass) ──────────────
    # A missing charge is treated as default_charge (identity transform when 1),
    # so NaN-charge rows are grouped/deduplicated/ranked alongside charge-valid
    # rows instead of being handled as a separate unranked bucket.
    ca = result[charge_col_a].fillna(default_charge).to_numpy(dtype=float)
    cb = result[charge_col_b].fillna(default_charge).to_numpy(dtype=float)
    mz_a = result[col_a].to_numpy(dtype=float)
    mz_b = result[col_b].to_numpy(dtype=float)

    key_a = ca * mz_a - (ca - 1) * _MASS_H
    key_b = cb * mz_b - (cb - 1) * _MASS_H

    if tolerance > 0:
        key_a = np.round(key_a / tolerance) * tolerance
        key_b = np.round(key_b / tolerance) * tolerance

    if symmetric:
        key_a, key_b = np.minimum(key_a, key_b), np.maximum(key_a, key_b)

    result["_key_a"] = key_a
    result["_key_b"] = key_b

    # ── Sort so best-ranked rows come first, then deduplicate ─────────────────
    sort_rank = result[ranking_col].replace(unranked_value, float("inf"))
    result["_sort_rank"] = sort_rank

    result = (
        result
        .sort_values("_sort_rank")
        .drop_duplicates(subset=["_key_a", "_key_b"], keep="first")
        .drop(columns=["_key_a", "_key_b", "_sort_rank"])
    )

    # Restore clean index, ranked rows first
    ranked = result[result[ranking_col] != unranked_value].sort_values(ranking_col)
    unranked_rows = result[result[ranking_col] == unranked_value]
    result = pd.concat([ranked, unranked_rows], ignore_index=True)

    # ── Optionally reset rankings to consecutive integers ─────────────────────
    if reset_ranking:
        # Recompute TFFC masses from the surviving rows
        ca2 = result[charge_col_a].fillna(default_charge).to_numpy(dtype=float)
        cb2 = result[charge_col_b].fillna(default_charge).to_numpy(dtype=float)
        mz_a2 = result[col_a].to_numpy(dtype=float)
        mz_b2 = result[col_b].to_numpy(dtype=float)

        sym_a = ca2 * mz_a2 - (ca2 - 1) * _MASS_H
        sym_b = cb2 * mz_b2 - (cb2 - 1) * _MASS_H
        if tolerance > 0:
            sym_a = np.round(sym_a / tolerance) * tolerance
            sym_b = np.round(sym_b / tolerance) * tolerance

        # Canonical (min, max) so TFFC(X*,Y*) and TFFC(Y*,X*) share the same rank slot
        result["_sym_a"] = np.minimum(sym_a, sym_b)
        result["_sym_b"] = np.maximum(sym_a, sym_b)

        def _group_min(x: pd.Series) -> float:
            valid = x[x != unranked_value]
            return float(valid.min()) if len(valid) > 0 else float("inf")

        result["_group_rank"] = result.groupby(["_sym_a", "_sym_b"])[ranking_col].transform(_group_min)

        unique_groups = (
            result[["_sym_a", "_sym_b", "_group_rank"]]
            .drop_duplicates(subset=["_sym_a", "_sym_b"])
            .sort_values("_group_rank")
        )
        ranked_groups = unique_groups[unique_groups["_group_rank"] != float("inf")].copy()
        unranked_groups = unique_groups[unique_groups["_group_rank"] == float("inf")].copy()
        ranked_groups["_new_rank"] = range(1, len(ranked_groups) + 1)
        unranked_groups["_new_rank"] = unranked_value

        group_map = pd.concat([ranked_groups, unranked_groups])[["_sym_a", "_sym_b", "_new_rank"]]
        result = result.merge(group_map, on=["_sym_a", "_sym_b"], how="left")
        result[ranking_col] = result["_new_rank"].astype(result[ranking_col].dtype)
        result = result.drop(columns=["_sym_a", "_sym_b", "_group_rank", "_new_rank"])

    return result


# ── Example / quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame({
        "m/z A":    [159.092, 159.092, 181.109, 181.109, 215.140],
        "m/z B":    [181.109, 181.109, 159.092, 159.092, 159.092],
        "Covariance": [5556.2, 5556.2, 5556.2, 5556.2, -2294.1],
        "Ranking":  [1183, 500, 1183, 750, 138541],
        "charge_A": [1, 2, 2, 1, 1],
        "charge_B": [2, 1, 1, 2, 1],
        "deconv_method": ["ffc_line", "ffc_line", "ffc_line", "ffc_line", "ffc_line"],
    })

    print("=== Input ===")
    print(sample.to_string(index=False))

    print("\n=== Merged (exact, non-symmetric) ===")
    out = merge_duplicate_ffcs(sample)
    print(out.to_string(index=False))

    print("\n=== Merged (exact, symmetric) ===")
    out_sym = merge_duplicate_ffcs(sample, symmetric=True)
    print(out_sym.to_string(index=False))

    print("\n=== Merged (non-symmetric, reset_ranking=True) ===")
    out_reset = merge_duplicate_ffcs(sample, reset_ranking=True)
    print(out_reset.to_string(index=False))

    print("\n=== Merged (symmetric, reset_ranking=True) ===")
    out_sym_reset = merge_duplicate_ffcs(sample, symmetric=True, reset_ranking=True)
    print(out_sym_reset.to_string(index=False))

    # ── merge_duplicate_ffcs_charge demo ─────────────────────────────────────
    # FFC (159.092, 181.109) appears with two charge assignments:
    #   charge (1,2): TFFC = (159.092, 362.210)   ranking 1183
    #   charge (2,1): TFFC = (317.177, 181.109)   ranking 500  ← different TFFC → kept separately
    # FFC (181.109, 159.092) with charge (1,2): TFFC = (181.109, 317.177) → same neutral masses
    #   as (2,1) of the reverse pair above (symmetric), but different assignment → depends on symmetric flag
    sample_charge = pd.DataFrame({
        "m/z A":    [159.092, 159.092, 181.109, 181.109, 215.140],
        "m/z B":    [181.109, 181.109, 159.092, 159.092, 159.092],
        "Covariance": [5556.2, 5556.2, 5556.2, 5556.2, -2294.1],
        "Ranking":  [1183,    500,     1183,    750,     138541],
        "charge_A": [1,       2,       2,       1,       1],
        "charge_B": [2,       1,       1,       2,       1],
        "deconv_method": ["ffc_line"] * 5,
    })

    print("\n=== Charge-aware merge (non-symmetric) ===")
    out_c = merge_duplicate_ffcs_charge(sample_charge)
    print(out_c.to_string(index=False))

    print("\n=== Charge-aware merge (symmetric) ===")
    out_cs = merge_duplicate_ffcs_charge(sample_charge, symmetric=True)
    print(out_cs.to_string(index=False))
