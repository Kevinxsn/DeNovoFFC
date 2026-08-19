"""
FFC Line-Cluster Detection
============================
Detects collinear structure in fragment-fragment correlation (FFC) maps.

Core idea
---------
In a 2D FFC map of (m/z_A, m/z_B) pairs, fragments that originate from
the same precursor with charge split (z_A, z_B) satisfy:

    z_A * m/z_A  +  z_B * m/z_B  ≈  M_parent

For each candidate charge split (i, j), we compute v = i·x + j·y for
every FFC point (x, y), then look for *clusters* of similar v-values.
Dense clusters are evidence that those points share a common parental
line at mass ≈ v.

Pipeline
--------
1. ``detect_line_clusters``  — scan all (i, j) splits, cluster v-values
2. ``compute_parent_offsets`` — express cluster centres as offsets from
   a known precursor mass  (Parent+X column)
3. ``merge_nearby_clusters``  — group clusters whose Parent+X values are
   within a tolerance (collapses isotopic / redundant detections)

External dependencies
---------------------
None beyond pandas / numpy.  The ``peptide`` and ``interpreter_modify``
imports are only needed at the caller level and are not used by the
library functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class LineCluster:
    """
    A single detected line-cluster for charge split (i, j).

    Attributes
    ----------
    i, j : int
        Charge-state multipliers (z_A, z_B).
    n_points : int
        Number of FFC points in this cluster.
    diameter : float
        Spread of the cluster (max_v - min_v).
    center : float
        Midpoint of the cluster ((max_v + min_v) / 2).
    min_v, max_v : float
        Extremes of v = i*x + j*y within the cluster.
    point_indices : ndarray
        Row indices (in the original DataFrame) of the member points.
    """
    i: int
    j: int
    n_points: int
    diameter: float
    center: float
    min_v: float
    max_v: float
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


# =============================================================================
# 2. CORE CLUSTERING
# =============================================================================

def _cluster_sorted_values(
    vals: np.ndarray,
    indices: np.ndarray,
    delta: float,
    min_size: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Single-linkage gap-based clustering on *pre-sorted* values.

    Two consecutive values belong to the same cluster if their difference
    is <= ``delta``.  Only clusters with at least ``min_size`` members are
    returned.

    Parameters
    ----------
    vals : 1-D array, **must be sorted ascending**
    indices : 1-D array of the same length -- original row indices
    delta : maximum gap between consecutive members
    min_size : minimum cluster membership

    Returns
    -------
    List of (member_indices, min_value, max_value) tuples.
    """
    if vals.size == 0:
        return []

    clusters: List[Tuple[np.ndarray, float, float]] = []
    start = 0

    for k in range(1, vals.size):
        if vals[k] - vals[k - 1] > delta:
            if k - start >= min_size:
                clusters.append((indices[start:k], vals[start], vals[k - 1]))
            start = k

    # Final segment
    if vals.size - start >= min_size:
        clusters.append((indices[start:], vals[start], vals[-1]))

    return clusters


def detect_line_clusters(
    df: pd.DataFrame,
    parent_charge: int,
    delta: float,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    enforce_sum_leq_charge: bool = True,
    min_cluster_size: int = 3,
    return_point_indices: bool = True,
) -> pd.DataFrame:
    """
    Scan all charge-split pairs (i, j) and detect clusters of consistent
    reconstructed mass v = i*x + j*y.

    Parameters
    ----------
    df : DataFrame
        Must contain ``col_a`` and ``col_b`` (the two m/z columns).
    parent_charge : int
        Maximum total charge to consider.
    delta : float
        Gap threshold for adjacent-value clustering (Da).
    col_a, col_b : str
        Column names for m/z of fragment A and B.
    enforce_sum_leq_charge : bool
        If True, skip (i, j) pairs where i + j > parent_charge.
    min_cluster_size : int
        Minimum number of points to form a valid cluster (>= 2).
    return_point_indices : bool
        Include an array of original DataFrame indices per cluster.

    Returns
    -------
    DataFrame sorted by (n_points desc, diameter asc) with columns:
        i, j, n_points, diameter, center, min_v, max_v,
        [point_indices if requested]

    Raises
    ------
    ValueError
        If parent_charge < 1, delta <= 0, or min_cluster_size < 2.
    KeyError
        If col_a or col_b are missing from the DataFrame.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if parent_charge < 1:
        raise ValueError("parent_charge must be >= 1")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")

    missing = [c for c in (col_a, col_b) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")

    # ── Extract clean numeric arrays ──────────────────────────────────────
    sub = df[[col_a, col_b]].dropna()
    x = sub[col_a].to_numpy(dtype=float)
    y = sub[col_b].to_numpy(dtype=float)
    orig_idx = sub.index.to_numpy()

    # ── Enumerate charge splits ───────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    for i in range(1, parent_charge + 1):
        for j in range(i, parent_charge + 1):
            if enforce_sum_leq_charge and i + j > parent_charge:
                continue

            v = i * x + j * y

            order = np.argsort(v)
            v_sorted = v[order]
            idx_sorted = orig_idx[order]

            for c_idx, min_v, max_v in _cluster_sorted_values(
                v_sorted, idx_sorted, delta, min_cluster_size
            ):
                row: Dict[str, Any] = {
                    "i": i,
                    "j": j,
                    "n_points": int(c_idx.size),
                    "diameter": float(max_v - min_v),
                    "center": float((max_v + min_v) / 2.0),
                    "min_v": float(min_v),
                    "max_v": float(max_v),
                }
                if return_point_indices:
                    row["point_indices"] = c_idx
                results.append(row)

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = (
            result_df
            .sort_values(["n_points", "diameter"], ascending=[False, True])
            .reset_index(drop=True)
        )

    return result_df


# =============================================================================
# 3. POST-PROCESSING: PARENT OFFSET & FILTERING
# =============================================================================

def compute_parent_offsets(
    clusters_df: pd.DataFrame,
    parent_mass: float,
    max_offset: float = 4.05,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Add a ``Parent+X`` column (cluster centre - parent mass) and an
    ``i+j`` column (total charge).  Filter to offsets below *max_offset*.

    Parameters
    ----------
    clusters_df : DataFrame
        Output of ``detect_line_clusters``.
    parent_mass : float
        Known precursor neutral mass (Da).
    max_offset : float
        Keep only clusters with Parent+X < max_offset.
    decimals : int
        Rounding precision for the offset column.

    Returns
    -------
    Filtered and augmented DataFrame, sorted by n_points descending.
    """
    df = clusters_df.copy()
    df["Parent+X"] = (df["center"] - parent_mass).round(decimals)
    df["i+j"] = df["i"] + df["j"]
    df = df[df["Parent+X"] < max_offset]
    return df.sort_values("n_points", ascending=False).reset_index(drop=True)


def filter_by_charge_sum(
    df: pd.DataFrame,
    target_charge: int,
    tolerance: int = 0,
    min_offset: Optional[float] = None,
) -> pd.DataFrame:
    """
    Keep only clusters whose total charge (i+j) is within *tolerance*
    of *target_charge*, optionally requiring Parent+X >= min_offset.

    Parameters
    ----------
    df : DataFrame
        Must have ``i+j`` and (optionally) ``Parent+X`` columns.
    target_charge : int
        Exact charge sum to select (e.g. the precursor charge).
    tolerance : int
        Allow i+j in [target_charge - tolerance, target_charge].
    min_offset : float or None
        If given, also require Parent+X >= min_offset.
    """
    mask = (df["i+j"] >= target_charge - tolerance) & (df["i+j"] <= target_charge)
    if min_offset is not None and "Parent+X" in df.columns:
        mask &= df["Parent+X"] >= min_offset
    return df[mask].reset_index(drop=True)


# =============================================================================
# 4. CLUSTER MERGING (group nearby Parent+X values)
# =============================================================================

def merge_nearby_clusters(
    df: pd.DataFrame,
    gap_threshold: float,
    offset_col: str = "Parent+X",
) -> pd.DataFrame:
    """
    Merge rows whose ``offset_col`` values are within *gap_threshold* of
    each other (gap-based grouping on the sorted column).

    Aggregation rules
    -----------------
    - min_v -> min,  max_v -> max,  center -> mean
    - Parent+X -> mean,  n_points -> sum,  diameter -> mean
    - i, j -> mode (most frequent value)
    - point_indices -> concatenated into a single list

    Parameters
    ----------
    df : DataFrame
        Output of ``compute_parent_offsets``.
    gap_threshold : float
        Maximum gap between consecutive sorted offset values to be
        considered the same group.
    offset_col : str
        Column to group on.
    """
    if offset_col not in df.columns:
        raise KeyError(f"Column '{offset_col}' not found in DataFrame")

    df2 = df.copy()
    df2[offset_col] = pd.to_numeric(df2[offset_col])
    df2 = df2.sort_values(offset_col).reset_index(drop=True)

    # Assign group IDs by gap
    gaps = df2[offset_col].diff().abs()
    df2["_grp"] = gaps.gt(gap_threshold).cumsum()

    # ── Build aggregation dict dynamically ────────────────────────────────
    def _mode(s):
        m = s.mode()
        return m.iat[0] if not m.empty else s.iloc[0]

    def _merge_indices(s):
        out = []
        for val in s.dropna():
            if isinstance(val, (list, tuple, np.ndarray)):
                out.extend(list(val))
            else:
                out.append(val)
        return out

    agg: Dict[str, Any] = {
        "min_v": "min",
        "max_v": "max",
        "center": "mean",
        offset_col: "mean",
    }

    optional = {
        "n_points": "sum",
        "diameter": "mean",
        "i": _mode,
        "j": _mode,
        "point_indices": _merge_indices,
    }
    for col, func in optional.items():
        if col in df2.columns:
            agg[col] = func

    grouped = (
        df2.groupby("_grp", as_index=False)
        .agg(agg)
        .sort_values(offset_col)
        .reset_index(drop=True)
    )

    return grouped


# =============================================================================
# 5. CONVENIENCE: FULL PIPELINE
# =============================================================================

def run_line_finding(
    ffc_df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    delta: float = 0.05,
    min_cluster_size: int = 3,
    max_offset: float = 4.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    merge_gap: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full line-finding pipeline in one call.

    Parameters
    ----------
    ffc_df : DataFrame
        Raw FFC data with m/z columns.
    parent_charge : int
        Precursor charge state.
    parent_mass : float
        Known precursor neutral mass (Da).
    delta : float
        Clustering gap threshold.
    min_cluster_size : int
        Minimum cluster membership.
    max_offset : float
        Maximum Parent+X offset to retain.
    col_a, col_b : str
        Column names for the two m/z values.
    merge_gap : float or None
        If given, merge nearby clusters by Parent+X with this gap.

    Returns
    -------
    dict with keys:
        "clusters"       -- all detected clusters with offsets
        "charge_matched" -- subset where i+j == parent_charge
        "merged"         -- (only if merge_gap is not None) grouped clusters
    """
    # Step 1: detect
    raw = detect_line_clusters(
        ffc_df,
        parent_charge=parent_charge,
        delta=delta,
        col_a=col_a,
        col_b=col_b,
        min_cluster_size=min_cluster_size,
    )

    # Step 2: parent offsets
    clusters = compute_parent_offsets(raw, parent_mass, max_offset=max_offset)

    # Step 3: charge-matched subset
    charge_matched = filter_by_charge_sum(clusters, parent_charge)

    result: Dict[str, pd.DataFrame] = {
        "clusters": clusters,
        "charge_matched": charge_matched,
    }

    # Step 4 (optional): merge
    if merge_gap is not None and not clusters.empty:
        result["merged"] = merge_nearby_clusters(clusters, gap_threshold=merge_gap)

    return result


# =============================================================================
# 6. DATA LOADING HELPERS
# =============================================================================

STANDARD_COLUMNS = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]


def load_ffc_whitespace(
    path: str,
    columns: Optional[List[str]] = None,
    skiprows: int = 1,
) -> pd.DataFrame:
    """Load an FFC file with whitespace-separated columns."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=skiprows, header=None, engine="python")
    df.columns = columns or STANDARD_COLUMNS[: df.shape[1]]
    return df


def load_ffc_excel(
    path: str,
    sheet_name: str,
    mz_col_a: str = "m/z fragment 1",
    mz_col_b: str = "m/z fragment 2",
) -> pd.DataFrame:
    """
    Load FFC data from an Excel sheet.  Renames the m/z columns to the
    standard names so downstream functions work without extra arguments.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    keep = [mz_col_a, mz_col_b] + [
        c for c in ["Covariance", "Partial Cov.", "Score", "Ranking"]
        if c in df.columns
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={mz_col_a: "m/z A", mz_col_b: "m/z B"})
    return df


def prepare_ffc_data(
    df: pd.DataFrame,
    top_n: Optional[int] = None,
    min_score: float = 0.0,
    remove_unranked: bool = True,
) -> pd.DataFrame:
    """
    Standard filtering/sorting applied before analysis.

    - Remove rows with Score <= min_score
    - Remove rows with Ranking == -1 (unranked)
    - Sort by Ranking ascending
    - Keep only the top N rows
    """
    df = df.copy()

    if "Score" in df.columns and min_score > 0:
        df = df[df["Score"] > min_score]

    if "Ranking" in df.columns:
        df["Ranking"] = df["Ranking"].fillna(-1).astype(int)
        if remove_unranked:
            df = df[df["Ranking"] != -1]
        df = df.sort_values("Ranking")

    if top_n is not None:
        df = df.head(top_n)

    return df.reset_index(drop=True)


# =============================================================================
# 7. MAIN — EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # ── Configuration ─────────────────────────────────────────────────────
    
    EXCEL_PATH = (
        "/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/data/"
        "Covariance Scoring Tables 10000 Scans.xlsx"
    )
    SHEET_NAME = "VEADIAGHGQEVLIR-mz536-3_cov"
    PARENT_CHARGE = 3
    PARENT_MASS = 1608.87   # neutral mass of precursor
    #PARENT_MASS = 1887.036239
    #PARENT_MASS = 1380.85609
    #PARENT_MASS = 3766.83688
    #PARENT_MASS = 4494.60433
    #PARENT_MASS = 4007.4216300000007
    #PARENT_MASS = 4275.1248 
    TOP_N = 300
    DELTA = 0.01
    MIN_CLUSTER_SIZE = 3
    

    # ── Load & prepare ────────────────────────────────────────────────────
    #ffc_df = load_ffc_excel(EXCEL_PATH, SHEET_NAME)
    data_path = '/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt'
    ffc_df = pd.read_csv(data_path, sep=r"\s+", skiprows=1, header=None, engine="python")
    ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)

    # ── Run full pipeline ─────────────────────────────────────────────────
    results = run_line_finding(
        ffc_df,
        parent_charge=PARENT_CHARGE,
        parent_mass=PARENT_MASS,
        delta=DELTA,
        min_cluster_size=MIN_CLUSTER_SIZE,
        merge_gap=0.5,  # set to None to skip merging
    )

    # ── Inspect results ───────────────────────────────────────────────────
    clusters = results["clusters"]
    charge_matched = results["charge_matched"]

    print("=== All clusters (top 50) ===")
    print(clusters.head(50))

    print(f"\n=== Charge-matched (i+j == {PARENT_CHARGE}) ===")
    print(charge_matched.head(50))

    # Near-charge clusters with offset >= -1
    near_charge = filter_by_charge_sum(
        clusters, target_charge=PARENT_CHARGE, tolerance=2, min_offset=-1.0
    )
    print(f"\n=== Near-charge clusters (i+j >= {PARENT_CHARGE - 2}, offset >= -1) ===")
    print(near_charge.head(50))

    if "merged" in results:
        merged = results["merged"]
        merged = merged.sort_values("n_points", ascending=False)
        print("\n=== Merged clusters (top 50) ===")
        print(merged[["min_v", "max_v", "center", "n_points", "diameter", "Parent+X"]].head(50))

    print(f"\n=== Raw FFC data (top 10) ===")
    print(ffc_df.head(10))

    # ── Optional: export to Excel ─────────────────────────────────────────
    # with pd.ExcelWriter("lines_output.xlsx", engine="openpyxl") as writer:
    #     clusters.to_excel(writer, sheet_name="All Clusters", index=False)
    #     charge_matched.to_excel(writer, sheet_name="Charge Matched", index=False)
    #     if "merged" in results:
    #         results["merged"].to_excel(writer, sheet_name="Merged", index=False)