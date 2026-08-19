"""
Spurious Parental FFC Finder
============================
Identifies FFCs that lie on the parental line (shift ≈ 0) but cannot be
annotated to any b/y ion breaking point.

A "breaking point" requires BOTH fragment masses to match a known b/y ion
(within ``threshold`` Da) for at least one (i, j) charge-split assignment.
FFCs that never satisfy this condition on any valid parental assignment are
called "spurious".

Algorithm
---------
Unlike the greedy-line pipeline, this script classifies each FFC
**independently** using a direct mass check:

    For each FFC (mz_A, mz_B):
        for every (i, j) with i + j == parent_charge:
            v = i * mz_A + j * mz_B
            if |v - parent_mass| < parental_shift_threshold:
                → this FFC is "on the parental line" under (i, j)
                → try to annotate adj_A, adj_B against b/y ions
                → if BOTH match → not spurious, stop

This avoids all greedy machinery (no Sort-and-Split, no cluster voting,
no removal), which means:
  • Each FFC is evaluated against ALL valid (i, j) assignments.
  • Adding more FFCs never changes the classification of already-seen ones
    (superset property is preserved).
  • Runtime is O(N × parent_charge) instead of O(N × parent_charge × log N).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from annotation import (
    MASS_H,
    _build_theoretical_ions,
    _find_all_matches,
)


# =============================================================================
# Core function
# =============================================================================

def find_spurious_parental_ffcs(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    iso_range: int = 0,
    threshold: float = 0.05,
    parental_shift_threshold: float = 0.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
) -> Tuple[List[int], pd.DataFrame]:
    """
    Find spurious FFCs on the parental line.

    Parameters
    ----------
    ffc_df : DataFrame
        FFC data (pre-filtered/sorted as needed).
    pep : Pep
        Peptide object with ion_mass() and AA_array.
    parent_charge : int
        Precursor charge state.
    parent_mass : float
        Precursor reconstructed mass reference:  i*mz_A + j*mz_B ≈ parent_mass.
        Same convention as annotation_greedy.py (i.e. M + parent_charge * H,
        no extra PROTON term).
    iso_range : int
        Number of isotope variants to consider during b/y matching.
    threshold : float
        Mass tolerance (Da) for b/y ion matching.
    parental_shift_threshold : float
        Maximum |i*mz_A + j*mz_B - parent_mass| for an FFC to be considered
        on the parental line under a given (i, j) assignment.
        Default 0.05 Da captures the typical ±0.02 Da reconstruction spread.
    col_a, col_b : str
        Column names for the two m/z values in ffc_df.
    ranking_col : str
        Column name for FFC ranking (lower = better).

    Returns
    -------
    spurious_rankings : list[int]
        Sorted Rankings of all spurious FFCs (empty if none found).
    spurious_df : DataFrame
        Subset of ffc_df rows for spurious FFCs, with extra columns:
            n_parental_assignments  – number of (i,j) pairs that placed this
                                      FFC on the parental line
            repr_i, repr_j          – (i, j) of the first such assignment
            repr_line_mass          – reconstructed mass under that (i, j)
            adj_mass_A, adj_mass_B  – singly-charged masses under repr (i, j)
    """
    _EXTRA_COLS = [
        "n_parental_assignments", "repr_i", "repr_j",
        "repr_line_mass", "adj_mass_A", "adj_mass_B",
    ]
    empty_df = pd.DataFrame(columns=list(ffc_df.columns) + _EXTRA_COLS)

    if ffc_df.empty:
        return [], empty_df

    # All (i, j) pairs with i + j == parent_charge, i >= 1, j >= 1
    parental_pairs = [
        (i, parent_charge - i) for i in range(1, parent_charge)
    ]
    if not parental_pairs:
        return [], empty_df

    ions = _build_theoretical_ions(pep, iso_range)
    has_ranking = ranking_col in ffc_df.columns

    mz_a_arr = ffc_df[col_a].to_numpy(dtype=float)
    mz_b_arr = ffc_df[col_b].to_numpy(dtype=float)

    # Pre-compute reconstructed masses for every (i, j) pair — shape (n_pairs, N)
    recon = np.stack(
        [i * mz_a_arr + j * mz_b_arr for (i, j) in parental_pairs],
        axis=0,
    )  # (n_pairs, N)

    # For each FFC: which (i, j) assignments put it on the parental line?
    on_parental = np.abs(recon - parent_mass) < parental_shift_threshold
    # Rows with at least one valid assignment
    candidate_mask = on_parental.any(axis=0)

    spurious_rows = []
    df_indices = ffc_df.index.to_numpy()

    for k in np.where(candidate_mask)[0]:
        ffc_row = ffc_df.iloc[k]
        mz_a = float(mz_a_arr[k])
        mz_b = float(mz_b_arr[k])

        # Collect valid (i, j) assignments for this FFC
        assignments = [
            (parental_pairs[p][0], parental_pairs[p][1], float(recon[p, k]))
            for p in range(len(parental_pairs))
            if on_parental[p, k]
        ]

        # Try to find a full annotation under any assignment
        fully_annotated = False
        for i, j, v in assignments:
            adj_a = i * mz_a - (i - 1) * MASS_H
            adj_b = j * mz_b - (j - 1) * MASS_H

            matches_a = _find_all_matches(adj_a, ions, threshold)
            matches_b = _find_all_matches(adj_b, ions, threshold)

            for ma in matches_a:
                if ma["base_name"] is None:
                    continue
                for mb in matches_b:
                    if mb["base_name"] is not None:
                        fully_annotated = True
                        break
                if fully_annotated:
                    break
            if fully_annotated:
                break

        if not fully_annotated:
            repr_i, repr_j, repr_v = assignments[0]
            adj_a_repr = repr_i * mz_a - (repr_i - 1) * MASS_H
            adj_b_repr = repr_j * mz_b - (repr_j - 1) * MASS_H

            extra: dict = {
                "n_parental_assignments": len(assignments),
                "repr_i":                repr_i,
                "repr_j":                repr_j,
                "repr_line_mass":        round(repr_v, 4),
                "adj_mass_A":            round(adj_a_repr, 4),
                "adj_mass_B":            round(adj_b_repr, 4),
            }
            spurious_rows.append(ffc_row.to_dict() | extra)

    if not spurious_rows:
        return [], empty_df

    spurious_df = pd.DataFrame(spurious_rows).reset_index(drop=True)
    spurious_rankings: List[int] = []
    if has_ranking:
        spurious_rankings = sorted(int(r) for r in spurious_df[ranking_col])

    return spurious_rankings, spurious_df


# =============================================================================
# Full parental-line annotation
# =============================================================================

def _sort_ion_names(names):
    """Sort ion names like ['b10', 'b3', 'y5'] numerically."""
    def _key(n):
        try:
            return (n[0], int(n[1:]))
        except (IndexError, ValueError):
            return (n, 0)
    return sorted(names, key=_key)


def parental_ffc_annotations(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    iso_range: int = 0,
    threshold: float = 0.05,
    parental_shift_threshold: float = 0.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
) -> pd.DataFrame:
    """
    Return all FFCs on the parental line with their annotation status.

    For each FFC that satisfies  |i*mz_A + j*mz_B - parent_mass| <
    ``parental_shift_threshold`` under any valid (i, j) with i+j == parent_charge:

    * If BOTH fragments match a known b/y ion under at least one (i, j):
      – ``b_ions``        : comma-separated b ion names found (e.g. "b5, b12")
      – ``breaking_points``: semicolon-separated "base_A/base_B" pairs
        (e.g. "b5/y10; b5/y10+1")
    * Otherwise (spurious):
      – ``b_ions``        : "spurious"
      – ``breaking_points``: "spurious"

    All valid (i, j) assignments are tried; a single full match anywhere
    across all assignments makes the FFC non-spurious.

    Parameters
    ----------
    (same as find_spurious_parental_ffcs)

    Returns
    -------
    DataFrame with one row per parental-line FFC, sorted by ``ranking_col``
    (if present).  Columns added beyond the original ffc_df columns:
        n_parental_assignments, repr_i, repr_j, repr_line_mass,
        adj_mass_A, adj_mass_B, b_ions, breaking_points
    """
    _EXTRA_COLS = [
        "n_parental_assignments", "repr_i", "repr_j",
        "repr_line_mass", "adj_mass_A", "adj_mass_B",
        "b_ions", "breaking_points",
    ]
    empty_df = pd.DataFrame(columns=list(ffc_df.columns) + _EXTRA_COLS)

    if ffc_df.empty:
        return empty_df

    parental_pairs = [(i, parent_charge - i) for i in range(1, parent_charge)]
    if not parental_pairs:
        return empty_df

    ions = _build_theoretical_ions(pep, iso_range)
    has_ranking = ranking_col in ffc_df.columns

    mz_a_arr = ffc_df[col_a].to_numpy(dtype=float)
    mz_b_arr = ffc_df[col_b].to_numpy(dtype=float)

    recon = np.stack(
        [i * mz_a_arr + j * mz_b_arr for (i, j) in parental_pairs], axis=0
    )
    on_parental = np.abs(recon - parent_mass) < parental_shift_threshold
    candidate_mask = on_parental.any(axis=0)

    result_rows = []

    for k in np.where(candidate_mask)[0]:
        ffc_row = ffc_df.iloc[k]
        mz_a = float(mz_a_arr[k])
        mz_b = float(mz_b_arr[k])

        assignments = [
            (parental_pairs[p][0], parental_pairs[p][1], float(recon[p, k]))
            for p in range(len(parental_pairs))
            if on_parental[p, k]
        ]

        b_ions_found: set = set()
        breaking_points_found: set = set()

        for i, j, v in assignments:
            adj_a = i * mz_a - (i - 1) * MASS_H
            adj_b = j * mz_b - (j - 1) * MASS_H

            matches_a = _find_all_matches(adj_a, ions, threshold)
            matches_b = _find_all_matches(adj_b, ions, threshold)

            for ma in matches_a:
                if ma["base_name"] is None:
                    continue
                for mb in matches_b:
                    if mb["base_name"] is None:
                        continue
                    breaking_points_found.add(
                        f"{ma['base_name']}/{mb['base_name']}"
                    )
                    for bn in (ma["base_name"], mb["base_name"]):
                        if bn.startswith("b"):
                            b_ions_found.add(bn)

        is_spurious = len(breaking_points_found) == 0

        repr_i, repr_j, repr_v = assignments[0]
        adj_a_repr = repr_i * mz_a - (repr_i - 1) * MASS_H
        adj_b_repr = repr_j * mz_b - (repr_j - 1) * MASS_H

        row_dict = ffc_row.to_dict()
        row_dict.update({
            "n_parental_assignments": len(assignments),
            "repr_i":                repr_i,
            "repr_j":                repr_j,
            "repr_line_mass":        round(repr_v, 4),
            "adj_mass_A":            round(adj_a_repr, 4),
            "adj_mass_B":            round(adj_b_repr, 4),
            "b_ions": (
                "spurious" if is_spurious
                else ", ".join(_sort_ion_names(b_ions_found))
            ),
            "breaking_points": (
                "spurious" if is_spurious
                else "; ".join(sorted(breaking_points_found))
            ),
        })
        result_rows.append(row_dict)

    if not result_rows:
        return empty_df

    result_df = pd.DataFrame(result_rows).reset_index(drop=True)
    if has_ranking and ranking_col in result_df.columns:
        result_df = result_df.sort_values(ranking_col).reset_index(drop=True)

    return result_df


# =============================================================================
# Charge-independent merged view
# =============================================================================

def merge_parental_annotations(
    annot_df: pd.DataFrame,
    merge_tol: float = 0.05,
    ranking_col: str = "Ranking",
    adj_col_a: str = "adj_mass_A",
    adj_col_b: str = "adj_mass_B",
) -> pd.DataFrame:
    """
    Collapse symmetric (A, B) / (B, A) pairs in a ``parental_ffc_annotations``
    output into one row per unique fragment-mass pair.

    Each row's ``adj_mass_A`` / ``adj_mass_B`` are charge-1 equivalent m/z
    values (``repr_i * mzA - (repr_i-1)*H``), so they are already
    charge-independent.  Two rows are considered the same pair when their
    canonical (min, max) adj_mass values are within ``merge_tol`` of each
    other.  The best-ranked (lowest ``ranking_col``) row is kept.

    The returned DataFrame replaces ``adj_mass_A`` / ``adj_mass_B`` with
    ``mz1_lo`` / ``mz1_hi`` (smaller value first) and drops the raw m/z and
    charge columns that are no longer meaningful after the merge.

    Parameters
    ----------
    annot_df : DataFrame
        Output of ``parental_ffc_annotations``.
    merge_tol : float
        Tolerance in Da for grouping near-identical charge-1 masses.
    ranking_col : str
        Column used to select the best row within each group (lower = better).
    adj_col_a, adj_col_b : str
        Names of the charge-independent mass columns.

    Returns
    -------
    DataFrame with one row per unique (mz1_lo, mz1_hi) pair, sorted by
    ``ranking_col``.  Columns ``mz1_lo`` and ``mz1_hi`` replace the original
    raw m/z and charge columns.
    """
    if annot_df.empty or adj_col_a not in annot_df.columns or adj_col_b not in annot_df.columns:
        return annot_df.copy()

    a = annot_df[adj_col_a].values.astype(float)
    b = annot_df[adj_col_b].values.astype(float)

    result = annot_df.copy()
    result["mz1_lo"] = np.minimum(a, b).round(4)
    result["mz1_hi"] = np.maximum(a, b).round(4)

    # Sequential gap-based grouping on sorted (lo, hi) pairs.
    # Fixed-grid rounding would split values near a bin boundary even when
    # their gap is well within merge_tol; sorting and checking consecutive
    # gaps avoids that problem entirely.
    tmp = (result[["mz1_lo", "mz1_hi", ranking_col]]
           .sort_values(["mz1_lo", "mz1_hi"])
           .reset_index())          # keep original index to map back
    lo = tmp["mz1_lo"].values
    hi = tmp["mz1_hi"].values

    gid, group_ids = 0, [0]
    for k in range(1, len(tmp)):
        if (lo[k] - lo[k - 1] > merge_tol) or (abs(hi[k] - hi[k - 1]) > merge_tol):
            gid += 1
        group_ids.append(gid)
    tmp["_group"] = group_ids

    # For each group keep the row with the lowest ranking value
    best_orig_idx = (tmp.groupby("_group")[ranking_col]
                     .idxmin()          # index into tmp
                     .map(tmp["index"]) # map back to original result index
                     .values)

    result = (result.loc[best_orig_idx]
              .drop(columns=[adj_col_a, adj_col_b,
                              "m/z A", "m/z B",
                              "repr_i", "repr_j"],
                    errors="ignore")
              .sort_values(ranking_col)
              .reset_index(drop=True))

    return result


# =============================================================================
# Example / quick test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import interpreter_modify  # noqa: F401
    import peptide
    from line_finding import prepare_ffc_data
    from merge_ffcs import merge_duplicate_ffcs

    # ── Configuration ─────────────────────────────────────────────────────
    #DATA_PATH   = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/VEA_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/KWK6+NCE20_with_intensity"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/YLE3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/HGT3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.NeuropeptideY_Z6_NCE25_300_ions"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/KWK5_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/KWK6_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/HAD_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/YPS6_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/LLG6_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/deconv/LLG6+_combine_replaced.txt"
    #DATA_PATH = '/home/sux002/data/mms_with_intensity/VEA_15_Z3'
    DATA_PATH = '/home/sux002/data/deconvoluted/merged/HAD_33_Z4_E16_XIAOWEN_merged.tsv'
    
    
    #PEP_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
    #PEP_SEQ     = "VEADIAGHGQEVLIR"
    PEP_SEQ      = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    #PEP_SEQ = "YLEFISDAIIHVLHSK"
    #PEP_SEQ = "HGTVVLTALGGILK"
    #PEP_SEQ = "YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY"
    #PEP_SEQ = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    
    #CHARGE      = 3
    CHARGE      = 4
    #CHARGE = 6
    #CHARGE = 5
    
    #PARENT_MASS =  667.90419 * 6.  #KWK
    #PARENT_MASS = 1608.87      # i*mzA + j*mzB reference (M + Z*H) VEA
    #PARENT_MASS = 1887.036239 #YLE
    #PARENT_MASS = 1380.85609 #HGT
    #PARENT_MASS  = 941.96162 * 4
    #PARENT_MASS = 712.52139 * 6 #YPS
    #PARENT_MASS = 749.43703 * 6 #LLG 4,496.62218
    PARENT_MASS = 0
    
    TOP_N       = 10000
    ISO_RANGE   = 4
    THRESHOLD   = 0.05
    PARENTAL_SHIFT_THRESHOLD = 0.05

    # ── Build peptide & load data ─────────────────────────────────────────
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    #pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20="NH3")
    PARENT_MASS = pep.pep_mass
    
    print(f"Peptide: {PEP_SEQ}  charge={CHARGE}  parent_mass={PARENT_MASS}")

    #ffc_df = pd.read_csv(DATA_PATH, sep=r"\s+", skiprows=1, header=None, engine="python")
    #ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", 'intensity A', 'intensity B']
    #ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    ffc_df = pd.read_csv(DATA_PATH, sep="\t")
    
    #print(ffc_df[ffc_df['ffc_df'] is in []])
    
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)
    ffc_df = merge_duplicate_ffcs(ffc_df)
    print(f"FFC rows after filter: {len(ffc_df)}")

    # ── Run: spurious only ────────────────────────────────────────────────
    rankings, spurious_df = find_spurious_parental_ffcs(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=PARENT_MASS,
        iso_range=ISO_RANGE,
        threshold=THRESHOLD,
        parental_shift_threshold=PARENTAL_SHIFT_THRESHOLD,
    )

    print(f"\nSpurious parental FFCs: {len(spurious_df)}")
    print(f"Rankings: {rankings}")

    if not spurious_df.empty:
        print("\nSpurious FFC table:")
        show_cols = [
            c for c in [
                "m/z A", "m/z B", "Ranking",
                "repr_i", "repr_j", "repr_line_mass",
                "adj_mass_A", "adj_mass_B", "n_parental_assignments",
            ]
            if c in spurious_df.columns
        ]
        print(spurious_df[show_cols].to_string(index=False))

    # ── Run: all parental FFCs with annotation ────────────────────────────
    print("\n" + "=" * 60)
    annot_df = parental_ffc_annotations(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=PARENT_MASS,
        iso_range=ISO_RANGE,
        threshold=0.01,
        parental_shift_threshold=PARENTAL_SHIFT_THRESHOLD,
    )

    print(f"\nAll parental-line FFCs: {len(annot_df)}")
    n_spurious = (annot_df["b_ions"] == "spurious").sum()
    print(f"  Spurious : {n_spurious}")
    print(f"  Annotated: {len(annot_df) - n_spurious}")

    if not annot_df.empty:
        show_cols = [
            c for c in [
                "m/z A", "m/z B", "Ranking",
                "repr_i", "repr_j",
                "adj_mass_A", "adj_mass_B",
                "b_ions", "breaking_points",
            ]
            if c in annot_df.columns
        ]
        print(annot_df[show_cols].to_string(index=False))
