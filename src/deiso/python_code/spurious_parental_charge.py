"""
Charge-Aware Spurious Parental FFC Finder
==========================================
Like ``spurious_parental.py`` but uses the ``charge_A`` / ``charge_B``
columns already present in the deconvolution output instead of re-deriving
all possible (i, j) assignments from scratch.

Difference vs. spurious_parental.py
-------------------------------------
``spurious_parental.parental_ffc_annotations`` tries EVERY (i, j) pair
where i + j == parent_charge for each FFC.  This script trusts the charge
assignment that the deconvolution pipeline already produced:

    1. Only rows where charge_A + charge_B == parent_charge are considered
       (parental-line FFCs as identified by the deconvolution).
    2. For each such row the single assigned (charge_A, charge_B) is used
       to compute the charge-1 equivalent masses and look up b/y ions.

This gives a more targeted evaluation: instead of asking "can this FFC be
annotated under *any* valid parental charge split?", it asks "can this FFC
be annotated under the charge split the pipeline actually assigned?"

Output columns are identical to ``parental_ffc_annotations`` so
``merge_parental_annotations`` from ``spurious_parental`` can be applied
directly to the result.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from annotation import (
    MASS_H,
    _build_theoretical_ions,
    _find_all_matches,
)
from spurious_parental import _sort_ion_names


# =============================================================================
# Core charge-aware annotation
# =============================================================================

def parental_ffc_annotations_from_charges(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    iso_range: int = 0,
    threshold: float = 0.05,
    parental_shift_threshold: float = 0.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    charge_col_a: str = "charge_A",
    charge_col_b: str = "charge_B",
    ranking_col: str = "Ranking",
) -> pd.DataFrame:
    """
    Return all deconvolution-assigned parental FFCs with their annotation status.

    Only rows where ``charge_A + charge_B == parent_charge`` are considered.
    Each such row is additionally verified by the parental shift check
    (|charge_A * mz_A + charge_B * mz_B - parent_mass| < parental_shift_threshold).
    The single (charge_A, charge_B) pair from the deconvolution is used
    for b/y matching — no other (i, j) combinations are tried.

    Parameters
    ----------
    ffc_df : DataFrame
        Deconvolution output with columns including ``col_a``, ``col_b``,
        ``charge_col_a``, ``charge_col_b``.
    pep : Pep
        Peptide object with ion masses.
    parent_charge : int
        Precursor charge state (used to filter parental-line rows).
    parent_mass : float
        Precursor reconstruction reference: i*mzA + j*mzB ≈ parent_mass.
    iso_range : int
        Isotope variants for b/y matching.
    threshold : float
        Mass tolerance (Da) for b/y ion matching.
    parental_shift_threshold : float
        Maximum |i*mzA + j*mzB - parent_mass| to accept a row as parental.
    col_a, col_b : str
        m/z column names.
    charge_col_a, charge_col_b : str
        Column names carrying the deconvolution charge assignments.
    ranking_col : str
        Ranking column (lower = better).

    Returns
    -------
    DataFrame with the same extra columns as ``parental_ffc_annotations``:
        charge_A, charge_B, repr_line_mass, adj_mass_A, adj_mass_B,
        b_ions, breaking_points
    Sorted by ``ranking_col``.
    """
    _EXTRA_COLS = [
        "repr_line_mass", "adj_mass_A", "adj_mass_B",
        "b_ions", "breaking_points",
    ]
    empty_df = pd.DataFrame(columns=list(ffc_df.columns) + _EXTRA_COLS)

    if ffc_df.empty:
        return empty_df
    if charge_col_a not in ffc_df.columns or charge_col_b not in ffc_df.columns:
        raise KeyError(
            f"Input must contain charge columns {charge_col_a!r} and {charge_col_b!r}. "
            f"Available: {list(ffc_df.columns)}"
        )

    ions = _build_theoretical_ions(pep, iso_range)
    has_ranking = ranking_col in ffc_df.columns

    charge_a = ffc_df[charge_col_a].to_numpy(dtype=float)
    charge_b = ffc_df[charge_col_b].to_numpy(dtype=float)
    mz_a_arr = ffc_df[col_a].to_numpy(dtype=float)
    mz_b_arr = ffc_df[col_b].to_numpy(dtype=float)

    # Keep only parental-line rows: charge_A + charge_B == parent_charge
    parental_mask = np.round(charge_a + charge_b).astype(int) == parent_charge
    # Also apply the shift check using the assigned charges
    recon = charge_a * mz_a_arr + charge_b * mz_b_arr
    shift_mask = np.abs(recon - parent_mass) < parental_shift_threshold
    candidate_idx = np.where(parental_mask & shift_mask)[0]

    result_rows = []

    for k in candidate_idx:
        ffc_row = ffc_df.iloc[k]
        i = int(round(charge_a[k]))
        j = int(round(charge_b[k]))
        mz_a = float(mz_a_arr[k])
        mz_b = float(mz_b_arr[k])
        v = float(recon[k])

        adj_a = i * mz_a - (i - 1) * MASS_H
        adj_b = j * mz_b - (j - 1) * MASS_H

        matches_a = _find_all_matches(adj_a, ions, threshold)
        matches_b = _find_all_matches(adj_b, ions, threshold)

        b_ions_found: set = set()
        breaking_points_found: set = set()

        for ma in matches_a:
            if ma["base_name"] is None:
                continue
            for mb in matches_b:
                if mb["base_name"] is None:
                    continue
                breaking_points_found.add(f"{ma['base_name']}/{mb['base_name']}")
                for bn in (ma["base_name"], mb["base_name"]):
                    if bn.startswith("b"):
                        b_ions_found.add(bn)

        is_spurious = len(breaking_points_found) == 0

        row_dict = ffc_row.to_dict()
        row_dict.update({
            "repr_line_mass":  round(v, 4),
            "adj_mass_A":      round(adj_a, 4),
            "adj_mass_B":      round(adj_b, 4),
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
# Example / quick test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import interpreter_modify  # noqa: F401
    import peptide
    from line_finding import prepare_ffc_data
    from merge_ffcs import merge_duplicate_ffcs_charge
    from spurious_parental import merge_parental_annotations

    # ── Configuration ─────────────────────────────────────────────────────
    DATA_PATH = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted/HAD_merged.tsv")

    PEP_SEQ = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    CHARGE = 4
    PARENT_MASS = 0

    TOP_N       = 2000
    ISO_RANGE   = 4
    THRESHOLD   = 0.05
    PARENTAL_SHIFT_THRESHOLD = 0.05

    # ── Build peptide & load data ─────────────────────────────────────────
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    PARENT_MASS = pep.pep_mass

    print(f"Peptide: {PEP_SEQ}  charge={CHARGE}  parent_mass={PARENT_MASS}")

    ffc_df = pd.read_csv(DATA_PATH, sep="\t")

    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)
    ffc_df = merge_duplicate_ffcs_charge(ffc_df, symmetric=True, reset_ranking=True)
    print(f"FFC rows after filter: {len(ffc_df)}")

    # ── Run: charge-assigned parental FFCs with annotation ────────────────
    annot_df = parental_ffc_annotations_from_charges(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=PARENT_MASS,
        iso_range=ISO_RANGE,
        threshold=THRESHOLD,
        parental_shift_threshold=PARENTAL_SHIFT_THRESHOLD,
    )

    n_total = len(annot_df)
    n_spurious = int((annot_df["b_ions"] == "spurious").sum()) if n_total else 0
    print(f"\nAll parental-line FFCs (charge-assigned): {n_total}")
    print(f"  Spurious : {n_spurious}")
    print(f"  Annotated: {n_total - n_spurious}")

    if n_total:
        show_cols = [
            c for c in [
                "m/z A", "m/z B", "Ranking",
                "charge_A", "charge_B",
                "adj_mass_A", "adj_mass_B",
                "b_ions", "breaking_points",
            ]
            if c in annot_df.columns
        ]
        print(annot_df[show_cols].to_string(index=False))

    # ── Charge-independent merged view ────────────────────────────────────
    if n_total:
        print("\n" + "=" * 60)
        print("Charge-independent merged view  (symmetric pairs collapsed)")
        print("=" * 60)

        merged = merge_parental_annotations(annot_df)
        n_merged_spurious = int((merged["b_ions"] == "spurious").sum()) if "b_ions" in merged.columns else 0
        print(f"\nUnique pairs: {len(merged)}")
        print(f"  Spurious : {n_merged_spurious}")
        print(f"  Annotated: {len(merged) - n_merged_spurious}")
        print(merged[['Ranking', 'b_ions', 'mz1_lo', 'mz1_hi', 'Ranking']])

