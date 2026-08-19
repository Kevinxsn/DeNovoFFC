#!/usr/bin/env python3
"""
Two-stage deconvolution pipeline (orientation-agnostic line finding)
====================================================================
Descendant of ``deiso/python_code/deconv_all_greedy.py`` +
``deconv_all_greedy_xiaowen.py``, folded into one module.

  Stage 1 -- FFC-line-based deconvolution (``deconv_stage1_oa``)
      Lines are found by ``detect_lines_oa``, the orientation-agnostic detector
      that ``line_based_annotation`` uses.  Each line carries a canonical charge
      split; each FFC's true per-side charges are resolved against it.  The
      isotope-envelope deconvolution that follows (monoisotopic refinement via
      ``theo_patt``) is the same machinery as before, reused unchanged.

  Stage 2 -- TopFD fallback for the FFCs no line claimed
      Their m/z values are written as a 1D peak list, deconvolved by TopFD with
      Xiaowen Liu's fine-tuned parameters, and mapped back onto the FFC rows.

  Merge -- Stage-1 and Stage-2 rows are combined into one table with a
      consistent column layout, written to the ``merge/`` folder.

What is different from the old pipeline
---------------------------------------
1. **One pre-processing pass.**  ``preprocess_ffc_oa`` runs once, here, and its
   output is what Stage 1 receives.  Previously the pipeline and Stage 1 each
   prepared their own frame (differently -- one doubled, one did not), so their
   ``ffc_index`` spaces did not correspond and residual detection had to fall
   back on matching (m/z A, m/z B) pairs.  With a single shared frame, residual
   detection is a set difference over row positions: exact, and cheap.
2. **No column doubling.**  See ``line_detection_oa`` -- the detector no longer
   needs both orderings of every FFC, so ``top_selected`` now counts distinct
   physical FFCs.
3. **Xiaowen's TopFD parameters are the default**, not an injected override.
4. **Output layout.**  Merged tables go to ``<output_dir>/merge/``; the
   per-stage intermediates and TopFD's working files go to
   ``<output_dir>/intermediate/<prefix>/`` so a batch run over many datasets
   does not collide or clutter.

Usage
-----
    python deconv_pipeline_oa.py <ffc_file> --parent-charge 4 \\
        --parent-mass 3769.86 --prefix HAD_33_Z4_E16

but the normal entry point is ``run_deconvolution.py``, which resolves the
peptide parameters from the dataset name for you.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap)

import topfd_xiaowen  # noqa: E402

from line_detection_oa import preprocess_ffc_oa  # noqa: E402
from deconv_stage1_oa import deconvolute_ffc_by_lines_oa  # noqa: E402


#: Sub-folders created under ``output_dir``.
MERGE_SUBDIR = "merge"
INTERMEDIATE_SUBDIR = "intermediate"

#: Default column names for a raw FFC table (header row skipped).
DEFAULT_COL_NAMES = [
    "m/z A", "m/z B", "Covariance", "Partial Cov.",
    "Score", "Ranking", "intensity A", "intensity B",
]


# =====================================================================
# STAGE 2 HELPERS: TopFD fallback
# =====================================================================

def _collect_peaks(
    frame: pd.DataFrame,
    mz_col_a: str,
    mz_col_b: str,
    intensity_col_a: Optional[str],
    intensity_col_b: Optional[str],
    merge_tol: float = 0.001,
) -> pd.DataFrame:
    """Pool both m/z columns of ``frame`` into a deduplicated 1D peak list.

    Peaks within ``merge_tol`` (1 mDa) collapse to the most intense one.
    Returns columns ``["mz", "intensity"]``, ready to write for TopFD.
    """
    if frame.empty:
        return pd.DataFrame(columns=["mz", "intensity"])

    mz_a = frame[mz_col_a].to_numpy(dtype=float)
    mz_b = frame[mz_col_b].to_numpy(dtype=float)
    if intensity_col_a and intensity_col_a in frame.columns:
        int_a = frame[intensity_col_a].to_numpy(dtype=float)
    else:
        int_a = np.ones(len(frame), dtype=float)
    if intensity_col_b and intensity_col_b in frame.columns:
        int_b = frame[intensity_col_b].to_numpy(dtype=float)
    else:
        int_b = np.ones(len(frame), dtype=float)

    all_mz = np.concatenate([mz_a, mz_b])
    all_int = np.concatenate([int_a, int_b])
    order = np.argsort(all_mz)
    all_mz, all_int = all_mz[order], all_int[order]

    unique_mz = [all_mz[0]]
    unique_int = [all_int[0]]
    for k in range(1, len(all_mz)):
        if all_mz[k] - unique_mz[-1] <= merge_tol:
            if all_int[k] > unique_int[-1]:
                unique_int[-1] = all_int[k]
                unique_mz[-1] = all_mz[k]
        else:
            unique_mz.append(all_mz[k])
            unique_int.append(all_int[k])

    return pd.DataFrame({"mz": unique_mz, "intensity": unique_int})


def _write_topfd_input(peaks_df: pd.DataFrame, output_path: str) -> None:
    """Write a peak list in the space-separated format TopFD expects."""
    peaks_df[["mz", "intensity"]].to_csv(
        output_path, sep=" ", index=False, header=False)


def _dedup_env_by_peak_idx(env_path: str) -> pd.DataFrame:
    """Deduplicate a TopFD ``.env`` file by PEAK_IDX, keeping the highest
    THEO_INTE per peak.  Mirrors ``dedup_by_peak_idx.py``."""
    best: Dict[str, dict] = {}
    with open(env_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            key = row["PEAK_IDX"]
            if key not in best or float(row["THEO_INTE"]) > float(best[key]["THEO_INTE"]):
                best[key] = row
    return pd.DataFrame(list(best.values()), columns=fieldnames)


def _annotate_residual_ffcs(
    residual_ffcs: pd.DataFrame,
    env_df: pd.DataFrame,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    mz_tol: float = 1e-4,
) -> pd.DataFrame:
    """Attach TopFD's deconvolution result to each residual FFC row.

    Mirrors ``annotate_mms.py``: for each m/z, look up the matching ORIG_MZ in
    the env table and append THEO_MONO_MZ / THEO_MONO_MASS / THEO_INTE_SUM /
    THEO_CHARGE, suffixed ``_1`` for side A and ``_2`` for side B.
    """
    ANNO_COLS = ["THEO_MONO_MZ", "THEO_MONO_MASS", "THEO_INTE_SUM", "THEO_CHARGE"]

    if env_df.empty:
        for col in ANNO_COLS:
            residual_ffcs[col + "_1"] = float("nan")
            residual_ffcs[col + "_2"] = float("nan")
        return residual_ffcs

    env = env_df.copy()
    env["ORIG_MZ"] = env["ORIG_MZ"].astype(float)
    env["_mz_key"] = env["ORIG_MZ"].round(6)
    env_indexed = env.set_index("_mz_key")

    def lookup(mz: float) -> dict:
        key = round(mz, 6)
        if key in env_indexed.index:
            row = env_indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {col: row[col] for col in ANNO_COLS}
        candidates = env_indexed[abs(env_indexed["ORIG_MZ"] - mz) < mz_tol]
        if not candidates.empty:
            row = candidates.iloc[0]
            return {col: row[col] for col in ANNO_COLS}
        return {col: float("nan") for col in ANNO_COLS}

    anno1 = pd.DataFrame([lookup(mz) for mz in
                          residual_ffcs[mz_col_a].to_numpy(dtype=float)])
    anno1 = anno1.rename(columns={c: c + "_1" for c in ANNO_COLS})
    anno2 = pd.DataFrame([lookup(mz) for mz in
                          residual_ffcs[mz_col_b].to_numpy(dtype=float)])
    anno2 = anno2.rename(columns={c: c + "_2" for c in ANNO_COLS})

    return pd.concat([residual_ffcs.reset_index(drop=True),
                      anno1.reset_index(drop=True),
                      anno2.reset_index(drop=True)], axis=1)


# =====================================================================
# INPUT SANITY
# =====================================================================

def _check_intensities(
    df: pd.DataFrame,
    mz_col_a: str,
    mz_col_b: str,
    intensity_col_a: Optional[str],
    intensity_col_b: Optional[str],
) -> None:
    """Warn if the input's intensity columns are incomplete, and say why it matters.

    Intensity is not decoration here.  The Stage-1 envelope refinement scores
    candidate isotope patterns against measured intensities, so a peak with no
    intensity yields no monoisotopic mass; and TopFD **segfaults** outright on a
    peak list containing NaN intensities, taking Stage 2 with it.

    That is a defect in the upstream `mms_with_intensity` merge -- the step that
    joins the raw FFC table to the 1D peak intensities -- not in anything this
    pipeline does, and no deconvolution setting can repair it.  Measured on the
    four ``HAD_33_Z4_E15_I*`` datasets: 41-63 % of their peaks carry no intensity
    at all, which produced a 67-90 % unplaced rate in Stage 1 and a TopFD crash.
    ``HAD_33_Z4_E16``, from the same peptide, has complete intensities and runs
    clean.

    Deliberately a warning, not an error: a partially-covered file still yields
    usable rows for the peaks that do have intensities, and stopping would throw
    those away too.  The fix belongs upstream.
    """
    cols = [c for c in (intensity_col_a, intensity_col_b) if c and c in df.columns]
    if not cols:
        return
    a, b = df[cols[0]], df[cols[-1]]
    n_bad = int((a.isna() | b.isna()).sum())
    if not n_bad:
        return

    peaks = pd.concat([
        pd.DataFrame({"mz": df[mz_col_a], "i": a}),
        pd.DataFrame({"mz": df[mz_col_b], "i": b}),
    ])
    covered = peaks.groupby("mz")["i"].apply(lambda s: s.notna().any())
    n_peaks, n_uncovered = len(covered), int((~covered).sum())

    print(f"  [WARN] {n_bad} of {len(df)} rows ({100.0 * n_bad / len(df):.1f}%) "
          f"have a missing intensity, and {n_uncovered} of {n_peaks} distinct "
          f"peaks have none at all.")
    print("         This is an upstream mms_with_intensity merge problem, not a "
          "deconvolution setting.")
    print("         Expect a high 'no monoisotopic mass' rate below, and a "
          "possible TopFD crash in Stage 2.")


# =====================================================================
# MERGE
# =====================================================================

def _merge_results(
    stage1_replaced: pd.DataFrame,
    stage1_annotated: pd.DataFrame,
    stage2_annotated: pd.DataFrame,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
) -> pd.DataFrame:
    """Combine Stage-1 and Stage-2 rows into one table.

    ``m/z A`` / ``m/z B`` hold the deconvolved monoisotopic m/z at the inferred
    charge (``mono_mass / charge + proton``) -- the same convention on both
    sides, since TopFD's THEO_MONO_MZ already is that.  ``charge_A`` /
    ``charge_B`` carry the inferred charges (line geometry for Stage 1,
    THEO_CHARGE for Stage 2), and ``deconv_method`` says which stage a row came
    from.  The pre-deconvolution values are preserved as ``mz_A_original`` /
    ``mz_B_original``.
    """
    parts: List[pd.DataFrame] = []

    if not stage1_replaced.empty:
        s1 = stage1_replaced.copy()
        s1["mz_A_original"] = stage1_annotated[mz_col_a].values
        s1["mz_B_original"] = stage1_annotated[mz_col_b].values
        s1["charge_A"] = stage1_annotated["charge_A"].values
        s1["charge_B"] = stage1_annotated["charge_B"].values
        s1["deconv_method"] = "ffc_line"
        parts.append(s1)

    if not stage2_annotated.empty:
        s2 = stage2_annotated.copy()
        s2["mz_A_original"] = s2[mz_col_a].copy()
        s2["mz_B_original"] = s2[mz_col_b].copy()
        if "THEO_MONO_MZ_1" in s2.columns:
            s2[mz_col_a] = s2["THEO_MONO_MZ_1"].fillna(s2[mz_col_a])
        if "THEO_MONO_MZ_2" in s2.columns:
            s2[mz_col_b] = s2["THEO_MONO_MZ_2"].fillna(s2[mz_col_b])
        s2["charge_A"] = (s2["THEO_CHARGE_1"].astype(float)
                          if "THEO_CHARGE_1" in s2.columns else float("nan"))
        s2["charge_B"] = (s2["THEO_CHARGE_2"].astype(float)
                          if "THEO_CHARGE_2" in s2.columns else float("nan"))
        s2["deconv_method"] = "topfd"
        parts.append(s2)

    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0].reset_index(drop=True)

    common = set(parts[0].columns) & set(parts[1].columns)
    ordered = [c for c in parts[0].columns if c in common]
    return pd.concat([p[ordered] for p in parts], ignore_index=True)


# =====================================================================
# PUBLIC ENTRY POINT
# =====================================================================

def run_deconvolution(
    ffc_path: str,
    parent_charge: int,
    parent_mass: float,
    output_dir: str,
    output_prefix: str = "combined",
    # ── FFC column mapping ──
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = "intensity A",
    intensity_col_b: Optional[str] = "intensity B",
    ranking_col: str = "Ranking",
    pcov_col: str = "Partial Cov.",
    # ── Stage 1: line finding (orientation-agnostic) ──
    delta: float = 0.02,
    delta_unit: str = "da",
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    top_lines: Optional[int] = None,
    merge_tolerance: float = 0.002,
    merge_over_fetch: int = 4,
    silent_drop: bool = False,
    fast_detector: bool = True,
    # ── Stage 1: deconvolution back end ──
    theo_patt_path: Optional[str] = None,
    mz_tol: float = 0.01,
    max_shift: Optional[int] = None,
    top_selected: int = 15000,
    top_bg: int = 15000,
    # ── Stage 2: TopFD (Xiaowen-tuned) ──
    run_topfd: bool = True,
    topfd_bin: str = "topfd",
    topfd_max_charge: Optional[int] = None,
    topfd_max_mass: Optional[float] = None,
    topfd_error_tol: float = 0.04,
    topfd_sn_ratio: float = 0.1,
    topfd_ecscore_cutoff: float = 0.0,
    topfd_disable_frag_num_filter: bool = True,
    topfd_max_miss_peak_num: Optional[int] = None,
    topfd_disable_filter_by_mz: bool = False,
    topfd_full_pool: bool = True,
    topfd_env_vars: Optional[Dict[str, str]] = None,
    # ── Input format ──
    skiprows: int = 1,
    col_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Run the full two-stage deconvolution for one dataset.

    ``topfd_max_charge`` and ``topfd_max_mass`` default to ``parent_charge`` and
    ``parent_mass`` -- a fragment cannot out-charge or out-weigh its precursor,
    and those are the values the tuned parameter set was calibrated around.

    ``topfd_max_miss_peak_num`` and ``topfd_disable_filter_by_mz`` default to
    the values the **shipped v1.8.1 binary** accepts (flag omitted / filter
    kept).  Xiaowen's rebuilt ``debug_mms`` binary is the only one that
    understands them; set ``topfd_max_miss_peak_num=4`` and
    ``topfd_disable_filter_by_mz=True`` when pointing at it.

    Returns a dict of the stage frames plus ``stats`` and ``lines``.
    """
    merge_dir = os.path.join(output_dir, MERGE_SUBDIR)
    work_dir = os.path.join(output_dir, INTERMEDIATE_SUBDIR, output_prefix)
    os.makedirs(merge_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    if topfd_max_charge is None:
        topfd_max_charge = parent_charge
    if topfd_max_mass is None:
        topfd_max_mass = parent_mass
    if top_bg < top_selected:
        raise ValueError(f"top_bg ({top_bg}) must be >= top_selected ({top_selected})")

    # ── Load ─────────────────────────────────────────────────────────────
    col_names = col_names or DEFAULT_COL_NAMES
    df = pd.read_csv(ffc_path, sep="\t", skiprows=skiprows,
                     header=None, engine="python")
    df.columns = col_names[:len(df.columns)]
    print(f"Loaded {len(df)} FFC rows from {ffc_path}")
    _check_intensities(df, mz_col_a, mz_col_b, intensity_col_a, intensity_col_b)

    # ── Pre-process ONCE; both stages share this frame ───────────────────
    print("\nPre-processing (oa: symmetric merge, no column doubling)")
    bg_df = preprocess_ffc_oa(
        df,
        mz_col_a=mz_col_a, mz_col_b=mz_col_b,
        ranking_col=ranking_col, pcov_col=pcov_col,
        merge_tolerance=merge_tolerance, top_n=top_bg,
        over_fetch=merge_over_fetch,
    )
    selected_df = bg_df.head(top_selected).reset_index(drop=True)
    print(f"  selected subset: {len(selected_df)} rows "
          f"(background pool: {len(bg_df)})")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: FFC-line-based deconvolution (orientation-agnostic lines)")
    print("=" * 60)
    print(f"  delta={delta} {delta_unit}   min_ffc={min_ffc_number}   "
          f"line_tol={line_tol if line_tol is not None else delta}   "
          f"top_lines={top_lines or len(selected_df)}")

    result_s1 = deconvolute_ffc_by_lines_oa(
        selected_df, bg_df,
        parent_charge=parent_charge,
        parent_mass=parent_mass,
        mz_col_a=mz_col_a, mz_col_b=mz_col_b,
        intensity_col_a=intensity_col_a, intensity_col_b=intensity_col_b,
        delta=delta, delta_unit=delta_unit,
        min_ffc_number=min_ffc_number, line_tol=line_tol,
        top_lines=top_lines,
        theo_patt_path=theo_patt_path, mz_tol=mz_tol, max_shift=max_shift,
        silent_drop=silent_drop,
        fast_detector=fast_detector,
    )

    s1_annotated = result_s1["annotated"]
    s1_replaced = result_s1["replaced"]
    s1_line_map = result_s1["line_map"]
    lines = result_s1["lines"]

    # Residual = the selected rows Stage 1 did not *place*.  Landing on a line
    # is necessary but not sufficient: the line supplies the charge, and the
    # envelope refinement still has to recover a monoisotopic mass from it.
    # When it cannot, it returns NaN, and `replaced` computes
    # `NaN / charge + proton` -- a row with no deconvolved m/z, which is no
    # more placed than one no line claimed.  Such rows are handed to TopFD
    # instead and dropped from the merged output.
    #
    # This is not hypothetical: on the four HAD_33_Z4_E15_I* datasets the
    # envelope step fails for 67-90% of line-assigned rows, and letting those
    # through put NaN m/z into the deliverable, which then crashed the
    # downstream annotation's KD-tree dedup.
    if s1_replaced.empty:
        s1_ok = pd.Series(dtype=bool)
        placed = set()
    else:
        s1_ok = (s1_replaced[mz_col_a].notna() & s1_replaced[mz_col_b].notna())
        placed = set(s1_annotated.loc[s1_ok.values, "ffc_index"].astype(int))

    n_unplaced_rows = int((~s1_ok).sum()) if len(s1_ok) else 0
    if n_unplaced_rows:
        print(f"\n  [warn] {n_unplaced_rows} of {len(s1_replaced)} line-assigned "
              f"rows got no monoisotopic mass from the envelope refinement; "
              f"they fall through to TopFD")
        s1_replaced = s1_replaced[s1_ok.values].reset_index(drop=True)
        s1_annotated_kept = s1_annotated[s1_ok.values].reset_index(drop=True)
    else:
        s1_annotated_kept = s1_annotated

    residual_mask = np.array([k not in placed for k in range(len(selected_df))])
    n_line_assigned = len(selected_df) - int(residual_mask.sum())
    n_residual = int(residual_mask.sum())
    print(f"\n  Line-assigned FFCs: {n_line_assigned}")
    print(f"  Residual FFCs:      {n_residual}")

    for name, frame in (("s1_annotated", s1_annotated),
                        ("s1_replaced", s1_replaced),
                        ("s1_line_map", s1_line_map)):
        frame.to_csv(os.path.join(work_dir, f"{output_prefix}_{name}.tsv"),
                     sep="\t", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2
    # ══════════════════════════════════════════════════════════════════════
    s2_annotated = pd.DataFrame()

    if run_topfd and n_residual > 0:
        print("\n" + "=" * 60)
        print("STAGE 2: TopFD fallback (Xiaowen-tuned parameters)")
        print("=" * 60)

        # With topfd_full_pool the FULL selected pool is handed to TopFD so it
        # has more envelope context; only the residual rows get annotated
        # either way, and Stage 1's results are untouched.
        pool = selected_df if topfd_full_pool else selected_df[residual_mask]
        peaks = _collect_peaks(pool, mz_col_a, mz_col_b,
                               intensity_col_a, intensity_col_b)
        label = "full-pool" if topfd_full_pool else "residual-only"
        print(f"  TopFD input peaks ({label}): {len(peaks)}")

        if not peaks.empty:
            topfd_dir = os.path.join(work_dir, "topfd")
            os.makedirs(topfd_dir, exist_ok=True)
            peak_list_path = os.path.join(topfd_dir, "residual.pkl")
            _write_topfd_input(peaks, peak_list_path)
            print(f"  Peak list written: {peak_list_path}")

            try:
                env_path = topfd_xiaowen.run_topfd_xiaowen(
                    peak_list_path,
                    topfd_bin=topfd_bin,
                    max_charge=topfd_max_charge,
                    error_tol=topfd_error_tol,
                    env_extra_args=topfd_env_vars,
                    max_mass=topfd_max_mass,
                    sn_ratio=topfd_sn_ratio,
                    ecscore_cutoff=topfd_ecscore_cutoff,
                    disable_frag_num_filter=topfd_disable_frag_num_filter,
                    max_miss_peak_num=topfd_max_miss_peak_num,
                    disable_filter_by_mz=topfd_disable_filter_by_mz,
                )
                print(f"  TopFD env file: {env_path}")

                env_dedup = _dedup_env_by_peak_idx(env_path)
                print(f"  Deduped env entries: {len(env_dedup)}")
                env_dedup.to_csv(
                    os.path.join(topfd_dir, "deconv_dedup_residual.env"),
                    sep="\t", index=False)

                residual_ffcs = (selected_df[residual_mask].copy()
                                 .reset_index(drop=True))
                s2_annotated = _annotate_residual_ffcs(
                    residual_ffcs, env_dedup,
                    mz_col_a=mz_col_a, mz_col_b=mz_col_b)

                n_both = 0
                if "THEO_MONO_MZ_1" in s2_annotated.columns:
                    n_both = int((s2_annotated["THEO_MONO_MZ_1"].notna()
                                  & s2_annotated["THEO_MONO_MZ_2"].notna()).sum())
                print(f"  Residual FFCs annotated: {len(s2_annotated)}")
                print(f"  Both sides matched:      {n_both}")

            except FileNotFoundError as exc:
                print(f"  [WARNING] TopFD failed: {exc}\n  Skipping Stage 2.")
            except subprocess.CalledProcessError as exc:
                print(f"  [WARNING] TopFD returned an error: {exc}\n  Skipping Stage 2.")

        if not s2_annotated.empty:
            s2_annotated.to_csv(
                os.path.join(work_dir, f"{output_prefix}_s2_annotated.tsv"),
                sep="\t", index=False)
    elif not run_topfd:
        print("\n[Stage 2 skipped: run_topfd=False]")
    else:
        print("\n[Stage 2 skipped: no residual FFCs]")

    # ══════════════════════════════════════════════════════════════════════
    # MERGE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MERGE: combining Stage 1 + Stage 2")
    print("=" * 60)

    n_lines_pool = min(top_lines, len(selected_df)) if top_lines else len(selected_df)
    merged = _merge_results(s1_replaced, s1_annotated_kept, s2_annotated,
                            mz_col_a=mz_col_a, mz_col_b=mz_col_b)
    print(f"  Total merged rows: {len(merged)}")
    if "deconv_method" in merged.columns:
        for method, count in merged["deconv_method"].value_counts().items():
            print(f"    {method}: {count}")

    merged_path = os.path.join(merge_dir, f"{output_prefix}_merged.tsv")
    merged.to_csv(merged_path, sep="\t", index=False)
    print(f"  Written: {merged_path}")

    stats = {
        "total_ffc_rows": len(df),
        "preprocessed_rows": len(bg_df),
        "selected_subset": len(selected_df),
        "top_lines": n_lines_pool,
        "n_lines_parental": sum(1 for ln in lines if ln.kind == "parental"),
        "n_lines_residual": sum(1 for ln in lines if ln.kind != "parental"),
        "stage1_line_assigned": n_line_assigned,
        "stage1_output_rows": len(s1_replaced),
        "stage1_unplaced_rows": n_unplaced_rows,
        "stage2_residual": n_residual,
        "stage2_output_rows": len(s2_annotated),
        "merged_total": len(merged),
        "merged_path": merged_path,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return {
        "stage1_annotated": s1_annotated,
        "stage1_replaced": s1_replaced,
        "stage1_line_map": s1_line_map,
        "stage2_annotated": s2_annotated,
        "merged": merged,
        "lines": lines,
        "stats": stats,
    }


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Two-stage deconvolution with orientation-agnostic lines")
    ap.add_argument("ffc_file", help="Path to the raw FFC table (TSV)")
    ap.add_argument("--output-dir", default="./deconv_output")
    ap.add_argument("--prefix", default="combined")

    ap.add_argument("--parent-charge", type=int, required=True)
    ap.add_argument("--parent-mass", type=float, required=True)

    # Stage 1: line finding
    ap.add_argument("--delta", type=float, default=0.02,
                    help="Sort-and-Split gap for line detection (default: 0.02)")
    ap.add_argument("--delta-unit", choices=["da", "ppm"], default="da",
                    help="unit for --delta (default: da)")
    ap.add_argument("--min-ffc", type=int, default=3,
                    help="minimum distinct FFCs to keep a line (default: 3)")
    ap.add_argument("--line-tol", type=float, default=None,
                    help="FFC-on-line tolerance for removal (default: --delta)")
    ap.add_argument("--top-lines", type=int, default=2000,
                    help="detect lines on the best-ranked N FFCs only; the "
                         "lines found are then matched against the rest. "
                         "0 means all of --top-selected (default: %(default)s, "
                         "from the Main-set sweep -- see deconvolution.md)")
    ap.add_argument("--merge-tolerance", type=float, default=0.002,
                    help="m/z proximity for the symmetric duplicate merge")
    ap.add_argument("--merge-over-fetch", type=int, default=4,
                    help="merge this multiple of --top-bg best-ranked rows "
                         "(the merge does not scale to the full raw table)")
    ap.add_argument("--silent-drop", action="store_true",
                    help="keep only one line membership per FFC")
    ap.add_argument("--reference-detector", action="store_true",
                    help="use line_based_annotation's reference detector "
                         "instead of the vectorised one (same lines, far "
                         "slower at this top-N)")

    # Stage 1: deconvolution back end
    ap.add_argument("--theo-patt", default=_paths.THEO_PATT)
    ap.add_argument("--mz-tol", type=float, default=0.01)
    ap.add_argument("--max-shift", type=int, default=None)
    ap.add_argument("--top-selected", type=int, default=15000)
    ap.add_argument("--top-bg", type=int, default=15000)
    ap.add_argument("--ranking-col", default="Ranking")
    ap.add_argument("--pcov-col", default="Partial Cov.")

    # Stage 2: TopFD
    ap.add_argument("--no-topfd", action="store_true")
    ap.add_argument("--topfd-bin", default=_paths.TOPFD_BIN)
    ap.add_argument("--topfd-max-charge", type=int, default=None,
                    help="TopFD -c (default: --parent-charge)")
    ap.add_argument("--topfd-max-mass", type=float, default=None,
                    help="TopFD -m (default: --parent-mass)")
    ap.add_argument("--topfd-error-tol", type=float, default=0.04)
    ap.add_argument("--topfd-sn-ratio", type=float, default=0.1)
    ap.add_argument("--topfd-ecscore-cutoff", type=float, default=0.0)
    ap.add_argument("--topfd-keep-frag-num-filter", action="store_true",
                    help="do NOT pass -d")
    ap.add_argument("--topfd-max-miss-peak-num", type=int, default=-1,
                    help="TopFD --max-miss-peak-num; negative omits the flag "
                         "(required for the shipped v1.8.1 binary)")
    ap.add_argument("--topfd-disable-filter-by-mz", action="store_true",
                    help="pass --disable-filter-by-mz (custom build only)")
    ap.add_argument("--topfd-full-pool", dest="topfd_full_pool",
                    action="store_true", default=True)
    ap.add_argument("--no-topfd-full-pool", dest="topfd_full_pool",
                    action="store_false")

    # Input format
    ap.add_argument("--skiprows", type=int, default=1)
    ap.add_argument("--intensity-col-a", default="intensity A")
    ap.add_argument("--intensity-col-b", default="intensity B")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_deconvolution(
        ffc_path=args.ffc_file,
        parent_charge=args.parent_charge,
        parent_mass=args.parent_mass,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        intensity_col_a=args.intensity_col_a,
        intensity_col_b=args.intensity_col_b,
        ranking_col=args.ranking_col,
        pcov_col=args.pcov_col,
        delta=args.delta,
        delta_unit=args.delta_unit,
        min_ffc_number=args.min_ffc,
        line_tol=args.line_tol,
        top_lines=args.top_lines or None,
        merge_tolerance=args.merge_tolerance,
        merge_over_fetch=args.merge_over_fetch,
        silent_drop=args.silent_drop,
        fast_detector=not args.reference_detector,
        theo_patt_path=args.theo_patt,
        mz_tol=args.mz_tol,
        max_shift=args.max_shift,
        top_selected=args.top_selected,
        top_bg=args.top_bg,
        run_topfd=not args.no_topfd,
        topfd_bin=args.topfd_bin,
        topfd_max_charge=args.topfd_max_charge,
        topfd_max_mass=args.topfd_max_mass,
        topfd_error_tol=args.topfd_error_tol,
        topfd_sn_ratio=args.topfd_sn_ratio,
        topfd_ecscore_cutoff=args.topfd_ecscore_cutoff,
        topfd_disable_frag_num_filter=not args.topfd_keep_frag_num_filter,
        topfd_max_miss_peak_num=(args.topfd_max_miss_peak_num
                                 if args.topfd_max_miss_peak_num >= 0 else None),
        topfd_disable_filter_by_mz=args.topfd_disable_filter_by_mz,
        topfd_full_pool=args.topfd_full_pool,
        skiprows=args.skiprows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
