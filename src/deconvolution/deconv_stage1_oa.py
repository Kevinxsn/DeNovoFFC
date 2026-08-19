#!/usr/bin/env python3
"""
Stage 1 — FFC-line-based deconvolution with orientation-agnostic line finding
=============================================================================
Descendant of ``deiso/python_code/deconv_combine_greedy.py``.  The
**deconvolution back end is reused verbatim** from ``deconv_combine_revised``
(charge-1 transform, ``theo_patt`` monoisotopic refinement, background
augmentation, the "replaced" view); only the front end -- how lines are found
and how each FFC's per-side charges are read off them -- is new.

What changed vs. ``deconv_combine_greedy``
------------------------------------------
========================  ================================  ============================
step                      old (greedy)                      new (oa)
========================  ================================  ============================
pre-processing            ``double_with_swapped_columns``   symmetric duplicate merge
                          (every FFC entered twice)         (mirrors collapsed to one row)
line detection            ``greedy_lines``                  ``detect_lines_oa``
                          counts on the literal (i, j)      counts distinct FFCs on the
                                                            canonical (min, max) charge
per-side charge           the line's (i, j), as stored      resolved per FFC, by which
                                                            orientation reconstructs
                                                            closest to the line mass
gap threshold             Da only                           Da or ppm (``delta_unit``)
========================  ================================  ============================

Everything downstream of the charge assignment -- Steps 3-5 -- is unchanged, and
that matters: the line only tells us the *charge* of each side.  The observed
peak on a line may itself be a +1/+2 isotope, so the monoisotopic mass still has
to be recovered by the existing envelope / ``theo_patt`` machinery.

Input contract
--------------
Unlike its ancestor this function does **not** pre-process its input.  It takes
``selected_df`` and ``bg_df`` already prepared (by
``line_detection_oa.preprocess_ffc_oa``) with a fresh RangeIndex, where
``selected_df`` is the head of ``bg_df``.  Doing the pre-processing once, in the
caller, is what lets ``ffc_index`` mean the same thing on both sides of the
call -- the old split (pipeline prepared one frame, Stage 1 silently prepared a
different one) is why residual detection there had to match on m/z pairs rather
than on indices.

Charge-multiplicity policy (``silent_drop``)
--------------------------------------------
A single FFC can be claimed by more than one line (a parental line and a
residual internal-fragment line, say).  ``silent_drop=True`` keeps only the
first claim (master lines are iterated first, so they win); ``silent_drop=False``
(the default, as in the previous pipeline) keeps every membership, so an FFC on
N lines yields N rows distinguished by ``line_id``.

Note this is no longer entangled with the doubling.  Previously the same
physical hypothesis appeared twice -- as (X, Y) on an (i, j) line *and* as
(Y, X) on a (j, i) line -- purely because of ``double_with_swapped_columns``.
With mirrors collapsed, a second row now means a genuinely second line.

Outputs
-------
Same contract as ``deconv_combine.deconvolute_ffc_by_lines``:
``annotated`` / ``replaced`` / ``line_map`` / ``assigned_indices``, plus
``lines`` (the detected ``Line`` objects, master first) for reporting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap)

from greedy_line import Line  # noqa: E402

# Deconvolution back end -- reused unchanged, so masses stay identical to the
# existing pipeline.
from deconv_combine_revised import (  # noqa: E402
    LineAssignment,
    _to_charge1_mz,
    _deconvolute_pooled,
    _load_lookup,
    PROTON_MASS,
)

from line_detection_oa import (  # noqa: E402
    detect_lines,
    describe_lines,
    expand_line_membership,
    resolve_orientation_array,
    MASTER_LADDER_SHIFT,
)


# =============================================================================
# 1. LINE -> CHARGE ASSIGNMENT  (orientation resolved per FFC)
# =============================================================================

def assignments_from_oa_lines(
    all_lines: Sequence[Line],
    ffc_df: pd.DataFrame,
    parent_mass: float,
    n_selected: int,
    *,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    shift_decimals: int = 3,
    silent_drop: bool = False,
) -> List[LineAssignment]:
    """Flatten ``Line`` objects into the ``LineAssignment`` list Steps 3-5 eat.

    The one substantive difference from
    ``deconv_combine_greedy.assignments_from_greedy_lines``: each assignment's
    ``(i, j)`` is the **per-FFC resolved orientation**, not the line's stored
    canonical ``(min, max)``.  A line pools both orientations, so taking the
    canonical pair literally would hand roughly half the members their charges
    the wrong way round -- and since the charge is exactly what Stage 1 exists
    to determine, that error would propagate straight into every downstream
    mass.

    Members outside the top ``n_selected`` rows are dropped, mirroring the
    original ``a.ffc_index < n_selected`` filter.
    """
    assignments: List[LineAssignment] = []
    seen: set = set()

    for line_id, ln in enumerate(all_lines):
        if ln.member_indices.size == 0:
            continue  # generated satellites carry no points
        shift = float(ln.mass - parent_mass)
        is_parental = (ln.kind == "parental")
        # matched_offset mirrors the old field: 0.0 for a parental line,
        # otherwise the (rounded) Parent+X shift this line sits at.
        matched_offset = 0.0 if is_parental else round(shift, shift_decimals)

        idx = np.asarray(ln.member_indices, dtype=int)
        rows = ffc_df.loc[idx]
        i_used, j_used = resolve_orientation_array(
            ln,
            rows[mz_col_a].to_numpy(dtype=float),
            rows[mz_col_b].to_numpy(dtype=float),
        )

        for ffc_idx, i_u, j_u in zip(idx.tolist(), i_used.tolist(), j_used.tolist()):
            if silent_drop:
                if ffc_idx in seen:
                    continue
                seen.add(ffc_idx)
            assignments.append(
                LineAssignment(
                    ffc_index=int(ffc_idx),
                    i=int(i_u),
                    j=int(j_u),
                    line_id=line_id,
                    line_offset=shift,
                    matched_offset=float(matched_offset),
                    line_n_points=int(ln.n_on_line),
                    is_parental=is_parental,
                )
            )

    return [a for a in assignments if a.ffc_index < n_selected]


# =============================================================================
# 2. PUBLIC ENTRY POINT
# =============================================================================

def deconvolute_ffc_by_lines_oa(
    selected_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    *,
    # ── column mapping ──
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    # ── line-finding parameters ──
    delta: float = 0.02,
    delta_unit: str = "da",
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    top_lines: Optional[int] = None,
    # ── deconvolution back-end parameters (unchanged) ──
    theo_patt_path: Optional[str] = None,
    mz_tol: float = 0.02,
    max_shift: Optional[int] = None,
    shift_decimals: int = 3,
    # ── charge-multiplicity policy ──
    silent_drop: bool = False,
    fast_detector: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Full FFC-line-based deconvolution, orientation-agnostic front end.

    ``selected_df`` are the FFCs that get deconvolved and returned;
    ``bg_df`` (its superset, same RangeIndex on the shared prefix) supplies the
    background peaks the envelope refinement augments with.  Both must already
    be pre-processed -- see the module docstring.

    ``delta`` is the Sort-and-Split gap for line detection, read as Da or ppm
    according to ``delta_unit``.  ``line_tol`` (the FFC-on-line removal
    tolerance) defaults to ``delta``.

    ``top_lines`` restricts *line detection* to the best-ranked N FFCs; the
    lines found are then matched against the rest of ``selected_df``
    (``expand_line_membership``).  Detection cost grows steeply with its input
    while the deconvolution itself is linear, so this is the knob that trades
    detection time against how deep into the ranking lines are looked for.
    ``None`` (default) detects on everything selected.

    ``fast_detector=False`` swaps in ``line_internal_annotation_oa_v2``'s
    reference implementation.  Same lines, far slower at this pipeline's top-N
    (see ``line_detection_oa`` section 2); useful for spot-checking.
    """
    if delta_unit not in ("da", "ppm"):
        raise ValueError(f"delta_unit must be 'da' or 'ppm', got {delta_unit!r}")
    for col in (mz_col_a, mz_col_b):
        for name, frame in (("selected_df", selected_df), ("bg_df", bg_df)):
            if col not in frame.columns:
                raise KeyError(f"{name} must contain column {col!r}")
    if len(bg_df) < len(selected_df):
        raise ValueError(
            f"bg_df ({len(bg_df)}) must be at least as large as selected_df "
            f"({len(selected_df)})"
        )

    n_selected = len(selected_df)

    # ── Steps 1-2: orientation-agnostic detection + charge assignment ────────
    # Detection may run on a smaller pool than the set being deconvolved; the
    # lines it finds are then tested against the rest (expand_line_membership).
    n_lines_pool = min(top_lines, n_selected) if top_lines else n_selected
    line_df = selected_df.head(n_lines_pool)

    line_list, master = detect_lines(
        line_df,
        parent_charge=parent_charge,
        delta=delta,
        min_ffc_number=min_ffc_number,
        col_a=mz_col_a,
        col_b=mz_col_b,
        line_tol=line_tol,
        delta_unit=delta_unit,
        fast=fast_detector,
    )
    all_lines: List[Line] = list(master) + list(line_list)

    n_added = 0
    if n_lines_pool < n_selected and all_lines:
        extra = expand_line_membership(
            all_lines, selected_df,
            line_tol if line_tol is not None else delta,
            col_a=mz_col_a, col_b=mz_col_b, delta_unit=delta_unit,
            skip_first=n_lines_pool,
        )
        for k, labels in extra.items():
            ln = all_lines[k]
            ln.member_indices = np.unique(
                np.concatenate([np.asarray(ln.member_indices, dtype=int),
                                np.asarray(labels, dtype=int)]))
            ln.n_on_line = int(ln.member_indices.size)
            n_added += len(labels)

    if verbose:
        if n_lines_pool < n_selected:
            print(f"  lines detected on the top {n_lines_pool} of "
                  f"{n_selected} FFCs; {n_added} further FFCs matched onto them")
        describe_lines(master, parent_mass, "MasterLines (parental)")
        describe_lines(line_list, parent_mass,
                       "GreedyLines (residual, internal-fragment lines)")

    assignments = assignments_from_oa_lines(
        all_lines, selected_df, parent_mass, n_selected,
        mz_col_a=mz_col_a, mz_col_b=mz_col_b,
        shift_decimals=shift_decimals, silent_drop=silent_drop,
    )

    if not assignments:
        empty = pd.DataFrame()
        return {
            "annotated": empty, "replaced": empty, "line_map": empty,
            "assigned_indices": set(), "lines": all_lines,
        }

    # =====================================================================
    # Steps 3-5 below are reused VERBATIM from deconv_combine -- only the
    # source of `assignments` changed.
    # =====================================================================

    line_map_df = pd.DataFrame([
        {
            "ffc_index": a.ffc_index,
            "line_id": a.line_id,
            "i": a.i,
            "j": a.j,
            "line_offset": a.line_offset,
            "matched_offset": a.matched_offset,
            "line_n_points": a.line_n_points,
            "is_parental": a.is_parental,
        }
        for a in assignments
    ])

    base = selected_df.iloc[line_map_df["ffc_index"].values].reset_index(drop=True)
    annotated = pd.concat(
        [base, line_map_df.reset_index(drop=True).drop(columns=["ffc_index"])],
        axis=1,
    )
    annotated["ffc_index"] = line_map_df["ffc_index"].values

    # ── Step 3: charge-1-equivalent transform ─────────────────────────────
    annotated["mz1_A"] = [
        _to_charge1_mz(mz, z)
        for mz, z in zip(annotated[mz_col_a].values, annotated["i"].values)
    ]
    annotated["mz1_B"] = [
        _to_charge1_mz(mz, z)
        for mz, z in zip(annotated[mz_col_b].values, annotated["j"].values)
    ]

    # ── Step 4: isotope-envelope deconvolution on the charge-1 axis ───────
    mono_lookup, env_lookup = _load_lookup(theo_patt_path)

    if intensity_col_a and intensity_col_a in annotated.columns:
        int_a = annotated[intensity_col_a].values.astype(float)
    else:
        int_a = np.ones(len(annotated), dtype=float)
    if intensity_col_b and intensity_col_b in annotated.columns:
        int_b = annotated[intensity_col_b].values.astype(float)
    else:
        int_b = np.ones(len(annotated), dtype=float)

    # Background peak pool: both m/z columns of bg_df, replicated once per
    # charge seen in the assignments so the charge-1 transform can be applied.
    all_charges = np.unique(np.concatenate([
        annotated["i"].values.astype(int),
        annotated["j"].values.astype(int),
    ]))
    full_mz_a = bg_df[mz_col_a].values.astype(float)
    full_mz_b = bg_df[mz_col_b].values.astype(float)
    if intensity_col_a and intensity_col_a in bg_df.columns:
        full_int_a = bg_df[intensity_col_a].values.astype(float)
    else:
        full_int_a = np.ones(len(bg_df), dtype=float)
    if intensity_col_b and intensity_col_b in bg_df.columns:
        full_int_b = bg_df[intensity_col_b].values.astype(float)
    else:
        full_int_b = np.ones(len(bg_df), dtype=float)

    all_bg_mz = np.concatenate([full_mz_a, full_mz_b])
    all_bg_int = np.concatenate([full_int_a, full_int_b])

    bg_mz_arr = np.concatenate([all_bg_mz for _ in all_charges])
    bg_int_arr = np.concatenate([all_bg_int for _ in all_charges])
    bg_charge_arr = np.concatenate(
        [np.full(len(all_bg_mz), z, dtype=int) for z in all_charges]
    )

    (mono_mz1_a, mono_mass_a, sim_a, env_id_a, env_members_a,
     mono_mz1_b, mono_mass_b, sim_b, env_id_b, env_members_b) = _deconvolute_pooled(
        annotated["mz1_A"].values.astype(float),
        annotated["mz1_B"].values.astype(float),
        int_a, int_b,
        mono_lookup, env_lookup, mz_tol,
        max_shift=max_shift,
        charge_a=annotated["i"].values.astype(int),
        charge_b=annotated["j"].values.astype(int),
        bg_mz=bg_mz_arr,
        bg_int=bg_int_arr,
        bg_charges=bg_charge_arr,
    )

    annotated["charge_A"] = annotated["i"]
    annotated["charge_B"] = annotated["j"]
    annotated["envelope_id_A"] = env_id_a
    annotated["envelope_id_B"] = env_id_b
    annotated["deconvoluted_mz_A"] = np.round(mono_mz1_a, 4)
    annotated["deconvoluted_mz_B"] = np.round(mono_mz1_b, 4)
    annotated["monoisotopic_mass_A"] = np.round(mono_mass_a, 4)
    annotated["monoisotopic_mass_B"] = np.round(mono_mass_b, 4)
    annotated["isotope_similarity_A"] = np.round(sim_a, 4)
    annotated["isotope_similarity_B"] = np.round(sim_b, 4)
    annotated["envelope_members_A"] = [str(d) for d in env_members_a]
    annotated["envelope_members_B"] = [str(d) for d in env_members_b]
    annotated["line_kind"] = [
        all_lines[lid].kind for lid in annotated["line_id"].values
    ]
    annotated["line_shift"] = np.round(
        annotated["line_offset"].values.astype(float), 4)
    # ``is_parental`` (and ``kind == "parental"``) only means the line's charges
    # SUM to the precursor charge -- the detector's parental pass filters on
    # ``i + j == parent_charge``, which a neutral-loss line at, say, -116 Da
    # also satisfies.  The true master ladder is the near-zero-shift subset,
    # exactly as line_based.md section 8 defines it.  Both flags are kept: the
    # charge sum is what made the assignment, the shift is what makes it the
    # parental ladder.
    annotated["is_master_ladder"] = (
        annotated["line_shift"].abs() < MASTER_LADDER_SHIFT)

    # ── Step 5: build the "replaced" view ─────────────────────────────────
    replaced = base.copy()
    replaced[mz_col_a] = (annotated["monoisotopic_mass_A"].values
                          / annotated["charge_A"] + PROTON_MASS)
    replaced[mz_col_b] = (annotated["monoisotopic_mass_B"].values
                          / annotated["charge_B"] + PROTON_MASS)
    replaced["line_id"] = annotated["line_id"].values

    return {
        "annotated": annotated.reset_index(drop=True),
        "replaced": replaced.reset_index(drop=True),
        "line_map": line_map_df.reset_index(drop=True),
        "assigned_indices": set(int(x) for x in line_map_df["ffc_index"].unique()),
        "lines": all_lines,
    }


# =============================================================================
# 3. SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    from line_detection_oa import preprocess_ffc_oa

    # Charge-3 precursor; a (1, 2) parental line of 4 complementary FFCs, one of
    # them reported mirrored, plus off-line noise.  theo_patt_path=None => the
    # monoisotopic refiner uses its lowest-peak fallback, enough to exercise the
    # full wiring.
    PARENT_MASS = 1000.0
    pairs = [(600.0, 200.0), (500.0, 250.0), (400.0, 300.0), (150.0, 700.0)]
    noise = [(123.4, 555.5), (321.0, 444.4), (810.0, 90.0)]

    rows, rank = [], 1
    for a, b in pairs + noise:
        rows.append({"m/z A": a, "m/z B": b, "Covariance": 1.0,
                     "Partial Cov.": 1.0, "Score": 1.0, "Ranking": rank,
                     "intensity A": 100.0, "intensity B": 100.0})
        rank += 1
    ffc = pd.DataFrame(rows)

    prepped = preprocess_ffc_oa(ffc, top_n=100)
    out = deconvolute_ffc_by_lines_oa(
        prepped, prepped, parent_charge=3, parent_mass=PARENT_MASS,
        intensity_col_a="intensity A", intensity_col_b="intensity B",
        delta=0.02, min_ffc_number=3, theo_patt_path=None,
    )

    a = out["annotated"]
    print("\n=== annotated (key cols) ===")
    cols = ["ffc_index", "m/z A", "m/z B", "charge_A", "charge_B",
            "monoisotopic_mass_A", "monoisotopic_mass_B", "line_kind"]
    print(a[cols].to_string(index=False))
    print("\nassigned_indices:", sorted(out["assigned_indices"]))

    # The mirrored FFC (150, 700) must get (2, 1) while the rest get (1, 2).
    got = {(round(r["m/z A"], 1), round(r["m/z B"], 1)):
           (int(r["charge_A"]), int(r["charge_B"])) for _, r in a.iterrows()}
    assert got[(600.0, 200.0)] == (1, 2), got
    assert got[(150.0, 700.0)] == (2, 1), got
    assert len(out["assigned_indices"]) == 4
    assert a["monoisotopic_mass_A"].notna().all()
    assert a["monoisotopic_mass_B"].notna().all()
    print("\nAll smoke-test assertions passed.")
