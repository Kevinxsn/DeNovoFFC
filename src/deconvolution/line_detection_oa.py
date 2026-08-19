#!/usr/bin/env python3
"""
Orientation-agnostic line finding for the deconvolution pipeline
================================================================
This module is the *only* place where the new pipeline decides what a line is.
It bolts the line finder used by ``line_based_annotation`` onto the
deconvolution's Stage 1, replacing the older
``greedy_line.greedy_lines`` + ``double_with_swapped_columns`` front end.

Three things come together here:

1. **The "oa" pre-processing** (``preprocess_ffc_oa``) -- symmetric duplicate
   merging instead of column doubling.
2. **The detector itself** (``detect_lines_oa``) -- *imported*, not copied,
   from ``line_based_annotation/line_internal_annotation_oa_v2.py``, so the
   deconvolution and the annotation always cluster identically.
3. **Per-FFC orientation resolution** (``resolve_orientation``) -- the step
   that turns a canonical line charge ``(min, max)`` back into the correct
   per-side charges for one FFC.

Why the old front end was problematic
-------------------------------------
``greedy_lines`` counted a line's members on the *literal* ``(i, j)`` charge
split, so a line and its mirror ``(j, i)`` -- the same physical line, because
the deconvolution's "m/z A" / "m/z B" ordering is arbitrary -- were two
different candidates.  Two consequences:

* **Split evidence.**  A line whose FFCs happened to be reported in both
  orderings had its support divided between ``(i, j)`` and ``(j, i)``, and
  either half could then fall under ``min_ffc_number`` and be dropped entirely.
* **The doubling hack.**  ``deconv_combine_greedy`` compensated by running
  ``double_with_swapped_columns`` first, so every FFC (X, Y) also entered as
  (Y, X).  That guarantees both orientations exist, but it doubles the row
  count, halves the effective depth of any ``top_selected`` cutoff, and leaves
  each physical FFC deconvolved twice under two mirror-image charge
  hypotheses.

``detect_lines_oa`` fixes this at the source: it pools ``i*x + j*y`` over all
``(i, j)``, clusters by mass, and counts *distinct* FFCs on the **canonical**
charge ``(min(i,j), max(i,j))``, pooling both orientations.  No charge
information is read from the input at all -- the charge of each side is
*recovered* from the line the FFC lands on.  With that, doubling is not just
unnecessary, it is actively harmful (it would feed the detector two copies of
every point), so the pre-processing collapses mirrors instead of creating them.

See ``line_based_annotation/line_based.md`` section 4 for the full rationale
and section 11 for the gotchas.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap)

from greedy_line import Line, _enumerate_charges, satellite_lines  # noqa: E402
from merge_ffcs import merge_duplicate_ffcs_charge  # noqa: E402
from line_finding import prepare_ffc_data  # noqa: E402

# The detector is IMPORTED, never copied: line_based_annotation owns it, and
# this pipeline is meant to track it.  If _oa_v2 gains a better detector, the
# deconvolution picks it up with no edit here.  ``detect_lines_oa`` is kept as
# the reference implementation; ``detect_lines`` below is a vectorised rewrite
# of the same algorithm, checked against it by ``assert_detectors_agree``.
from line_internal_annotation_oa_v2 import (  # noqa: E402
    detect_lines_oa,
    _most_informative_line_oa,
    _remove_lines_oa,
    MASS_H,
)

#: A detected "parental" line is one whose charges SUM to the precursor charge
#: -- that is the filter the detector's parental pass uses, and a neutral-loss
#: line at, say, -116 Da satisfies it too.  The true **master ladder** is the
#: near-zero-shift subset, identified by |shift| < this, not by charge sum.
#: Same convention (and same 0.5 Da) as ``line_based.md`` section 8.
MASTER_LADDER_SHIFT = 0.5

__all__ = [
    "preprocess_ffc_oa",
    "MASTER_LADDER_SHIFT",
    "expand_line_membership",
    "detect_lines",
    "detect_lines_oa",
    "most_informative_line_fast",
    "assert_detectors_agree",
    "resolve_orientation",
    "resolve_orientation_array",
    "describe_lines",
    "MASS_H",
]


# =============================================================================
# 1. PRE-PROCESSING  (replaces double_with_swapped_columns)
# =============================================================================

def preprocess_ffc_oa(
    df: pd.DataFrame,
    *,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    pcov_col: Optional[str] = "Partial Cov.",
    merge_tolerance: float = 0.002,
    top_n: Optional[int] = None,
    over_fetch: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """The ``line_based_annotation`` pre-processing, applied to *raw* FFC data.

    Order of operations (each step is documented below because the order is
    load-bearing)::

        positive Partial Cov.  ->  best-ranked over_fetch*top_n rows
          ->  merge_duplicate_ffcs_charge(symmetric=True, reset_ranking=True)
          ->  prepare_ffc_data(top_n)

    **Positive partial covariance first.**  Inherited from the old pipeline and
    kept deliberately: filtering before the ranking is renumbered means the cut
    can never be defeated by going deeper into ``top_n``.

    **Then a pre-cut, because the merge does not scale.**  ``line_based_annotation``
    merges its whole table -- it works on *deconvolved* tables of ~15 k rows.
    Raw FFC tables are far bigger (KWK: 7.4 M rows), and the merge's KD-tree
    proximity pass over all of them is intractable.  So the rows are sorted by
    ranking and only the best ``over_fetch * top_n`` are merged.

    That is safe because the only thing the merge removes is a mirror duplicate:
    a pair whose two rows both rank below the cut is unaffected, and where one
    mirror is inside the cut and the other outside, the merge would have kept
    the better-ranked one -- which is the one inside. So the surviving top
    ``top_n`` is the same set, and the pre-cut only shifts where the boundary
    falls. ``over_fetch=4`` leaves ample margin (2 would already suffice if
    *every* row were mirrored), and the cut widens automatically if the merge
    still leaves fewer than ``top_n`` rows.

    **Symmetric merge, not doubling.**  ``merge_duplicate_ffcs_charge`` with
    ``symmetric=True`` collapses FFC(X, Y) and FFC(Y, X) into a single row and
    renumbers the rankings over the survivors.  This is the exact call
    ``line_internal_annotation_oa_v2``'s ``__main__`` makes.

    *A note on the charge columns.*  That function keys on the charge-corrected
    neutral mass ``z*mz - (z-1)*MASS_H``.  Raw FFC data has no charge columns --
    charges are the very thing this pipeline is about to infer -- so they are
    added as NaN, which the function maps to ``default_charge=1``.  At charge 1
    the transform is the identity, so the TFFC key degenerates to the raw m/z
    pair and the call reduces to a symmetric raw-m/z proximity dedup.  That is
    the right behaviour here: pre-deconvolution there is no charge to be aware
    of, and the mirror collapse -- the part we actually need -- is unaffected.

    **Then the top-N cutoff.**  ``prepare_ffc_data`` drops unranked rows
    (``Ranking == -1``), sorts by ranking, and keeps the head.  Because the
    merge already renumbered, "top 15000" now means 15000 *distinct physical*
    FFCs rather than 7500 doubled ones.

    Returns a DataFrame with a fresh RangeIndex, so positional ``ffc_index``
    values used downstream address it directly.
    """
    work = df.copy()   # never mutate the caller's frame (charge cols are added)

    if pcov_col and pcov_col in work.columns:
        n_before = len(work)
        work = work[work[pcov_col] > 0].copy()
        if verbose:
            print(f"  positive {pcov_col!r}: {len(work)} / {n_before} rows kept "
                  f"({n_before - len(work)} dropped)")
    elif verbose:
        print(f"  [warn] column {pcov_col!r} not found -- "
              f"positive partial-covariance filter NOT applied")

    # Charge columns are absent on raw FFC input; NaN -> default_charge=1 makes
    # the TFFC transform the identity (see the docstring).
    for col in ("charge_A", "charge_B"):
        if col not in work.columns:
            work[col] = np.nan

    if ranking_col in work.columns:
        work = work[work[ranking_col] != -1].sort_values(ranking_col)
    n_ranked = len(work)

    # Merge only as deep as needed; widen if mirrors ate into the top_n.
    cut = n_ranked if top_n is None else min(n_ranked, max(top_n * over_fetch, top_n))
    while True:
        merged = merge_duplicate_ffcs_charge(
            work.head(cut),
            col_a=mz_col_a, col_b=mz_col_b,
            ranking_col=ranking_col,
            tolerance=merge_tolerance,
            symmetric=True,
            reset_ranking=True,
        )
        if top_n is None or len(merged) >= top_n or cut >= n_ranked:
            break
        cut = min(n_ranked, cut * 2)
    if verbose:
        print(f"  symmetric duplicate merge: {len(merged)} rows from the best "
              f"{cut} of {n_ranked} ranked (tolerance={merge_tolerance})")
    work = merged

    # The synthesised charge columns have served their purpose; dropping them
    # keeps the output schema identical to the input's.
    work = work.drop(columns=[c for c in ("charge_A", "charge_B")
                             if c in work.columns and work[c].isna().all()])

    work = prepare_ffc_data(work, top_n=top_n)
    if verbose:
        print(f"  after prepare_ffc_data(top_n={top_n}): {len(work)} rows")
    return work


# =============================================================================
# 2. THE DETECTOR, VECTORISED
# =============================================================================
# ``_most_informative_line_oa`` loops over Sort-and-Split clusters in Python and
# builds a ``Counter`` per cluster.  At the top-N the *annotation* runs
# (200-2000 FFCs) that is milliseconds; at the top-N the *deconvolution* runs
# (15000) it is not -- the pooled mass set is |pairs| x N entries and the greedy
# peeling calls it once per line found, so the Python-level per-cluster overhead
# dominates and the whole detection lands in the hours.
#
# ``most_informative_line_fast`` computes exactly the same quantities with
# numpy group-by operations instead: one argsort for the clustering, one for the
# (cluster, charge) grouping, then reduceat/bincount.  It is a *reimplementation,
# not a redefinition* -- every tie-break, every count and the returned Line are
# specified to match ``_most_informative_line_oa`` element for element, and
# ``assert_detectors_agree`` checks that on real data.  The reference stays
# imported and is the oracle.

def most_informative_line_fast(ffc_map, parent_charge, delta, min_ffc_number,
                               col_a="m/z A", col_b="m/z B",
                               charge_filter=None, delta_unit="da"):
    """Vectorised ``_most_informative_line_oa``.  Same result, same tie-breaks.

    Pools ``i*x + j*y`` over all admissible ``(i, j)``, gap-clusters the pooled
    masses, and for each cluster finds the dominant **canonical** charge
    ``(min(i,j), max(i,j))`` -- so a line and its mirror are one candidate.  The
    winning cluster is the one with the most *distinct* FFCs on its dominant
    charge, ties broken by tighter mass spread and then by lower mass.
    """
    sub = ffc_map[[col_a, col_b]].dropna()
    if sub.empty:
        return None
    x = sub[col_a].to_numpy(dtype=float)
    y = sub[col_b].to_numpy(dtype=float)
    row_idx = sub.index.to_numpy()
    pairs = _enumerate_charges(parent_charge, charge_filter)
    if not pairs:
        return None

    n = x.size
    masses = np.concatenate([i * x + j * y for i, j in pairs])
    i_arr = np.concatenate([np.full(n, i, dtype=np.int64) for i, _ in pairs])
    j_arr = np.concatenate([np.full(n, j, dtype=np.int64) for _, j in pairs])
    pos_arr = np.tile(np.arange(n, dtype=np.int64), len(pairs))
    lo = np.minimum(i_arr, j_arr)
    hi = np.maximum(i_arr, j_arr)

    # ── Sort-and-Split, as cluster ids rather than a list of arrays ──────────
    order = np.argsort(masses, kind="mergesort")
    m_s = masses[order]
    if m_s.size == 1:
        cid_s = np.zeros(1, dtype=np.int64)
    else:
        gaps = np.diff(m_s)
        thr = (np.abs(m_s[:-1]) * delta * 1e-6) if delta_unit == "ppm" else delta
        cid_s = np.concatenate([[0], np.cumsum(gaps > thr)]).astype(np.int64)
    lo_s, hi_s, pos_s = lo[order], hi[order], pos_arr[order]
    n_clusters = int(cid_s[-1]) + 1

    # ── Group by (cluster, canonical charge) ─────────────────────────────────
    # The elements are already ordered by cluster (mass order), so a stable sort
    # on the charge key alone leaves each group contiguous and clusters in
    # ascending order -- which is what makes "first cluster wins a tie" below
    # match the reference's iteration over ascending-mass clusters.
    span = int(parent_charge) + 1
    gkey = cid_s * (span * span) + lo_s * span + hi_s
    gsort = np.argsort(gkey, kind="mergesort")
    gk = gkey[gsort]
    starts = np.flatnonzero(np.concatenate([[True], gk[1:] != gk[:-1]]))
    n_groups = starts.size
    g_first = gsort[starts]
    g_cid = cid_s[g_first]
    g_lo = lo_s[g_first]
    g_hi = hi_s[g_first]
    g_count = np.diff(np.append(starts, gk.size))

    # Per-group mass statistics (duplicated orientations included, exactly as
    # the reference's ``masses[cl][on]`` does).
    m_g = m_s[gsort]
    g_sum = np.add.reduceat(m_g, starts)
    g_min = np.minimum.reduceat(m_g, starts)
    g_max = np.maximum.reduceat(m_g, starts)

    # Distinct FFCs per group: one FFC can enter the same canonical group twice
    # (once per orientation), which is why the reference takes ``np.unique``.
    p_g = pos_s[gsort]
    inv = np.repeat(np.arange(n_groups), g_count)
    dsort = np.lexsort((p_g, inv))
    inv_d, p_d = inv[dsort], p_g[dsort]
    new = np.concatenate([[True], (inv_d[1:] != inv_d[:-1]) | (p_d[1:] != p_d[:-1])])
    g_distinct = np.bincount(inv_d[new], minlength=n_groups)

    # ── Dominant charge per cluster: max (count, lo+hi, -lo) ─────────────────
    # lexsort's last key is primary.
    dom_order = np.lexsort((g_lo, -(g_lo + g_hi), -g_count, g_cid))
    dom_first = np.flatnonzero(
        np.concatenate([[True], g_cid[dom_order][1:] != g_cid[dom_order][:-1]]))
    dom = dom_order[dom_first]                      # one group index per cluster
    dom_cid = g_cid[dom]

    # ── Winning cluster: max (distinct, -spread); ties -> lowest-mass cluster ─
    spread = g_max[dom] - g_min[dom]
    best = dom[np.lexsort((dom_cid, spread, -g_distinct[dom]))[0]]

    distinct = int(g_distinct[best])
    if distinct < min_ffc_number:
        return None

    # Full cluster size (all charges in that cluster), as the reference reports.
    cluster_size = int(np.count_nonzero(cid_s == g_cid[best]))
    sel = gsort[starts[best]: starts[best] + g_count[best]]
    member_pos = np.unique(pos_s[sel])
    return Line(i=int(g_lo[best]), j=int(g_hi[best]),
                mass=float(g_sum[best] / g_count[best]),
                cluster_size=cluster_size, n_on_line=distinct,
                member_indices=row_idx[member_pos])


def detect_lines(ffc_df, parent_charge, delta, min_ffc_number,
                 col_a="m/z A", col_b="m/z B", line_tol=None,
                 delta_unit="da", max_lines=10000, fast=True):
    """``detect_lines_oa`` with a pluggable inner loop.

    ``fast=True`` (default) uses ``most_informative_line_fast``; ``fast=False``
    calls ``line_internal_annotation_oa_v2._most_informative_line_oa``, the
    reference.  The surrounding greedy peeling -- clear the parental
    (charge-sum) lines and their isotope/neutral satellites first, then peel
    residual internal lines -- is identical either way, and so is the return
    value ``(line_list, master)``.
    """
    if line_tol is None:
        line_tol = delta
    pick = most_informative_line_fast if fast else _most_informative_line_oa

    master = []
    work = ffc_df.copy()
    for _ in range(max_lines):
        ln = pick(work, parent_charge, delta, min_ffc_number, col_a, col_b,
                  charge_filter=lambda i, j: (i + j) == parent_charge,
                  delta_unit=delta_unit)
        if ln is None:
            break
        ln.kind = "parental"
        master.append(ln)
        nxt = _remove_lines_oa(work, [ln], line_tol, col_a, col_b, delta_unit)
        if len(nxt) == len(work):
            break
        work = nxt

    sats = satellite_lines(master)
    work = _remove_lines_oa(ffc_df, master + sats, line_tol, col_a, col_b,
                            delta_unit)

    line_list = []
    for _ in range(max_lines):
        ln = pick(work, parent_charge, delta, min_ffc_number, col_a, col_b,
                  delta_unit=delta_unit)
        if ln is None:
            break
        line_list.append(ln)
        nxt = _remove_lines_oa(work, [ln], line_tol, col_a, col_b, delta_unit)
        if len(nxt) == len(work):
            break
        work = nxt

    return line_list, master


def expand_line_membership(lines, ffc_df, line_tol, col_a="m/z A",
                           col_b="m/z B", delta_unit="da", skip_first=0):
    """Attach FFCs *outside* the detection window to the lines already found.

    Line detection is quadratic-ish in its input, so it is often run on a
    smaller pool (``top_lines``) than the set actually being deconvolved
    (``top_selected``).  The lines are then a hypothesis about the whole
    spectrum, and the FFCs below the detection cutoff still have to be tested
    against them -- otherwise raising ``top_selected`` buys nothing.

    Membership uses the same predicate the greedy peeling uses to *remove* an
    FFC from the map: ``|i*x + j*y - mass| <= line_tol`` in **either**
    orientation.  Rows ``0 .. skip_first-1`` are left alone -- inside the
    detection window the detector's own cluster membership stands, and this
    tolerance test is a looser thing that would otherwise add members the
    detector deliberately did not give the line.

    Each extra FFC joins **one** line only, the one it reconstructs closest to,
    mirroring the exclusivity greedy peeling enforces inside the window.

    Returns ``{line_index: array_of_row_labels}`` for the added members only.
    """
    extra: Dict[int, list] = {}
    if not lines or len(ffc_df) <= skip_first:
        return extra

    tail = ffc_df.iloc[skip_first:]
    x = tail[col_a].to_numpy(dtype=float)
    y = tail[col_b].to_numpy(dtype=float)
    idx = tail.index.to_numpy()

    best_dev = np.full(x.size, np.inf)
    best_line = np.full(x.size, -1, dtype=int)
    for k, ln in enumerate(lines):
        t = abs(ln.mass) * line_tol * 1e-6 if delta_unit == "ppm" else line_tol
        dev = np.minimum(np.abs(ln.i * x + ln.j * y - ln.mass),
                         np.abs(ln.j * x + ln.i * y - ln.mass))
        better = (dev <= t) & (dev < best_dev)
        best_dev[better] = dev[better]
        best_line[better] = k

    for k in np.unique(best_line[best_line >= 0]):
        extra[int(k)] = idx[best_line == k].tolist()
    return extra


def assert_detectors_agree(ffc_df, parent_charge, delta=0.02, min_ffc_number=3,
                           col_a="m/z A", col_b="m/z B", delta_unit="da",
                           verbose: bool = True) -> None:
    """Fail loudly if the fast detector and the reference ever disagree.

    Compares the full ``(line_list, master)`` output line by line: canonical
    charge, mass, cluster size, member count and the member index set.  Run this
    whenever the reference implementation changes.
    """
    ref_list, ref_master = detect_lines(
        ffc_df, parent_charge, delta, min_ffc_number,
        col_a=col_a, col_b=col_b, delta_unit=delta_unit, fast=False)
    fast_list, fast_master = detect_lines(
        ffc_df, parent_charge, delta, min_ffc_number,
        col_a=col_a, col_b=col_b, delta_unit=delta_unit, fast=True)

    for label, ref, got in (("master", ref_master, fast_master),
                            ("line_list", ref_list, fast_list)):
        assert len(ref) == len(got), (
            f"{label}: reference found {len(ref)} lines, fast found {len(got)}")
        for k, (r, g) in enumerate(zip(ref, got)):
            assert (r.i, r.j) == (g.i, g.j), f"{label}[{k}] charge {r.i, r.j} vs {g.i, g.j}"
            assert abs(r.mass - g.mass) < 1e-9, f"{label}[{k}] mass {r.mass} vs {g.mass}"
            assert r.cluster_size == g.cluster_size, f"{label}[{k}] cluster_size"
            assert r.n_on_line == g.n_on_line, f"{label}[{k}] n_on_line"
            assert set(r.member_indices.tolist()) == set(g.member_indices.tolist()), \
                f"{label}[{k}] members"
    if verbose:
        print(f"  detectors agree: {len(ref_master)} master + {len(ref_list)} "
              f"residual lines identical")


# =============================================================================
# 3. PER-FFC ORIENTATION RESOLUTION
# =============================================================================

def resolve_orientation(line: Line, mz_a: float, mz_b: float) -> Tuple[int, int]:
    """Return this FFC's true per-side charges ``(i_used, j_used)`` on ``line``.

    ``detect_lines_oa`` stores one *canonical* charge ``(min, max)`` per line,
    pooling the ``(i, j)`` and ``(j, i)`` orientations.  A given member may sit
    on either, so the concrete assignment is decided per FFC: whichever of
    ``i*mzA + j*mzB`` and ``j*mzA + i*mzB`` reconstructs closer to the line
    mass wins.

    This is byte-for-byte the rule ``coverage_table_internal`` applies when it
    fills a cell (``line_internal_annotation_oa_v2.py`` around line 1787), which
    is what keeps the charges this pipeline writes out consistent with the
    charges the annotation later reads back in.
    """
    i, j = int(line.i), int(line.j)
    if abs((j * mz_a + i * mz_b) - line.mass) < abs((i * mz_a + j * mz_b) - line.mass):
        return j, i
    return i, j


def resolve_orientation_array(
    line: Line, mz_a: np.ndarray, mz_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised ``resolve_orientation`` over many members of one line."""
    i, j = int(line.i), int(line.j)
    mz_a = np.asarray(mz_a, dtype=float)
    mz_b = np.asarray(mz_b, dtype=float)
    swap = (np.abs((j * mz_a + i * mz_b) - line.mass)
            < np.abs((i * mz_a + j * mz_b) - line.mass))
    i_used = np.where(swap, j, i).astype(int)
    j_used = np.where(swap, i, j).astype(int)
    return i_used, j_used


# =============================================================================
# 4. REPORTING
# =============================================================================

def describe_lines(
    lines: Sequence[Line],
    parent_mass: float,
    title: str = "lines",
    max_rows: int = 60,
) -> pd.DataFrame:
    """One row per line: canonical charge, mass, shift vs. parent, #members.

    ``shift = mass - parent_mass``; ~0 marks the master ladder, negative an
    internal / neutral-loss line (``line_based.md`` section 1).  ``ladder``
    flags the near-zero-shift lines -- ``kind == "parental"`` alone does not,
    see ``MASTER_LADDER_SHIFT``.
    """
    rows = []
    for k, ln in enumerate(lines):
        shift = float(ln.mass) - parent_mass
        rows.append({
            "line_id": k,
            "kind": ln.kind,
            "i": ln.i,
            "j": ln.j,
            "charge_sum": ln.i + ln.j,
            "mass": round(float(ln.mass), 4),
            "shift": round(shift, 4),
            "ladder": abs(shift) < MASTER_LADDER_SHIFT,
            "n_ffc": int(ln.n_on_line),
            "cluster_size": int(ln.cluster_size),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        print(f"\n=== {title} ({len(frame)}) ===")
        print(frame.head(max_rows).to_string(index=False))
        if len(frame) > max_rows:
            print(f"    ... {len(frame) - max_rows} more")
    else:
        print(f"\n=== {title} (none) ===")
    return frame


# =============================================================================
# 5. SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    # A charge-3 precursor with a (1, 2) parental line whose members are
    # reported in BOTH orientations -- the case the old front end split in two.
    PARENT_MASS = 1000.0
    forward = [(600.0, 200.0), (500.0, 250.0), (400.0, 300.0)]   # 1*a + 2*b = 1000
    mirrored = [(200.0, 400.0)]                                   # 2*a + 1*b = 800? no:
    # 1*200 + 2*400 = 1000 as (1,2) too -- build a true mirror instead:
    mirrored = [(b, a) for a, b in [(700.0, 150.0)]]              # (150, 700): 2*150+1*700=1000

    rows, rank = [], 1
    for a, b in forward + mirrored:
        rows.append({"m/z A": a, "m/z B": b, "Covariance": 1.0,
                     "Partial Cov.": 1.0, "Score": 1.0, "Ranking": rank,
                     "intensity A": 100.0, "intensity B": 100.0})
        rank += 1
    ffc = pd.DataFrame(rows)

    prepped = preprocess_ffc_oa(ffc, top_n=100)
    line_list, master = detect_lines(
        prepped, parent_charge=3, delta=0.02, min_ffc_number=3,
    )
    describe_lines(master, PARENT_MASS, "MasterLines (parental)")
    describe_lines(line_list, PARENT_MASS, "GreedyLines (residual)")
    assert_detectors_agree(prepped, parent_charge=3)

    assert master, "the parental line should have been found"
    ln = master[0]
    assert (ln.i, ln.j) == (1, 2), f"canonical charge should be (1, 2), got ({ln.i}, {ln.j})"
    assert ln.n_on_line == 4, f"all 4 FFCs should be on it, got {ln.n_on_line}"

    # The mirrored member must come back as (2, 1), not the canonical (1, 2).
    assert resolve_orientation(ln, 600.0, 200.0) == (1, 2)
    assert resolve_orientation(ln, 150.0, 700.0) == (2, 1)
    print("\nAll smoke-test assertions passed.")
