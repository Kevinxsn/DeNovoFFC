"""
Greedy FFC Line Finding  (MostInformativeLine / GreedyLines)
============================================================
New version of the line-detection method for fragment-fragment
correlation (FFC) maps, implementing the ``MostInformativeLine`` /
``GreedyLines`` pseudocode.

What changed vs. the original ``line_finding.py``
-------------------------------------------------
1. **Symmetric charge enumeration in a single pass.**
   ``MostInformativeLine`` enumerates  i = 1..parentCharge-1,
   j = 1..parentCharge-i.  Because i and j range *independently*, a single
   pass over fixed (x, y) now covers both (i, j) and (j, i) orderings
   (e.g. (1,2) and (2,1)).  This removes the need to run detection twice
   with x/y swapped.

2. **One combined mass set, clustered together.**
   Every quintet  ( mass_{i,j}(X,Y), i, j, X, Y )  for every FFC and every
   admissible (i, j) is pooled into a single ``massSet``, then partitioned
   by Sort-and-Split(delta).  The largest cluster's most-frequent (i, j)
   charge state defines the line; its mass is the mean reconstructed mass
   over that (i, j) sub-population.

3. **Greedy peeling.**
   ``GreedyLines`` clears the parental lines together with their isotopic
   and neutral-loss satellites (``MasterLines``), then repeatedly extracts
   the most informative line from the residual map, removing its FFCs each
   round, until no informative line remains.

Reconstructed mass convention
------------------------------
    mass_{i,j}(X, Y) = i * (m/z_X) + j * (m/z_Y)

This matches the original module's  v = i*x + j*y  (no proton term).  The
proton convention only shifts the *absolute* value of every line by the
same amount, so it does not affect clustering, the (i, j) charge sums that
flag a line as "parental", or the *relative* isotopic/neutral offsets used
to build the satellite lines.  A proton offset can be applied later, only
where the line mass is compared to a true neutral parent mass.
        --> FLAGGED: confirm whether you want a proton term folded into
            ``reconstructed_mass`` (see ``PROTON`` below, default 0.0).

Satellite (isotopic / neutral) offsets are charge-independent in this
reconstructed-mass space:
  * an isotope step of fragment X shifts m/z_X by +1.00235/i, so v shifts
    by i * (1.00235/i) = +1.00235  -> isotopic lines at  M + k*1.00235
  * a neutral loss of mass D from fragment X shifts m/z_X by -D/i, so v
    shifts by -D  -> neutral lines at  M - D
        --> FLAGGED: isotope spacing fixed at 1.00235 Da (per project
            convention); neutral-loss set and isotope k-range are
            configurable below -- confirm the set you want cleared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Conventions / constants  (FLAGGED for confirmation)
# -----------------------------------------------------------------------------

#: Optional proton term folded into the reconstructed mass.  Default 0.0 keeps
#: exact parity with the original  v = i*x + j*y.  Set to a proton mass if you
#: want neutral-mass-referenced lines.
PROTON: float = 0.0

#: Isotope peak spacing used throughout the project.
ISOTOPE_SPACING: float = 1.00235

#: Default neutral losses to clear as satellite lines (monoisotopic masses).
DEFAULT_NEUTRAL_LOSSES: Dict[str, float] = {
    "H2O": 18.010565,
    "NH3": 17.026549,
    "CO": 27.994915,
}

#: Default isotope satellite offsets (in units of ISOTOPE_SPACING) to clear,
#: relative to the monoisotopic parental line.
DEFAULT_ISOTOPE_K: Tuple[int, ...] = (1, 2, 3)


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------

@dataclass
class Line:
    """
    An informative line  i*x + j*y = mass , encoded by the triple (i, j, mass).

    Attributes
    ----------
    i, j : int
        Charge-state multipliers (z_X, z_Y) defining the line.
    mass : float
        Mean reconstructed mass over the dominant-(i, j) members of the cluster.
    cluster_size : int
        Size of the full Sort-and-Split cluster the line was drawn from
        (this is the quantity compared against ``min_ffc_number``, matching the
        pseudocode's |maxCluster|).
    n_on_line : int
        Number of FFCs whose dominant-(i, j) reconstruction lies in the cluster
        (i.e. members actually on the (i, j) line).
    member_indices : ndarray
        Original FFC-map row indices of the on-line members.
    member_rankings : ndarray
        Rankings of those members (empty if no ranking column was present).
    kind : str
        Provenance tag: "parental", "isotopic", "neutral", or "greedy".
    """
    i: int
    j: int
    mass: float
    cluster_size: int = 0
    n_on_line: int = 0
    member_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    member_rankings: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    kind: str = "greedy"

    def as_triple(self) -> Tuple[int, int, float]:
        return (self.i, self.j, self.mass)


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def reconstructed_mass(x: np.ndarray, y: np.ndarray, i: int, j: int) -> np.ndarray:
    """mass_{i,j}(X, Y) = i*x + j*y  (+ optional PROTON term)."""
    return i * x + j * y + PROTON


def _enumerate_charges(
    parent_charge: int,
    charge_filter: Optional[Callable[[int, int], bool]] = None,
) -> List[Tuple[int, int]]:
    """
    Admissible (i, j) pairs:  i = 1..parentCharge-1,  j = 1..parentCharge-i.

    i and j range independently, so both (i, j) and (j, i) appear when
    i != j and i + j <= parentCharge.  An optional ``charge_filter(i, j)``
    further restricts the set (used to isolate parental lines).
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(1, parent_charge):
        for j in range(1, parent_charge - i + 1):
            if charge_filter is None or charge_filter(i, j):
                pairs.append((i, j))
    return pairs


def _sort_and_split(values: np.ndarray, delta: float) -> List[np.ndarray]:
    """
    Sort-and-Split(delta) clustering.

    Single-linkage gap clustering: after sorting, a gap > ``delta`` between
    consecutive values starts a new cluster.  Returns a list of arrays of
    *positions into the sorted order*.  No minimum-size filtering is applied
    here -- that comparison happens against ``min_ffc_number`` in the caller.
    """
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    v_sorted = values[order]
    if n == 1:
        return [order]
    gaps = np.diff(v_sorted)
    split_after = np.where(gaps > delta)[0] + 1
    groups_sorted = np.split(np.arange(n), split_after)
    # Map each group of sorted-positions back to original positions.
    return [order[g] for g in groups_sorted]


# -----------------------------------------------------------------------------
# MostInformativeLine
# -----------------------------------------------------------------------------

def most_informative_line(
    ffc_map: pd.DataFrame,
    parent_charge: int,
    delta: float,
    min_ffc_number: int,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    charge_filter: Optional[Callable[[int, int], bool]] = None,
    count_on_line_only: bool = True,
) -> Optional[Line]:
    """
    Identify the single most informative line in ``ffc_map``.

    Pseudocode
    ----------
        massSet <- {}
        for each FFC (X, Y):
            for i = 1..parentCharge-1:
                for j = 1..parentCharge-i:
                    add ( mass_{i,j}(X,Y), i, j, X, Y ) to massSet
        clusters  <- Sort-and-Split(delta) over massSet
        maxCluster <- largest cluster
        (i, j)    <- most frequent charge state in maxCluster
        mass      <- mean mass_{i,j} over the (i, j) members of maxCluster
        Line      <- (i, j, mass)
        return Line if |maxCluster| >= minFFCnumber else 0

    Parameters
    ----------
    charge_filter : callable (i, j) -> bool, optional
        Restrict the enumerated charge states (e.g. ``lambda i, j: i+j == Z``
        to find parental lines only).
    count_on_line_only : bool
        If False (default, literal pseudocode), the ``min_ffc_number`` test
        uses the full cluster size |maxCluster|.  If True, it uses the number
        of FFCs actually on the dominant (i, j) line.

    Returns
    -------
    Line, or None if no cluster meets ``min_ffc_number`` (the "return 0" case).
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

    # Build the pooled mass set as parallel arrays:
    #   masses[k], i_arr[k], j_arr[k], ffcpos[k]  (ffcpos -> position in `sub`)
    n = x.size
    masses_blocks: List[np.ndarray] = []
    i_blocks: List[np.ndarray] = []
    j_blocks: List[np.ndarray] = []
    pos_blocks: List[np.ndarray] = []
    base_pos = np.arange(n)

    for (i, j) in pairs:
        masses_blocks.append(reconstructed_mass(x, y, i, j))
        i_blocks.append(np.full(n, i, dtype=int))
        j_blocks.append(np.full(n, j, dtype=int))
        pos_blocks.append(base_pos)

    masses = np.concatenate(masses_blocks)
    i_arr = np.concatenate(i_blocks)
    j_arr = np.concatenate(j_blocks)
    pos_arr = np.concatenate(pos_blocks)

    # Sort-and-Split over the whole pooled set.
    clusters = _sort_and_split(masses, delta)
    if not clusters:
        return None

    # Largest cluster (tie-break: tighter mass spread).
    def _spread(members: np.ndarray) -> float:
        mv = masses[members]
        return float(mv.max() - mv.min())

    max_cluster = max(clusters, key=lambda m: (m.size, -_spread(m)))
    cluster_size = int(max_cluster.size)

    # Most frequent (i, j) within the cluster.
    cl_i = i_arr[max_cluster]
    cl_j = j_arr[max_cluster]
    cl_pos = pos_arr[max_cluster]
    cl_mass = masses[max_cluster]

    counts = Counter(zip(cl_i.tolist(), cl_j.tolist()))
    # tie-break: highest count, then larger total charge (closer to a full
    # parental reconstruction), then smaller i for determinism.
    dom_i, dom_j = max(
        counts.keys(),
        key=lambda ij: (counts[ij], ij[0] + ij[1], -ij[0]),
    )

    on_line = (cl_i == dom_i) & (cl_j == dom_j)
    line_mass = float(cl_mass[on_line].mean())
    member_pos = cl_pos[on_line]
    member_indices = row_idx[member_pos]

    member_rankings = np.array([], dtype=int)
    if ranking_col in ffc_map.columns:
        member_rankings = ffc_map.loc[member_indices, ranking_col].to_numpy()

    n_on_line = int(on_line.sum())

    test_count = n_on_line if count_on_line_only else cluster_size
    if test_count < min_ffc_number:
        return None

    return Line(
        i=int(dom_i),
        j=int(dom_j),
        mass=line_mass,
        cluster_size=cluster_size,
        n_on_line=n_on_line,
        member_indices=member_indices,
        member_rankings=member_rankings,
        kind="greedy",
    )


# -----------------------------------------------------------------------------
# Line membership / removal
# -----------------------------------------------------------------------------

def ffcs_on_line(
    ffc_map: pd.DataFrame,
    line: Line,
    tol: float,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
) -> np.ndarray:
    """Original row indices whose (i, j) reconstruction is within ``tol`` of the
    line mass:  |i*x + j*y - mass| <= tol."""
    sub = ffc_map[[col_a, col_b]].dropna()
    if sub.empty:
        return np.array([], dtype=int)
    v = reconstructed_mass(
        sub[col_a].to_numpy(dtype=float),
        sub[col_b].to_numpy(dtype=float),
        line.i,
        line.j,
    )
    mask = np.abs(v - line.mass) <= tol
    return sub.index.to_numpy()[mask]


def _remove_lines(
    ffc_map: pd.DataFrame,
    lines: Sequence[Line],
    tol: float,
    col_a: str,
    col_b: str,
    also_use_members: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Drop every FFC lying on any of ``lines``.  Returns (residual_map,
    removed_indices).  Union of the on-line predicate and the recorded member
    indices guarantees forward progress in the greedy loop."""
    to_drop: set = set()
    for ln in lines:
        idx = ffcs_on_line(ffc_map, ln, tol, col_a, col_b)
        to_drop.update(idx.tolist())
        if also_use_members and ln.member_indices.size:
            to_drop.update(ln.member_indices.tolist())
    keep = ~ffc_map.index.isin(to_drop)
    return ffc_map[keep], np.array(sorted(to_drop), dtype=int)


# -----------------------------------------------------------------------------
# MasterLines : parental lines + isotopic + neutral satellites
# -----------------------------------------------------------------------------

def find_parental_lines(
    ffc_map: pd.DataFrame,
    parent_charge: int,
    delta: float,
    min_ffc_number: int,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    line_tol: Optional[float] = None,
    max_lines: int = 1000,
) -> List[Line]:
    """
    Greedily extract all informative *parental* lines (i + j == parentCharge).

    Each found parental line's FFCs are removed before searching for the next,
    so multiple parental lines (e.g. charge splits (1,2) and (2,1) of the same
    precursor charge) are all recovered.
    """
    if line_tol is None:
        line_tol = delta
    work = ffc_map.copy()
    parental: List[Line] = []
    for _ in range(max_lines):
        ln = most_informative_line(
            work, parent_charge, delta, min_ffc_number,
            col_a=col_a, col_b=col_b, ranking_col=ranking_col,
            charge_filter=lambda i, j: (i + j) == parent_charge,
        )
        if ln is None:
            break
        ln.kind = "parental"
        parental.append(ln)
        work, removed = _remove_lines(work, [ln], line_tol, col_a, col_b)
        if removed.size == 0:
            break
    return parental


def satellite_lines(
    parental: Sequence[Line],
    isotope_k: Sequence[int] = DEFAULT_ISOTOPE_K,
    neutral_losses: Optional[Dict[str, float]] = None,
    isotope_spacing: float = ISOTOPE_SPACING,
) -> List[Line]:
    """
    Build the isotopic and neutral-loss satellite lines for each parental line.

    Same (i, j); mass offset only:
        isotopic :  mass + k * isotope_spacing   (k in isotope_k)
        neutral  :  mass - D                      (D in neutral_losses)
    """
    if neutral_losses is None:
        neutral_losses = DEFAULT_NEUTRAL_LOSSES
    sats: List[Line] = []
    for p in parental:
        for k in isotope_k:
            sats.append(Line(p.i, p.j, p.mass + k * isotope_spacing, kind="isotopic"))
        for _name, d in neutral_losses.items():
            sats.append(Line(p.i, p.j, p.mass - d, kind="neutral"))
    return sats


def build_master_lines(
    ffc_map: pd.DataFrame,
    parent_charge: int,
    delta: float,
    min_ffc_number: int,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    line_tol: Optional[float] = None,
    isotope_k: Sequence[int] = DEFAULT_ISOTOPE_K,
    neutral_losses: Optional[Dict[str, float]] = None,
) -> List[Line]:
    """
    MasterLines = all informative parental lines + their isotopic and neutral
    lines.  Returned with ``kind`` tags so callers can inspect provenance.
    """
    parental = find_parental_lines(
        ffc_map, parent_charge, delta, min_ffc_number,
        col_a=col_a, col_b=col_b, ranking_col=ranking_col, line_tol=line_tol,
    )
    sats = satellite_lines(parental, isotope_k=isotope_k, neutral_losses=neutral_losses)
    return parental + sats


# -----------------------------------------------------------------------------
# GreedyLines
# -----------------------------------------------------------------------------

def greedy_lines(
    ffc_map: pd.DataFrame,
    parent_charge: int,
    delta: float,
    min_ffc_number: int,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    line_tol: Optional[float] = None,
    use_master_lines: bool = True,
    isotope_k: Sequence[int] = DEFAULT_ISOTOPE_K,
    neutral_losses: Optional[Dict[str, float]] = None,
    max_lines: int = 10000,
    return_master: bool = False,
    report_satellites: bool = False,
):
    """
    GreedyLines pseudocode:

        LineList    <- []
        MasterLines <- parental lines + isotopic + neutral lines
        remove FFCs on MasterLines from FFCmap
        while informative lines remain in FFCmap:
            Line <- MostInformativeLine(FFCmap, ...)
            append Line; remove FFCs on Line from FFCmap
        return LineList

    Parameters
    ----------
    line_tol : float, optional
        Tolerance for deciding which FFCs lie on a line during removal.
        Defaults to ``delta``.
    use_master_lines : bool
        If False, skip the parental/satellite pre-clearing step (greedy from
        the raw map).
    return_master : bool
        If True, return (LineList, MasterLines) instead of just LineList.
    report_satellites : bool
        The isotopic/neutral satellite lines are *generated* (predicted from
        each detected parental line), not detected -- they carry no point
        counts and exist only to clear predictable echo FFCs from the map.
        By default the returned MasterLines contains only the **detected**
        parental lines (the ones with real point counts).  Set this True to
        also include the generated satellites for auditing what got cleared.
    """
    if line_tol is None:
        line_tol = delta

    work = ffc_map.copy()
    master: List[Line] = []

    if use_master_lines:
        # Detected parental lines (real point counts) ...
        parental = find_parental_lines(
            work, parent_charge, delta, min_ffc_number,
            col_a=col_a, col_b=col_b, ranking_col=ranking_col, line_tol=line_tol,
        )
        # ... and their generated isotopic/neutral satellites (no point counts,
        # used only for clearing).
        sats = satellite_lines(
            parental, isotope_k=isotope_k, neutral_losses=neutral_losses
        )
        work, _ = _remove_lines(work, parental + sats, line_tol, col_a, col_b)
        master = (parental + sats) if report_satellites else parental

    line_list: List[Line] = []
    for _ in range(max_lines):
        ln = most_informative_line(
            work, parent_charge, delta, min_ffc_number,
            col_a=col_a, col_b=col_b, ranking_col=ranking_col,
        )
        if ln is None:
            break
        line_list.append(ln)
        work, removed = _remove_lines(work, [ln], line_tol, col_a, col_b)
        if removed.size == 0:  # safety: guarantee progress
            break

    if return_master:
        return line_list, master
    return line_list


# -----------------------------------------------------------------------------
# Pretty-print helper
# -----------------------------------------------------------------------------

def lines_to_frame(
    lines: Sequence[Line],
    parent_mass: Optional[float] = None,
    max_rankings: int = 5,
) -> pd.DataFrame:
    """Tabulate a list of Line objects (drops the heavy member-index arrays but
    keeps the per-line rankings).

    The ``rankings`` column lists the Ranking of every FFC on the line, sorted
    ascending (best first), truncated to ``max_rankings`` entries.  Generated
    satellite lines have no points, so their list is empty.

    Parameters
    ----------
    parent_mass : float, optional
        Known/actual parental mass.  If given, a signed ``shift`` column is
        added (placed right after ``mass``):

            shift = line.mass - parent_mass

        Interpretation: ~0 for the true parental line, +k*1.00235 for isotope
        echoes, -(neutral loss) for neutral lines, and negative for the
        sub-parental internal lines (how much mass is "missing" vs the full
        precursor).  This is diagnostic only -- it never affects detection.
        Omitted entirely when ``parent_mass`` is None.
    max_rankings : int
        Maximum number of rankings to show per line (best/lowest first).
        Defaults to 5.
    """
    rows = []
    for ln in lines:
        row = {
            "kind": ln.kind,
            "i": ln.i,
            "j": ln.j,
            "i+j": ln.i + ln.j,
            "mass": round(ln.mass, 4),
        }
        if parent_mass is not None:
            row["shift"] = round(ln.mass - parent_mass, 4)
        row["cluster_size"] = ln.cluster_size
        row["n_on_line"] = ln.n_on_line
        all_rankings = sorted(int(r) for r in ln.member_rankings)
        row["rankings"] = all_rankings[:max_rankings]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_shifts(
    lines: Sequence[Line],
    parent_mass: float,
    ndigits: int = 2,
) -> List[float]:
    """
    Return the sorted, de-duplicated list of line shifts relative to a known
    parental mass:

        shift = line.mass - parent_mass

    Each shift is rounded to ``ndigits`` decimals first, so shifts that are
    really close merge into a single value (e.g. 1.0008 and 0.9997 both -> 1.0).

    Pass any sequence of lines -- e.g. ``collect_shifts(master, M)``,
    ``collect_shifts(residual, M)``, or ``collect_shifts(master + residual, M)``
    to pool both tables.
    """
    if parent_mass is None:
        raise ValueError("collect_shifts requires a parent_mass.")
    # `+ 0.0` normalizes any -0.0 to 0.0 so it prints cleanly.
    shifts = {round(ln.mass - parent_mass, ndigits) + 0.0 for ln in lines}
    return sorted(shifts)


def double_with_swapped_columns(df, col1, col2, extra_swap_pairs=None, reset_index=True):
    """
    Return a dataframe with original rows plus duplicated rows where
    the values in col1 and col2 are swapped.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col1 : str
        First column name.
    col2 : str
        Second column name.
    extra_swap_pairs : list of (str, str), optional
        Additional column pairs to swap in the duplicated rows alongside
        (col1, col2).  Only pairs where both columns exist in df are swapped.
        Use this to keep paired columns (e.g. intensity A / intensity B)
        consistent with the m/z swap.
    reset_index : bool
        Whether to reset index after concatenation.

    Returns
    -------
    pd.DataFrame
        Dataframe with doubled number of rows.
    """
    swapped_df = df.copy()

    swapped_df[[col1, col2]] = swapped_df[[col2, col1]].values

    if extra_swap_pairs:
        for c1, c2 in extra_swap_pairs:
            if c1 in swapped_df.columns and c2 in swapped_df.columns:
                swapped_df[[c1, c2]] = swapped_df[[c2, c1]].values

    result = pd.concat([df, swapped_df], axis=0)

    if reset_index:
        result = result.reset_index(drop=True)

    return result


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from line_finding import load_ffc_excel, prepare_ffc_data  # reuse loaders

    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/HAD_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/VEA_merged.tsv"
    DATA_PATH = "/home/sux002/projects/2D_spec_dict/deiso/data/LLG6+_with_intensity.txt"
    PARENT_CHARGE = 6
    DELTA = 0.005
    MIN_FFC = 3
    TOP_N = 1000
    MAX_RANKINGS = 20       # max rankings shown per line in the output table
    #PARENT_MASS = 1608.87  # set to None to omit the shift column VEA
    #PARENT_MASS = 1887.036239 # YLE
    PARENT_MASS = 4496.62218
    #PARENT_MASS = 941.96162 * 4

    #ffc_df = pd.read_csv(DATA_PATH, sep=r"\s+", skiprows=1, header=None, engine="python")
    #ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    ffc_df = pd.read_csv(DATA_PATH, sep='\t')
    
    
    ffc_df = double_with_swapped_columns(ffc_df, 'm/z A', 'm/z B')
    from merge_ffcs import merge_duplicate_ffcs


    
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)
    ffc_df = merge_duplicate_ffcs(ffc_df)

    lines, master = greedy_lines(
        ffc_df,
        parent_charge=PARENT_CHARGE,
        delta=DELTA,
        min_ffc_number=MIN_FFC,
        return_master=True,
    )

    print("=== MasterLines (parental + satellites) ===")
    print(lines_to_frame(master, parent_mass=PARENT_MASS, max_rankings=MAX_RANKINGS).to_string(index=False))
    print("\n=== GreedyLines (residual, internal-fragment lines) ===")
    print(lines_to_frame(lines, parent_mass=PARENT_MASS, max_rankings=MAX_RANKINGS).to_string(index=False))

    if PARENT_MASS is not None:
        print("\n=== Merged shifts (rounded to 3 digits) ===")
        print(collect_shifts(master + lines, PARENT_MASS))