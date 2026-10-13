"""
Line-Based Annotation -- Charge-Independent, Orientation-Agnostic Detection
==========================================================================
Variant of ``line_internal_annotation_ppm.py`` that recovers every FFC's
charge from line detection instead of trusting the deconvolution's per-side
charge, and treats a line ``(i, j)`` and its mirror ``(j, i)`` as one physical
line (the m/z-A / m/z-B ordering from deconvolution is arbitrary).

Two things differ from the ppm module:
  * Detection (``detect_lines_oa``) ignores ``charge_A``/``charge_B``, pools
    both orientations of every FFC when counting a line's members, and stores
    a canonical ``(i, j) = (min, max)`` per line; each member's true
    orientation is resolved per-FFC in the coverage table.
  * The line-detection ``delta`` defaults to 0.02 Da (still Da for now; to be
    moved to ppm later), needed to bridge the small reconstruction spread of
    mirror-paired FFCs.

Fragment matching stays on the same relative **ppm** window as the ppm module.
The coverage-table output (columns, rows, cell format) is otherwise identical;
only *which* lines are detected can change.

Why ppm: the observed m/z carries a roughly *relative* mass error, so a fixed
Da window over-tolerates heavy/multiply-charged fragments and under-tolerates
light ones.  A ppm window is also charge-invariant here -- since
``adj_mass ~= i * mz``, a ppm window on ``adj_mass`` is the same relative
window as on the raw m/z, so it lines up with ffc_annotator's ppm-on-m/z.

Everything else (fragment index, pairing tiers, internal/secondary/neutral-loss
labelling, mock FFCs, coverage table, information rows) is identical.

Motivation
----------
``annotation_greedy.py`` keys each coverage-table column by the greedy line
``(i, j, shift)`` and then annotates each side of an FFC against b/y ions
*plus a neutral-loss offset equal to the line shift*.  For a fragment that
sits on a sub-parental (internal) line, that produces awkward labels such as

    b22, (y11+1)-348.155

i.e. "b22 paired with y11 minus 348 Da".  What that fragment really is, for
a 33-residue peptide, is the *internal* fragment spanning y-residues 4..11
(y11 with its first three C-terminal residues removed, whose mass is the
348 Da that was "lost").  This module instead builds the theoretical
*internal* ions directly and reports the clean label

    b22, i(4:11)

Internal-ion notation
---------------------
An internal fragment is an amino-acyl (b-type) fragment spanning residues
``s..e`` (1-based, inclusive) with ``1 < s`` and ``e < n``.  It is labelled in
absolute whole-peptide residue coordinates ``i(s:e)`` -- the same convention
as ``theoretical_mms_annotation/ffc_annotator.py`` -- so the label reads the
same regardless of which side of the FFC the fragment is on.  For example the
residues-26..30 fragment paired with b25 reads ``i(26:30)`` (not the
y-series-relative ``i(4:8)``).

The complementary bond that indexes the coverage-table row is always the
cleavage *between* the two adjacent fragments, i.e. after residue
``left.e``, giving row ``"b{left.e}, y{n-left.e}"`` -- the same row a genuine
parental ``b/y`` pair would occupy.

Pipeline
--------
1. Run ``detect_lines_oa()`` (orientation-agnostic) on the FFC map.
2. Build the theoretical fragment index (b, y, internal) once.
3. For every FFC assigned to a line ``(i, j, mass)``:
     - convert each observed m/z to its singly-charged [M+H]+ mass using the
       line's per-side charge (adj_mass_A uses i, adj_mass_B uses j),
     - find the best *compatible* (adjacent, or complementary terminal) pair
       of theoretical fragments,
     - order the pair N->C, label any internal side, and drop it into the
       coverage cell for its bond and its ``(i, j, shift)`` column.
4. Assemble the coverage table (same row/column shape as annotation_greedy).
"""

from __future__ import annotations

import os
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# The greedy-line detector and peptide model live in the deiso package.
_DEISO = os.path.join(os.path.dirname(__file__), "..", "deiso", "python_code")
sys.path.insert(0, os.path.abspath(_DEISO))

from greedy_line import (  # noqa: E402
    Line, _enumerate_charges, _sort_and_split, satellite_lines,
)
import peptide  # noqa: E402
from peptide import AA  # noqa: E402

# ── Physical constants (kept identical to annotation.py) ──────────────────────
ISOTOPE_OFFSET = 1.00335   # Da, spacing between isotopic peaks
MASS_H = 1.00784           # Da, used for charge-state deconvolution of m/z
PROTON = AA.element_masses["proton"]   # 1.00725, used inside pep.ion_mass()

#: Default relative mass tolerance (parts per million) for fragment lookup.
DEFAULT_TOL_PPM = 20.0

#: Default tolerance (ppm) for identifying *mock* FFCs.  Deliberately looser
#: than DEFAULT_TOL_PPM and independent of it: a mock is probed from the line's
#: *shift*, which is a cluster mean carrying the line-detection error, not a
#: single measured peak -- so it sits further from theory than a real FFC does.
DEFAULT_MOCK_TOL_PPM = 100.0

#: A line whose |shift| is below this (Da) carries no real neutral loss -- it is
#: the parental/isotope line.  This is a *structural* Da test on the line shift,
#: not a matching tolerance, so it stays absolute even in the ppm variant.
MIN_LOSS_DA = 0.05

#: Default Sort-and-Split gap (Da) for line detection.  Wider than the
#: ppm-module default so mirror-paired FFCs, whose reconstructions differ
#: by a few mDa, land in one cluster.  (To be migrated to ppm later.)
DEFAULT_DELTA = 0.02


# =============================================================================
# 1. COLUMN KEY  (unchanged from annotation_greedy._col_key)
# =============================================================================

def _col_key(i: int, j: int, shift: float, decimals: int = 3,
             parent_tol: float = 0.005) -> str:
    """Format a ``(i, j, shift)`` triple as a readable column label.

    (1, 3, 0.0018)  -> "(1,3) parent+0.002"
    (2, 2, 1.0023)  -> "(2,2) +1.002"
    (1, 2, -348.15) -> "(1,2) -348.155"

    ``decimals`` sets the numeric shift precision (3 dp by default).  A
    genuinely near-zero shift (|shift| < ``parent_tol``, default 0.005) is
    labelled ``parent`` but *still shows its deviation* -- e.g.
    "parent+0.002" -- so the parental line's residual from the exact precursor
    mass stays visible.  A larger offset like +0.006 shows plainly as "+0.006".
    """
    num = f"{shift:+.{decimals}f}"
    shift_str = f"parent{num}" if abs(shift) < parent_tol else num
    return f"({i},{j}) {shift_str}"


# =============================================================================
# 2. THEORETICAL FRAGMENT INDEX (b, y, internal)
# =============================================================================

#: Named neutral losses considered on any fragment (at most one per fragment),
#: matching the ordered set used by theoretical_mms_annotation/ffc_annotator.py.
NEUTRAL_LOSSES: Dict[str, float] = {
    "H2O": 18.01056,
    "NH3": 17.02655,
    "CO": 27.99491,
}


@dataclass(frozen=True)
class Fragment:
    """A theoretical fragment.

    kind      : 'b', 'y', or 'internal'
    s, e      : 1-based inclusive residue span covered by the fragment.
                b_k -> (1, k);  y_k -> (n-k+1, n);  internal -> (s, e).
    mzH       : monoisotopic singly-charged [M+H]+ mass, *including* any
                named neutral loss already subtracted.
    num       : ion number for terminal ions (b{num} / y{num}); 0 for internal.
    loss_name : '' for an intact fragment, else one of NEUTRAL_LOSSES
                ('H2O' / 'NH3' / 'CO').  At most one loss per fragment.
    """
    kind: str
    s: int
    e: int
    mzH: float
    num: int = 0
    loss_name: str = ""


def build_fragment_index(pep, iso_range: int = 0,
                         neutral_losses: Optional[Dict[str, float]] = None):
    """Build the theoretical fragment set and a sorted [M+H]+ lookup.

    The set covers terminal (b/y) and internal ions, each both intact and
    carrying a single named neutral loss (H2O / NH3 / CO).  That lets a
    fragment on a loss line be named for what it actually is -- e.g.
    ``i(2:6)-H2O`` or ``(b2)-CO`` -- instead of falling back to the opaque
    "terminal ion minus the raw line shift" form.

    Returns
    -------
    (mz_grid, frag_grid, iso_grid, n)
        Three parallel, m/z-sorted arrays -- monoisotopic-or-isotope-shifted
        [M+H]+ mass, the backing ``Fragment``, and the isotope index -- plus
        the peptide length ``n``.  Bisect ``mz_grid`` for tolerance windows.
    """
    if neutral_losses is None:
        neutral_losses = NEUTRAL_LOSSES
    n = len(pep.AA_array)

    # prefix[k] = sum of residue masses of residues 1..k (b-type, no water)
    prefix = [0.0] * (n + 1)
    for k in range(1, n + 1):
        prefix[k] = prefix[k - 1] + pep.AA_array[k - 1].get_mass()

    # Intact fragments first: (kind, s, e, num, [M+H]+)
    base: List[Tuple[str, int, int, int, float]] = []
    for k in range(1, n):                       # b-ions: residues 1..k
        base.append(("b", 1, k, k, pep.ion_mass(f"b{k}")))
    for k in range(1, n):                       # y-ions: residues (n-k+1)..n
        base.append(("y", n - k + 1, n, k, pep.ion_mass(f"y{k}")))
    for s in range(2, n):                       # internal: 1 < s <= e < n
        for e in range(s, n):
            base.append(("internal", s, e, 0, (prefix[e] - prefix[s - 1]) + PROTON))

    # Expand each into its intact form plus one-loss variants.
    frags: List[Fragment] = []
    for kind, s, e, num, mzH in base:
        frags.append(Fragment(kind, s, e, mzH, num=num, loss_name=""))
        for lname, lmass in neutral_losses.items():
            if mzH - lmass <= 0:
                continue
            frags.append(Fragment(kind, s, e, mzH - lmass, num=num, loss_name=lname))

    # Expand isotopes onto a flat, m/z-sorted grid.
    records: List[Tuple[float, Fragment, int]] = []
    for f in frags:
        for iso in range(iso_range + 1):
            records.append((f.mzH + iso * ISOTOPE_OFFSET, f, iso))
    records.sort(key=lambda r: r[0])

    mz_grid = np.array([r[0] for r in records], dtype=float)
    frag_grid = [r[1] for r in records]
    iso_grid = np.array([r[2] for r in records], dtype=int)
    return mz_grid, frag_grid, iso_grid, n


@dataclass
class Candidate:
    frag: Fragment
    iso: int
    dev: float
    loss: float = 0.0   # neutral-loss (line shift) applied to a terminal ion

    @property
    def rank(self) -> int:
        """Plausibility class (lower = preferred):

            0  intact terminal b/y ion
            1  intact internal ion
            2  terminal ion with one named neutral loss   -> "(b2)-CO"
            3  internal ion with one named neutral loss   -> "i(2:6)-H2O"
            4  terminal ion minus the raw line shift      -> "(y8)-113.064"

        Named-loss interpretations therefore win over the opaque raw-shift
        fallback, while intact ions still win over everything.
        """
        if self.loss > 1e-6:
            return 4
        terminal = self.frag.kind in ("b", "y")
        if not self.frag.loss_name:
            return 0 if terminal else 1
        return 2 if terminal else 3


def _candidates(adj_mass: float, mz_grid, frag_grid, iso_grid,
                tol_ppm: float) -> List[Candidate]:
    """All theoretical fragments within ``tol_ppm`` (relative) of a [M+H]+ mass.

    The window is ``adj_mass * tol_ppm * 1e-6``.  Because ``adj_mass ~= i*mz``,
    this is the same relative window as a ppm tolerance on the raw m/z.
    """
    tol_da = abs(adj_mass) * tol_ppm * 1e-6
    lo = bisect_left(mz_grid, adj_mass - tol_da)
    hi = bisect_right(mz_grid, adj_mass + tol_da)
    out = []
    for k in range(lo, hi):
        out.append(Candidate(frag_grid[k], int(iso_grid[k]),
                             abs(adj_mass - mz_grid[k])))
    return out


def _shifted_terminal_candidates(adj_mass: float, neutral_loss: float,
                                 mz_grid, frag_grid, iso_grid,
                                 tol_ppm: float) -> List[Candidate]:
    """Terminal (b/y) ions carried onto a loss line: a fragment that lost
    ``neutral_loss`` Da matches an intact b/y ion whose mass is
    ``adj_mass + neutral_loss``.  Mirrors annotation_greedy's
    ``_build_ions_for_line`` fallback, so lines whose shift is an awkward
    neutral loss (e.g. -17.99 water, -113.06 residue) still annotate as
    ``(y8)-113.064`` rather than going unexplained.  Isotope variants are
    handled implicitly by the isotope-expanded index.

    Only *intact* terminal ions are shifted -- stacking the raw line shift on
    top of an already named neutral loss would be a double loss."""
    out = []
    for c in _candidates(adj_mass + neutral_loss, mz_grid, frag_grid, iso_grid,
                         tol_ppm):
        if c.frag.kind in ("b", "y") and not c.frag.loss_name:
            out.append(Candidate(c.frag, c.iso, c.dev, loss=neutral_loss))
    return out


# =============================================================================
# 3. PAIRING + LABELLING
# =============================================================================

def _adjacent(f1: Fragment, f2: Fragment) -> bool:
    """True if the two fragments meet at exactly one backbone bond (f1 then
    f2, N->C) with no overlap.  Parental b/y pairs and terminal+internal
    pairs both satisfy this."""
    return f1.e + 1 == f2.s


_RAW_SHIFT_RANK = 4   # Candidate.rank for the "(b/y) - raw line shift" fallback


def _is_secondary(f1: Fragment, f2: Fragment) -> bool:
    """True if the two fragments form a *secondary* pair (Yalcin et al. 1996):
    one terminal b-ion and one terminal y-ion that are non-overlapping but
    NOT adjacent -- i.e. a stretch of residues between them is unaccounted for
    (e.g. b9 / y6 on a 16-mer, with residue 10 in the gap).  Two same-type
    terminal ions always overlap, so a valid non-overlapping terminal pair is
    necessarily one b and one y."""
    if {f1.kind, f2.kind} != {"b", "y"}:
        return False
    non_overlapping = f1.e < f2.s or f2.e < f1.s
    adjacent = _adjacent(f1, f2) or _adjacent(f2, f1)
    return non_overlapping and not adjacent


def _best_pair(cand_a: List[Candidate], cand_b: List[Candidate],
               consider_secondary: bool = True):
    """Pick the most plausible compatible pair of candidates.

    Geometry tiers (lower = preferred):
        0  adjacent pair explained without the raw-shift fallback
           (parental / internal / neutral-loss)
        1  secondary pair -- two terminal b/y ions with a gap between them
           (only when ``consider_secondary``; never uses the raw-shift
           fallback), considered only as a last resort before the raw shift
        2  adjacent pair that needs the raw-shift fallback ``(b/y)-shift``

    Within a tier: fewer internal ions / neutral losses (rank sum), then lower
    isotope, then lower mass deviation.  ``cand_a`` come from side A (charge
    i), ``cand_b`` from side B (charge j).

    Returns ``(cand_a, cand_b)`` (original A/B association preserved) or
    ``None`` if nothing is compatible.
    """
    best = None
    best_key = None
    for ca in cand_a:
        for cb in cand_b:
            adjacent = _adjacent(ca.frag, cb.frag) or _adjacent(cb.frag, ca.frag)
            uses_raw = ca.rank == _RAW_SHIFT_RANK or cb.rank == _RAW_SHIFT_RANK
            if adjacent:
                tier = 2 if uses_raw else 0
            elif (consider_secondary and not uses_raw
                  and _is_secondary(ca.frag, cb.frag)):
                tier = 1
            else:
                continue
            key = (tier, ca.rank + cb.rank, ca.iso + cb.iso, ca.dev + cb.dev)
            if best_key is None or key < best_key:
                best_key = key
                best = (ca, cb)
    return best


# -----------------------------------------------------------------------------
# ffc_annotator-style FFC classification (parental / internal / secondary /
# unmatched / unexplained).  Used only for the information rows below the main
# table -- it deliberately excludes the raw-shift fallback so the categories
# match theoretical_mms_annotation/ffc_annotator.py's definitions:
#   * unmatched   -- at least one side matches no theoretical b/y/internal ion
#   * unexplained -- both sides match, but the pair is not a valid theoretical
#                    FFC (parental / internal / secondary)
# (The main-table cells are unaffected; they still use the raw-shift fallback.)
# -----------------------------------------------------------------------------

#: Tie-break order for the "line type" row (most-common category per line).
_CATEGORY_ORDER = ("parental", "internal", "secondary", "unexplained", "unmatched")


def _mod_marker(cand: "Candidate") -> str:
    """Per-side modification marker (mirrors ffc_annotator._mod_marker):
      '*'  the side carries a named neutral loss (takes precedence);
      '~'  the side carries an isotopic shift but no loss;
      ''   the side is a bare monoisotopic ion.
    A side with both a loss and an isotope is marked '*' only, so each side
    contributes at most one character -- giving suffixes '', '*', '~', '**',
    '~~', '*~', '~*' (e.g. (b2-CO, y13) -> 'parental*').
    """
    if cand.frag.loss_name:
        return "*"
    if cand.iso > 0:
        return "~"
    return ""


def _valid_ffc_category(fa: Fragment, fb: Fragment,
                        consider_secondary: bool) -> Optional[str]:
    """Category of a valid theoretical FFC pair, or None if (fa, fb) is not a
    valid FFC (mirrors ffc_annotator.is_compatible + classify_ffc_relationship,
    minus the modification suffix):

        parental   two terminal ions, adjacent (complementary b/y)
        internal   one terminal + one internal, adjacent
        secondary  two terminal ions (one b, one y), non-overlapping, gap
        None       everything else (two losses, internal-internal, overlapping
                   or same-type terminals, non-adjacent internal, ...)
    """
    if fa.loss_name and fb.loss_name:          # at most one neutral loss / pair
        return None
    a_term = fa.kind in ("b", "y")
    b_term = fb.kind in ("b", "y")
    adjacent = _adjacent(fa, fb) or _adjacent(fb, fa)
    if a_term and b_term:
        if adjacent:
            return "parental"
        if consider_secondary and _is_secondary(fa, fb):
            return "secondary"
        return None
    if adjacent:                               # at least one internal
        if fa.kind == "internal" and fb.kind == "internal":
            return None                        # internal-internal is not an FFC
        return "internal"
    return None


def _classify_ffc(cand_a: List[Candidate], cand_b: List[Candidate],
                  consider_secondary: bool) -> str:
    """Classify one FFC (given its raw-shift-free per-side candidate lists)
    into one of the five ffc_annotator categories."""
    if not cand_a or not cand_b:
        return "unmatched"
    best_cat = None
    best_key = None
    best_pair = None
    for ca in cand_a:
        for cb in cand_b:
            cat = _valid_ffc_category(ca.frag, cb.frag, consider_secondary)
            if cat is None:
                continue
            tier = 1 if cat == "secondary" else 0
            key = (tier, ca.rank + cb.rank, ca.iso + cb.iso, ca.dev + cb.dev)
            if best_key is None or key < best_key:
                best_key = key
                best_cat = cat
                best_pair = (ca, cb)
    if best_cat is None:
        return "unexplained"
    # Per-side modification suffix, ordered N->C (the b-side first, matching
    # ffc_annotator's X-then-Y order after b-in-A/y-in-B placement).
    left, right = sorted(best_pair, key=lambda c: c.frag.s)
    return f"{best_cat}{_mod_marker(left)}{_mod_marker(right)}"


def _base_category(category: str) -> str:
    """Strip the per-side modification suffix: 'parental*~' -> 'parental'."""
    return category.rstrip("*~")


def _most_common_category(counter: Counter) -> Optional[str]:
    """Most frequent category on a line, including its modification suffix
    (e.g. 'parental*').  Ties are broken by base-category plausibility
    (``_CATEGORY_ORDER``) and then by the cleaner (shorter-suffix) variant."""
    if not counter:
        return None

    def _key(item):
        cat, count = item
        base = _base_category(cat)
        base_rank = (_CATEGORY_ORDER.index(base) if base in _CATEGORY_ORDER
                     else len(_CATEGORY_ORDER))
        return (count, -base_rank, -len(cat))

    return max(counter.items(), key=_key)[0]


def _mass_i(mz: float, charge: int) -> float:
    """Transformed per-side mass mass_i(X) = i * (X - proton)."""
    return charge * (mz - PROTON)


def _fpr_tuple(x_label: str, y_label: str, i: int, j: int, ranking: int,
               mz_a: float, mz_b: float) -> str:
    """Diagnostic tuple for an unmatched/unexplained FFC:
    ``(X, Y, (i,j), [rank], mass_i(X), mass_j(Y), mass_i(X)+mass_j(Y))``."""
    ma = _mass_i(mz_a, i)
    mb = _mass_i(mz_b, j)
    return (f"({x_label}, {y_label}, ({i},{j}), [{ranking}], "
            f"{ma:.3f}, {mb:.3f}, {ma + mb:.3f})")


def _internal_label(frag: Fragment, is_left: bool, n: int) -> str:
    """Internal-ion label in absolute whole-peptide residue coordinates
    ``i(s:e)``, where s..e are the 1-based residue positions the fragment
    spans.  This matches the theoretical FFC annotator's ``i(k:m)`` convention
    and is unambiguous regardless of which side the partner is on -- e.g. the
    residues-26..30 fragment paired with b25 reads ``i(26:30)`` rather than the
    y-series-relative ``i(4:8)``.

    (``is_left`` / ``n`` are retained for call-site compatibility; the label no
    longer depends on them.)
    """
    return f"i({frag.s}:{frag.e})"


def _decorate_loss(base: str, cand: "Candidate") -> str:
    """Attach a named neutral loss to a fragment label.

    Terminal ions are parenthesised so the loss reads against the whole ion
    (``(b2)-CO``); internal labels already carry their own brackets and take
    the loss directly (``i(2:6)-H2O``).
    """
    if not cand.frag.loss_name:
        return base
    if cand.frag.kind == "internal":
        return f"{base}-{cand.frag.loss_name}"
    return f"({base})-{cand.frag.loss_name}"


def _iso_tag(iso: int) -> str:
    """Isotopologue suffix: '' for monoisotopic, ' (+1)' / ' (+2)' otherwise.

    Without it, a match that landed on the +k isotope peak of a fragment
    prints identically to the monoisotopic match, so an isotope-satellite
    line (e.g. -226.16, the +1 echo of -227.17) would show the same labels
    as its parent line."""
    return f" (+{iso})" if iso else ""


def _placed_label(cand: "Candidate", is_left: bool, n: int) -> str:
    """Label for a fragment placed as part of a compatible pair.

    Internal ions use the complementary-frame ``i(a:b)``; a named neutral loss
    is appended (``i(2:6)-H2O``, ``(b2)-CO``).  The raw-shift fallback keeps
    annotation_greedy's ``(b/y{num})-{shift}`` form.  A non-zero isotope is
    flagged with a trailing ``(+k)``.
    """
    f = cand.frag
    if cand.loss > 1e-6:
        base = f"({f.kind}{f.num})-{round(cand.loss, 3)}"
    else:
        base = _internal_label(f, is_left, n) if f.kind == "internal" else f"{f.kind}{f.num}"
        base = _decorate_loss(base, cand)
    return base + _iso_tag(cand.iso)


def _lone_label(cand: "Candidate", n: int) -> str:
    """Label for a fragment matched on its own (half-break example), where the
    partner is unknown so there is no complementary frame.  Internal ions fall
    back to whole-peptide (b-frame) numbering ``i(s:e)``."""
    f = cand.frag
    if cand.loss > 1e-6:
        base = f"({f.kind}{f.num})-{round(cand.loss, 3)}"
    else:
        base = f"i({f.s}:{f.e})" if f.kind == "internal" else f"{f.kind}{f.num}"
        base = _decorate_loss(base, cand)
    return base + _iso_tag(cand.iso)


def _side_best_label(cands: List["Candidate"], n: int,
                     mz: Optional[float] = None) -> str:
    """Best standalone label for one side of an FFC.

    With no candidate the side reads "No Match"; if ``mz`` is supplied it is
    appended as ``No Match-<m/z>`` so an unexplained peak can still be looked
    up by hand from the diagnostic tuple.
    """
    if not cands:
        return "No Match" if mz is None else f"No Match-{mz:.4f}"
    best = min(cands, key=lambda c: (c.rank, c.iso, c.dev))
    return _lone_label(best, n)


def _t_tuple(mz_a: float, mz_b: float, i: int, j: int):
    """Transformed-FFC tuple: ``(i*mzA, j*mzB, i*mzA + j*mzB)`` -- the charge-
    weighted per-side masses and their reconstructed sum, for direct
    comparison against theoretical neutral masses."""
    ta = round(mz_a * i, 4)
    tb = round(mz_b * j, 4)
    return (ta, tb, round(ta + tb, 4))


# =============================================================================
# 4a. SUMMARY ROWS (FFC count, false-positive counts, examples)
# =============================================================================

def _build_summary_rows(
    all_columns,
    col_order: list,
    line_type_map: dict,
    bp_ffc_map: dict,
    unm_unexp_map: dict,
    act_inact_map: dict,
    aligned_map: dict,
    spurious_map: dict,
    ex_unexplained: dict,
    ex_unmatched: dict,
    max_ex: int,
) -> pd.DataFrame:
    """Build the per-line summary rows appended below the coverage table.

    Row order:
        line type                      (most-common FFC category on the line)
        #breakpoints/FFCs on a line
        #active/#inactive TFFCs        (internal columns only)
        aligned?                       (internal columns only: 1/0)
        #unmatched/#unexplained        (ffc_annotator-style FPR counts)
        First spurious FFC             (any column, blank if none)
        Unresolved FFCs 1..max_ex      (up to max_ex unexplained examples)
        Unmatched FFCs 1..max_ex       (up to max_ex unmatched examples)

    ``line type`` is one of parental / internal / secondary / unexplained /
    unmatched.  ``#unmatched/#unexplained`` shows ``{unmatched}/{unexplained}``.
    ``#active/#inactive TFFCs`` is ``{active}/{inactive}`` for internal
    columns, blank for parental.  ``aligned?`` is 1 when the internal line has
    enough inactive FFCs to align to the parental ladder, else 0 (blank for
    parental).  ``First spurious FFC`` shows the most highly ranked
    unmatched/unexplained FFC.  Each ``Unresolved``/``Unmatched FFCs`` example
    is a tuple ``(X, Y, (i,j), [rank], mass_i(X), mass_j(Y), sum)``; rows are
    blank where no such example exists.
    """

    def _count_row(data: dict, index_label: str, blank_if_zero: bool) -> pd.DataFrame:
        row = {
            col: (None if (blank_if_zero and not data.get(col)) else data.get(col))
            for col in col_order
        }
        return pd.DataFrame([row], index=[index_label]).reindex(columns=all_columns)

    def _expand(example_dict: dict, prefix: str) -> pd.DataFrame:
        rows = []
        for k in range(max_ex):
            rows.append({
                col: (example_dict[col][k]
                      if col in example_dict and k < len(example_dict[col]) else None)
                for col in col_order
            })
        return pd.DataFrame(
            rows, index=[f"{prefix} {k + 1}" for k in range(max_ex)]
        ).reindex(columns=all_columns)

    return pd.concat([
        _count_row(line_type_map,  "line type", blank_if_zero=False),
        _count_row(bp_ffc_map,     "#breakpoints/FFCs on a line", blank_if_zero=False),
        _count_row(act_inact_map,  "#active/#inactive TFFCs", blank_if_zero=False),
        _count_row(aligned_map,    "aligned?", blank_if_zero=False),
        _count_row(unm_unexp_map,  "#unmatched/#unexplained", blank_if_zero=False),
        _count_row(spurious_map,   "First spurious FFC", blank_if_zero=False),
        _expand(ex_unexplained, "Unresolved FFCs"),
        _expand(ex_unmatched,   "Unmatched FFCs"),
    ])


# =============================================================================
# 4. COVERAGE TABLE
# =============================================================================

def coverage_table_internal(
    ffc_df: pd.DataFrame,
    lines: Sequence[Line],
    pep,
    parent_mass: float,
    parent_charge: int = 1,
    iso_range: int = 0,
    tol_ppm: float = DEFAULT_TOL_PPM,
    mock_tol_ppm: float = DEFAULT_MOCK_TOL_PPM,
    shift_decimals: int = 3,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    parental_cols: Optional[Sequence[str]] = None,
    parent_shift_tol: float = 0.5,
    consider_secondary: bool = True,
    align_threshold: int = 1,
) -> pd.DataFrame:
    """Bond-by-bond coverage table keyed by ``(i, j, shift)`` with explicit
    internal-ion labels.

    Same row/column shape as ``annotation_greedy.coverage_table_greedy``:
    rows are backbone cleavage sites ``b_k, y_{n-k}``; columns are the
    detected lines.  Cell format is::

        {left}, {right}, [{chargeL}, {chargeR}], ({devL}, {devR}), [{rank}]

    where ``left``/``right`` are the two fragments ordered N->C and any
    internal side is written ``i(a:b)`` per ``_internal_label``.

    parental_cols : sequence of str, optional
        Column keys to treat as the parental (master-ladder) lines.  When
        omitted, columns whose shift is within ``parent_shift_tol`` Da of the
        precursor mass are used (i.e. the near-zero-shift lines).  These drive
        the ``#active/#inactive TFFCs`` overlap row and ``First spurious FFC``.
    consider_secondary : bool, default True
        If True, an FFC that has no adjacent (parental / internal /
        neutral-loss) explanation may be annotated as a *secondary* pair --
        two terminal b/y ions with a gap between them (e.g. ``b9, y6``) --
        which is preferred over the ``(b/y)-shift`` raw-shift fallback.  If
        False, secondary pairs are never formed (prior behaviour).
    tol_ppm : float, default DEFAULT_TOL_PPM (20)
        Relative tolerance for matching observed FFC peaks to theoretical
        fragments.
    mock_tol_ppm : float, default DEFAULT_MOCK_TOL_PPM (100)
        Relative tolerance used *only* when probing a line's shift for a mock
        (virtual) FFC.  Independent of ``tol_ppm`` and deliberately looser: the
        shift is a cluster mean carrying the line-detection error rather than a
        single measured peak, so it deviates further from theory than a real
        FFC.  Loosening this does not affect any observed-FFC annotation.
    """
    n = len(pep.AA_array)
    row_labels = [f"b{k}, y{n - k}" for k in range(1, n)]
    theo_masses = [
        (round(pep.ion_mass(f"b{k}"), 3), round(pep.ion_mass(f"y{n - k}"), 3))
        for k in range(1, n)
    ]
    row_index = {lbl: k for k, lbl in enumerate(row_labels, start=1)}
    has_ranking = ranking_col in ffc_df.columns

    mz_grid, frag_grid, iso_grid, _ = build_fragment_index(pep, iso_range)

    # ── Ordered, de-duplicated column list (+ actual shift per column) ─────
    col_order: List[str] = []
    seen: set = set()
    col_shift: Dict[str, float] = {}
    for ln in lines:
        if ln.member_indices.size == 0:
            continue
        actual_shift = ln.mass - parent_mass
        shift = round(actual_shift, shift_decimals)
        key = _col_key(ln.i, ln.j, shift, shift_decimals)
        if key not in seen:
            seen.add(key)
            col_order.append(key)
            col_shift[key] = actual_shift

    if not col_order:
        return pd.DataFrame(index=row_labels)

    # Parental (master-ladder) columns: explicit set, else near-zero shift.
    if parental_cols is not None:
        parental_set = {c for c in parental_cols if c in seen}
    else:
        parental_set = {c for c in col_order if abs(col_shift[c]) < parent_shift_tol}

    annot = {c: {r: None for r in row_labels} for c in col_order}
    iso_best = {c: {r: float("inf") for r in row_labels} for c in col_order}
    ffc_counts = {c: 0 for c in col_order}       # FFCs assigned to the line
    # Per-FFC placement bond (one entry per member FFC that forms a valid
    # pair); drives the #active/#inactive TFFC counts below.
    placed_bonds = {c: [] for c in col_order}

    # ffc_annotator-style per-line category tallies + example lists.
    cat_counts = {c: Counter() for c in col_order}       # category -> count
    MAX_EX = 3
    ex_unmatched = {c: [] for c in col_order}    # up to 3 unmatched tuples
    ex_unexplained = {c: [] for c in col_order}  # up to 3 unexplained tuples

    # Most highly ranked spurious FFC per line (any column): the best-ranked
    # (lowest Ranking) unmatched/unexplained FFC.  (rank_key, description);
    # rank_key sorts unranked (-1) last.
    spurious_best: Dict[str, Optional[Tuple[float, str]]] = {c: None for c in col_order}

    def _record_spurious(col, mz_a, mz_b, ranking, cand_a, cand_b):
        rank_key = ranking if ranking is not None and ranking >= 0 else float("inf")
        cur = spurious_best[col]
        if cur is not None and cur[0] <= rank_key:
            return
        desc = (
            f"({round(mz_a, 4)}, {round(mz_b, 4)}), rank {ranking}, "
            f"X: {_side_best_label(cand_a, n)}, Y: {_side_best_label(cand_b, n)}"
        )
        spurious_best[col] = (rank_key, desc)

    # ── Populate ──────────────────────────────────────────────────────────
    for ln in lines:
        if ln.member_indices.size == 0:
            continue
        i, j = ln.i, ln.j
        actual_shift = ln.mass - parent_mass
        shift = round(actual_shift, shift_decimals)
        col = _col_key(i, j, shift, shift_decimals)
        # Neutral loss carried by this line (positive = mass lost by a
        # fragment); used to admit terminal ions as ``(b/y) - loss``.
        neutral_loss = max(0.0, -actual_shift)

        for idx in ln.member_indices:
            if idx not in ffc_df.index:
                continue
            ffc_counts[col] += 1

            row = ffc_df.loc[idx]
            mz_a = float(row[col_a])
            mz_b = float(row[col_b])
            ranking = int(row[ranking_col]) if has_ranking else -1

            # Per-member orientation: this line merges (i,j) and (j,i); pick
            # the assignment that reconstructs closest to the line mass, so a
            # mirror-orientation FFC gets the correct per-side charges.
            if abs((j * mz_a + i * mz_b) - ln.mass) < abs((i * mz_a + j * mz_b) - ln.mass):
                iu, ju = j, i
            else:
                iu, ju = i, j
            adj_a = iu * mz_a - (iu - 1) * MASS_H
            adj_b = ju * mz_b - (ju - 1) * MASS_H

            # Raw-shift-free candidates: used for the ffc_annotator-style
            # classification below.
            base_a = _candidates(adj_a, mz_grid, frag_grid, iso_grid, tol_ppm)
            base_b = _candidates(adj_b, mz_grid, frag_grid, iso_grid, tol_ppm)

            # ── Classification (line type / unmatched / unexplained rows) ──
            category = _classify_ffc(base_a, base_b, consider_secondary)
            cat_counts[col][category] += 1
            if category in ("unmatched", "unexplained"):
                _record_spurious(col, mz_a, mz_b, ranking, base_a, base_b)
                tup = _fpr_tuple(
                    _side_best_label(base_a, n, mz_a),
                    _side_best_label(base_b, n, mz_b),
                    iu, ju, ranking, mz_a, mz_b,
                )
                bucket = ex_unmatched if category == "unmatched" else ex_unexplained
                if len(bucket[col]) < MAX_EX:
                    bucket[col].append(tup)

            # ── Main-table placement (unchanged: raw-shift + secondary) ────
            cand_a = list(base_a)
            cand_b = list(base_b)
            if neutral_loss > MIN_LOSS_DA:
                cand_a += _shifted_terminal_candidates(
                    adj_a, neutral_loss, mz_grid, frag_grid, iso_grid, tol_ppm)
                cand_b += _shifted_terminal_candidates(
                    adj_b, neutral_loss, mz_grid, frag_grid, iso_grid, tol_ppm)
            if not cand_a or not cand_b:
                continue                        # nothing to place
            pair = _best_pair(cand_a, cand_b, consider_secondary)
            if pair is None:
                continue

            ca, cb = pair
            # Attach the line charge to each side, then order the pair N->C.
            left, right = sorted(((ca, iu), (cb, ju)), key=lambda t: t[0].frag.s)
            (cL, chL), (cR, chR) = left, right

            bond = cL.frag.e                    # cleavage after residue cL.e
            bond_label = f"b{bond}, y{n - bond}"
            if bond_label not in row_index:
                continue
            # This FFC resolves to a real breakpoint -> record it for the
            # #active/#inactive tally (recorded even if a lower-isotope FFC
            # already won the cell, so every paired FFC is counted once).
            placed_bonds[col].append(bond_label)

            iso_sum = cL.iso + cR.iso
            if iso_sum >= iso_best[col][bond_label]:
                continue  # keep the existing, lower-isotope annotation

            label_L = _placed_label(cL, is_left=True, n=n)
            label_R = _placed_label(cR, is_left=False, n=n)
            annot[col][bond_label] = (
                f"{label_L}, {label_R}, [{chL}, {chR}], "
                f"({round(cL.dev, 3)}, {round(cR.dev, 3)}), [{ranking}]"
            )
            iso_best[col][bond_label] = iso_sum

    # ── Mock (virtual) FFCs : what does each loss line's shift mean? ───────
    # For a line at shift -L, probe the virtual FFC (X, ParentPair - X): if the
    # shift itself is a terminal ion X, its complement completes a b/y pair and
    # names the cleavage the line is missing.  E.g. shift -229.11 on a 15-mer
    # resolves to (b2, y13) -- "this is the line that lost b2".
    # ParentPair = b_k + y_{n-k}, identical for every k.
    #
    # The lost fragment X may itself be MULTIPLY charged: at charge z its
    # reconstruction contribution is L, so its [M+H]+ mass is L-(z-1)*proton.
    # We probe every charge 1..(parent_charge-2) and keep the best-matching
    # mock, so e.g. a -915.5 shift resolves to y9 at charge 2, not just a
    # charge-1 fragment.  The complement is taken at charge 1.
    parent_pair = pep.ion_mass("b1") + pep.ion_mass(f"y{n - 1}")
    mock_charges = range(1, max(2, parent_charge - 1))   # 1 .. parent_charge-2
    mock_cells: Dict[str, Dict[str, str]] = {}
    for c in col_order:
        loss = -col_shift[c]
        if loss <= MIN_LOSS_DA:     # parental / isotope lines: nothing to probe
            continue
        best = None   # (key, cL, cR, chL, chR)
        for z in mock_charges:
            x_mass = loss - (z - 1) * PROTON     # lost fragment [M+H]+ at charge z
            if x_mass <= 0:
                continue
            # Mocks use their own, looser tolerance (see DEFAULT_MOCK_TOL_PPM).
            mock_a = _candidates(x_mass, mz_grid, frag_grid, iso_grid, mock_tol_ppm)
            mock_b = _candidates(parent_pair - x_mass, mz_grid, frag_grid,
                                 iso_grid, mock_tol_ppm)
            # A mock probes the shift as a complementary (adjacent) b/y pair,
            # so secondary geometry never applies here.
            pair = _best_pair(mock_a, mock_b, consider_secondary=False)
            if pair is None:
                continue
            ca, cb = pair            # ca = lost (charge z), cb = complement (charge 1)
            (cL, chL), (cR, chR) = sorted(((ca, z), (cb, 1)),
                                          key=lambda t: t[0].frag.s)
            key = (ca.rank + cb.rank, ca.iso + cb.iso, ca.dev + cb.dev)
            if best is None or key < best[0]:
                best = (key, cL, cR, chL, chR)
        if best is None:
            continue
        _, cL, cR, chL, chR = best
        bond_label = f"b{cL.frag.e}, y{n - cL.frag.e}"
        if bond_label not in row_index:
            continue
        mock_cells.setdefault(c, {})[bond_label] = (
            f"*{_placed_label(cL, True, n)}, {_placed_label(cR, False, n)}, "
            f"[{chL}, {chR}], ({round(cL.dev, 3)}, {round(cR.dev, 3)}), [mock]"
        )

    # ── Assemble DataFrame ────────────────────────────────────────────────
    table = pd.DataFrame(annot, columns=col_order)
    table.index = row_labels
    table = table[table.isna().sum().sort_values().index]     # fullest first

    # Real (observed) coverage per bond, captured BEFORE the mock overlay so a
    # bond supported only by mock FFCs can be flagged with a trailing '*'.
    real_covered = table.notna().any(axis=1)

    # ── Overlay mock FFCs (into empty cells only) ─────────────────────────
    # A mock never displaces observed data; it is tagged '*' + '[mock]' but
    # counts as an annotated cell in the breakpoint / line tallies below.
    for c, cells in mock_cells.items():
        for r, val in cells.items():
            if c in table.columns and pd.isna(table.at[r, c]):
                table.at[r, c] = val

    # Per-bond annotated-cell count -> "Breakpoint Coverage" column value.
    # Per-line annotated-cell count -> "#breakpoints" in the combined row.
    # Both are mock-inclusive (computed before the AA / theoretical-mass
    # columns are attached, so only the line columns are counted).
    bond_count = table.notna().sum(axis=1).astype(int)
    line_bp_count = table.notna().sum(axis=0).astype(int)

    table["theoretical mass"] = theo_masses
    # AA column: one residue per bond row (residues 1..n-1).  The final residue
    # (residue n) has no bond row of its own, so it is parked on the
    # "#breakpoints/FFCs on a line" summary row below.
    table["AA"] = [pep.AA_array[k].acid_name for k in range(len(row_labels))]

    # ── Master-ladder overlap : #active / #inactive TFFCs ─────────────────
    # B = cleavage-site rows covered by any parental (master-ladder) column.
    # Counted PER FFC (per TFFC), so #active + #inactive == #FFC on the line:
    #   #inactive = member FFCs whose breakpoint is already on the master
    #               ladder (redundant with a parental line -> no new mass)
    #   #active   = every other member FFC (a new virtual-b-ion breakpoint, or
    #               an unmatched/unexplained FFC that resolved to no breakpoint)
    # Mocks are virtual and are not member FFCs, so they never enter this count.
    master_rows = {
        r for r in row_labels
        for pc in parental_set if annot[pc][r] is not None
    }
    inactive_map: Dict[str, Optional[int]] = {}
    active_map: Dict[str, Optional[int]] = {}
    for c in col_order:
        if c in parental_set:
            inactive_map[c] = None   # overlap is defined for internal lines only
            active_map[c] = None
            continue
        inactive = sum(1 for b in placed_bonds[c] if b in master_rows)
        inactive_map[c] = inactive
        active_map[c] = ffc_counts[c] - inactive

    # Combined summary-row cells.
    line_type_map = {c: _most_common_category(cat_counts[c]) for c in col_order}
    bp_ffc_map = {
        c: f"{int(line_bp_count.get(c, 0))}/{ffc_counts.get(c, 0)}"
        for c in col_order
    }
    unm_unexp_map = {
        c: f"{cat_counts[c].get('unmatched', 0)}/{cat_counts[c].get('unexplained', 0)}"
        for c in col_order
    }
    act_inact_map = {
        c: (None if c in parental_set
            else f"{active_map[c]}/{inactive_map[c]}")
        for c in col_order
    }
    # An internal line "aligns" (1) to the parental ladder when it has at least
    # ``align_threshold`` inactive FFCs (FFCs whose breakpoint is already on a
    # parental line); otherwise 0.  Blank for parental columns.
    aligned_map = {
        c: (None if c in parental_set
            else (1 if inactive_map[c] >= align_threshold else 0))
        for c in col_order
    }

    spurious_map = {c: (v[1] if v else None) for c, v in spurious_best.items()}
    summary = _build_summary_rows(
        table.columns, col_order,
        line_type_map, bp_ffc_map, unm_unexp_map,
        act_inact_map, aligned_map, spurious_map,
        ex_unexplained, ex_unmatched, MAX_EX,
    )
    table = pd.concat([table, summary])

    # Park the final residue (residue n) on the first summary row (directly
    # below the last bond row) so the AA column has no gap.
    first_summary_row = "line type"
    if first_summary_row in table.index:
        table.at[first_summary_row, "AA"] = pep.AA_array[-1].acid_name

    # ── Breakpoint Coverage column ────────────────────────────────────────
    # Per-bond count of annotated cells (mock-inclusive); "0" if the bond has
    # no support.  A bond supported *only* by mock FFCs gets a trailing '*'
    # (e.g. "1*").  The header percentage is the fraction of the n-1 backbone
    # cleavage sites with a non-zero count.
    n_covered = int((bond_count > 0).sum())
    cov_pct = n_covered / len(row_labels) * 100
    col_name = f"Breakpoint Coverage={cov_pct:.1f}%"
    bp_marks: Dict[str, str] = {}
    for r in row_labels:
        cnt = int(bond_count[r])
        if cnt == 0:
            bp_marks[r] = "0"
        elif bool(real_covered.get(r, False)):
            bp_marks[r] = str(cnt)
        else:
            bp_marks[r] = f"{cnt}*"       # mock-only support
    table[col_name] = pd.Series(bp_marks)

    lead = ["AA", col_name]
    ordered = lead + [c for c in table.columns if c not in lead]
    return table[ordered]


# =============================================================================
# 5. PIPELINE RUNNER
# =============================================================================

# =============================================================================
# 4c. ORIENTATION-AGNOSTIC LINE DETECTION (charge-independent)
# =============================================================================
# The detector ignores the deconvolution per-side charge and recovers each
# FFC's charge from the line it lands on.  A line (i, j) and its mirror (j, i)
# at the same shift are the SAME physical line, so cluster membership is
# counted over the canonical charge (min, max), pooling both orientations; each
# member keeps its own orientation, resolved per-FFC in the coverage table.

def _sort_and_split_oa(values, delta, delta_unit="da"):
    """Sort-and-Split gap clustering with a Da or ppm gap threshold.

    ``delta_unit='da'``  -> a fixed absolute gap of ``delta`` Da.
    ``delta_unit='ppm'`` -> a relative gap of ``delta`` ppm, evaluated against
    the (lower) reconstructed mass of each gap.  Because the pooled masses sit
    near the precursor mass, this scales the clustering tolerance with mass the
    same way the ppm fragment matching does (e.g. 0.02 Da at 4000 Da ~= 5 ppm).
    """
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    v_sorted = values[order]
    if n == 1:
        return [order]
    gaps = np.diff(v_sorted)
    if delta_unit == "ppm":
        thr = np.abs(v_sorted[:-1]) * delta * 1e-6
    else:
        thr = delta
    split_after = np.where(gaps > thr)[0] + 1
    groups_sorted = np.split(np.arange(n), split_after)
    return [order[g] for g in groups_sorted]


def _most_informative_line_oa(ffc_map, parent_charge, delta, min_ffc_number,
                              col_a, col_b, charge_filter=None, delta_unit="da"):
    """Orientation-agnostic MostInformativeLine.

    Pools ``i*x + j*y`` over all admissible (i, j), clusters by mass, and for
    the winning cluster counts *distinct* FFCs on the dominant *canonical*
    charge ``(min(i,j), max(i,j))`` -- so (i, j) and (j, i) count as one line.
    Returns a ``Line`` with the canonical (i, j) and unique member indices, or
    None if no cluster reaches ``min_ffc_number`` distinct FFCs.
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
    m_blocks, i_blocks, j_blocks, p_blocks = [], [], [], []
    base_pos = np.arange(n)
    for (i, j) in pairs:
        m_blocks.append(i * x + j * y)
        i_blocks.append(np.full(n, i, dtype=int))
        j_blocks.append(np.full(n, j, dtype=int))
        p_blocks.append(base_pos)
    masses = np.concatenate(m_blocks)
    i_arr = np.concatenate(i_blocks)
    j_arr = np.concatenate(j_blocks)
    pos_arr = np.concatenate(p_blocks)
    can_lo = np.minimum(i_arr, j_arr)          # canonical (min, max) charge
    can_hi = np.maximum(i_arr, j_arr)

    clusters = _sort_and_split_oa(masses, delta, delta_unit)
    if not clusters:
        return None

    best = None   # ((distinct, -spread), cluster, (clo, chi), on_line)
    for cl in clusters:
        clo = can_lo[cl]
        chi = can_hi[cl]
        cpos = pos_arr[cl]
        counts = Counter(zip(clo.tolist(), chi.tolist()))
        dom = max(counts.keys(),
                  key=lambda k: (counts[k], k[0] + k[1], -k[0]))
        on = (clo == dom[0]) & (chi == dom[1])
        distinct = int(np.unique(cpos[on]).size)
        spread = (float(masses[cl][on].max() - masses[cl][on].min())
                  if on.any() else 0.0)
        key = (distinct, -spread)
        if best is None or key > best[0]:
            best = (key, cl, dom, on)

    (distinct, _), cl, dom, on = best
    if distinct < min_ffc_number:
        return None
    member_pos = np.unique(pos_arr[cl][on])
    line_mass = float(masses[cl][on].mean())
    return Line(i=int(dom[0]), j=int(dom[1]), mass=line_mass,
                cluster_size=int(cl.size), n_on_line=distinct,
                member_indices=row_idx[member_pos])


def _remove_lines_oa(ffc_map, lines, tol, col_a, col_b, delta_unit="da"):
    """Drop every FFC lying on any of ``lines`` in *either* orientation
    (|i*x+j*y - mass| <= tol OR |j*x+i*y - mass| <= tol), plus the recorded
    member indices.  ``tol`` is Da or ppm per ``delta_unit`` (ppm is taken
    against each line's own mass).  Returns the residual map."""
    x = ffc_map[col_a].to_numpy(dtype=float)
    y = ffc_map[col_b].to_numpy(dtype=float)
    idx = ffc_map.index.to_numpy()
    drop: set = set()
    for ln in lines:
        t = abs(ln.mass) * tol * 1e-6 if delta_unit == "ppm" else tol
        v1 = ln.i * x + ln.j * y
        v2 = ln.j * x + ln.i * y
        mask = (np.abs(v1 - ln.mass) <= t) | (np.abs(v2 - ln.mass) <= t)
        drop.update(idx[mask].tolist())
        if ln.member_indices.size:
            drop.update(ln.member_indices.tolist())
    return ffc_map[~ffc_map.index.isin(drop)]


def detect_lines_oa(ffc_df, parent_charge, delta, min_ffc_number,
                    col_a="m/z A", col_b="m/z B", line_tol=None,
                    delta_unit="da", max_lines=10000):
    """Charge-independent, orientation-agnostic GreedyLines.

    ``delta`` (and the FFC-on-line ``line_tol``, which defaults to it) are Da
    or ppm depending on ``delta_unit`` ("da" default, or "ppm").

    Returns ``(line_list, master)`` with the same meaning as
    ``greedy_line.greedy_lines(return_master=True)``: ``master`` are the
    parental (charge-sum) lines cleared first together with their isotopic /
    neutral satellites, ``line_list`` the residual internal lines.
    """
    if line_tol is None:
        line_tol = delta

    master = []
    work = ffc_df.copy()
    for _ in range(max_lines):
        ln = _most_informative_line_oa(
            work, parent_charge, delta, min_ffc_number, col_a, col_b,
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
        ln = _most_informative_line_oa(
            work, parent_charge, delta, min_ffc_number, col_a, col_b,
            delta_unit=delta_unit)
        if ln is None:
            break
        line_list.append(ln)
        nxt = _remove_lines_oa(work, [ln], line_tol, col_a, col_b, delta_unit)
        if len(nxt) == len(work):
            break
        work = nxt

    return line_list, master


# =============================================================================
# 5. PIPELINE RUNNER
# =============================================================================

def run_internal_annotation(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    delta: float = DEFAULT_DELTA,
    delta_unit: str = "da",
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    iso_range: int = 0,
    tol_ppm: float = DEFAULT_TOL_PPM,
    mock_tol_ppm: float = DEFAULT_MOCK_TOL_PPM,
    shift_decimals: int = 3,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    consider_secondary: bool = True,
    align_threshold: int = 1,
):
    """Run orientation-agnostic ``detect_lines_oa`` + ``coverage_table_internal``.

    The deconvolution's per-side charge is ignored; detection recovers each
    FFC's charge and merges (i, j)/(j, i) mirror lines.  ``delta_unit`` selects
    whether the detection ``delta`` is a fixed Da gap (default, 0.02 Da) or a
    relative ppm gap.  Returns ``(coverage_table, all_lines)``.
    """
    if delta_unit not in ("da", "ppm"):
        raise ValueError(f"delta_unit must be 'da' or 'ppm', got {delta_unit!r}")
    line_list, master = detect_lines_oa(
        ffc_df, parent_charge, delta, min_ffc_number,
        col_a=col_a, col_b=col_b, line_tol=line_tol, delta_unit=delta_unit,
    )
    all_lines = master + line_list

    # Master ladder = the near-zero-shift parental lines.
    parental_cols = {
        _col_key(ln.i, ln.j, round(ln.mass - parent_mass, shift_decimals),
                 shift_decimals)
        for ln in master
        if abs(ln.mass - parent_mass) < 0.5
    }

    table = coverage_table_internal(
        ffc_df,
        all_lines,
        pep,
        parent_mass,
        parent_charge=parent_charge,
        iso_range=iso_range,
        tol_ppm=tol_ppm,
        mock_tol_ppm=mock_tol_ppm,
        shift_decimals=shift_decimals,
        col_a=col_a,
        col_b=col_b,
        ranking_col=ranking_col,
        parental_cols=parental_cols,
        consider_secondary=consider_secondary,
        align_threshold=align_threshold,
    )
    return table, all_lines


# =============================================================================
# 6. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from line_finding import prepare_ffc_data          # noqa: E402
    from merge_ffcs import merge_duplicate_ffcs         # noqa: E402
    from merge_ffcs import merge_duplicate_ffcs_charge

    # ── Configuration (HAD, charge 4 -- matches 4+.xlsx) ───────────────────
    #DATA_PATH = os.path.join(_DEISO, "result", "HAD_merged.tsv")
    DATA_PATH = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted/merged/KWK_37_Z5_E28_2_XIAOWEN_merged.tsv")
    
    #PEP_SEQ = "VEADIAGHGQEVLIR"
    #PEP_SEQ = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    #PEP_SEQ = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    #PEP_SEQ = "YLEFISDAIIHVLHSK"
    PEP_SEQ="KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
    
    CHARGE = 5
    TOP_N = 500
    DELTA = 0.02
    DELTA_UNIT = "da"          # "da" (0.02 Da gap) or "ppm" (relative gap)
    MIN_FFC = 3
    ISO_RANGE = 1
    TOL_PPM = 20.0
    MOCK_TOL_PPM = 100.0        # looser, independent tolerance for mock FFCs
    OUTPUT_SHEET = "KWK3+_line_based_oa"
    OUTPUT_EXCEL = os.path.join(os.path.dirname(__file__), f"{OUTPUT_SHEET}.xlsx")
    

    #pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20='NH3')
    
    parent_mass = pep.pep_mass
    print(f"Peptide length: {len(pep.AA_array)}   Precursor mass: {parent_mass:.4f}")

    ffc_df = pd.read_csv(DATA_PATH, sep="\t")
    ffc_df = merge_duplicate_ffcs_charge(ffc_df, symmetric = True, reset_ranking = True)
    print(f"FFC rows after filter/merge: {len(ffc_df)}")
    print(ffc_df.head())
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)

    cov_table, all_lines = run_internal_annotation(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=parent_mass,
        delta=DELTA,
        delta_unit=DELTA_UNIT,
        min_ffc_number=MIN_FFC,
        iso_range=ISO_RANGE,
        tol_ppm=TOL_PPM,
        mock_tol_ppm=MOCK_TOL_PPM,
    )

    print(f"\nDetected lines: {len(all_lines)}")
    data_rows = cov_table.iloc[: len(pep.AA_array) - 1]
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(data_rows.to_string())

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        cov_table.to_excel(writer, sheet_name=OUTPUT_SHEET,
                           index_label=f"#FFC={TOP_N}")
        ws = writer.sheets[OUTPUT_SHEET]
        for col_cells in ws.columns:
            ws.column_dimensions[col_cells[0].column_letter].width = 32
    print(f"\nSaved to {OUTPUT_EXCEL} [{OUTPUT_SHEET}]")
