"""
Line-Based Annotation -- Charge-Independent, Orientation-Agnostic Detection (v2)
=================================================================================
Variant of ``line_internal_annotation_oa.py`` (itself built on
``line_internal_annotation_ppm.py``) that recovers every FFC's charge from
line detection instead of trusting the deconvolution's per-side charge, and
treats a line ``(i, j)`` and its mirror ``(j, i)`` as one physical line (the
m/z-A / m/z-B ordering from deconvolution is arbitrary).

New in this version:
  * ``Master TFFC-set (coverage=XX%)`` column, placed right after
    ``Breakpoint Coverage=XX%``.  Per bond row it is a binary 0/1: 1 if that
    breakpoint is represented by a *master TFFC* -- an FFC that (a) sits on a
    parental (master-ladder) line and (b) classifies as a clean ``parental``
    FFC with no modification suffix (no ``*`` neutral loss, no ``~`` isotope
    tag).  Special parental-line members that are actually a neutral-loss or
    isotopic variant (``parental*``, ``parental*~``, ``parental~``, ...) do
    NOT count.  The header percentage is the fraction of the n-1 backbone
    bonds with a master TFFC.
  * ``#incorrect TFFCs`` summary row, placed right after ``line
    type``.  Meaningful only on columns whose ``line type`` has base category
    ``parental`` (bare ``parental`` or any of its ``*``/``~`` variants); it
    counts how many of that column's member FFCs are NOT a clean, unmodified
    ``parental`` match.  Blank on every other column.
  * ``loose_mock`` -- a charge-free mock probe (see the mock block in
    ``coverage_table_internal``).  The original probe sweeps the lost
    fragment's charge and so can only find a *charged* lost fragment; a line
    whose charges already sum to the precursor charge lost a NEUTRAL residue
    run and was silently skipped.
  * ``#reconstructions/correct reconstructions`` summary row (opt-in via
    ``reconstruct=True``), placed right after ``#incorrect TFFCs``.
    The breakpoints carried by the parental (master-ladder) lines are handed
    to the longest-path de novo sequencer in
    ``../reconstruct/path_finding.py``; the cell reports how many tied
    reconstructions it returned and how many of those
    ``path_finding.is_reconstruction_correct`` accepts -- e.g. ``3/1`` means
    three reconstructions, one of them correct, ``2/0`` means two with none
    correct.  It describes the whole peptide rather than any single line, so
    it occupies ONE cell (on the first parental column), not a full row.
    ``considering_spurious=True`` additionally feeds the unmatched /
    unexplained FFCs found on those parental lines, converted to
    charge-independent neutral masses.  See ``_reconstruction_counts`` for the
    mandatory water-basis conversion (path_finding assumes a water
    C-terminus; ``end_h20='NH3'`` peptides must be converted or every
    reconstruction comes back incorrect).

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

#: Smallest residue mass (glycine).  A line whose alignment shift is below this
#: cannot be a run of residues, so it names no cleavage and gets no mock.
MIN_RESIDUE_MASS = 57.02146

#: Neutral water mass.  ``reconstruct/path_finding.py`` hard-codes a water
#: C-terminus (``SOURCE_NODE = (0.0, 18.01056)``), so every mass handed to it
#: must be expressed on a *water* basis -- see ``_reconstruction_counts``.
WATER = 18.01056

#: Where ``reconstruct/path_finding.py`` and its precomputed compound-mass
#: table (``edges.json``, built by ``edges.py``) live.  Imported lazily, only
#: when ``reconstruct=True``, so this module keeps working without them.
_RECONSTRUCT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reconstruct")
)
DEFAULT_EDGES_PATH = os.path.join(_RECONSTRUCT_DIR, "edges.json")

#: Diagnostic: how many banking rounds the last ``_tffc_analysis`` needed before
#: the master TFFC-set stopped growing (see section 4c).  Read it after a run to
#: confirm the iteration converged rather than hitting ``max_rounds``.
LAST_CONVERGENCE_ROUNDS: int = 0

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
# 4b. DE NOVO RECONSTRUCTION FROM THE PARENTAL (MASTER) LADDER
# =============================================================================
# Feeds the breakpoints carried by the *parental* lines into the longest-path
# de novo sequencer in ``../reconstruct/path_finding.py`` and reports how many
# tied reconstructions it finds and how many of those are actually correct
# (``path_finding.is_reconstruction_correct``, checked against the known
# sequence).
#
# The water-basis conversion
# --------------------------
# path_finding hard-codes a WATER C-terminus: ``SOURCE_NODE = (0.0, 18.01056)``
# and both ``format_gapped_reconstruction`` and ``is_reconstruction_correct``
# mirror the y-axis through ``total_mass - WATER``.  Peptides built with
# ``end_h20='NH3'`` (a C-terminal amide) carry NH3 instead, so their neutral
# mass and every y-ion sit 0.984 Da (= H2O - NH3) low.  Handing those numbers
# over unconverted silently breaks the search -- verified on KWK (NH3): the
# unconverted spectrum still yields a longest path, but 0 of them are correct,
# while the water-basis spectrum gives 1/1.
#
# We sidestep the terminus entirely for the *theoretical* side by deriving both
# axes from residue prefix sums:
#     b-side value  = prefix[k]                       (residues 1..k)
#     y-side value  = total_water - prefix[k]         (residues k+1..n + H2O)
#     total_water   = prefix[n] + H2O
# Observed *spurious* masses (used only when ``considering_spurious``) can't be
# derived that way -- they are measured, and we don't know which side of the
# pair is b-like and which is y-like -- so each is offered under BOTH readings:
# as-is (b-like) and shifted by (H2O - terminus) (y-like, corrected onto the
# water basis).  For a water-terminated peptide the correction is 0 and the two
# readings collapse into one, matching path_finding's own demo.


def _reconstruction_counts(
    pep,
    break_bonds: Sequence[int],
    spurious_masses: Sequence[Tuple[float, float]],
    considering_spurious: bool = False,
    edges_path: Optional[str] = None,
    threshold: float = 0.001,
    max_paths: int = 1000,
) -> str:
    """Run the de novo longest-path search and return ``"{n_paths}/{n_correct}"``.

    ``break_bonds`` are 1-based cleavage indices ``k`` (bond ``b{k}, y{n-k}``)
    supported by parental-line FFCs; each contributes its b- and y-axis mass.
    ``spurious_masses`` are ``(mass_A, mass_B)`` charge-independent neutral
    masses of the unmatched/unexplained FFCs on the parental lines -- folded in
    only when ``considering_spurious`` is True.

    Returns e.g. ``"3/1"`` (three tied longest reconstructions, one correct) or
    ``"2/0"`` (two, none correct).  ``"0/0"`` when no path reaches the sink.
    """
    if edges_path is None:
        edges_path = DEFAULT_EDGES_PATH
    if _RECONSTRUCT_DIR not in sys.path:
        sys.path.append(_RECONSTRUCT_DIR)
    import path_finding as pf                     # noqa: E402  (lazy)
    from edges import load_edges                  # noqa: E402  (lazy)

    residues = [a.get_mass() for a in pep.AA_array]
    n = len(residues)
    prefix = [0.0] * (n + 1)
    for k in range(1, n + 1):
        prefix[k] = prefix[k - 1] + residues[k - 1]
    total_water = prefix[n] + WATER

    # (H2O - actual terminus): 0 for a normal peptide, +0.984 for a C-amide.
    term_mass = (WATER if pep.end_h20 is True
                 else AA.element_masses[pep.end_h20])
    y_correction = WATER - term_mass

    values = {0.0, WATER, total_water}
    for k in break_bonds:
        if 0 < k < n:
            values.add(prefix[k])                     # b-side (N-terminal)
            values.add(total_water - prefix[k])       # y-side, water basis
    if considering_spurious:
        for m_a, m_b in spurious_masses:
            for v in (m_a, m_b):
                if v <= 0:
                    continue
                values.add(v)                         # read as a b-side mass
                if y_correction:
                    values.add(v + y_correction)      # read as a y-side mass

    # path_finding's convention: build the grid from the lower half of the
    # spectrum only (the upper half is the mirror image of it).
    spec = sorted(values)
    mid = len(spec) // 2 + 1
    lower_half = sorted(spec[:mid])

    nodes = pf.construct_graph_nodes(lower_half)
    graph = pf.build_adjacency_graph(
        nodes=nodes,
        allowed_masses=load_edges(edges_path),
        total_peptide_mass=total_water,
        threshold=threshold,
    )
    max_len, all_paths = pf.find_all_longest_paths(
        graph, start_node=pf.SOURCE_NODE, max_paths=max_paths,
    )
    if max_len == -1 or not all_paths:
        return "0/0"

    pep_seq = "".join(a.acid_name for a in pep.AA_array)
    n_correct = sum(
        1 for _, node_seq in all_paths
        if pf.is_reconstruction_correct(node_seq[:-1], total_water, pep_seq)
    )
    return f"{len(all_paths)}/{n_correct}"


# =============================================================================
# 4d. RECONSTRUCTION FROM THE WHOLE MASTER TFFC-SET
# =============================================================================

def _draw_reconstruction_graph(spectrum, hero, candidates, incorrect_coords,
                               ion_labels, total, save_path, title=""):
    """Render the 2D map with the correct path highlighted.

    Thin wrapper over ``../reconstruct/path_graph_revised.visualize_all_paths``
    -- imported lazily, and with a non-interactive matplotlib backend forced,
    so nothing here needs a display and the module keeps working when
    matplotlib is absent and ``graph=False``.

    The *hero* is always the **correct** reconstruction (not merely the first
    one found); every other tied path is drawn as a candidate.  Incorrect
    master TFFCs are passed as ``spurious_masses`` so they render in red, which
    makes it obvious at a glance where a wrong coordinate sits relative to the
    true ladder.
    """
    if _RECONSTRUCT_DIR not in sys.path:
        sys.path.append(_RECONSTRUCT_DIR)
    import matplotlib                                  # noqa: E402  (lazy)
    matplotlib.use("Agg", force=False)
    import path_graph_revised                          # noqa: E402  (lazy)
    import path_finding as pf                          # noqa: E402  (lazy)

    aa_map = {v: k for k, v in pf.amino_acid_masses_merge.items()}
    # A coordinate and its complement are the same breakpoint; flag both so the
    # bad one is visible whichever half of the map it lands on.
    spurious = sorted({c for c in incorrect_coords} |
                      {total - c for c in incorrect_coords})
    path_graph_revised.visualize_all_paths(
        spectrum,
        spurious_masses=spurious,
        candidate_paths=candidates or None,
        correct_path=hero,
        aa_map=aa_map,
        ion_labels=ion_labels,
        show_aa_mass=False,
        pep_mass=total,
        title=title,
        save_path=save_path,
    )


def _master_set_reconstruction(
    pep,
    bonds: Sequence[int],
    incorrect_coords: Sequence[float],
    edges_path: Optional[str] = None,
    threshold: float = 0.02,
    max_paths: int = 1000,
    graph: bool = False,
    graph_path: Optional[str] = None,
    graph_title: str = "",
):
    """De novo reconstruction from the master TFFC-set (parental + internal).

    ``bonds`` are the backbone cleavages the master set covers; each is fed at
    its **exact** ``prefix[k]`` so fragment-mass error cannot blur the search --
    the same simplification ``_reconstruction_counts`` uses for the parental
    baseline.  ``incorrect_coords`` are master TFFCs that matched no bond at
    all; they go in at their **measured** folded value, because they are really
    in the set and the reconstruction has to cope with them.

    Rather than converting everything onto a water basis (what
    ``_reconstruction_counts`` does), this rebases ``path_finding`` itself:
    ``SOURCE_NODE`` is set to the peptide's real terminus for the duration of
    the call.  ``construct_graph_nodes``, ``format_gapped_reconstruction`` and
    ``is_reconstruction_correct`` all read that global when called, so measured
    coordinates need no per-side b/y decision.  The global is restored
    afterwards.

    Returns ``(n_paths, any_correct, rows)``; ``rows`` is one dict per tied
    reconstruction with its gapped string and correctness.
    """
    if edges_path is None:
        edges_path = DEFAULT_EDGES_PATH
    if _RECONSTRUCT_DIR not in sys.path:
        sys.path.append(_RECONSTRUCT_DIR)
    import path_finding as pf                          # noqa: E402  (lazy)
    from edges import load_edges                       # noqa: E402  (lazy)

    residues = [a.get_mass() for a in pep.AA_array]
    n = len(residues)
    prefix = [0.0] * (n + 1)
    for k in range(1, n + 1):
        prefix[k] = prefix[k - 1] + residues[k - 1]
    term = WATER if pep.end_h20 is True else AA.element_masses[pep.end_h20]
    total = prefix[n] + term

    values = {0.0, term, total}
    for k in bonds:
        if 0 < k < n:
            values.add(prefix[k])                 # exact N-terminal coordinate
            values.add(total - prefix[k])         # its complement
    for c in incorrect_coords:
        if 0 < c < total:
            values.add(c)
            values.add(total - c)

    spec = sorted(values)
    mid = len(spec) // 2 + 1
    lower_half = sorted(spec[:mid])

    pep_seq = "".join(a.acid_name for a in pep.AA_array)
    saved_source = pf.SOURCE_NODE
    hero, others = None, []
    try:
        pf.SOURCE_NODE = (0.0, term)              # rebase for this call
        dag = pf.build_adjacency_graph(
            nodes=pf.construct_graph_nodes(lower_half),
            allowed_masses=load_edges(edges_path),
            total_peptide_mass=total,
            threshold=threshold,
        )
        max_len, all_paths = pf.find_all_longest_paths(
            dag, start_node=pf.SOURCE_NODE, max_paths=max_paths)
        if max_len == -1 or not all_paths:
            return 0, False, []
        rows = []
        for idx, (_masses, node_seq) in enumerate(all_paths, start=1):
            stripped = node_seq[:-1]              # drop the Sink sentinel
            ok = pf.is_reconstruction_correct(stripped, total, pep_seq)
            rows.append({
                "#": idx,
                "reconstruction": pf.format_gapped_reconstruction(stripped, total),
                "correct": "correct" if ok else "",
            })
            # The hero is the CORRECT path, not simply the first one found.
            if ok and hero is None:
                hero = stripped
            else:
                others.append(stripped)
    finally:
        pf.SOURCE_NODE = saved_source

    # ── Optional 2D map (one extra file; the table is untouched) ──────────
    # Drawn only when a correct reconstruction exists -- the figure's whole
    # point is to show where the correct path runs relative to the candidates.
    if graph and hero is not None:
        ion_labels = {}
        for k in bonds:
            if 0 < k < n:
                ion_labels[prefix[k]] = f"b{k}"
                ion_labels[total - prefix[k]] = f"y{n - k}"
        _draw_reconstruction_graph(
            lower_half, hero, others, list(incorrect_coords), ion_labels,
            total, graph_path or "master_set_reconstruction.png", graph_title)

    return len(rows), any(r["correct"] for r in rows), rows


# =============================================================================
# 4c. TFFC PRODUCTIVITY -- non-productive / unambiguous / ambiguous
# =============================================================================
# Per-line breakdown of what each line's *active* TFFCs would contribute to a de
# novo reconstruction, using the extension algorithm of
# ``../reconstruct/master_reconstruction.py`` (see ``reconstruct.md`` section 6).
# We import and reuse that module rather than reimplementing it; the import is
# lazy, which also avoids a circular import (master_reconstruction imports THIS
# module at load time).
#
# Definitions (``reconstruct.md`` section 6)
# ------------------------------------------
# A TFFC is folded to a single breakpoint coordinate, and has two *images* --
# "trust Y" = fold(Y - proton) and "trust X" = fold(X - proton) -- because one
# TFFC alone cannot say which of its sides is the N-terminal fragment.  Each
# image is tested for *goodness*: does adding it strictly increase the weight of
# the heaviest path through the 2D-DAG?
#
#   non-productive  neither image is good -- it buys no new sequence
#   unambiguous     exactly one image is good -- that placement is forced
#   ambiguous       both images are good -- the TFFC is real but its side is
#                   undecidable; taking it costs one false breakpoint
#
# Relationship to the #active/#inactive row
# -----------------------------------------
# The three counts partition this line's **active** TFFCs, so
# ``#non-productive + #unambiguous + #ambiguous == #active`` on every line.
# *Inactive* is NOT recomputed here by coordinate coincidence (what
# master_reconstruction does); we reuse this module's existing verdict -- an FFC
# whose breakpoint already sits on the parental ladder -- so the two rows stay
# consistent and the parental lines remain the reference throughout.
#
# The reference is the PARENTAL LADDER, frozen
# -------------------------------------------
# Both the inactive test and the goodness test are made against the parental
# lines alone.  Nothing this classification looks at is ever added back into the
# reference, which is what makes the row well-defined:
#
#   * **order-independent** -- the verdict for a TFFC depends only on itself and
#     on the parental set, never on which lines or FFCs were visited first;
#   * **line-independent** -- a spurious line cannot change another line's
#     counts, so every line can be reported (including `unmatched` /
#     `unexplained` ones) and the decision to *use* a line is taken separately
#     from its inactive count (the `aligned?` row / ``align_threshold``);
#   * **consistent with `#active/#inactive`**, which already uses the parental
#     ladder as its reference.
#
# master_reconstruction extends its master set as it goes (that is its job --
# it is building a reconstruction, not describing lines), and its
# ``--inactive_vs parental`` option exists precisely because the growing
# reference makes an earlier decision silence a later TFFC (reconstruct.md
# section 6b).  This row takes that one step further and freezes the reference
# for the goodness test as well, so the numbers describe each line on its own
# merits.
#
# No mock TFFCs are counted here.  A mock is virtual, not a member FFC, so
# counting it would break the ``== #active`` invariant -- a line can therefore
# read 0/0/1 here and still be productive in master_reconstruction via its mock.
#
# Tolerances are ppm, applied per coordinate (see ``_build_dag_ppm``).  A fixed
# Da window is at once too loose for light coordinates and too tight for heavy
# ones: measured on KWK's junk `(1,4) -214.459` line, a fixed window already
# manufactures productive TFFCs at 0.04 Da (1/2/0) and 0.05 Da (0/3/0), whereas
# per-coordinate ppm holds it at 3/0/0 across 10-30 ppm.

def _build_dag_ppm(spectrum, sorted_allowed, total_mass, source, tol_ppm, sink):
    """``master_reconstruction.build_dag`` with a **relative** edge tolerance.

    Identical edge semantics (Down / Right / Sink, every gap must be an allowed
    compound residue mass) except that the mass window is ``ref * tol_ppm``
    rather than a fixed number of Da.  ``ref`` is the *larger* endpoint of the
    gap -- a gap's uncertainty is dominated by the heavier of the two
    coordinates it spans -- and the whole precursor mass for the sink edge.

    Why relative: a breakpoint coordinate is a measured fragment mass, and
    deconvolved fragments carry a roughly relative error.  One fixed Da window
    is therefore simultaneously too loose for the light coordinates (where it
    lets junk lines score) and too tight for the heavy ones.
    """
    coords = sorted(set(list(spectrum) + [source[0], source[1]]))
    nodes = [(x1, x2) for x1 in coords for x2 in coords
             if x1 != x2 or (x1, x2) == source]
    graph = {n: [] for n in nodes}
    node_set = set(nodes)
    if not sorted_allowed:
        return graph

    min_edge, max_edge = sorted_allowed[0], sorted_allowed[-1]
    max_coord = coords[-1]
    max_tol = abs(max_coord) * tol_ppm * 1e-6      # widest window in play

    def allowed(gap: float, ref: float) -> Optional[float]:
        tol = abs(ref) * tol_ppm * 1e-6
        if gap < min_edge - tol or gap > max_edge + tol:
            return None
        idx = bisect_left(sorted_allowed, gap)
        for k in (idx, idx - 1):
            if 0 <= k < len(sorted_allowed) and abs(sorted_allowed[k] - gap) <= tol:
                return sorted_allowed[k]
        return None

    def coords_above(val: float):
        return coords[bisect_left(coords, val + min_edge - max_tol):]

    for node in nodes:
        x1, x2 = node
        out = graph[node]
        max_current = max(x1, x2)

        remainder = total_mass - x1 - x2
        if remainder > 0:
            m = allowed(remainder, total_mass)
            if m is not None:
                out.append((sink, m, "sink"))

        for t1 in coords_above(x1):
            gap = t1 - x1
            if gap > max_coord - x1 + max_tol:
                break
            m = allowed(gap, t1)
            if m is not None:
                tgt = (t1, x2)
                if tgt in node_set and max_current < max(t1, x2):
                    out.append((tgt, m, "down"))

        for t2 in coords_above(x2):
            gap = t2 - x2
            if gap > max_coord - x2 + max_tol:
                break
            m = allowed(gap, t2)
            if m is not None:
                tgt = (x1, t2)
                if tgt in node_set and max_current < max(x1, t2):
                    out.append((tgt, m, "right"))

    return graph


def _tffc_analysis(
    pep,
    col_ffcs: Dict[str, list],
    col_order: Sequence[str],
    parental_set,
    active_of,
    inactive_of,
    master_align_threshold: int = 1,
    edges_path: Optional[str] = None,
    tol_ppm: float = 30.0,
    coord_tol_ppm: float = 30.0,
    mock_tol_ppm: float = DEFAULT_MOCK_TOL_PPM,
    converge: bool = True,
    max_rounds: int = 20,
    accept_ambiguity: bool = False,
):
    """Classify each line's active TFFCs and extend the master TFFC-set.

    ``col_ffcs[c]`` is a list of ``[x, y, bond_label, category]`` -- one entry
    per member FFC of column ``c``, where ``x``/``y`` are the singly-protonated
    per-side masses.  ``active_of(col, entry)`` says whether that FFC counts as
    active (its breakpoint is not already on the parental ladder);
    ``inactive_of(col)`` gives the column's inactive count, which gates whether
    the line may join the master set.

    Every verdict is taken against the frozen parental set, so the result does
    not depend on the order columns or FFCs are visited.

    ``converge`` runs the iterative rounds on top of that: bank the unambiguous
    TFFCs, re-derive aligned/active/unambiguous against the grown set, repeat
    until the master TFFC-set stops increasing (at most ``max_rounds``).

    Returns ``(cells, cells_final, extra_bonds, eligible)``:
      * ``cells``       -- round-1 (frozen parental)
                           ``{column: "{non_productive}/{unambiguous}/{ambiguous}"}``
                           for non-parental columns, ``None`` for parental ones;
      * ``cells_final`` -- the same after convergence, or ``None`` when
                           ``converge=False``;
      * ``extra_bonds`` -- backbone bond indices contributed to the master
                           TFFC-set by the eligible internal / neutral-loss
                           lines (mocks included);
      * ``eligible``    -- ``{column: 'internal'|'loss'}`` for the lines that
                           passed both gates, for reporting.
    """
    if edges_path is None:
        edges_path = DEFAULT_EDGES_PATH
    if _RECONSTRUCT_DIR not in sys.path:
        sys.path.append(_RECONSTRUCT_DIR)
    import master_reconstruction as mr               # noqa: E402  (lazy)
    from edges import load_edges                     # noqa: E402  (lazy)

    term = mr.terminus_mass(pep)
    total = mr.residue_prefixes(pep)[-1] + term
    source = (0.0, term)
    allowed = sorted(load_edges(edges_path))

    class _MasterSetPpm(mr.MasterSet):
        """``MasterSet`` whose "same TFFC" test is relative, not a fixed Da.

        Two coordinates coincide when they agree to ``coord_tol_ppm``, measured
        against the larger of the two (so the window never shrinks below what
        either coordinate's own error allows).  ``self.tol`` keeps a nominal
        absolute value because the inherited ``add``/``coords`` use it only to
        reject coordinates sitting on a terminus.
        """

        def find(self, r: float) -> Optional["mr.MasterEntry"]:
            c = mr.fold(r, self.total)
            for e in self.entries:
                if abs(e.coord - c) <= max(abs(c), abs(e.coord)) * coord_tol_ppm * 1e-6:
                    return e
            return None

        def snapshot(self) -> "_MasterSetPpm":
            copy = _MasterSetPpm(self.total, self.tol)
            copy.entries = [mr.MasterEntry(e.coord, e.weight, list(e.origins))
                            for e in self.entries]
            return copy

    # ── Bond coordinates, and SNAPPING onto them ──────────────────────────
    # Every coordinate that enters the master set is first snapped to the exact
    # mass of the backbone bond it matches.  Without this the set holds raw
    # measured values, and the DAG's edge test -- which allows a gap only within
    # ``tol_ppm`` OF THE GAP -- cannot connect them: a mock sits ~75 ppm off its
    # bond (it is a line's *median* alignment shift, not a measured peak), so a
    # ~163 Da first edge with a 0.005 Da window is simply absent and the node is
    # DEAD.  Measured on YPS: 14 of 15 master coordinates were off-exact, the
    # heaviest path read 14 instead of 16, and a false-but-precise coordinate
    # (168.091, 6.7 ppm from P+A) out-scored the true-but-imprecise b1 -- which
    # is what made the (1,4) -1271.545 TFFCs look ambiguous.  Snapped, the same
    # false coordinate buys only a tie and the ambiguity disappears.
    #
    # NOTE this uses ``prefix``, i.e. the known sequence, so the classification
    # below is "how would this behave with exact masses" rather than a strictly
    # de novo measure.  That is the deliberate trade: tolerance is not what we
    # are trying to study at this stage.  A coordinate matching no bond is left
    # exactly as measured -- those are the wrongly-added TFFCs, and the
    # reconstruction has to live with them as they really are.
    prefix = mr.residue_prefixes(pep)
    n_res = len(pep.AA_array)
    folded_bonds = [(mr.fold(prefix[k], total), k) for k in range(1, n_res)]

    def _bond_of(coord: float, tol_ppm_: float) -> Optional[int]:
        """Backbone bond index whose breakpoint coordinate matches ``coord``.

        Folding is a bijection on bonds: bond k's two masses are prefix[k] and
        total - prefix[k], and fold() picks one of them, so a folded coordinate
        names exactly one cleavage.

        The ppm window is referenced to the **unfolded** mass,
        ``max(coord, total - coord)``, NOT to the folded coordinate.  Folding
        moves a coordinate but not its error: a y30 measured at 3043 Da with an
        8 ppm error (0.025 Da) folds onto a 958 Da coordinate still carrying
        0.025 Da, which is 27 ppm *of the folded value*.  Referencing the folded
        coordinate would demand 27 ppm to accept an 8 ppm measurement, and the
        TFFC would be attributed to no bond at all and silently vanish from the
        column.  Since we cannot know which side of the pair was the measured
        one, the larger side is the safe reference.  Mis-attribution is not a
        risk: consecutive bonds are >= 57 Da (Gly) apart, and the nearest bond
        wins.
        """
        best = None
        for fk, k in folded_bonds:
            ref = max(abs(fk), abs(total - fk))
            if abs(fk - coord) <= ref * tol_ppm_ * 1e-6:
                d = abs(fk - coord)
                if best is None or d < best[0]:
                    best = (d, k)
        return best[1] if best else None

    def _snap(coord: float, tol_ppm_: float) -> float:
        """Exact bond coordinate for ``coord``, or ``coord`` if it matches none."""
        k = _bond_of(coord, tol_ppm_)
        return mr.fold(prefix[k], total) if k is not None else coord

    # ── Seed the master set from the parental lines ───────────────────────
    # ``seed_master(source='annotated')`` policy: every parental-line FFC the
    # annotator resolved, snapped onto its bond.
    master = _MasterSetPpm(total, total * coord_tol_ppm * 1e-6)
    for c in col_order:
        if c not in parental_set:
            continue
        for x, y, _bond, category in col_ffcs.get(c, []):
            if category in ("unmatched", "unexplained"):
                continue
            master.add(_snap(x - PROTON, coord_tol_ppm), 1.0)

    # Heaviest-path scorer over the ppm DAG.  Weights are uniform here (nothing
    # is ever added to the frozen reference), so the coordinate set alone keys
    # the cache.
    _cache: Dict[tuple, float] = {}

    def _score(coords, weights) -> float:
        key = tuple(round(c, 4) for c in coords)
        if key not in _cache:
            graph = _build_dag_ppm(coords, allowed, total, source, tol_ppm,
                                   mr.SINK)
            best, _paths = mr.find_all_heaviest_paths(graph, weights, source,
                                                      max_paths=1)
            _cache[key] = -1.0 if best == float("-inf") else best
        return _cache[key]

    # ── Classify every active TFFC against the FROZEN parental set ────────
    # The master set is never extended here, so:
    #   * every TFFC is judged against the same non-speculative reference --
    #     the parental ladder -- exactly like the #active/#inactive row;
    #   * the result is independent of the order lines and FFCs are visited,
    #     and of which other lines happen to be present;
    #   * a spurious line cannot influence any other line's counts, so no line
    #     filtering is needed -- the numbers are reported for every line and
    #     the decision to *use* a line is taken separately, from its inactive
    #     count (the `aligned?` row / `align_threshold`).
    # This is master_reconstruction's ``--inactive_vs parental`` taken one step
    # further: there it fixes only the inactive test while goodness still runs
    # against the growing set; here goodness is frozen too.
    base_weights = master.weight_map()
    base = _score(master.coords(), base_weights)

    def _classify(entry):
        """(class, good_images) for one TFFC.

        The two images are the TFFC's two possible placements -- "trust Y"
        (fold(Y - proton)) and "trust X" (fold(X - proton)) -- since one TFFC
        alone cannot say which side is the N-terminal fragment.  An image is
        *good* if adding it strictly increases the heaviest path's weight.
        Note a mirrored FFC (Y, X) yields the same pair of images swapped, so
        it always receives the same verdict -- mirrors need no special casing.
        """
        x, y = entry[0], entry[1]
        h = _snap(mr.fold(y - PROTON, total), coord_tol_ppm)
        v = _snap(mr.fold(x - PROTON, total), coord_tol_ppm)
        good_h = _score(master.coords([h]), base_weights) > base + 1e-9
        good_v = _score(master.coords([v]), base_weights) > base + 1e-9
        if good_h and good_v:
            return "ambiguous", [h, v]
        if good_h:
            return "unambiguous", [h]
        if good_v:
            return "unambiguous", [v]
        return "non_productive", []

    verdicts: Dict[str, list] = {}      # col -> [(entry, class, good_images)]
    cells: Dict[str, Optional[str]] = {}
    for c in col_order:
        if c in parental_set:
            cells[c] = None             # parental lines seed the set, not extend it
            continue
        rows = [(e,) + _classify(e)
                for e in col_ffcs.get(c, []) if active_of(c, e)]
        verdicts[c] = rows
        counts = Counter(r[1] for r in rows)
        cells[c] = (f"{counts['non_productive']}/{counts['unambiguous']}"
                    f"/{counts['ambiguous']}")

    # ── Which non-parental lines may join the master TFFC-set ─────────────
    # Two gates, both required (see _extended_master_bonds' caller):
    #   * the line must be one the algorithm can align -- a pure `internal`
    #     line or a parental line carrying one named neutral `loss`.
    #     `classify_line` puts everything else in `other`: negative A, an A
    #     between a loss and a residue, an isotope satellite, and -- the case
    #     called out explicitly -- an internal line that ALSO carries a neutral
    #     loss, which would need the 2D (shift1, shift2) alignment this version
    #     does not implement (reconstruct.md section 5);
    #   * `mark_isotope_satellites` then demotes any line sitting k*1.00335
    #     below a stronger one (reconstruct.md section 4).
    # A is taken from the FFC masses (median of ParentPair - (X + Y)), not from
    # the detection shift, which carries an extra (i + j - z)*proton.
    parent_pair = total + 2 * PROTON
    recs: Dict[str, "mr.LineRec"] = {}
    for c in col_order:
        entries = col_ffcs.get(c, [])
        if c in parental_set or not entries:
            continue
        a_shift = mr._median([parent_pair - (e[0] + e[1]) for e in entries])
        rec = mr.LineRec(key=c, i=0, j=0, mass=0.0, shift=0.0,
                         a_shift=a_shift, a_from_detection=0.0)
        rec.tffcs = [mr.TFFC(x=e[0], y=e[1]) for e in entries]
        rec.category, rec.detail = mr.classify_line(a_shift, allowed)
        recs[c] = rec
    mr.mark_isotope_satellites(list(recs.values()))

    # ── Extend the master TFFC-set with the eligible lines' breakpoints ───
    # An eligible line contributes:
    #   * each unambiguous TFFC's single good image (a forced placement);
    #   * both images of an ambiguous TFFC (master_reconstruction's default
    #     ``--ambiguous_policy both``) -- one of the two is false by
    #     construction, which is the documented cost of using it;
    #   * its MOCK: the breakpoint named by the line's own shift.  The line is
    #     missing a fragment of mass A, so the cleavage that produced it is a
    #     real breakpoint at fold(A).  It needs no image decision (both images
    #     collapse onto the same cleavage) and applies only to `internal`
    #     lines -- a neutral loss is not a cleavage (reconstruct.md section 6).
    # Non-productive TFFCs contribute nothing: their images either duplicate a
    # breakpoint already in the set or are unreachable in the DAG.

    def _resolve(banked: Sequence[Tuple[float, str, str]]):
        """Turn banked ``(coordinate, kind)`` pairs into a master-set summary.

        Returns ``(bond_sources, incorrect)`` where

          * ``bond_sources[k]`` -- the kinds ('unambiguous' / 'ambiguous' /
            'mock') of every TFFC that put backbone bond ``k`` into the master
            set.  The multiplicity matters: a bond supported by a *single*
            ambiguous TFFC rests on a coin flip, and §8e drops it from the
            reconstruction on that basis.
          * ``incorrect`` -- ``(coordinate, kind, source column)`` for every
            banked coordinate matching **no** backbone bond.  Those are TFFCs we
            added wrongly; they are still real entries in the master set, so the
            reconstruction has to live with them.  The source column is kept so
            a bad TFFC can be traced back to the line that proposed it.

        Mocks keep the looser window (A is a line-level median, not a measured
        peak -- see DEFAULT_MOCK_TOL_PPM): on KWK the A = 314.161 (KW) mock
        sits 0.013 Da == 41 ppm off prefix[2].
        """
        bond_sources: Dict[int, List[str]] = {}
        incorrect: List[Tuple[float, str, str]] = []
        for coord, kind, src in banked:
            tol = mock_tol_ppm if kind == "mock" else coord_tol_ppm
            bond = _bond_of(coord, tol)
            if bond is None:
                incorrect.append((coord, kind, src))
            else:
                bond_sources.setdefault(bond, []).append(kind)
        return bond_sources, incorrect

    def _line_can_contribute(c: str) -> bool:
        rec = recs.get(c)
        return rec is not None and rec.category in ("internal", "loss")

    def _mock_coord(c: str) -> Optional[float]:
        """The line's mock breakpoint, snapped onto its bond, or None.

        The line is missing a fragment of mass A, so the cleavage that produced
        it is a real breakpoint at fold(A).  No image decision is needed -- both
        images collapse onto the same cleavage.

        Admitted in two cases:
          * the line is a pure ``internal`` line (A is a residue run), or
          * ``fold(A)`` lands on a real backbone cleavage under
            ``mock_tol_ppm``, whatever ``classify_line`` made of the line.

        The second case is the same criterion the main table's ``loose_mock``
        already uses, and it is needed because A can be read two ways at once.
        YPS's ``(2,4) -180.072`` has A = 180.077, which is simultaneously
        "a residue + NH3" (so ``classify_line`` files it as `other`, an internal
        line carrying a loss, which this version cannot align) and *exactly*
        ``y1`` = total - prefix[35] = 180.090, i.e. the line lost the whole
        C-terminal y1 as a neutral.  The second reading is the true one, and
        gating on ``classify_line`` alone threw away b35.
        """
        rec = recs.get(c)
        if rec is None or rec.a_shift < mr.MIN_AA_MASS - 0.05:
            return None
        coord = mr.fold(rec.a_shift, total)
        if rec.category == "internal" or _bond_of(coord, mock_tol_ppm) is not None:
            return _snap(coord, mock_tol_ppm)
        return None

    # =========================================================================
    # ITERATIVE ROUNDS -- grow the master TFFC-set until it stops growing
    # =========================================================================
    # Round 1 above judged everything against the frozen parental set.  But the
    # concepts *aligned* / *active* / *unambiguous* are all relative to the
    # master set, so once round 1 has added anything they must be re-derived:
    # a TFFC that was ambiguous may now be forced (one of its two images is
    # already in the set, so only the other can still add weight), a line that
    # was not aligned may now have an inactive TFFC and become eligible, and a
    # TFFC that was non-productive may become productive.  So we repeat rounds
    # until the set stops increasing.
    #
    # Only UNAMBIGUOUS TFFCs are banked during the iteration; ambiguous ones are
    # held back and re-tested every round, which is the whole point -- banking a
    # forced placement is what can disambiguate them.  Whatever is still
    # ambiguous at convergence is genuinely undecidable from this data.
    #
    # Line order: **most populated line first**, every round.  The outcome can
    # in principle depend on the order the lines are visited, so the order is
    # fixed and principled -- the line carrying the most FFCs is the best
    # corroborated, so its forced placements are the safest to bank first.
    if not converge:
        banked: List[Tuple[float, str, str]] = []
        eligible: Dict[str, str] = {}
        for c, rec in recs.items():
            if inactive_of(c) < master_align_threshold:
                continue
            if not _line_can_contribute(c):
                mc = _mock_coord(c)     # mock-only line
                if mc is not None:
                    banked.append((mc, "mock", c))
                continue
            eligible[c] = rec.category
            for _e, cls, imgs in verdicts.get(c, []):
                if cls == "ambiguous" and not accept_ambiguity:
                    continue            # atomic pair: all or nothing
                for img in imgs:
                    banked.append((img, cls, c))
            mc = _mock_coord(c)
            if mc is not None:
                banked.append((mc, "mock", c))
        return (cells, None, _resolve(banked), eligible)

    work = master.snapshot()
    pending: Dict[str, list] = {c: [r[0] for r in rows]
                                for c, rows in verdicts.items()}
    n_unamb = {c: 0 for c in pending}
    n_amb = {c: 0 for c in pending}
    n_unprod = {c: 0 for c in pending}
    banked: List[Tuple[float, str, str]] = []  # (coord, kind, source column)
    mocks_added: set = set()            # lines whose mock is already banked
    eligible: Dict[str, str] = {}

    # Most populated line first; the key is stable so the order is deterministic.
    line_order = sorted(pending, key=lambda c: (-len(col_ffcs.get(c, [])), c))

    def _images_of(entry):
        """The TFFC's two placements, each snapped onto its bond (see _snap)."""
        return (_snap(mr.fold(entry[1] - PROTON, total), coord_tol_ppm),
                _snap(mr.fold(entry[0] - PROTON, total), coord_tol_ppm))

    # Rounds are **batch**, not incremental: every TFFC in a round is judged
    # against the set as it stood at the START of that round, and the winners
    # are banked together at the end of it.  That is what makes a round
    # order-independent -- no TFFC can be silenced or promoted by another TFFC
    # decided moments earlier in the same round.  (Judging against the set as it
    # grows within a round would reintroduce exactly the order dependence the
    # frozen round-1 reference was chosen to avoid.)
    rounds_used = 0
    for _round in range(max_rounds):
        rounds_used = _round + 1
        w = work.weight_map()
        s0 = _score(work.coords(), w)

        # Eligibility is re-derived every round against the frozen set: a line
        # not aligned in round 1 can acquire an inactive TFFC later and qualify.
        round_eligible = []
        for c in line_order:
            # A line qualifies if it can contribute FFCs (pure internal / loss)
            # OR only a mock -- a shift landing on a real cleavage is a real
            # breakpoint even when classify_line cannot align the line itself.
            if not _line_can_contribute(c) and _mock_coord(c) is None:
                continue
            n_inactive = sum(
                1 for e in col_ffcs.get(c, [])
                if any(img in work for img in _images_of(e))
            )
            if n_inactive >= master_align_threshold:
                round_eligible.append(c)

        picks = []          # (col, entry, image) -- unambiguous this round
        new_mocks = []      # (col, coord)
        for c in round_eligible:
            if _line_can_contribute(c):
                eligible[c] = recs[c].category
            if c not in mocks_added:
                mocks_added.add(c)
                mc = _mock_coord(c)
                if mc is not None:
                    new_mocks.append((c, mc))
            if not _line_can_contribute(c):
                continue                # mock-only line: no FFC contributions
            for e in pending[c]:
                h, v = _images_of(e)
                if (h in work) or (v in work):
                    continue            # inactive; settled at convergence
                good_h = _score(work.coords([h]), w) > s0 + 1e-9
                good_v = _score(work.coords([v]), w) > s0 + 1e-9
                if good_h == good_v:
                    continue            # ambiguous or non-productive: hold it
                picks.append((c, e, h if good_h else v))

        if not picks and not new_mocks:
            rounds_used = _round          # this round banked nothing
            break                         # master TFFC-set has converged

        for _c, mc in new_mocks:
            banked.append((mc, "mock", _c))
            work.add(mc, 1.0)
        for c, e, img in picks:
            banked.append((img, "unambiguous", c))
            work.add(img, 1.0)
            pending[c].remove(e)
            n_unamb[c] += 1

    global LAST_CONVERGENCE_ROUNDS
    LAST_CONVERGENCE_ROUNDS = rounds_used

    # ── Settle whatever never got banked, against the converged set ───────
    # A TFFC that turned *inactive* during the iteration adds nothing new, which
    # is operationally the same as non-productive, so it is counted there -- that
    # keeps the converged row summing to #active, like the round-1 row.
    w = work.weight_map()
    s0 = _score(work.coords(), w)
    for c in pending:
        for e in pending[c]:
            h, v = _images_of(e)
            if (h in work) or (v in work):
                n_unprod[c] += 1
                continue
            good_h = _score(work.coords([h]), w) > s0 + 1e-9
            good_v = _score(work.coords([v]), w) > s0 + 1e-9
            if good_h and good_v:
                n_amb[c] += 1
                # An ambiguous TFFC's two images are ATOMIC: its evidence is
                # "one of these two", never "this one".  Banking a single image
                # would be inventing a decision the data does not support, and
                # banking both is the write-up's ``ambiguous_policy=both``.  So
                # it is all or nothing, controlled by ``accept_ambiguity``.
                if c in eligible and accept_ambiguity:
                    banked.append((h, "ambiguous", c))
                    banked.append((v, "ambiguous", c))
            elif good_h or good_v:
                n_unamb[c] += 1         # only reachable on a non-eligible line
                if c in eligible:
                    banked.append((h if good_h else v, "unambiguous", c))
            else:
                n_unprod[c] += 1

    cells_final: Dict[str, Optional[str]] = {}
    for c in col_order:
        if c in parental_set:
            cells_final[c] = None
            continue
        cells_final[c] = f"{n_unprod[c]}/{n_unamb[c]}/{n_amb[c]}"

    return cells, cells_final, _resolve(banked), eligible


# =============================================================================
# 4a. SUMMARY ROWS (FFC count, false-positive counts, examples)
# =============================================================================

def _build_summary_rows(
    all_columns,
    col_order: list,
    line_type_map: dict,
    incorrect_master_map: dict,
    bp_ffc_map: dict,
    unm_unexp_map: dict,
    act_inact_map: dict,
    aligned_map: dict,
    spurious_map: dict,
    ex_unexplained: dict,
    ex_unmatched: dict,
    max_ex: int,
    reconstruct_map: Optional[dict] = None,
    productivity_map: Optional[dict] = None,
    converged_map: Optional[dict] = None,
    master_recon_row: bool = False,
) -> pd.DataFrame:
    """Build the per-line summary rows appended below the coverage table.

    Row order:
        line type                      (most-common FFC category on the line)
        #incorrect TFFCs               (parental cols: FFCs that are not a
                                        clean parental match; Master TFFC-set
                                        col: TFFCs added in error)
        #reconstructions/correct reconstructions  (only when reconstruct=True;
                                        a single cell, not a per-line row)
        #breakpoints/FFCs on a line
        #active/#inactive TFFCs        (internal columns only)
        #unproductive/unambiguous/ambiguous  (internal columns only; only when
                                        productivity=True)
        #unproductive/unambiguous/ambiguous (converged)  (same, after the
                                        iterative rounds; only when converge=True)
        aligned?                       (internal cols: aligned/non-aligned)
        #unmatched/#unexplained        (ffc_annotator-style FPR counts)
        First spurious FFC             (any column, blank if none)
        Unresolved FFCs 1..max_ex      (up to max_ex unexplained examples)
        Unmatched FFCs 1..max_ex       (up to max_ex unmatched examples)

    ``line type`` is one of parental / internal / secondary / unexplained /
    unmatched.  ``#incorrect TFFCs`` is only meaningful on columns
    whose ``line type`` is (base) ``parental`` -- it counts the FFCs on that
    line that are NOT a clean, unmodified ``parental`` match (i.e. every
    ``parental*``/``parental~``/``parental*~`` and any other category on that
    column); blank on every other column.
    ``#reconstructions/correct reconstructions`` is emitted only when
    ``reconstruct_map`` is given.  It describes the whole peptide rather than
    any one line, so it occupies a *single cell* (``{n_paths}/{n_correct}``,
    e.g. ``3/1``) on the first parental column, blank elsewhere.
    ``#unmatched/#unexplained`` shows
    ``{unmatched}/{unexplained}``.  ``#active/#inactive TFFCs`` is
    ``{active}/{inactive}`` for internal columns, blank for parental.
    ``#unproductive/unambiguous/ambiguous`` (only when ``productivity_map`` is
    given) splits those *active* TFFCs by what they would contribute to a de
    novo reconstruction, and sums to ``#active``; blank for parental.
    ``aligned?`` reads ``aligned`` when the internal line has enough inactive
    FFCs to align to the parental ladder, else ``non-aligned`` (blank for
    parental).  ``First spurious
    FFC`` shows the most highly ranked unmatched/unexplained FFC.  Each
    ``Unresolved``/``Unmatched FFCs`` example is a tuple
    ``(X, Y, (i,j), [rank], mass_i(X), mass_j(Y), sum)``; rows are blank where
    no such example exists.
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

    parts = [
        _count_row(line_type_map,  "line type", blank_if_zero=False),
        _count_row(incorrect_master_map, "#incorrect TFFCs", blank_if_zero=False),
    ]
    if master_recon_row:
        parts.append(_count_row({}, "#reconstructions/correct?",
                                blank_if_zero=False))
    if reconstruct_map is not None:
        parts.append(_count_row(reconstruct_map,
                                "#reconstructions/correct reconstructions",
                                blank_if_zero=False))
    parts += [
        _count_row(bp_ffc_map,     "#breakpoints/FFCs on a line", blank_if_zero=False),
        _count_row(act_inact_map,  "#active/#inactive TFFCs", blank_if_zero=False),
    ]
    if productivity_map is not None:
        parts.append(_count_row(productivity_map,
                                "#unproductive/unambiguous/ambiguous",
                                blank_if_zero=False))
    if converged_map is not None:
        parts.append(_count_row(converged_map,
                                "#unproductive/unambiguous/ambiguous (converged)",
                                blank_if_zero=False))
    return pd.concat(parts + [
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
    reconstruct: bool = False,
    considering_spurious: bool = False,
    edges_path: Optional[str] = None,
    reconstruct_threshold: float = 0.001,
    reconstruct_max_paths: int = 1000,
    productivity: bool = False,
    productivity_tol_ppm: float = 30.0,
    productivity_coord_tol_ppm: float = 30.0,
    extend_master_set: bool = False,
    master_align_threshold: int = 1,
    converge: bool = True,
    max_convergence_rounds: int = 20,
    loose_mock: bool = False,
    accept_ambiguity: bool = False,
    graph: bool = False,
    graph_path: Optional[str] = None,
    tffc_analysis_hook=None,                  # [DeNovoFFC]
    master_set_reconstruction_hook=None,      # [DeNovoFFC]
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
    reconstruct : bool, default False
        Emit the ``#reconstructions/correct reconstructions`` summary row by
        running ``../reconstruct/path_finding.py``'s longest-path de novo
        search over the breakpoints carried by the parental (master-ladder)
        lines.  The row is omitted entirely when False, and neither
        ``path_finding`` nor ``edges.json`` is imported/loaded.
    considering_spurious : bool, default False
        Only meaningful with ``reconstruct=True``.  When True the
        unmatched/unexplained ("spurious") FFCs on the parental lines are also
        fed to the search, converted to charge-independent neutral masses; when
        False they are ignored and only properly annotated parental
        breakpoints are used.
    edges_path : str, optional
        Path to the precomputed compound-mass table (``edges.json``);
        defaults to ``DEFAULT_EDGES_PATH``.
    reconstruct_threshold : float, default 0.001
        Mass tolerance (Da) for the reconstruction graph's edges.
    reconstruct_max_paths : int, default 1000
        Cap on the number of tied longest paths collected.
    productivity : bool, default False
        Emit the ``#unproductive/unambiguous/ambiguous`` summary row, splitting
        each line's *active* TFFCs by what they would contribute to a de novo
        reconstruction (``../reconstruct/master_reconstruction.py``'s goodness
        test, judged against the frozen parental ladder).  Omitted when False.
    productivity_tol_ppm, productivity_coord_tol_ppm : float, default 20
        DAG edge tolerance and "already in the master set" tolerance for that
        classification, in **ppm** -- applied per coordinate, not as one fixed
        Da window.  Deconvolved fragments carry a roughly relative mass error,
        so a single Da window is at once too loose for the light coordinates
        and too tight for the heavy ones.  20 ppm spans ~0.01 Da at 500 Da to
        ~0.04 Da at 2000 Da, bracketing the 0.03 Da that ``reconstruct.md``
        section 7 validated on KWK.
    extend_master_set : bool, default False
        Let the aligned **internal** and **neutral-loss** lines contribute to
        the ``Master TFFC-set`` column, on top of the parental lines: their
        productive TFFCs plus each internal line's mock.  Excluded by
        construction are lines that are internal *and* carry a neutral loss
        (they need a 2D alignment this version does not implement) and isotope
        satellites.  Off by default -- it needs ``edges.json`` and the de novo
        machinery; when False the column stays parental-only.
    master_align_threshold : int, default 1
        Inactive FFCs an internal / loss line must have -- i.e. FFCs whose
        breakpoint already sits on the master TFFC-set, proving the line really
        is aligned to it -- before it may join that set.  Re-derived every round
        against the *grown* set, so a line can qualify later than round 1.
    converge : bool, default True
        Build the master TFFC-set **iteratively in rounds** rather than in a
        single pass: bank the unambiguous TFFCs, re-derive
        aligned/active/unambiguous against the grown set, repeat until the set
        stops increasing.  Adds the
        ``#unproductive/unambiguous/ambiguous (converged)`` row beneath the
        round-1 row, and makes ``extend_master_set`` use the converged set.
    max_convergence_rounds : int, default 20
        Safety cap on those rounds (convergence is typically 2-4).
    loose_mock : bool, default False
        Add a **charge-free** mock probe alongside the charge sweep.  The sweep
        can only reach a lost fragment that is itself charged (it assumes
        ``z = parent_charge - (i + j) >= 1``), so a line whose charges already
        sum to the precursor charge -- one that lost a *neutral* run of residues
        -- gets no mock at all.  The loose probe takes the alignment shift A
        straight from the FFC masses (no proton term, hence no charge guess) and
        probes ``A + proton``; it still has to match a terminal ion whose
        complement completes an adjacent b/y pair.  Such cells are tagged
        ``[mock-loose]``, and a charged reading wins any tie.  Recovers e.g.
        LLG's ``(2,4) -113.080`` and YLE's ``(1,2) -163.059``, both b1.
    accept_ambiguity : bool, default False
        Let **ambiguous** TFFCs into the master TFFC-set.  An ambiguous TFFC's
        two images are atomic -- its evidence is "one of these two", never
        "this one" -- so with True BOTH are banked (one is false by
        construction: ``reconstruct.md`` section 6a) and with False neither is,
        for the column, the reconstruction and the incorrect-TFFC tally alike.
        Default False: measured over seven peptides, ambiguous TFFCs bought one
        usable bond (LLG) and supplied four wrong coordinates, three of which
        cost YPS its correct reconstruction.
    graph : bool, default False
        When a **correct** reconstruction exists, also render the 2D map
        (``../reconstruct/path_graph_revised.visualize_all_paths``) with the
        correct path as the hero, the other tied paths as candidates, and the
        incorrect master TFFCs marked spurious.  Produces one extra image file
        and changes nothing in the table.  Nothing is drawn when no
        reconstruction is correct.
    graph_path : str, optional
        Where to write it; defaults to ``master_set_reconstruction.png``.
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

    # Bonds represented by a *master TFFC*: an FFC on a parental (master-
    # ladder) column whose ffc_annotator-style category is a bare "parental"
    # (no "*" neutral-loss / "~" isotope suffix).  Drives the "Master
    # TFFC-set" column below.
    master_bond_set: set = set()

    # Reconstruction inputs, harvested from the parental (master-ladder)
    # columns only.  ``parental_break_bonds`` are the 1-based cleavage indices
    # k that a properly annotated parental FFC resolved to;
    # ``parental_spurious_masses`` are the charge-independent neutral masses of
    # the unmatched/unexplained FFCs there (used only when
    # ``considering_spurious``).
    parental_break_bonds: set = set()
    parental_spurious_masses: List[Tuple[float, float]] = []

    # Per-column TFFC records for the productivity classification:
    # [x, y, bond_label, category] per member FFC, where x/y are the
    # singly-protonated per-side masses.  PROTON (not MASS_H) is used so the
    # complementarity X + Y == ParentPair - A holds exactly, matching
    # master_reconstruction.build_line_records.
    col_ffcs: Dict[str, list] = {c: [] for c in col_order}

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
            # TFFC record for the productivity classification; its bond is
            # filled in below if this FFC places.
            ffc_rec = [iu * mz_a - (iu - 1) * PROTON,
                       ju * mz_b - (ju - 1) * PROTON, None, category]
            col_ffcs[col].append(ffc_rec)
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
                # Spurious FFC on a parental line -> keep its charge-independent
                # neutral masses for the (optional) reconstruction search.
                if col in parental_set:
                    parental_spurious_masses.append(
                        (_mass_i(mz_a, iu), _mass_i(mz_b, ju))
                    )

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
            ffc_rec[2] = bond_label
            if col in parental_set:
                if category == "parental":
                    master_bond_set.add(bond_label)
                if category not in ("unmatched", "unexplained"):
                    # A properly annotated parental FFC pins this cleavage --
                    # feed it to the reconstruction search.
                    parental_break_bonds.add(bond)

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
    #
    # ``loose_mock`` -- why the charge sweep is not enough
    # ---------------------------------------------------
    # Sweeping z is really guessing the charge that converts the *detection*
    # shift into the true alignment shift A, and the algebra fixes it exactly:
    #     shift = -A + (i + j - parent_charge)*proton   =>   z = parent_charge - (i + j)
    # So the sweep's 1..(parent_charge-2) range silently assumes the lost piece
    # is CHARGED (z >= 1), i.e. that the line's own charges leave room for it.
    # A line whose charges already sum to the full precursor charge lost a
    # *neutral* run of residues, needs z = 0, and can never be reached:
    #     LLG  (2,4) -113.080   i+j = 6 = parent_charge -> z = 0
    #     YLE  (1,2) -163.059   i+j = 3 = parent_charge -> z = 0
    # Both are real: A = 113.080 and 163.058, whose [M+H]+ (A + proton) match b1
    # to 0.004 / 0.006 Da.
    #
    # With ``loose_mock`` we add one extra, charge-FREE probe: take A straight
    # from the FFC masses (A = ParentPair - (X + Y), median over the line, which
    # carries no proton term at all) and probe ``A + proton``.  Everything else
    # is unchanged -- the probe still has to match a theoretical terminal ion
    # whose complement completes an adjacent b/y pair, which is a far stronger
    # filter than the charge bookkeeping it replaces.  Since we work in
    # charge-independent masses, the charge of the lost piece is irrelevant.
    parent_pair = pep.ion_mass("b1") + pep.ion_mass(f"y{n - 1}")
    mock_charges = range(1, max(2, parent_charge - 1))   # 1 .. parent_charge-2

    # Charge-independent alignment shift per line, straight from the FFC masses.
    col_a_shift: Dict[str, Optional[float]] = {}
    for c in col_order:
        vals = sorted(parent_pair - (e[0] + e[1]) for e in col_ffcs.get(c, []))
        col_a_shift[c] = vals[len(vals) // 2] if vals else None

    mock_cells: Dict[str, Dict[str, str]] = {}
    for c in col_order:
        loss = -col_shift[c]
        if loss <= MIN_LOSS_DA:     # parental / isotope lines: nothing to probe
            continue
        # (x_mass, charge label, came_from_loose_probe)
        probes = [(loss - (z - 1) * PROTON, z, False) for z in mock_charges]
        a_ffc = col_a_shift.get(c)
        if (loose_mock and a_ffc is not None
                and a_ffc >= MIN_RESIDUE_MASS - 0.05):
            probes.append((a_ffc + PROTON, 1, True))

        best = None   # (key, cL, cR, chL, chR, is_loose)
        for x_mass, z, is_loose in probes:
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
            # Prefer a charged (strict) reading when it is equally good: the
            # loose probe is the fallback, so it loses ties.
            key = (ca.rank + cb.rank, ca.iso + cb.iso, ca.dev + cb.dev,
                   int(is_loose))
            if best is None or key < best[0]:
                best = (key, cL, cR, chL, chR, is_loose)
        if best is None:
            continue
        _, cL, cR, chL, chR, is_loose = best
        bond_label = f"b{cL.frag.e}, y{n - cL.frag.e}"
        if bond_label not in row_index:
            continue
        tag = "mock-loose" if is_loose else "mock"
        mock_cells.setdefault(c, {})[bond_label] = (
            f"*{_placed_label(cL, True, n)}, {_placed_label(cR, False, n)}, "
            f"[{chL}, {chR}], ({round(cL.dev, 3)}, {round(cR.dev, 3)}), [{tag}]"
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
    # #incorrect master TFFCs -- only meaningful on columns whose line type is
    # (base) "parental": the FFCs on that column that are NOT a clean,
    # unmodified "parental" match (loss/isotope variants and anything else).
    # Blank everywhere else.
    incorrect_master_map = {
        c: (ffc_counts[c] - cat_counts[c].get("parental", 0)
            if _base_category(line_type_map[c] or "") == "parental"
            else None)
        for c in col_order
    }
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
    # Spelled out rather than 1/0 -- "aligned" / "non-aligned" reads directly in
    # the sheet without having to remember the polarity.
    aligned_map = {
        c: (None if c in parental_set
            else ("aligned" if inactive_map[c] >= align_threshold
                  else "non-aligned"))
        for c in col_order
    }

    # ── TFFC analysis: productivity split + master-set extension ─────────
    # Both features share one pass (same frozen parental reference, same DAG
    # scorer), so run it if either is requested.
    productivity_map = None
    converged_map = None
    extra_master_bonds: set = set()
    bond_sources: Dict[int, List[str]] = {}
    incorrect_coords: List[Tuple[float, str, str]] = []
    if productivity or extend_master_set:
        def _is_active(col, entry):
            bond_label = entry[2]
            return bond_label is None or bond_label not in master_rows

        # [DeNovoFFC] An explicit hook replaces the research tree's habit of
        # swapping the module-level function (combining_*/_CaptureTffc):
        # same call, same arguments, no global state touched.
        _tffc = tffc_analysis_hook or _tffc_analysis       # [DeNovoFFC]
        cells, cells_final, (bond_sources, incorrect_coords), _eligible = _tffc(
            pep, col_ffcs, col_order, parental_set,
            _is_active, lambda c: inactive_map.get(c) or 0,
            master_align_threshold=master_align_threshold,
            edges_path=edges_path,
            tol_ppm=productivity_tol_ppm,
            coord_tol_ppm=productivity_coord_tol_ppm,
            mock_tol_ppm=mock_tol_ppm,
            converge=converge,
            max_rounds=max_convergence_rounds,
            accept_ambiguity=accept_ambiguity,
        )
        if productivity:
            productivity_map = cells
            converged_map = cells_final
        if extend_master_set:
            extra_master_bonds = {f"b{k}, y{n - k}" for k in bond_sources}
        else:
            bond_sources, incorrect_coords = {}, []

    # ── De novo reconstruction from the parental ladder (opt-in) ──────────
    # Describes the whole peptide, not one line, so the result occupies a
    # single cell -- placed on the first parental column (the master ladder
    # the breakpoints came from), blank everywhere else.
    reconstruct_map = None
    if reconstruct:
        cell = _reconstruction_counts(
            pep,
            sorted(parental_break_bonds),
            parental_spurious_masses,
            considering_spurious=considering_spurious,
            edges_path=edges_path,
            threshold=reconstruct_threshold,
            max_paths=reconstruct_max_paths,
        )
        anchor = next((c for c in table.columns if c in parental_set),
                      col_order[0])
        reconstruct_map = {c: (cell if c == anchor else None) for c in col_order}

    spurious_map = {c: (v[1] if v else None) for c, v in spurious_best.items()}
    summary = _build_summary_rows(
        table.columns, col_order,
        line_type_map, incorrect_master_map, bp_ffc_map, unm_unexp_map,
        act_inact_map, aligned_map, spurious_map,
        ex_unexplained, ex_unmatched, MAX_EX,
        reconstruct_map=reconstruct_map,
        productivity_map=productivity_map,
        converged_map=converged_map,
        master_recon_row=bool(reconstruct and extend_master_set),
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

    # ── Master TFFC-set column ────────────────────────────────────────────
    # Binary 0/1 per bond: 1 if the bond is represented by a master TFFC.
    #
    # Always included: a clean, unmodified "parental" FFC on a parental /
    # master-ladder column (neutral-loss "*" and isotopic "~" parental variants
    # do not count).
    #
    # With ``extend_master_set``, also the breakpoints contributed by the
    # aligned internal and neutral-loss lines -- their productive TFFCs plus
    # each internal line's mock.  A line joins only if it is a *pure* internal
    # or *pure* one-neutral-loss line (an internal line that also carries a loss
    # is excluded, as is an isotope satellite) AND it has at least
    # ``master_align_threshold`` inactive FFCs proving it aligns to the parental
    # ladder.  See ``_tffc_analysis``.
    master_bond_set = master_bond_set | extra_master_bonds
    n_master = sum(1 for r in row_labels if r in master_bond_set)
    master_pct = n_master / len(row_labels) * 100
    master_col_name = f"Master TFFC-set (coverage={master_pct:.1f}%)"
    table[master_col_name] = pd.Series(
        {r: ("1" if r in master_bond_set else "0") for r in row_labels}
    )

    # ── Reconstruction from the WHOLE master TFFC-set ─────────────────────
    # Reported in the Master TFFC-set column, since that is the set it is built
    # from.  Two cells:
    #   #incorrect TFFCs           -- master TFFCs matching no backbone bond
    #   #reconstructions/correct?  -- e.g. "8/yes"
    #
    # Whether ambiguous TFFCs are in the master set at all is decided upstream,
    # by ``accept_ambiguity`` (see ``_tffc_analysis``): an ambiguous TFFC's two
    # images are atomic -- its evidence is "one of these two", never "this one"
    # -- so they are banked together or not at all.  By the time we get here the
    # master set is simply the master set, and every bond in it is fed.
    if reconstruct and extend_master_set:
        parental_bonds = {
            int(r.split(",")[0][1:]) for r in master_bond_set
            if r not in extra_master_bonds
        }
        recon_bonds = set(parental_bonds) | set(bond_sources)
        # [DeNovoFFC] An explicit replacement for the final search (the opt-in
        # RBDP annotation route, src/rbdp/annotation_adapter.py).  It receives
        # the same arguments plus this call's context; None keeps the original.
        _recon = _master_set_reconstruction
        if master_set_reconstruction_hook is not None:
            import functools as _ft
            _recon = _ft.partial(master_set_reconstruction_hook, context=dict(
                ffc_df=ffc_df, lines=list(lines), parent_mass=parent_mass,
                parent_charge=parent_charge, tol_ppm=tol_ppm))
        n_paths, any_ok, recon_rows = _recon(
            pep, sorted(recon_bonds), [c for c, _k, _s in incorrect_coords],
            edges_path=edges_path,
            threshold=reconstruct_threshold,
            max_paths=reconstruct_max_paths,
            graph=graph,
            graph_path=graph_path,
            graph_title=f"{''.join(a.acid_name for a in pep.AA_array)}  "
                        f"master TFFC-set reconstruction",
        )
        table.at["#incorrect TFFCs", master_col_name] = len(incorrect_coords)
        table.at["#reconstructions/correct?", master_col_name] = (
            f"{n_paths}/{'yes' if any_ok else 'no'}")
        # Carried out-of-band so the caller can write it to a second sheet.
        table.attrs["reconstructions"] = pd.DataFrame(
            recon_rows, columns=["#", "reconstruction", "correct"])
        table.attrs["master_bonds"] = sorted(recon_bonds)
        # (coordinate, how it was added, which line proposed it) -- enough to
        # trace a bad TFFC back to its origin; see the `incorrect_tffcs` sheet.
        table.attrs["incorrect_tffcs"] = [
            {"coord": round(c, 4), "kind": k, "line": src}
            for c, k, src in incorrect_coords
        ]

    lead = ["AA", col_name, master_col_name]
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
    reconstruct: bool = False,
    considering_spurious: bool = False,
    edges_path: Optional[str] = None,
    reconstruct_threshold: float = 0.001,
    reconstruct_max_paths: int = 1000,
    productivity: bool = False,
    productivity_tol_ppm: float = 30.0,
    productivity_coord_tol_ppm: float = 30.0,
    extend_master_set: bool = False,
    master_align_threshold: int = 1,
    converge: bool = True,
    max_convergence_rounds: int = 20,
    loose_mock: bool = False,
    accept_ambiguity: bool = False,
    graph: bool = False,
    graph_path: Optional[str] = None,
    master_set_reconstruction_hook=None,      # [DeNovoFFC]
):
    """Run orientation-agnostic ``detect_lines_oa`` + ``coverage_table_internal``.

    The deconvolution's per-side charge is ignored; detection recovers each
    FFC's charge and merges (i, j)/(j, i) mirror lines.  ``delta_unit`` selects
    whether the detection ``delta`` is a fixed Da gap (default, 0.02 Da) or a
    relative ppm gap.  Returns ``(coverage_table, all_lines)``.

    ``reconstruct=True`` adds the ``#reconstructions/correct reconstructions``
    summary row, running the de novo longest-path search in
    ``../reconstruct/path_finding.py`` over the parental ladder's breakpoints;
    ``considering_spurious=True`` additionally feeds it the unmatched /
    unexplained FFCs found on those parental lines.

    ``productivity=True`` adds the ``#unproductive/unambiguous/ambiguous`` row,
    splitting each line's active TFFCs by what they would contribute to a de
    novo reconstruction.  Judged against the frozen parental ladder, so the
    result is order-independent (see ``_productivity_counts``).
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
        reconstruct=reconstruct,
        considering_spurious=considering_spurious,
        edges_path=edges_path,
        reconstruct_threshold=reconstruct_threshold,
        reconstruct_max_paths=reconstruct_max_paths,
        productivity=productivity,
        productivity_tol_ppm=productivity_tol_ppm,
        productivity_coord_tol_ppm=productivity_coord_tol_ppm,
        extend_master_set=extend_master_set,
        master_align_threshold=master_align_threshold,
        converge=converge,
        max_convergence_rounds=max_convergence_rounds,
        loose_mock=loose_mock,
        accept_ambiguity=accept_ambiguity,
        graph=graph,
        graph_path=graph_path,
        master_set_reconstruction_hook=master_set_reconstruction_hook,  # [DeNovoFFC]
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
    RECONSTRUCT = True          # emit "#reconstructions/correct reconstructions"
    CONSIDERING_SPURIOUS = False   # also feed parental-line spurious FFCs
    PRODUCTIVITY = True         # emit "#unproductive/unambiguous/ambiguous"
    EXTEND_MASTER_SET = True    # let aligned internal/loss lines join Master TFFC-set
    MASTER_ALIGN_THRESHOLD = 1  # inactive FFCs such a line needs to qualify
    LOOSE_MOCK = True           # charge-free mock probe as well as the charge sweep
    ACCEPT_AMBIGUITY = False    # let ambiguous TFFC pairs into the master set
    GRAPH = True                # draw the 2D map when a reconstruction is correct
    OUTPUT_SHEET = "KWK3+_line_based_oa_v2"
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
        reconstruct=RECONSTRUCT,
        considering_spurious=CONSIDERING_SPURIOUS,
        productivity=PRODUCTIVITY,
        extend_master_set=EXTEND_MASTER_SET,
        master_align_threshold=MASTER_ALIGN_THRESHOLD,
        loose_mock=LOOSE_MOCK,
        accept_ambiguity=ACCEPT_AMBIGUITY,
        graph=GRAPH,
        graph_path=os.path.join(os.path.dirname(__file__),
                                f"{OUTPUT_SHEET}_graph.png"),
    )

    print(f"\nDetected lines: {len(all_lines)}")
    data_rows = cov_table.iloc[: len(pep.AA_array) - 1]
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(data_rows.to_string())

    recon = cov_table.attrs.get("reconstructions")
    if recon is not None and not recon.empty:
        n_ok = int((recon["correct"] != "").sum())
        print(f"\nMaster-set reconstructions: {len(recon)}  "
              f"({n_ok} correct)   incorrect TFFCs: "
              f"{len(cov_table.attrs.get('incorrect_tffcs', []))}")
        for _, r in recon.head(5).iterrows():
            print(f"  {r['#']:>3}. {r['reconstruction']}"
                  f"{'   <-- CORRECT' if r['correct'] else ''}")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        cov_table.to_excel(writer, sheet_name=OUTPUT_SHEET,
                           index_label=f"#FFC={TOP_N}")
        ws = writer.sheets[OUTPUT_SHEET]
        for col_cells in ws.columns:
            ws.column_dimensions[col_cells[0].column_letter].width = 32
        if recon is not None:
            recon.to_excel(writer, sheet_name="reconstructions", index=False)
            ws2 = writer.sheets["reconstructions"]
            ws2.column_dimensions["A"].width = 6
            ws2.column_dimensions["B"].width = 140
            ws2.column_dimensions["C"].width = 12
    print(f"\nSaved to {OUTPUT_EXCEL} [{OUTPUT_SHEET}]"
          + ("  + [reconstructions]" if recon is not None else ""))
