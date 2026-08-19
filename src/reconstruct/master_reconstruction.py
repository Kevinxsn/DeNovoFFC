#!/usr/bin/env python
"""
master_reconstruction.py -- end-to-end DeNovoMMS reconstruction that combines
the *parental*, *neutral-loss* and *internal* lines of a line-annotated FFC map
into a single **master TFFC-set**, then reads the peptide off the heaviest path
of the 2D-DAG built on that set.

This is the script that glues the three pieces that already existed separately:

  * ``../line_based_annotation/line_internal_annotation_oa_v2.py``
        detects the lines in a deconvolved FFC map (orientation-agnostic,
        charge-independent) and tells us what each line *is*.
  * ``internal_line.py``
        the Step-1 prototype: classify one internal-line TFFC against a
        parental reconstruction as inactive / non-productive / unambiguous /
        ambiguous.
  * ``path_finding.py``
        builds the 2D-DAG over a spectrum and returns the longest source->sink
        path(s), plus the gapped-reconstruction / correctness helpers.

and implements the full algorithm written up in ``reconstruct.md``.


The algorithm
=============

1.  **Detect lines** (``detect_lines_oa``) and resolve every FFC's per-side
    charge from the line it sits on, giving each FFC a pair of singly-
    protonated masses ``(X, Y)``.

2.  **Align every line against the parental line.**  For a TFFC on a line, the
    alignment shift in ion-mass coordinates is

        A = ParentPair - (X + Y)          ParentPair = M + 2*proton

    ``A`` is estimated per line as the *median* over its member FFCs, which is
    more accurate than the line-detection cluster mean (that one also carries a
    ``(i+j-z)*proton`` offset whenever the line's charges do not sum to the
    precursor charge -- e.g. the VEA "shift=229.118" line really aligns by
    mass(VE)=228.111).

3.  **Keep only the line types we currently understand** -- parental
    (``A ~ 0``), parental with **one neutral loss** (``A ~ H2O / NH3 / CO``),
    and **pure internal** (``A >= smallest residue mass``, and ``A`` is a
    plausible residue-run mass).  Everything else is skipped: positive shifts,
    internal+loss combinations that would need the 2D ``(shift1, shift2)``
    alignment of ``reconstruct.md`` section 9, junk lines, and +1/+2 **isotope
    satellites** of a stronger line (``mark_isotope_satellites``).

4.  **Gate on ``inactive_count``.**  A line is only used if at least
    ``inactive_count`` (default 1) of its TFFCs are *inactive*, i.e. their
    shift-image already coincides with a TFFC on the master set.  Inactive
    TFFCs prove the line really is aligned to the parental line; a line with
    none of them is unanchored and is dropped.

5.  **Extend the master set** (``reconstruct.md`` section 6).  For each kept
    line, repeatedly:
      - classify each remaining TFFC ``(X, Y)`` by how many of its two virtual
        images ``(X+A, Y)`` and ``(X, Y+A)`` are *good* (a virtual TFFC is good
        iff adding it strictly increases the weight of the heaviest path);
      - add the good image of an *unambiguous* TFFC (weight 1);
      - if none is unambiguous, add both images of an *ambiguous* TFFC, each
        with weight 1/2;
      - stop when only inactive / non-productive TFFCs remain.
    A **mock TFFC** is appended to every internal line whose ``A`` is at least
    the smallest residue mass: it is the breakpoint the line's own shift names
    (the fragment the line lost), and it needs no alignment decision.

6.  **Reconstruct**: build the 2D-DAG on the final master set and report every
    heaviest path as a gapped N->C reconstruction, with a correctness check
    against the known sequence.

7.  **Blame the failure** (``diagnose_failure``, a ground-truth debugging aid):
    if none of the heaviest paths is correct, find the master TFFCs that are
    out-scoring the truth and trace each back to the FFC and the line that put
    it there.  A path built only from true breakpoints is correct by
    definition, so banning the false coordinates the winners use -- and
    repeating -- always terminates with a "remove these and the truth wins"
    certificate.


Coordinate conventions (read this before touching the maths)
============================================================

Everything lives in ``path_finding``'s *neutral* coordinates: a breakpoint
after residue k is the pair ``(prefix_k, terminus + suffix_k)`` whose two
components always sum to ``total = residue_sum + terminus``.  An observed
singly-protonated mass ``X`` becomes the coordinate ``X - proton``.

Because the two components of a TFFC are complementary, **one number is enough
to describe a TFFC**: we store the *folded* coordinate ``min(r, total - r)``.
This collapses the b/y ambiguity that a single TFFC can never resolve on its
own, and it is exactly the "lower half of the spectrum" that
``path_finding``'s graph construction wants.  The two virtual images of an
internal-line TFFC then reduce to a pleasantly simple statement:

    horizontal image (X+A, Y)  ==  "trust Y"  ==  fold(Y - proton)
    vertical   image (X, Y+A)  ==  "trust X"  ==  fold(X - proton)

so the noisy line shift ``A`` never enters the coordinates of a virtual TFFC
(it is only used to *recognise* the line and to place the mock).

**C-terminus.**  ``path_finding`` hard-codes a water C-terminus in
``SOURCE_NODE = (0.0, 18.01056)`` and mirrors the y-axis through
``total - WATER``; an ``end_h20='NH3'`` peptide (e.g. KWK) is 0.984 Da off on
every y-ion and fails *silently*.  Rather than converting the masses onto a
water basis, this module rebases ``path_finding`` itself by pointing
``pf.SOURCE_NODE`` at ``(0.0, terminus_mass)`` and passing
``total = residue_sum + terminus_mass``.  ``construct_graph_nodes``,
``format_gapped_reconstruction`` and ``is_reconstruction_correct`` all read
that global, so the whole search becomes terminus-agnostic and observed masses
can be used directly, without ever having to decide which side of an FFC is the
b-ion.  Only H2O and NH3 termini are supported (that is all the data has).

The terminus is an **input** (``--terminus``), not something the algorithm
recovers.  It is worth understanding why: every graph coordinate is a *measured*
mass (``X - proton``), and the precursor mass is measured too, so moving the
terminus by 0.984 Da only re-labels how that fixed total splits into
``residues + terminus``.  The one place it touches the search is the source
node's y-coordinate -- i.e. the single "right" move out of the source.  Hence
``--terminus auto`` (evaluate both, keep the heavier path) mostly reports a
*tie*: measured on the three test peptides with ``--parental_source all``, VEA
prefers H2O by 1 unit of path weight while KWK and LLG are exact ties.  It looks
decisive with ``--parental_source annotated`` only because the annotator matches
against theoretical y-ions and therefore already knows the answer.  Treat auto
mode as a diagnostic, and pass the terminus when you know it.


**Line finding decides the answer.**  ``--top_n``, ``--delta`` and ``--delta_unit``
control *which lines exist*, and they matter more than anything downstream.  The
defaults here match ``annotation_pepline/run_annotation_reconstruct.py``
(top 500, 20 ppm, tol 30 ppm); at ``line_internal_annotation_oa_v2``'s own
defaults (0.02 Da / 20 ppm) KWK yields 16 lines instead of 25, the
``A = 128.093`` line that reveals b1 is never detected, and the run stops six
breakpoints short.  Greedy peeling is non-monotonic in ``--top_n``, so a
different cutoff gives *different* lines, not just more of them.

Usage
-----
    python master_reconstruction.py                      # KWK 5+, N=500
    python master_reconstruction.py --top_n 800 --delta 20 --delta_unit ppm
    python master_reconstruction.py --top_n 300 --inactive_count 2
    python master_reconstruction.py --ambiguous_policy skip --require_annot
    python master_reconstruction.py --seq VEADIAGHGQEVLIR --charge 3 \
        --terminus h2o --data /path/to/VEA_merged.tsv --top_n 100 \
        --threshold 0.02 --coord_tol 0.02
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_LBA = os.path.abspath(os.path.join(_HERE, "..", "line_based_annotation"))
_DEISO = os.path.abspath(os.path.join(_HERE, "..", "deiso", "python_code"))

# Import the annotation module FIRST: it puts ``deiso/python_code`` at the head
# of sys.path, so ``peptide`` resolves to the canonical copy there (this
# directory holds a duplicate) and both modules end up sharing one AA table.
sys.path.insert(0, _LBA)
sys.path.insert(0, _DEISO)
import line_internal_annotation_oa_v2 as lba       # noqa: E402
import peptide                                     # noqa: E402
from peptide import AA                             # noqa: E402
from line_finding import prepare_ffc_data          # noqa: E402
from merge_ffcs import merge_duplicate_ffcs_charge  # noqa: E402

sys.path.append(_HERE)
import path_finding as pf                          # noqa: E402
from edges import load_edges                       # noqa: E402

import pandas as pd                                # noqa: E402


# =============================================================================
# 0. CONSTANTS
# =============================================================================

PROTON = AA.element_masses["proton"]          # 1.00725
WATER = AA.element_masses["H2O"]              # 18.01056
NH3 = AA.element_masses["NH3"]                # 17.02655
CO = 27.99491

#: Named single neutral losses a "parental + one neutral loss" line may carry.
NEUTRAL_LOSS_SHIFTS: Dict[str, float] = {"H2O": WATER, "NH3": NH3, "CO": CO}

#: Smallest residue mass (glycine).  A line whose alignment shift is below this
#: cannot be a residue run, and gets no mock TFFC.
MIN_AA_MASS = min(lba.AA.amino_acid_masses[a] for a in "ARNDCEQGHILKMFPSTWYV")

DEFAULT_EDGES = os.path.join(_HERE, "edges.json")
DEFAULT_DATA = "/home/sux002/data/deconvoluted/merged/KWK_37_Z5_E28_2_XIAOWEN_merged.tsv"
DEFAULT_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"

#: Line categories this version of the algorithm knows how to align.
USABLE_CATEGORIES = ("parental", "loss", "internal")


# =============================================================================
# 1. PEPTIDE / COORDINATE HELPERS
# =============================================================================

def terminus_mass(pep) -> float:
    """Mass of the C-terminal group (H2O for a free acid, NH3 for an amide)."""
    if pep.end_h20 is True:
        return WATER
    return AA.element_masses[pep.end_h20]


def residue_prefixes(pep) -> List[float]:
    """``prefix[k]`` = summed residue mass of residues 1..k (``prefix[0] = 0``).

    Derived from ``AA_array`` rather than ``pep.ion_mass`` so the C-terminus
    never enters -- ``pep.seq_mass`` subtracts water unconditionally and is
    wrong by 0.984 Da for an amidated peptide.
    """
    prefix = [0.0]
    for aa in pep.AA_array:
        prefix.append(prefix[-1] + aa.get_mass())
    return prefix


def fold(r: float, total: float) -> float:
    """Fold a breakpoint coordinate into the lower half ``(0, total/2]``.

    A TFFC's two coordinates ``r`` and ``total - r`` describe the same
    breakpoint; which of them is the b-side cannot be decided from one TFFC, so
    we canonicalise on the smaller one.
    """
    return r if r <= total - r else total - r


# =============================================================================
# 2. THE MASTER TFFC-SET
# =============================================================================

@dataclass
class Origin:
    """Where one master TFFC came from -- enough to blame it later (§9).

    ``tffc`` is the FFC (or mock) that proposed the coordinate, so a culprit can
    be traced all the way back to its m/z pair, charges and ranking in the FFC
    map.  ``rejected`` is the *other* image of the same TFFC, i.e. the placement
    the algorithm decided against -- for a wrong unambiguous call that is the
    coordinate it should have picked instead.
    """
    line_key: str
    kind: str = "observed"        # observed | mock | parental
    decision: str = "seed"        # seed | unambiguous | ambiguous
    tffc: Optional["TFFC"] = None
    rejected: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.line_key}:{self.kind}"


@dataclass
class MasterEntry:
    """One TFFC in the master set, as a folded breakpoint coordinate."""
    coord: float
    weight: float = 1.0
    origins: List[Origin] = field(default_factory=list)


class MasterSet:
    """The growing master TFFC-set, stored as folded breakpoint coordinates.

    Two coordinates within ``tol`` are the same TFFC.  Weights follow
    ``reconstruct.md``: 1 for a real or unambiguously placed TFFC, 1/2 for each
    half of an ambiguous pair (the larger weight wins if a coordinate is
    proposed twice).
    """

    def __init__(self, total: float, tol: float):
        self.total = total
        self.tol = tol
        self.entries: List[MasterEntry] = []

    # ── lookup ────────────────────────────────────────────────────────────
    def find(self, r: float) -> Optional[MasterEntry]:
        c = fold(r, self.total)
        for e in self.entries:
            if abs(e.coord - c) <= self.tol:
                return e
        return None

    def __contains__(self, r: float) -> bool:
        return self.find(r) is not None

    def __len__(self) -> int:
        return len(self.entries)

    # ── mutation ──────────────────────────────────────────────────────────
    def add(self, r: float, weight: float = 1.0,
            origin: Optional[Origin] = None) -> bool:
        """Add breakpoint ``r``.  Returns True if it was genuinely new."""
        c = fold(r, self.total)
        if c <= self.tol or c >= self.total - self.tol:
            return False                      # a terminus, not a breakpoint
        hit = self.find(c)
        if hit is not None:
            hit.weight = max(hit.weight, weight)
            if origin:
                hit.origins.append(origin)
            return False
        self.entries.append(MasterEntry(c, weight, [origin] if origin else []))
        return True

    def snapshot(self) -> "MasterSet":
        """Frozen copy -- used as the fixed reference for the inactive test."""
        copy = MasterSet(self.total, self.tol)
        copy.entries = [MasterEntry(e.coord, e.weight, list(e.origins))
                        for e in self.entries]
        return copy

    # ── views ─────────────────────────────────────────────────────────────
    def coords(self, extra: Optional[Sequence[float]] = None) -> List[float]:
        """Sorted folded coordinates, optionally with candidate coords mixed in."""
        vals = [e.coord for e in self.entries]
        if extra:
            for r in extra:
                c = fold(r, self.total)
                if self.tol < c < self.total - self.tol:
                    vals.append(c)
        return sorted(vals)

    def weight_map(self, extra_weights: Optional[Dict[float, float]] = None) -> Dict[float, float]:
        wm = {e.coord: e.weight for e in self.entries}
        if extra_weights:
            for c, w in extra_weights.items():
                wm[fold(c, self.total)] = w
        return wm


# =============================================================================
# 3. 2D-DAG CONSTRUCTION AND WEIGHTED HEAVIEST PATH
# =============================================================================
# ``build_dag`` is a lean re-implementation of
# ``path_finding.build_adjacency_graph``: same edge semantics, but it takes the
# allowed-mass list **already sorted** instead of re-sorting 2.1 M floats (and
# re-allocating a numpy copy of them) on every call.  The goodness test calls
# this hundreds of times, so that matters.
#
# ``find_all_heaviest_paths`` generalises
# ``path_finding.find_all_longest_paths`` from "count the edges" to "sum the
# weights of the TFFCs the path steps onto".  With every weight at 1 the two
# agree exactly; fractional weights only appear for ambiguously-placed virtual
# TFFCs (1/2 each, per reconstruct.md).

SINK = pf.SINK_NODE


def build_dag(spectrum: Sequence[float], sorted_allowed: Sequence[float],
              total_mass: float, source: Tuple[float, float],
              threshold: float = 0.02) -> dict:
    """Directed acyclic graph over the 2D grid spanned by *spectrum*.

    Parameters
    ----------
    spectrum : sequence of float
        Folded breakpoint coordinates (the "lower half" of the 2D spectrum).
    sorted_allowed : sequence of float
        Compound residue masses (``edges.json``), ascending.
    total_mass : float
        ``residue_sum + terminus``; a node's remainder to the sink must be an
        allowed compound mass.
    source : (float, float)
        ``(0.0, terminus)`` -- injected into the grid so the search has a start.
    threshold : float
        Mass tolerance (Da) for an edge.
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

    def allowed(gap: float) -> Optional[float]:
        if gap < min_edge - threshold or gap > max_edge + threshold:
            return None
        idx = bisect_left(sorted_allowed, gap)
        if idx < len(sorted_allowed) and abs(sorted_allowed[idx] - gap) <= threshold:
            return sorted_allowed[idx]
        if idx > 0 and abs(sorted_allowed[idx - 1] - gap) <= threshold:
            return sorted_allowed[idx - 1]
        return None

    def coords_above(val: float) -> Sequence[float]:
        return coords[bisect_left(coords, val + min_edge - threshold):]

    for node in nodes:
        x1, x2 = node
        out = graph[node]
        max_current = max(x1, x2)

        remainder = total_mass - x1 - x2
        if remainder > 0:
            m = allowed(remainder)
            if m is not None:
                out.append((SINK, m, "sink"))

        for t1 in coords_above(x1):
            gap = t1 - x1
            if gap > max_coord - x1 + threshold:
                break
            m = allowed(gap)
            if m is not None:
                tgt = (t1, x2)
                if tgt in node_set and max_current < max(t1, x2):
                    out.append((tgt, m, "down"))

        for t2 in coords_above(x2):
            gap = t2 - x2
            if gap > max_coord - x2 + threshold:
                break
            m = allowed(gap)
            if m is not None:
                tgt = (x1, t2)
                if tgt in node_set and max_current < max(x1, t2):
                    out.append((tgt, m, "right"))

    return graph


def find_all_heaviest_paths(graph: dict, weights: Dict[float, float],
                            start_node, sink_node=SINK, max_paths: int = 1000,
                            eps: float = 1e-9):
    """All maximum-weight source->sink paths.

    A path's weight is the sum, over its moves, of the weight of the *newly
    reached* coordinate (``weights.get(coord, 1.0)``), plus 1 for the closing
    sink edge.  With uniform weights this is precisely the edge count that
    ``path_finding.find_all_longest_paths`` maximises.

    Returns ``(best_weight, [(mass_seq, node_seq), ...])`` with
    ``best_weight = -inf`` when no path reaches the sink.
    """
    NEG = float("-inf")
    memo: Dict = {}

    def edge_weight(neighbor, edge_type) -> float:
        if edge_type == "sink":
            return 1.0
        coord = neighbor[0] if edge_type == "down" else neighbor[1]
        return weights.get(coord, 1.0)

    def dfs(node):
        if node == sink_node:
            return (0.0, [([], [sink_node])])
        if node in memo:
            return memo[node]
        if node not in graph or not graph[node]:
            memo[node] = (NEG, [])
            return memo[node]

        best, best_paths = NEG, []
        for neighbor, mass, etype in graph[node]:
            nb_score, nb_paths = dfs(neighbor)
            if nb_score == NEG:
                continue
            cur = nb_score + edge_weight(neighbor, etype)
            if cur > best + eps:
                best, best_paths = cur, []
            elif cur < best - eps or len(best_paths) >= max_paths:
                continue
            for m_seq, n_seq in nb_paths:
                best_paths.append(([mass] + m_seq, [node] + n_seq))
                if len(best_paths) >= max_paths:
                    break

        memo[node] = (best, best_paths) if best > NEG else (NEG, [])
        return memo[node]

    return dfs(start_node)


class PathScorer:
    """Heaviest-path evaluator with a cache over coordinate sets.

    The goodness test asks the same question (``does adding coordinate c to
    this master set lengthen the heaviest path?``) hundreds of times, and two
    different TFFCs frequently propose the same breakpoint, so caching on the
    rounded coordinate set pays for itself.
    """

    def __init__(self, sorted_allowed, total, source, threshold, max_paths=1000):
        self.allowed = sorted_allowed
        self.total = total
        self.source = source
        self.threshold = threshold
        self.max_paths = max_paths
        self._cache: Dict[Tuple, float] = {}
        self.n_builds = 0

    def _key(self, coords, weights):
        return (tuple(round(c, 4) for c in coords),
                tuple(round(weights.get(c, 1.0), 3) for c in coords))

    def score(self, coords: Sequence[float], weights: Dict[float, float]) -> float:
        key = self._key(coords, weights)
        if key in self._cache:
            return self._cache[key]
        self.n_builds += 1
        graph = build_dag(coords, self.allowed, self.total, self.source,
                          self.threshold)
        best, _ = find_all_heaviest_paths(graph, weights, self.source,
                                          max_paths=1)
        val = -1.0 if best == float("-inf") else best
        self._cache[key] = val
        return val

    def paths(self, coords: Sequence[float], weights: Dict[float, float]):
        graph = build_dag(coords, self.allowed, self.total, self.source,
                          self.threshold)
        return find_all_heaviest_paths(graph, weights, self.source,
                                       max_paths=self.max_paths)


# =============================================================================
# 4. LINE RECORDS -- what every detected line is, and what TFFCs it carries
# =============================================================================

@dataclass
class TFFC:
    """One TFFC (observed FFC, or a virtual mock) on a line.

    x, y      : singly-protonated ion masses of the two sides.
    ch_x/ch_y : per-side charges recovered from the line.
    a_shift   : ``ParentPair - (x + y)`` -- this FFC's own alignment shift.
    kind      : 'observed' or 'mock'.
    category  : the annotator's category for this FFC (diagnostic only).
    """
    x: float
    y: float
    ch_x: int = 1
    ch_y: int = 1
    a_shift: float = 0.0
    kind: str = "observed"
    ranking: int = -1
    category: str = ""
    label: str = ""

    def images(self, total: float) -> Tuple[float, float]:
        """The two virtual-TFFC breakpoint coordinates (horizontal, vertical).

        Horizontal ``(X+A, Y)`` keeps the measured Y, vertical ``(X, Y+A)``
        keeps the measured X -- see the module docstring.
        """
        return (fold(self.y - PROTON, total), fold(self.x - PROTON, total))


@dataclass
class LineRec:
    """A detected line plus everything the reconstruction needs from it."""
    key: str
    i: int
    j: int
    mass: float
    shift: float                  # ln.mass - parent_mass (detection convention)
    a_shift: float                # alignment shift in ion-mass coordinates
    a_from_detection: float       # -shift + (i+j-z)*proton, for comparison
    tffcs: List[TFFC] = field(default_factory=list)
    category: str = "other"       # parental | loss | internal | other
    detail: str = ""              # e.g. 'NH3' for a loss line
    annot_type: Optional[str] = None
    n_inactive: int = 0
    used: bool = False
    reason: str = ""

    @property
    def n_ffc(self) -> int:
        return sum(1 for t in self.tffcs if t.kind == "observed")


def classify_line(a_shift: float, allowed_set, parent_tol: float = 0.5,
                  loss_tol: float = 0.05, edge_tol: float = 0.05
                  ) -> Tuple[str, str]:
    """Classify a line from its alignment shift alone (no ground truth).

    Returns ``(category, detail)`` with category in
    ``parental`` / ``loss`` / ``internal`` / ``other``.

    * ``|A| < parent_tol``                       -> parental line
    * ``A`` matches one named neutral loss        -> parental + one loss
    * ``A >= MIN_AA_MASS`` and ``A`` is a plausible compound residue mass
      (present in ``edges.json``)                 -> pure internal line
    * anything else (negative A, A between a loss and a residue, an internal
      line that *also* carries a neutral loss and would need a 2D
      ``(shift1, shift2)`` alignment) -> other, and it is skipped.
    """
    if abs(a_shift) < parent_tol:
        return "parental", "parent"
    for name, m in NEUTRAL_LOSS_SHIFTS.items():
        if abs(a_shift - m) <= loss_tol:
            return "loss", name
    if a_shift >= MIN_AA_MASS - edge_tol:
        idx = bisect_left(allowed_set, a_shift)
        for k in (idx - 1, idx):
            if 0 <= k < len(allowed_set) and abs(allowed_set[k] - a_shift) <= edge_tol:
                return "internal", f"{a_shift:.3f}"
        # A residue run heavier than the pre-computed table still counts.
        if a_shift > allowed_set[-1]:
            return "internal", f"{a_shift:.3f} (beyond edge table)"
        return "other", f"{a_shift:.3f} not a residue-run mass"
    return "other", f"{a_shift:.3f}"


#: 13C - 12C.  An FFC built on the first isotope peak of a fragment weighs this
#: much more, so the *line* it forms sits one notch lower in alignment shift.
ISOTOPE_OFFSET = 1.00335


def mark_isotope_satellites(records: List["LineRec"], tol: float = 0.02,
                            max_k: int = 2) -> None:
    """Demote internal lines that are isotope satellites of a stronger line.

    If a fragment on a line is picked up at its ``+1`` isotope peak, the whole
    line reappears at ``A - 1.00335``: same TFFCs, every coordinate 1 Da off.
    Shifting by that alignment produces virtual TFFCs that are a clean isotope
    spacing away from the truth -- close enough to look plausible, far enough to
    be wrong -- so these lines are false-positive factories.  (Measured on
    LLG 6+: the ``A=225.160`` satellite of the ``A=226.164`` line was the single
    reason the correct reconstruction was lost.)

    A line is a satellite when its ``A`` sits ``k * 1.00335`` *below* that of a
    line with at least as many FFCs.  The direction matters and makes the test
    specific: no pair of residue-run masses differs by 1.003 Da.
    """
    order = sorted(records, key=lambda r: (-r.n_ffc, r.key))
    for rec in records:
        if rec.category != "internal":
            continue
        for other in order:
            if other is rec or other.n_ffc < rec.n_ffc:
                continue
            for k in range(1, max_k + 1):
                if abs(other.a_shift - rec.a_shift - k * ISOTOPE_OFFSET) <= tol:
                    rec.category = "other"
                    rec.detail = (f"+{k} isotope satellite of {other.key} "
                                  f"(A={other.a_shift:.3f})")
                    break
            else:
                continue
            break                       # this line is settled; on to the next


def _median(vals: Sequence[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build_line_records(ffc_df, lines, pep, parent_mass: float,
                       parent_charge: int, parent_pair: float,
                       allowed_sorted, annotate: bool = True,
                       iso_range: int = 1, tol_ppm: float = 20.0,
                       consider_secondary: bool = True,
                       col_a: str = "m/z A", col_b: str = "m/z B",
                       ranking_col: str = "Ranking") -> List[LineRec]:
    """Turn detected ``Line`` objects into :class:`LineRec`s with their TFFCs.

    Each member FFC's orientation is resolved exactly as in
    ``coverage_table_internal`` (whichever of ``i*mzA + j*mzB`` /
    ``j*mzA + i*mzB`` lands closer to the line mass), then each side is
    converted to its singly-protonated mass.  ``annotate=True`` additionally
    runs the annotator's category classifier on every FFC -- used only for the
    diagnostic report and for the optional ``--parental_source annotated``
    filter, never to place a virtual TFFC.
    """
    mz_grid = frag_grid = iso_grid = None
    if annotate:
        mz_grid, frag_grid, iso_grid, _ = lba.build_fragment_index(pep, iso_range)

    has_ranking = ranking_col in ffc_df.columns
    records: List[LineRec] = []

    for ln in lines:
        if ln.member_indices.size == 0:
            continue
        shift = ln.mass - parent_mass
        key = lba._col_key(ln.i, ln.j, round(shift, 3), 3)
        rec = LineRec(
            key=key, i=ln.i, j=ln.j, mass=ln.mass, shift=shift,
            a_shift=0.0,
            a_from_detection=-shift + (ln.i + ln.j - parent_charge) * PROTON,
        )

        cats: Counter = Counter()
        for idx in ln.member_indices:
            if idx not in ffc_df.index:
                continue
            row = ffc_df.loc[idx]
            mz_a, mz_b = float(row[col_a]), float(row[col_b])
            ranking = int(row[ranking_col]) if has_ranking else -1

            if abs((ln.j * mz_a + ln.i * mz_b) - ln.mass) < \
               abs((ln.i * mz_a + ln.j * mz_b) - ln.mass):
                iu, ju = ln.j, ln.i
            else:
                iu, ju = ln.i, ln.j

            # Singly-protonated masses.  We use the proton mass here (not the
            # H-atom mass the annotator uses for its ppm matching) so that the
            # complementarity X + Y = ParentPair - A holds exactly.
            x = iu * mz_a - (iu - 1) * PROTON
            y = ju * mz_b - (ju - 1) * PROTON

            t = TFFC(x=x, y=y, ch_x=iu, ch_y=ju,
                     a_shift=parent_pair - (x + y), ranking=ranking)
            if annotate:
                ca = lba._candidates(x, mz_grid, frag_grid, iso_grid, tol_ppm)
                cb = lba._candidates(y, mz_grid, frag_grid, iso_grid, tol_ppm)
                t.category = lba._classify_ffc(ca, cb, consider_secondary)
                cats[t.category] += 1
                t.label = (f"{lba._side_best_label(ca, len(pep.AA_array), mz_a)} | "
                           f"{lba._side_best_label(cb, len(pep.AA_array), mz_b)}")
            rec.tffcs.append(t)

        if not rec.tffcs:
            continue
        rec.a_shift = _median([t.a_shift for t in rec.tffcs])
        rec.category, rec.detail = classify_line(rec.a_shift, allowed_sorted)
        if annotate:
            rec.annot_type = lba._most_common_category(cats)
        records.append(rec)

    return records


def add_mock(rec: LineRec, parent_pair: float, tol: float = 1e-9) -> Optional[TFFC]:
    """Append the line's **mock TFFC** and return it (None if not applicable).

    The mock is the TFFC named by the line's own shift: the line is missing a
    fragment of mass ``A``, so the cleavage that produced it is a genuine
    breakpoint at ``A``.  It needs no horizontal/vertical decision -- both of
    its images collapse onto the same breakpoint -- and it is only meaningful
    when ``A`` is at least one residue (a neutral-loss line's shift is not a
    cleavage).  Note the lost fragment may be N-terminal *or* C-terminal; the
    folded coordinate covers both cases.
    """
    if rec.category != "internal" or rec.a_shift < MIN_AA_MASS - 0.05:
        return None
    mock = TFFC(x=rec.a_shift + PROTON, y=parent_pair - rec.a_shift - PROTON,
                a_shift=0.0, kind="mock", label=f"mock@{rec.a_shift:.3f}")
    rec.tffcs.append(mock)
    return mock


# =============================================================================
# 5. THE EXTENSION ALGORITHM (reconstruct.md, section 6)
# =============================================================================

CLASS_INACTIVE = "inactive"
CLASS_NON_PRODUCTIVE = "non_productive"
CLASS_UNAMBIGUOUS = "unambiguous"
CLASS_AMBIGUOUS = "ambiguous"


def count_inactive(rec: LineRec, master: MasterSet) -> int:
    """How many of the line's TFFCs already land on the master set.

    This is the ``inactive_count`` gate: an inactive TFFC adds nothing new, but
    it *proves* the line is genuinely aligned against the parental line, which
    is exactly what makes its active TFFCs trustworthy.
    """
    n = 0
    for t in rec.tffcs:
        if t.kind != "observed":
            continue
        h, v = t.images(master.total)
        if (h in master) or (v in master):
            n += 1
    return n


def classify_tffc(t: TFFC, master: MasterSet, scorer: PathScorer,
                  base_score: float, reference: Optional[MasterSet] = None):
    """Classify one TFFC against the current master set.

    Returns ``(classification, good_images, scores)`` where ``good_images`` is
    the list of breakpoint coordinates worth adding.

    ``reference`` is the set the *inactive* test is made against; it defaults to
    ``master`` itself (the TFFC is inactive if either image coincides with
    anything already in the growing set).  Passing the frozen **parental** seed
    set instead (``--inactive_vs parental``) breaks a chain reaction: with the
    growing set as reference, a TFFC processed later is silenced by whatever an
    earlier TFFC happened to add, including a *speculative* half of an ambiguous
    pair -- so getting an earlier call right consumes the evidence, and a branch
    that got it wrong is left with more to score.  Judging every TFFC against
    the same fixed, non-speculative reference gives them all the same chance.
    Goodness is always tested against the real (growing) ``master``, so a TFFC
    whose image is already present simply scores no gain and is non-productive.
    """
    total = master.total
    ref = master if reference is None else reference
    h, v = t.images(total)

    if t.kind == "mock":
        # A mock is a single proposed breakpoint; there is nothing to
        # disambiguate (both images name the same cleavage).
        if h in ref:
            return CLASS_INACTIVE, [], {}
        s = scorer.score(master.coords([h]), master.weight_map())
        if s > base_score + 1e-9:
            return CLASS_UNAMBIGUOUS, [h], {"mock": s}
        return CLASS_NON_PRODUCTIVE, [], {"mock": s}

    h_on, v_on = (h in ref), (v in ref)
    if h_on or v_on:
        return CLASS_INACTIVE, [], {}

    s_h = scorer.score(master.coords([h]), master.weight_map())
    s_v = scorer.score(master.coords([v]), master.weight_map())
    good_h = s_h > base_score + 1e-9
    good_v = s_v > base_score + 1e-9
    scores = {"h": s_h, "v": s_v}

    if good_h and good_v:
        return CLASS_AMBIGUOUS, [h, v], scores
    if good_h:
        return CLASS_UNAMBIGUOUS, [h], scores
    if good_v:
        return CLASS_UNAMBIGUOUS, [v], scores
    return CLASS_NON_PRODUCTIVE, [], scores


def extend_with_line(rec: LineRec, master: MasterSet, scorer: PathScorer,
                     ambiguous_weight: float = 0.5, ambiguous_policy: str = "both",
                     max_rounds: int = 50, reference: Optional[MasterSet] = None,
                     log=print) -> int:
    """Run the iterative extension for one line.  Returns #TFFCs added.

    Per ``reconstruct.md``: prefer an unambiguous TFFC, fall back to an
    ambiguous one (both images, half weight each), re-classify everything after
    every addition (the master set has changed, so a TFFC that was
    non-productive a moment ago may have become good), and stop once only
    inactive / non-productive TFFCs are left.

    ``ambiguous_policy='skip'`` refuses the fallback and keeps only the
    unambiguous TFFCs.  An ambiguous TFFC contributes one true and one false
    breakpoint by construction -- that is what the 1/2 weight is for -- so
    skipping trades coverage for a cleaner master set.
    """
    remaining = list(rec.tffcs)
    added = 0

    for _ in range(max_rounds):
        base = scorer.score(master.coords(), master.weight_map())
        classified = []
        for t in remaining:
            cls, imgs, scores = classify_tffc(t, master, scorer, base, reference)
            classified.append((t, cls, imgs, scores))

        pick = next((c for c in classified if c[1] == CLASS_UNAMBIGUOUS), None)
        if pick is None and ambiguous_policy != "skip":
            pick = next((c for c in classified if c[1] == CLASS_AMBIGUOUS), None)
        if pick is None:
            for t, cls, _, sc in classified:
                log(f"      {_tffc_str(t)}  {cls}"
                    + (f"  scores={_fmt_scores(sc)}" if sc else ""))
            break

        t, cls, imgs, scores = pick
        weight = 1.0 if cls == CLASS_UNAMBIGUOUS else ambiguous_weight
        both = t.images(master.total)
        marks = []
        for c in imgs:
            # The image we did NOT take, recorded so a wrong call can be shown
            # next to the alternative it beat (§9).
            other = next((o for o in both if abs(o - c) > master.tol), None)
            new = master.add(c, weight=weight,
                             origin=Origin(rec.key, t.kind, cls, t, other))
            marks.append(f"{c:.3f}{'' if new else ' (dup)'}")
            added += int(new)
        log(f"      {_tffc_str(t)}  {cls.upper()}  scores={_fmt_scores(scores)}"
            f"  -> add {', '.join(marks)} (w={weight})")
        # Retire (X, Y) *and* its mirror (Y, X): the FFC map is symmetric, so
        # the same TFFC can appear twice with the sides swapped.
        remaining = [
            o for o in remaining
            if o is not t
            and not (abs(o.x - t.y) <= master.tol and abs(o.y - t.x) <= master.tol)
        ]

    return added


def _tffc_str(t: TFFC) -> str:
    tag = "mock" if t.kind == "mock" else f"({t.x:.3f}, {t.y:.3f})"
    extra = f" [{t.category}]" if t.category else ""
    return f"{tag:<26}{extra}"


def _fmt_scores(scores: Dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}:{v:g}" for k, v in scores.items()) + "}"


# =============================================================================
# 6. SEEDING THE MASTER SET FROM THE PARENTAL LINES
# =============================================================================

def seed_master(records: List[LineRec], master: MasterSet, pep,
                source: str = "annotated", prefix: Optional[List[float]] = None,
                iso_range: int = 1, tol_ppm: float = 20.0,
                consider_secondary: bool = True, log=print) -> None:
    """Fill the master set with the TFFCs carried by the parental lines.

    ``source``:
      * ``all``         -- every FFC on a parental line (fully de novo; the
                           spurious ones are in there too).
      * ``annotated``   -- only FFCs the annotator resolved to a valid pair,
                           using their *measured* masses (default).
      * ``theoretical`` -- as ``annotated``, but each accepted FFC contributes
                           the exact ``prefix[k]`` of the bond it resolved to.
                           Reproduces ``line_internal_annotation_oa_v2``'s
                           ``#reconstructions`` baseline.
    """
    if source == "theoretical":
        mz_grid, frag_grid, iso_grid, _ = lba.build_fragment_index(pep, iso_range)
        n = len(pep.AA_array)

    n_used = 0
    for rec in records:
        if rec.category != "parental":
            continue
        for t in rec.tffcs:
            if t.kind != "observed":
                continue
            if source != "all" and t.category in ("unmatched", "unexplained"):
                continue
            if source == "theoretical":
                ca = lba._candidates(t.x, mz_grid, frag_grid, iso_grid, tol_ppm)
                cb = lba._candidates(t.y, mz_grid, frag_grid, iso_grid, tol_ppm)
                pair = lba._best_pair(ca, cb, consider_secondary)
                if pair is None:
                    continue
                left = min(pair, key=lambda c: c.frag.s)
                bond = left.frag.e
                if not 0 < bond < n:
                    continue
                master.add(prefix[bond], 1.0,
                           origin=Origin(rec.key, "parental", "seed", t))
            else:
                master.add(t.x - PROTON, 1.0,
                           origin=Origin(rec.key, "parental", "seed", t))
            n_used += 1
    log(f"  seeded from {source} parental FFCs: {n_used} FFCs -> "
        f"{len(master)} distinct TFFCs")


# =============================================================================
# 7. REPORTING
# =============================================================================

def coverage_report(master: MasterSet, prefix: List[float], tol: float
                    ) -> Tuple[List[int], int, int]:
    """Map the master set onto true cleavage sites.

    Returns ``(covered_bonds, n_false, n_false_half)`` -- the sorted list of
    true bonds a master TFFC lands on, how many master TFFCs match no true bond
    at all, and how many of *those* were added at fractional weight.  The split
    matters: an ambiguous TFFC deliberately contributes one true and one false
    breakpoint at weight 1/2 each, so a half-weight false TFFC is the algorithm
    working as designed, while a full-weight one is a genuine mistake.
    """
    total = master.total
    n = len(prefix) - 1
    true_folded = {k: fold(prefix[k], total) for k in range(1, n)}
    covered, n_false, n_half = set(), 0, 0
    for e in master.entries:
        hit = [k for k, c in true_folded.items() if abs(c - e.coord) <= tol]
        if hit:
            covered.update(hit)
        else:
            n_false += 1
            n_half += int(e.weight < 1.0)
    return sorted(covered), n_false, n_half


def format_gapped(path, total: float, term: float, merge_tol: float,
                  label_tol: float) -> str:
    """Gapped N->C reconstruction of one path, with two separate tolerances.

    Same rendering as ``path_finding.format_gapped_reconstruction`` -- collapse
    the b-side and (mirrored) y-side breakpoints into one N->C ladder, then
    print each segment as a residue letter or as its mass -- except that the
    *breakpoint merging* and the *residue naming* get their own tolerances.
    Naming has to be the tighter of the two: with a 0.03 Da window a measured
    gap can sit closer to the wrong residue (K and Q are only 0.036 Da apart),
    and a confidently wrong letter is worse than an honest number.  The
    ``(sink)`` marker flags the one gap the two ladders never bridged.
    """
    if not path:
        return ""
    aa = pf.amino_acid_masses_merge
    m_res = total - term
    b_side = [x1 for x1, _ in path]
    y_side = [m_res - (x2 - term) for _, x2 in path]
    sink_prev, sink_next = max(b_side), min(y_side)
    breakpoints = pf._merge_close([0.0, m_res] + b_side + y_side, merge_tol)

    tokens, run = [], ""
    for prev, nxt in zip(breakpoints[:-1], breakpoints[1:]):
        gap = nxt - prev
        is_sink = (abs(prev - sink_prev) <= merge_tol
                   and abs(nxt - sink_next) <= merge_tol)
        best = min(aa.items(), key=lambda kv: abs(kv[1] - gap))
        letter = best[0] if abs(best[1] - gap) <= label_tol else None
        if letter:
            run += f"{letter}(sink)" if is_sink else letter
        else:
            if run:
                tokens.append(run)
                run = ""
            tokens.append(f"{gap:.3f}" + (" (sink)" if is_sink else ""))
    if run:
        tokens.append(run)
    return " - ".join(tokens)


def report_paths(scorer: PathScorer, master: MasterSet, pep_seq: str,
                 total: float, term: float, threshold: float,
                 label_tol: Optional[float] = None, log=print) -> Tuple[int, int]:
    """Print every heaviest path as a gapped reconstruction.  Returns (n, n_correct).

    Correctness is judged on *breakpoints* (``is_reconstruction_correct``),
    never on the printed letters.
    """
    if label_tol is None:
        label_tol = threshold
    best, paths = scorer.paths(master.coords(), master.weight_map())
    if best == float("-inf") or not paths:
        log("  no path reaches the sink")
        return 0, 0, 0.0
    log(f"  heaviest-path weight : {best:g}")
    log(f"  tied heaviest paths  : {len(paths)}")
    n_correct = 0
    for i, (_, node_seq) in enumerate(paths, start=1):
        stripped = node_seq[:-1]
        gapped = format_gapped(stripped, total, term, threshold, label_tol)
        ok = pf.is_reconstruction_correct(stripped, total, pep_seq,
                                          tolerance=threshold)
        n_correct += int(ok)
        if i <= 20:
            log(f"    [{i}] {gapped}{'   [CORRECT]' if ok else ''}")
    if len(paths) > 20:
        log(f"    ... {len(paths) - 20} more")
    log(f"  correct reconstructions: {n_correct}/{len(paths)}")
    return len(paths), n_correct, best


# =============================================================================
# 7a. WHY DID THE CORRECT RECONSTRUCTION LOSE?
# =============================================================================
# A ground-truth *debugging* aid, not part of the algorithm: when none of the
# heaviest paths is correct, some false master TFFC is carrying a heavier path
# than the true breakpoints can. This finds which ones, and traces each back to
# the FFC and the line that put it there.
#
# The search is a greedy peel: take the current heaviest paths, ban every false
# coordinate they use, repeat. It always terminates with a correct answer,
# because a path built only from true breakpoints is correct by definition -- so
# the banned set is a genuine "remove these and the truth wins" certificate,
# and each round's bans are exactly the TFFCs that were out-scoring it.


def _path_coords(node_seq, source: Tuple[float, float]) -> set:
    """Every spectrum coordinate a path touches, minus the source's own two."""
    used = {c for node in node_seq if node != SINK for c in node}
    return used - {source[0], source[1]}


def diagnose_failure(scorer: PathScorer, master: MasterSet, prefix: List[float],
                     total: float, term: float, pep_seq: str, tol: float,
                     threshold: float, label_tol: float,
                     max_rounds: int = 20, log=print) -> dict:
    """Report the master TFFCs that keep the correct reconstruction off the top.

    Returns ``{"blockers": [MasterEntry], "best_correct_weight": float,
    "restored": (n_paths, n_correct)}``.
    """
    n = len(prefix) - 1
    true_folded = [fold(prefix[k], total) for k in range(1, n)]

    def is_true(c: float) -> bool:
        return any(abs(c - t) <= tol for t in true_folded)

    by_coord = {e.coord: e for e in master.entries}
    weights = master.weight_map()
    false_entries = [e for e in master.entries if not is_true(e.coord)]

    log("\n" + "=" * 78)
    log("WHY THE CORRECT RECONSTRUCTION LOST  (ground-truth diagnostic)")
    log("=" * 78)
    if not false_entries:
        log("  every master TFFC is a true breakpoint -- nothing to blame.  The "
            "correct\n  path must be losing to a tie cap or a sink/edge "
            "tolerance issue, not to a\n  spurious TFFC.")
        return {"blockers": [], "best_correct_weight": 0.0, "restored": (0, 0)}

    # ── What is the best the truth can do on its own? ─────────────────────
    true_coords = sorted(e.coord for e in master.entries if is_true(e.coord))
    w_true, paths_true = scorer.paths(true_coords, weights)
    if w_true == float("-inf"):
        log("  no path at all can be built from the true breakpoints alone.")
        return {"blockers": [], "best_correct_weight": 0.0, "restored": (0, 0)}
    log(f"  best correct path weight : {w_true:g}   "
        f"(vs {scorer.score(master.coords(), weights):g} for the winner)")
    log(f"  {format_gapped(paths_true[0][1][:-1], total, term, threshold, label_tol)}")

    # ── Greedy peel: ban the false coordinates the winners rely on ────────
    coords = master.coords()
    banned: List[float] = []
    rounds: List[Tuple[int, List[float]]] = []
    restored = (0, 0)
    for rnd in range(1, max_rounds + 1):
        best, paths = scorer.paths(coords, weights)
        if best == float("-inf") or not paths:
            break
        n_ok = sum(1 for _, ns in paths
                   if pf.is_reconstruction_correct(ns[:-1], total, pep_seq,
                                                   tolerance=threshold))
        if n_ok:
            restored = (len(paths), n_ok)
            break
        used_false = sorted({c for _, ns in paths
                             for c in _path_coords(ns, scorer.source)
                             if c in by_coord and not is_true(c)})
        if not used_false:
            break
        rounds.append((rnd, used_false))
        banned.extend(used_false)
        coords = [c for c in coords if c not in set(used_false)]

    if not banned:
        log("  the winning paths use no false TFFC -- the correct path is being "
            "lost to\n  something else (tie cap, edge tolerance, or a "
            "breakpoint that is off by\n  more than --coord_tol).")
        return {"blockers": [], "best_correct_weight": w_true, "restored": restored}

    def bond_of(c: float, factor: float = 1.0) -> Optional[Tuple[int, float]]:
        """Nearest true bond within ``factor * tol``, as ``(k, signed error)``."""
        best = min(enumerate(true_folded, start=1),
                   key=lambda kt: abs(c - kt[1]))
        return (best[0], c - best[1]) if abs(c - best[1]) <= factor * tol else None

    log(f"\n  {len(banned)} master TFFC(s) block it. Removing them: "
        f"{restored[1]}/{restored[0]} correct.")
    for rnd, group in rounds:
        log(f"\n  -- round {rnd}: out-scoring the truth by "
            f"{len(group)} TFFC(s) --")
        for c in group:
            _log_blocker(by_coord[c], bond_of, log)

    log("\n  Lines involved: "
        + ", ".join(sorted({o.line_key for c in banned
                            for o in by_coord[c].origins})))
    return {"blockers": [by_coord[c] for c in banned],
            "best_correct_weight": w_true, "restored": restored}


def _near(coord: float, bond_of, factor: float = 4.0) -> str:
    """Flag a coordinate that is a real bond just outside ``--coord_tol``.

    Such a TFFC is not spurious at all -- it is the right breakpoint carried by
    a mass that drifted further than the tolerance allows, which is a
    ``--threshold`` / ``--coord_tol`` problem rather than a bad line.
    """
    if bond_of(coord):
        return ""
    hit = bond_of(coord, factor)
    if not hit:
        return ""
    k, err = hit
    return (f"   [NEAR MISS: this is b{k}, off by {err:+.3f} -- "
            f"beyond --coord_tol; loosen it rather than blaming the line]")


def _log_blocker(entry: MasterEntry, bond_of, log=print) -> None:
    """One culprit TFFC: its coordinate, and the FFC + line that proposed it.

    For an *ambiguous* origin the other image was added too (at half weight),
    so it is shown as "other half"; for an unambiguous one it was refused, and
    if that refused image turns out to be a real bond the algorithm simply
    picked the wrong side of the FFC.
    """
    log(f"  coord {entry.coord:.3f} (w={entry.weight:g}){_near(entry.coord, bond_of)}")
    for o in entry.origins:
        head = f"    from {o.line_key}   [{o.kind}, {o.decision}]"
        if o.rejected is not None:
            hit = bond_of(o.rejected)
            word = "other half" if o.decision == CLASS_AMBIGUOUS else "not taken"
            head += f"   {word}: {o.rejected:.3f}"
            head += (f" = b{hit[0]}  <- the true one" if hit
                     else _near(o.rejected, bond_of) or " (also false)")
        log(head)
        t = o.tffc
        if t is None or t.kind == "mock":
            log("      mock TFFC -- the line's own shift, not an observed FFC")
            continue
        log(f"      FFC (X, Y) = ({t.x:.4f}, {t.y:.4f})  charges [{t.ch_x}, "
            f"{t.ch_y}]  rank {t.ranking}")
        if t.category or t.label:
            log(f"      annotator: {t.category or '?'}   {t.label}")


# =============================================================================
# 8. PIPELINE
# =============================================================================
# The FFC map, the line detection and the edge table do not depend on the
# C-terminus at all (``detect_lines_oa`` never sees the peptide), so they are
# prepared once in ``_prepare`` and shared by however many terminus hypotheses
# ``_run_once`` is asked to evaluate.

@dataclass
class Context:
    """Terminus-independent inputs, computed once."""
    ffc_df: "pd.DataFrame"
    lines: list
    n_parental: int
    allowed: List[float]
    measured_total: float       # neutral precursor mass read off the parental line


def _prepare(args, log=print) -> Context:
    """Load the FFC map, detect lines, load the edge table."""
    ffc_df = pd.read_csv(args.data, sep="\t")
    ffc_df = merge_duplicate_ffcs_charge(ffc_df, symmetric=True, reset_ranking=True)
    ffc_df = prepare_ffc_data(ffc_df, top_n=args.top_n)
    log(f"FFC rows after merge + top-{args.top_n}: {len(ffc_df)}")

    line_list, master_lines = lba.detect_lines_oa(
        ffc_df, args.charge, args.delta, args.min_ffc,
        line_tol=args.line_tol, delta_unit=args.delta_unit)
    all_lines = master_lines + line_list
    log(f"Detected {len(all_lines)} lines "
        f"({len(master_lines)} at precursor charge / {len(line_list)} residual) "
        f"with delta={args.delta} {args.delta_unit}, min_ffc={args.min_ffc}")

    # The *first* precursor-charge line greedy peeling extracts is the most
    # informative one -- the true parental line -- and its reconstructed mass is
    # the precursor measurement: b_k + y_{n-k} summed at charge z, so
    # subtracting z protons gives the neutral peptide mass with no assumption
    # about the C-terminus.  That is what makes `--terminus auto` meaningful:
    # only the terminus, not the total, is being hypothesised.  (Do not pick by
    # mass or by member count -- the loss lines and the spurious positive-shift
    # lines also have i + j == z.)
    measured_total = float("nan")
    if master_lines:
        measured_total = master_lines[0].mass - args.charge * PROTON

    allowed = load_edges(args.edges)
    log(f"Loaded {len(allowed):,} compound residue masses from "
        f"{os.path.basename(args.edges)}")
    return Context(ffc_df, all_lines, len(master_lines), allowed, measured_total)


def _run_once(args, terminus: str, ctx: Context,
              total_override: Optional[float] = None, log=print) -> dict:
    """Run the whole algorithm under one C-terminus hypothesis."""
    t_start = time.time()

    # ── Peptide model ─────────────────────────────────────────────────────
    end = True if terminus.lower() in ("h2o", "water", "true") else "NH3"
    pep = peptide.Pep(f"[{args.seq}+{args.charge}H]{args.charge}+", end_h20=end)
    n = len(pep.AA_array)
    prefix = residue_prefixes(pep)
    term = terminus_mass(pep)
    # ``total`` is the *measured* neutral precursor mass; the terminus
    # hypothesis then splits it into residues + terminus.  Passing an override
    # keeps the total fixed while the terminus varies, which is the only
    # honest way to compare the two hypotheses.
    total = prefix[n] + term if total_override is None else total_override
    parent_mass = total + args.charge * PROTON    # protonated precursor
    parent_pair = total + 2 * PROTON              # b_k + y_{n-k}, any k

    # Rebase path_finding onto this peptide's C-terminus (see module docstring).
    pf.SOURCE_NODE = (0.0, term)
    source = pf.SOURCE_NODE

    log("=" * 78)
    log(f"Peptide      : {args.seq}  ({n} residues, charge {args.charge}, "
        f"C-term {'H2O' if end is True else 'NH3'})")
    log(f"Neutral mass : {total:.4f}   precursor {parent_mass:.4f}   "
        f"ParentPair {parent_pair:.4f}   residue sum {total - term:.4f}")
    log(f"Parameters   : top_n={args.top_n}  inactive_count={args.inactive_count}  "
        f"threshold={args.threshold} ({args.threshold / total * 1e6:.1f} ppm of M)"
        f"  coord_tol={args.coord_tol}")
    log(f"Line finding : delta={args.delta} {args.delta_unit}  "
        f"min_ffc={args.min_ffc}  tol_ppm={args.tol_ppm}  "
        f"iso_range={args.iso_range}")
    log("=" * 78)

    allowed = [m for m in ctx.allowed if m <= total]
    ffc_df, all_lines = ctx.ffc_df, ctx.lines
    log(f"\n{len(allowed):,} compound residue masses <= {total:.1f} Da; "
        f"{len(all_lines)} lines ({ctx.n_parental} at precursor charge)")

    records = build_line_records(
        ffc_df, all_lines, pep, parent_mass, args.charge, parent_pair, allowed,
        annotate=not args.no_annotate, iso_range=args.iso_range,
        tol_ppm=args.tol_ppm)
    if not args.keep_isotope_lines:
        mark_isotope_satellites(records, tol=args.coord_tol)

    # ── Line inventory ────────────────────────────────────────────────────
    log("\n" + "-" * 78)
    log("DETECTED LINES")
    log("-" * 78)
    log(f"{'line':<24}{'#FFC':>5}{'shift':>12}{'align A':>12}"
        f"{'A(detect)':>11}  {'category':<10}{'annot':<14}note")
    for rec in sorted(records, key=lambda r: (r.category != "parental", -r.n_ffc)):
        log(f"{rec.key:<24}{rec.n_ffc:>5}{rec.shift:>12.3f}{rec.a_shift:>12.3f}"
            f"{rec.a_from_detection:>11.3f}  {rec.category:<10}"
            f"{str(rec.annot_type):<14}{rec.detail}")

    # ── Seed the master set from the parental lines ───────────────────────
    master = MasterSet(total, args.coord_tol)
    scorer = PathScorer(allowed, total, source, args.threshold, args.max_paths)

    log("\n" + "-" * 78)
    log("STEP 1 -- MASTER SET FROM THE PARENTAL LINES")
    log("-" * 78)
    seed_master(records, master, pep, source=args.parental_source,
                prefix=prefix, iso_range=args.iso_range, tol_ppm=args.tol_ppm,
                consider_secondary=True, log=log)
    covered, n_false, _ = coverage_report(master, prefix, args.coord_tol)
    log(f"  master TFFCs = {len(master)}   true breakpoints covered = "
        f"{len(covered)}/{n - 1} ({len(covered) / (n - 1) * 100:.1f}%)   "
        f"false TFFCs = {n_false}")
    log(f"  bonds: {covered}")
    base_paths, base_correct, base_weight = report_paths(
        scorer, master, args.seq, total, term, args.threshold,
        args.label_tol, log)
    baseline = {
        "master": len(master), "bonds": len(covered), "false": n_false,
        "paths": base_paths, "correct": base_correct, "weight": base_weight,
    }

    # ── Select the internal / neutral-loss lines ──────────────────────────
    log("\n" + "-" * 78)
    log("STEP 2 -- LINE SELECTION")
    log("-" * 78)
    candidates = []
    for rec in records:
        if rec.category == "parental":
            rec.reason = "parental (already in the master set)"
            continue
        if rec.category not in USABLE_CATEGORIES:
            rec.reason = f"skipped: unsupported line type ({rec.detail})"
            continue
        if args.require_annot and rec.annot_type is not None and \
                lba._base_category(rec.annot_type) not in ("parental", "internal"):
            rec.reason = f"skipped: annotator calls it {rec.annot_type}"
            continue
        rec.n_inactive = count_inactive(rec, master)
        if rec.n_inactive < args.inactive_count:
            rec.reason = (f"skipped: only {rec.n_inactive} inactive TFFC(s), "
                          f"need {args.inactive_count}")
            continue
        rec.used = True
        rec.reason = f"USED ({rec.n_inactive} inactive TFFCs)"
        candidates.append(rec)

    for rec in records:
        if rec.category != "parental":
            log(f"  {rec.key:<24}{rec.category:<10}{rec.reason}")

    # Most strongly anchored lines first, then the richest ones.
    candidates.sort(key=lambda r: (-r.n_inactive, -r.n_ffc, r.key))

    # Fixed reference for the inactive test (see classify_tffc).  The parental
    # seed set is frozen here, before any internal-line TFFC can enter it.
    reference = master.snapshot() if args.inactive_vs == "parental" else None

    # ── Extend ────────────────────────────────────────────────────────────
    log("\n" + "-" * 78)
    log("STEP 3 -- EXTENDING THE MASTER SET")
    log("-" * 78)
    total_added = 0
    for rec in candidates:
        mock = add_mock(rec, parent_pair)
        log(f"\n  {rec.key}  (A={rec.a_shift:.3f}, {rec.n_ffc} FFCs"
            f"{', +mock' if mock else ''}, {rec.n_inactive} inactive)")
        added = extend_with_line(rec, master, scorer,
                                 ambiguous_weight=args.ambiguous_weight,
                                 ambiguous_policy=args.ambiguous_policy,
                                 max_rounds=args.max_rounds,
                                 reference=reference, log=log)
        total_added += added
        log(f"      -> {added} new master TFFC(s); master now {len(master)}")

    # ── Final reconstruction ──────────────────────────────────────────────
    log("\n" + "=" * 78)
    log("STEP 4 -- FINAL RECONSTRUCTION FROM THE MASTER TFFC-SET")
    log("=" * 78)
    covered2, n_false2, n_half2 = coverage_report(master, prefix, args.coord_tol)
    log(f"  master TFFCs = {len(master)} (+{total_added} from internal / "
        f"loss lines)")
    log(f"  true breakpoints covered = {len(covered2)}/{n - 1} "
        f"({len(covered2) / (n - 1) * 100:.1f}%)   false TFFCs = {n_false2} "
        f"(FPR {n_false2 / max(1, len(master)) * 100:.1f}%, "
        f"{n_half2} of them the half-weight half of an ambiguous pair)")
    log(f"  bonds: {covered2}")
    gained = sorted(set(covered2) - set(covered))
    log(f"  newly covered bonds: {gained}")

    true_folded = {k: fold(prefix[k], total) for k in range(1, n)}
    log(f"\n  {'coord':>12}{'w':>5}  {'bond':<8}{'err':>8}  origin")
    for e in sorted(master.entries, key=lambda x: x.coord):
        hits = [k for k, c in true_folded.items()
                if abs(c - e.coord) <= args.coord_tol]
        bond = f"b{hits[0]}" if hits else "--"
        err = f"{e.coord - true_folded[hits[0]]:+.3f}" if hits else ""
        log(f"  {e.coord:>12.3f}{e.weight:>5g}  {bond:<8}{err:>8}  "
            f"{', '.join(dict.fromkeys(str(o) for o in e.origins))}")
    n_paths, n_correct, weight = report_paths(
        scorer, master, args.seq, total, term, args.threshold,
        args.label_tol, log)

    # ── Blame analysis when the truth did not make the cut ────────────────
    blame = None
    if n_correct == 0 and n_paths and not args.no_diagnose:
        blame = diagnose_failure(
            scorer, master, prefix, total, term, args.seq, args.coord_tol,
            args.threshold, args.label_tol, log=log)

    # ── Summary ───────────────────────────────────────────────────────────
    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"{'':<26}{'parental only':>16}{'+ internal/loss':>18}")
    log(f"{'master TFFCs':<26}{baseline['master']:>16}{len(master):>18}")
    log(f"{'true bonds covered':<26}{baseline['bonds']:>16}{len(covered2):>18}"
        f"   (of {n - 1})")
    log(f"{'false TFFCs':<26}{baseline['false']:>16}{n_false2:>18}")
    log(f"{'heaviest paths':<26}{baseline['paths']:>16}{n_paths:>18}")
    log(f"{'correct reconstructions':<26}{baseline['correct']:>16}{n_correct:>18}")
    if blame and blame["blockers"]:
        lines_at_fault = sorted({o.line_key for e in blame["blockers"]
                                 for o in e.origins})
        log(f"\n{len(blame['blockers'])} TFFC(s) from {len(lines_at_fault)} "
            f"line(s) cost the correct reconstruction: "
            + ", ".join(lines_at_fault))
    log(f"\ngraph builds: {scorer.n_builds}   elapsed: {time.time() - t_start:.1f}s")

    return {
        "terminus": "H2O" if end is True else "NH3",
        "total": total, "term": term,
        "baseline": baseline,
        "final": {"master": len(master), "bonds": len(covered2),
                  "false": n_false2, "paths": n_paths, "correct": n_correct,
                  "weight": weight},
        "records": records, "master_set": master, "blame": blame,
    }


def run(args) -> dict:
    """Entry point.  Prepares the terminus-independent inputs, then evaluates
    either the requested C-terminus or -- with ``--terminus auto`` -- both.

    Auto mode holds the measured precursor mass fixed (read off the parental
    line) and lets only the terminus vary, then keeps whichever hypothesis
    supports the heavier path.  See ``--terminus``'s help text for how well
    that actually discriminates.
    """
    ctx = _prepare(args, log=print)

    if args.terminus != "auto":
        return _run_once(args, args.terminus, ctx, log=print)

    if ctx.measured_total != ctx.measured_total:      # NaN -> no parental line
        raise SystemExit("--terminus auto needs a parental line to read the "
                         "precursor mass from; none was detected.")

    trials = []
    for hypothesis in ("h2o", "nh3"):
        buf: List[str] = []
        res = _run_once(args, hypothesis, ctx, total_override=ctx.measured_total,
                        log=buf.append)
        trials.append((hypothesis, res, buf))

    def rank(item):
        _, r, _ = item
        return (r["final"]["weight"], r["baseline"]["weight"], r["final"]["master"])

    trials.sort(key=rank, reverse=True)
    (win_name, win_res, win_log), (lose_name, lose_res, _) = trials

    print("\n".join(win_log))
    print("\n" + "=" * 78)
    print("C-TERMINUS INFERENCE  (--terminus auto)")
    print("=" * 78)
    print(f"  measured neutral precursor mass: {ctx.measured_total:.4f} "
          f"(from the parental line, terminus-independent)")
    print(f"  {'hypothesis':<22}{'heaviest path':>15}{'baseline path':>15}"
          f"{'master TFFCs':>14}{'true bonds':>12}")
    for _, r, _ in trials:
        label = f"{r['terminus']} (res {r['total'] - r['term']:.3f})"
        print(f"  {label:<22}{r['final']['weight']:>15g}"
              f"{r['baseline']['weight']:>15g}"
              f"{r['final']['master']:>14}{r['final']['bonds']:>12}")
    margin = win_res["final"]["weight"] - lose_res["final"]["weight"]
    print(f"\n  -> picked {win_res['terminus']} over {lose_res['terminus']} "
          f"by {margin:g} path weight")
    if margin <= 0:
        print("     WARNING: no margin -- the data does not distinguish the two; "
              "pass --terminus explicitly.")
    return win_res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Master-TFFC-set de novo reconstruction "
                    "(parental + neutral-loss + internal lines).")
    # data / peptide
    p.add_argument("--data", default=DEFAULT_DATA, help="deconvolved FFC .tsv")
    p.add_argument("--seq", default=DEFAULT_SEQ, help="peptide sequence")
    p.add_argument("--charge", type=int, default=5, help="precursor charge")
    p.add_argument("--terminus", default="nh3", choices=["nh3", "h2o", "auto"],
                   help="C-terminal group (KWK is an amide -> nh3).  This is an "
                        "INPUT, not something the algorithm derives: it sets the "
                        "source node's y-coordinate and the residue/terminus "
                        "split of the precursor mass.  'auto' evaluates both and "
                        "keeps the heavier path -- see the module docstring for "
                        "how weak that signal is")
    # ── Line detection.  These are the knobs that decide *which lines exist*,
    # and they change the answer more than anything downstream: the pipeline
    # settings below (20 ppm / tol 30) find the KWK -129.093 line that reveals
    # b1, which the old 0.02 Da / 20 ppm settings miss entirely.
    p.add_argument("--top_n", type=int, default=500,
                   help="number of top-ranked FFCs fed to line detection.  "
                        "Greedy peeling is non-monotonic in this, so different "
                        "cutoffs give different lines, not just more of them")
    p.add_argument("--delta", type=float, default=20.0,
                   help="line-detection Sort-and-Split cluster gap, in the unit "
                        "set by --delta_unit (default 20 ppm, matching "
                        "annotation_pepline/run_annotation_reconstruct.py)")
    p.add_argument("--delta_unit", default="ppm", choices=["da", "ppm"],
                   help="unit for --delta: 'ppm' (relative, default) or 'da' "
                        "(fixed gap; the oa_v2 module's own default is 0.02 Da)")
    p.add_argument("--line_tol", type=float, default=None,
                   help="tolerance for calling an FFC a member of a detected "
                        "line (defaults to --delta, same unit)")
    p.add_argument("--min_ffc", type=int, default=3,
                   help="minimum distinct FFCs to keep a line (NumFFC).  A real "
                        "line with 2 members is dropped")
    p.add_argument("--iso_range", type=int, default=1,
                   help="isotope variants built into the theoretical fragment "
                        "index used for annotation")
    p.add_argument("--tol_ppm", type=float, default=30.0,
                   help="relative tolerance for matching an FFC side to a "
                        "theoretical fragment (annotation only)")
    # the algorithm
    p.add_argument("--inactive_count", type=int, default=1,
                   help="inactive TFFCs a line needs before it is used")
    p.add_argument("--parental_source", default="annotated",
                   choices=["annotated", "all", "theoretical"],
                   help="which parental-line FFCs seed the master set")
    p.add_argument("--require_annot", action="store_true",
                   help="also require the annotator's line type to be "
                        "parental/internal (belt and braces; the inactive "
                        "gate usually suffices)")
    p.add_argument("--no_annotate", action="store_true",
                   help="skip per-FFC annotation entirely (implies "
                        "--parental_source all)")
    p.add_argument("--ambiguous_weight", type=float, default=0.5,
                   help="weight given to each half of an ambiguous TFFC")
    p.add_argument("--inactive_vs", default="master",
                   choices=["master", "parental"],
                   help="what a TFFC's 'already known' (inactive) test compares "
                        "against: 'master' (default) = the growing set, so an "
                        "earlier TFFC can silence a later one; 'parental' = the "
                        "frozen parental seed set, giving every TFFC the same "
                        "fixed reference (see classify_tffc)")
    p.add_argument("--ambiguous_policy", default="both", choices=["both", "skip"],
                   help="'both' (reconstruct.md) adds both images of an "
                        "ambiguous TFFC at half weight -- one of which is false "
                        "by construction; 'skip' keeps only unambiguous TFFCs")
    p.add_argument("--keep_isotope_lines", action="store_true",
                   help="do NOT demote internal lines that look like the +1/+2 "
                        "isotope satellite of a stronger line "
                        "(see mark_isotope_satellites)")
    p.add_argument("--max_rounds", type=int, default=50)
    p.add_argument("--no_diagnose", action="store_true",
                   help="skip the 'why the correct reconstruction lost' blame "
                        "analysis that runs when no heaviest path is correct")
    # graph
    p.add_argument("--threshold", type=float, default=0.03,
                   help="mass tolerance (Da) for a 2D-DAG edge.  Real "
                        "deconvolved masses carry a *relative* error, so this "
                        "has to cover ~2x the worst per-side error: 0.03 Da is "
                        "about 8 ppm of the KWK precursor.  Too tight and true "
                        "breakpoints stop making edges; too loose and spurious "
                        "TFFCs start looking good (0.05 costs 2 false TFFCs)")
    p.add_argument("--coord_tol", type=float, default=0.03,
                   help="tolerance for 'this TFFC is already in the master set'")
    p.add_argument("--label_tol", type=float, default=0.02,
                   help="tolerance for naming a one-residue gap with its letter "
                        "in the printed reconstruction (see format_gapped)")
    p.add_argument("--max_paths", type=int, default=1000)
    p.add_argument("--edges", default=DEFAULT_EDGES)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.no_annotate:
        args.parental_source = "all"
    run(args)
