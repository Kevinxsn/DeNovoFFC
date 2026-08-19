#!/usr/bin/env python3
"""
De novo sequencing from an MMS (FFC) map -- no ground truth anywhere
====================================================================

This is the *real* de novo counterpart of
``../line_based_annotation/line_internal_annotation_oa_v2.py``.  The algorithm
is the same one that module documents (see ``line_based.md`` §8b-§8e):

    detect lines  ->  seed the master TFFC-set from the parental lines
                  ->  extend it with the aligned internal / neutral-loss lines,
                      in batch rounds, until it converges
                  ->  longest/heaviest path through the 2D-DAG  ->  sequence

but with the two things the annotation module is allowed to know removed:

1. **No peptide sequence.**  Every step that consulted ``pep`` -- the fragment
   index, the b/y/internal annotation, ``_bond_of`` / ``_snap`` against
   ``prefix[k]``, ``is_reconstruction_correct`` -- is gone.  The only inputs are
   the FFC table, the precursor charge, the precursor neutral mass, and which
   C-terminal group the chemistry produced.

2. **No exact masses.**  ``_oa_v2`` snapped every master coordinate onto its
   true ``prefix[k]`` before using it (``line_based.md`` §8e), which made the
   DAG's edge windows meaningless -- a snapped coordinate is exact, so a 0.001
   Da window is fine.  We cannot snap, so tolerance has to be carried
   explicitly.  ``line_based.md`` §8e names the de-novo-safe alternative and
   this module implements it, one step further:

       **every coordinate carries its own absolute uncertainty sigma**, derived
       from the mass that was actually measured, and every DAG edge window is
       ``sigma(a) + sigma(b)``.

Why per-coordinate sigma, and not one ppm number
------------------------------------------------
A breakpoint coordinate is *folded* into the lower half ``(0, total/2]``, and
folding moves a coordinate but not its error (``line_based.md`` §8c).  A y30
measured at 3043 Da with 8 ppm of error folds onto a 958 Da coordinate still
carrying 0.025 Da -- 27 ppm *there*.  Referencing a ppm window to the folded
value therefore demands 27 ppm to accept an 8 ppm measurement.  ``_oa_v2``
patched this in ``_bond_of`` by referencing ``max(c, total - c)``, a proxy for
"the mass that was really measured".  Here we do not need the proxy: we know
which side produced each coordinate, so we attach that side's own error to it
and never have to guess again.  The same machinery then handles, uniformly:

  * a **mock** -- a line's *median* alignment shift, so its error is a
    difference of two large masses (~``total * tol_ppm``), not a single peak's.
    ``_oa_v2`` needed a separate ``mock_tol_ppm`` knob for exactly this; here it
    falls out of the derivation (:func:`_mock_sigma`).
  * a coordinate **corroborated by several TFFCs** -- they are independent
    measurements of one breakpoint, so :class:`MasterSet` merges them by
    inverse-variance weighting.  The coordinate gets *more accurate and more
    precise* as evidence accumulates, which is the de novo replacement for
    snapping onto the truth.

Everything else is deliberately unchanged from ``_oa_v2``, and the pieces that
were already de novo are imported rather than re-implemented:
``detect_lines_oa`` (orientation-agnostic line finding), ``classify_line``
(is the alignment shift a residue run / a named neutral loss?),
``mark_isotope_satellites``, and ``find_all_heaviest_paths``.

Public API
----------
    result = run_denovo(ffc_df, parent_charge=5, parent_mass=4059.36,
                        terminus="NH3")
    result.reconstructions[0]["reconstruction"]   # 'KW - 516.342 - KJEKVG - ...'
    result.master.entries                          # the master TFFC-set
    result.lines                                   # per-line diagnostics

``run_denovo`` never sees a sequence.  Scoring a result against a known peptide
is a *separate* step and lives in ``evaluate.py``.
"""
from __future__ import annotations

import math
import os
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_DEISO = os.path.join(_ROOT, "deiso", "python_code")
_LBA = os.path.join(_ROOT, "line_based_annotation")
_RECON = os.path.join(_ROOT, "reconstruct")
for _p in (_DEISO, _LBA, _RECON):
    if _p not in sys.path:
        sys.path.append(_p)

# ── Constants (same values the rest of the codebase uses) ────────────────────
PROTON = 1.00725          # used to strip the ionising proton off a [M+H]+ mass
MASS_H = 1.00784          # used by the charge deconvolution: i*mz - (i-1)*H
WATER = 18.010565
AMMONIA = 17.026549
NH3 = 17.02655            # the c-ion increment over b (ETD/cz_series.py)
HYDROGEN = 1.00783        # the H atom ETD delivers with the electron
ISOTOPE_OFFSET = 1.00335  # 13C - 12C

DEFAULT_EDGES_PATH = os.path.join(_RECON, "edges.json")

#: Terminus name -> neutral mass of the C-terminal group.  ``residue sum +
#: terminus == neutral peptide mass``, so this is what makes the y-axis start in
#: the right place (``line_based.md`` §8a: getting it wrong fails *silently*).
TERMINI = {"H2O": WATER, "OH": WATER, "NH3": AMMONIA, "NH2": AMMONIA}


# =============================================================================
# 0. FRAGMENTATION: HCD (b/y) vs ETD (c/z-dot)
# =============================================================================
# The two are the *same geometry* -- an N-terminal fragment meeting a C-terminal
# fragment at one backbone bond -- and differ only in three numbers.  See
# ``../ETD/etd.md`` §2 and §4a, whose derivation this reproduces:
#
#              N-term ion        C-term ion            pair sum      line charge
#   HCD  b/y   b_k = prefix[k]   y_j = total-prefix[k]  M + 2p        i + j = z
#   ETD  c/z.  c_k = b_k + NH3   z._j = y_j + H - NH3   M + H + 2p    i + j = z - 1
#
# ``pair_h`` and ``side_nh3`` below are the only chemistry; everything else in
# this module is written against the *coordinate basis* they induce.
#
# ⚠ Why a coordinate basis at all.  The master-set machinery folds a bond's two
# masses onto one node, which needs the map "measured neutral -> coordinate" to
# be an involution.  For b/y it is: a neutral reads as prefix[k] or as
# total - prefix[k].  For c/z it is NOT -- until NH3 comes off every measured
# side, after which
#
#     u = x - NH3 - PROTON        c ion  ->  u = prefix[k]
#                                 z. ion ->  u = TOTAL_STAR - prefix[k]
#
# and the involution is restored against ``TOTAL_STAR = total + H - 2*NH3``.
# Taking ``fold()`` at face value on the raw c/z masses silently mixes two bases.
#
# ⚠ TOTAL_STAR is bookkeeping, not a mass.  Its implied terminus
# ``coord_term = term + H - 2*NH3`` comes out NEGATIVE (-15.03 for a free acid),
# which is harmless -- it is only ever the y-axis origin, and
# ``coord_total - coord_term == prefix[n]`` exactly, so the residue ladder the
# reconstruction prints is still the real one.
#
# ⚠ Two totals are in play at once (etd.md §4a).  Coordinates use
# ``coord_total``; the alignment shift ``A = ParentPair - (x + y)`` is computed
# from measured masses on the REAL basis and must use ``parent_pair``.  Mixing
# them is a ~33 Da error that reads as a bad line rather than as a bug.

@dataclass(frozen=True)
class Basis:
    """The fragmentation-dependent constants, resolved for one peptide."""
    mode: str                 # "HCD" | "ETD"
    charge_offset: int        # 0 for b/y; -1 for c/z (etd.md §2b)
    side_nh3: float           # subtracted from every measured side (0 | NH3)
    pair_h: float             # extra H in the pair sum (0 | HYDROGEN)
    total: float              # REAL neutral peptide mass
    term: float               # REAL C-terminal group

    @property
    def coord_total(self) -> float:
        """``total`` on the coordinate basis (``TOTAL_STAR`` for ETD)."""
        return self.total + self.pair_h - 2.0 * self.side_nh3

    @property
    def coord_term(self) -> float:
        """y-axis origin on the coordinate basis.  Negative for ETD."""
        return self.term + self.pair_h - 2.0 * self.side_nh3

    @property
    def parent_pair(self) -> float:
        """Sum of a complementary pair's two ``[M+H]+`` masses."""
        return self.total + self.pair_h + 2.0 * PROTON

    def line_charge(self, parent_charge: int) -> int:
        """Charge sum the backbone ladder sits on.

        ⚠ ETD's ``-1`` fails *silently* (etd.md §2b): electron transfer consumes
        one of the precursor's protons, so a c/z pair from [M+6H]6+ carries five
        charges.  The ladder then sits one electron (0.0006 Da) from where a b/y
        parental line would, i.e. at the same *mass* on a different *charge sum*,
        so ``detect_lines_oa``'s parental pass misses it and greedy peeling finds
        it instead -- splitting its evidence across its own isotope satellites,
        which carry more members than the monoisotopic line.  Nothing errors.
        """
        return max(parent_charge + self.charge_offset, 2)


#: ``fragmentation`` -> the chemistry.  ``HCD`` is the default everywhere.
FRAGMENTATION = {
    "HCD": dict(charge_offset=0, side_nh3=0.0, pair_h=0.0),
    "ETD": dict(charge_offset=-1, side_nh3=NH3, pair_h=HYDROGEN),
}
#: EThcD carries both ladders at nearly the same line mass on different charge
#: sums (etd.md §6b), so one detector pass cannot serve both.  Run it as ETD to
#: read the c/z ladder (much the stronger of the two) or as HCD for b/y.
FRAGMENTATION["ETHCD"] = FRAGMENTATION["ETD"]


def make_basis(fragmentation: str, parent_mass: float, terminus: str) -> Basis:
    key = fragmentation.upper()
    if key not in FRAGMENTATION:
        raise ValueError(f"fragmentation must be one of {sorted(FRAGMENTATION)}, "
                         f"got {fragmentation!r}")
    if terminus not in TERMINI:
        raise ValueError(f"terminus must be one of {sorted(TERMINI)}, "
                         f"got {terminus!r}")
    return Basis(mode=key, total=float(parent_mass), term=TERMINI[terminus],
                 **FRAGMENTATION[key])


# =============================================================================
# 1. COORDINATES AND THEIR UNCERTAINTY
# =============================================================================
# A "breakpoint coordinate" is the neutral mass of the N-terminal fragment
# produced by one backbone cleavage, folded into (0, total/2] -- a cleavage's
# two fragments have masses r and total - r and one FFC cannot say which is
# which, so we canonicalise on the smaller (master_reconstruction.fold).

def fold(r: float, total: float) -> float:
    """Fold a breakpoint coordinate into the lower half ``(0, total/2]``."""
    return r if r <= total - r else total - r


def coord_from_side(ion_mass: float, basis: Basis, tol_ppm: float,
                    precursor_ppm: float) -> Tuple[float, float]:
    """``([M+H]+ mass of one FFC side)`` -> ``(folded coordinate, sigma)``.

    Two things happen here, and they must not be confused.

    **The coordinate** is on the fragmentation's basis: strip the proton, then
    strip ``side_nh3`` (0 for b/y, NH3 for c/z), then fold against
    ``coord_total``.  That is the involution the master set needs.

    **Sigma is referenced to the mass that was actually measured** -- never to
    the folded coordinate, and never to the basis-shifted one.  ``m_meas`` is
    the fragment's measured neutral mass, known to ``tol_ppm`` *of itself*; the
    coordinate may be tens of Da away from it (c/z) or on the far side of the
    peptide (a mirrored side), but the error travels with the measurement, not
    with the number we chose to write down.  Mirroring additionally inherits
    whatever uncertainty the precursor mass carries.
    """
    m_meas = ion_mass - PROTON              # measured neutral fragment mass
    u = m_meas - basis.side_nh3             # same quantity, coordinate basis
    sig = abs(m_meas) * tol_ppm * 1e-6
    ct = basis.coord_total
    if u <= ct - u:
        return u, sig
    return ct - u, sig + basis.total * precursor_ppm * 1e-6


@dataclass
class Entry:
    """One breakpoint in the master TFFC-set.

    ``coord`` is an inverse-variance weighted mean over every TFFC that landed
    on this breakpoint, and ``sigma`` shrinks accordingly -- corroboration makes
    a coordinate both more accurate and more precise.  That is what stands in
    for ``_oa_v2``'s snap onto the exact ``prefix[k]``.
    """
    coord: float
    sigma: float
    n: int = 1
    kinds: List[str] = field(default_factory=list)
    origins: List[str] = field(default_factory=list)
    #: Best (lowest) FFC ranking among the TFFCs supporting this breakpoint.
    #: The deconvolution's own confidence measure, and the only quality signal
    #: the pipeline has that is not derived from the masses themselves.
    best_rank: int = -1


class MasterSet:
    """The growing master TFFC-set, as folded coordinates with uncertainties.

    Two coordinates are the same breakpoint when they agree to within the sum
    of their uncertainties.  This replaces ``MasterSet``'s fixed ``tol`` (and
    ``_oa_v2``'s ``_MasterSetPpm``'s relative one): a light, precisely measured
    coordinate gets a tight window and a mock gets a loose one, automatically.
    """

    def __init__(self, total: float, sigma_floor_ppm: float = 3.0,
                 mock_weight: float = 1.0, evidence_weight: float = 0.0,
                 evidence_cap: float = 1.0, rank_floor: float = 1.0,
                 rank_scale: float = 1.0):
        self.total = total
        #: Merging shrinks sigma without bound if left alone; the floor keeps a
        #: coordinate from claiming better accuracy than the instrument has.
        self.sigma_floor_ppm = sigma_floor_ppm
        #: A mock is weaker evidence than an observed FFC -- it is a line-level
        #: inference, not a measured peak -- so it may be given less than a full
        #: vote in the heaviest-path search.
        self.mock_weight = mock_weight
        #: Bonus per *additional* independent TFFC on one breakpoint, capped.
        #: A coordinate seen by several FFCs is far likelier to be a real
        #: cleavage than a lone one, and this is the only lever that lets the
        #: search prefer the corroborated ladder when paths tie.
        self.evidence_weight = evidence_weight
        self.evidence_cap = evidence_cap
        #: Rank-derived confidence: a breakpoint whose best supporting FFC is
        #: ranked ``rank_scale`` gets ``rank_floor`` of a vote, one ranked 0 gets
        #: a full vote, linearly in between.  ``rank_floor = 1`` disables it.
        self.rank_floor = rank_floor
        self.rank_scale = max(rank_scale, 1.0)
        self.entries: List[Entry] = []

    def rank_factor(self, rank: int) -> float:
        if self.rank_floor >= 1.0 or rank is None or rank < 0:
            return 1.0
        frac = min(max(rank / self.rank_scale, 0.0), 1.0)
        return 1.0 - (1.0 - self.rank_floor) * frac

    def weight_of(self, kind_or_kinds, n: int = 1, rank: int = -1) -> float:
        kinds = ([kind_or_kinds] if isinstance(kind_or_kinds, str)
                 else list(kind_or_kinds))
        base = self.mock_weight if kinds and all(k == "mock" for k in kinds) else 1.0
        base *= self.rank_factor(rank)
        return base + min(self.evidence_weight * (n - 1), self.evidence_cap)

    # ── lookup ──────────────────────────────────────────────────────────────
    def find(self, coord: float, sigma: float = 0.0) -> Optional[Entry]:
        best = None
        for e in self.entries:
            d = abs(e.coord - coord)
            if d <= e.sigma + sigma and (best is None or d < best[0]):
                best = (d, e)
        return best[1] if best else None

    def contains(self, coord: float, sigma: float = 0.0) -> bool:
        return self.find(coord, sigma) is not None

    def __len__(self) -> int:
        return len(self.entries)

    # ── mutation ────────────────────────────────────────────────────────────
    def add(self, coord: float, sigma: float, kind: str = "observed",
            origin: str = "", rank: int = -1) -> bool:
        """Add a breakpoint.  Returns True if it was genuinely new.

        A repeat measurement of an existing breakpoint is combined with it by
        inverse-variance weighting -- the textbook estimator for independent
        measurements of one quantity, which is exactly what two TFFCs on the
        same cleavage are.
        """
        c = fold(coord, self.total)
        if c <= sigma or c >= self.total - sigma:
            return False                      # a terminus, not a cleavage
        hit = self.find(c, sigma)
        if hit is not None:
            w_old = 1.0 / max(hit.sigma, 1e-9) ** 2
            w_new = 1.0 / max(sigma, 1e-9) ** 2
            hit.coord = (hit.coord * w_old + c * w_new) / (w_old + w_new)
            hit.sigma = max(1.0 / math.sqrt(w_old + w_new),
                            hit.coord * self.sigma_floor_ppm * 1e-6)
            hit.n += 1
            hit.kinds.append(kind)
            if rank >= 0 and (hit.best_rank < 0 or rank < hit.best_rank):
                hit.best_rank = rank
            if origin:
                hit.origins.append(origin)
            return False
        self.entries.append(Entry(c, max(sigma, c * self.sigma_floor_ppm * 1e-6),
                                  1, [kind], [origin] if origin else [], rank))
        return True

    def snapshot(self) -> "MasterSet":
        copy = MasterSet(self.total, self.sigma_floor_ppm, self.mock_weight,
                         self.evidence_weight, self.evidence_cap,
                         self.rank_floor, self.rank_scale)
        copy.entries = [Entry(e.coord, e.sigma, e.n, list(e.kinds),
                              list(e.origins), e.best_rank)
                        for e in self.entries]
        return copy

    # ── views ───────────────────────────────────────────────────────────────
    def view(self, extra=None):
        """``(sorted coords, {coord: sigma}, {coord: weight})``.

        ``extra`` is a list of ``(coord, sigma, kind[, rank])`` candidates mixed
        in without committing them -- that is how the goodness test asks "would
        adding this breakpoint buy anything?".
        """
        vals = [(e.coord, e.sigma, self.weight_of(e.kinds, e.n, e.best_rank))
                for e in self.entries]
        if extra:
            for item in extra:
                c, s, kind = item[0], item[1], item[2]
                rank = item[3] if len(item) > 3 else -1
                fc = fold(c, self.total)
                if s < fc < self.total - s:
                    vals.append((fc, s, self.weight_of(kind, 1, rank)))
        vals.sort()
        return ([c for c, _, _ in vals],
                {c: s for c, s, _ in vals},
                {c: w for c, _, w in vals})


# =============================================================================
# 2. THE 2D-DAG, WITH A PER-COORDINATE MASS WINDOW
# =============================================================================
# Same edge semantics as ``path_finding.build_adjacency_graph`` and
# ``master_reconstruction.build_dag`` (Down / Right / Sink; every gap must be an
# allowed compound residue mass), and the same shape as ``_oa_v2``'s
# ``_build_dag_ppm``.  The single change is the window:
#
#     _oa_v2:  tol = |heavier endpoint of the gap| * tol_ppm
#     here:    tol = sigma(a) + sigma(b)
#
# A gap is a difference of two measurements, so its uncertainty is the sum of
# theirs.  Deriving it this way is what lets a mock (error ~ total * tol_ppm)
# and a well-measured light fragment (error ~ 200 Da * tol_ppm, i.e. 60x
# smaller) coexist in one graph without either a window too tight to reach the
# mock or one so loose the light coordinates start making spurious edges.

def build_dag(coords: Sequence[float], sigma: Dict[float, float],
              sorted_allowed: Sequence[float], total: float,
              source: Tuple[float, float], sink: str,
              precursor_sigma: float) -> dict:
    """2D-DAG over ``coords``; ``sigma[c]`` is coordinate ``c``'s uncertainty."""
    sig = dict(sigma)
    all_coords = sorted(set(list(coords) + [source[0], source[1]]))
    sig.setdefault(source[0], 0.0)
    sig.setdefault(source[1], 0.0)          # the terminus is a definition, not
    for c in all_coords:                    # a measurement
        sig.setdefault(c, 0.0)

    nodes = [(x1, x2) for x1 in all_coords for x2 in all_coords
             if x1 != x2 or (x1, x2) == source]
    graph = {n: [] for n in nodes}
    node_set = set(nodes)
    if not sorted_allowed:
        return graph

    min_edge, max_edge = sorted_allowed[0], sorted_allowed[-1]
    max_coord = all_coords[-1]
    max_tol = 2.0 * max(sig.values()) if sig else 0.0

    def allowed(gap: float, tol: float) -> Optional[float]:
        if gap < min_edge - tol or gap > max_edge + tol:
            return None
        idx = bisect_left(sorted_allowed, gap)
        for k in (idx, idx - 1):
            if 0 <= k < len(sorted_allowed) and abs(sorted_allowed[k] - gap) <= tol:
                return sorted_allowed[k]
        return None

    def coords_above(val: float):
        return all_coords[bisect_left(all_coords, val + min_edge - max_tol):]

    for node in nodes:
        x1, x2 = node
        out = graph[node]
        max_current = max(x1, x2)

        remainder = total - x1 - x2
        if remainder > 0:
            m = allowed(remainder, sig[x1] + sig[x2] + precursor_sigma)
            if m is not None:
                out.append((sink, m, "sink"))

        for t1 in coords_above(x1):
            gap = t1 - x1
            if gap > max_coord - x1 + max_tol:
                break
            m = allowed(gap, sig[t1] + sig[x1])
            if m is not None:
                tgt = (t1, x2)
                if tgt in node_set and max_current < max(t1, x2):
                    out.append((tgt, m, "down"))

        for t2 in coords_above(x2):
            gap = t2 - x2
            if gap > max_coord - x2 + max_tol:
                break
            m = allowed(gap, sig[t2] + sig[x2])
            if m is not None:
                tgt = (x1, t2)
                if tgt in node_set and max_current < max(x1, t2):
                    out.append((tgt, m, "right"))

    return graph


class Scorer:
    """Heaviest-path evaluator over ``(coords, sigmas)``, with a cache.

    The goodness test asks the same question hundreds of times and different
    TFFCs frequently propose the same breakpoint, so caching pays for itself
    (this is ``master_reconstruction.PathScorer``, keyed on sigmas as well).
    """

    def __init__(self, sorted_allowed, total, source, sink, precursor_sigma,
                 max_paths: int = 1000):
        self.allowed = sorted_allowed
        self.total = total
        self.source = source
        self.sink = sink
        self.precursor_sigma = precursor_sigma
        self.max_paths = max_paths
        self._cache: Dict[tuple, float] = {}
        self.n_builds = 0

    def _key(self, coords, sigma, weights):
        return (tuple(round(c, 4) for c in coords),
                tuple(round(sigma.get(c, 0.0), 5) for c in coords),
                tuple(round(weights.get(c, 1.0), 3) for c in coords))

    def _graph(self, coords, sigma):
        self.n_builds += 1
        return build_dag(coords, sigma, self.allowed, self.total, self.source,
                         self.sink, self.precursor_sigma)

    def score(self, coords, sigma, weights) -> float:
        from master_reconstruction import find_all_heaviest_paths
        key = self._key(coords, sigma, weights)
        if key in self._cache:
            return self._cache[key]
        best, _ = find_all_heaviest_paths(self._graph(coords, sigma), weights,
                                          self.source, sink_node=self.sink,
                                          max_paths=1)
        val = -1.0 if best == float("-inf") else best
        self._cache[key] = val
        return val

    def paths(self, coords, sigma, weights):
        from master_reconstruction import find_all_heaviest_paths
        return find_all_heaviest_paths(self._graph(coords, sigma), weights,
                                       self.source, sink_node=self.sink,
                                       max_paths=self.max_paths)


# =============================================================================
# 3. LINES -> TFFCs
# =============================================================================

@dataclass
class Tffc:
    """One member FFC of a line, as two singly-protonated side masses."""
    x: float
    y: float
    ch_x: int
    ch_y: int
    ranking: int = -1

    def images(self, basis: "Basis", tol_ppm: float, precursor_ppm: float):
        """The TFFC's two possible placements, each with its own sigma.

        A TFFC folds to one breakpoint but has two *images* -- "trust Y" and
        "trust X" -- because one FFC alone cannot say which side is the
        N-terminal fragment (``line_based.md`` §8b).  On a *parental* line the
        two images coincide, since ``x + y == basis.parent_pair`` by
        construction -- which holds for c/z too, on its own basis.
        """
        return (coord_from_side(self.y, basis, tol_ppm, precursor_ppm),
                coord_from_side(self.x, basis, tol_ppm, precursor_ppm))


@dataclass
class LineInfo:
    """A detected line and everything the reconstruction needs from it."""
    key: str
    i: int
    j: int
    mass: float
    a_shift: float                # alignment shift, from the FFC masses
    tffcs: List[Tffc] = field(default_factory=list)
    category: str = "other"       # parental | loss | internal | other
    detail: str = ""
    # filled in by the iteration, for the report
    n_inactive: int = 0
    n_both_known: int = 0
    eligible: bool = False
    n_banked: int = 0
    n_unambiguous: int = 0
    n_ambiguous: int = 0
    n_unproductive: int = 0
    mock: Optional[Tuple[float, float]] = None    # (coord, sigma)
    mock_refused: str = ""

    @property
    def n_ffc(self) -> int:
        return len(self.tffcs)

    @property
    def best_rank(self) -> int:
        ranks = [t.ranking for t in self.tffcs if t.ranking >= 0]
        return min(ranks) if ranks else -1

    def mock_ok(self, secondary_gate: float) -> bool:
        """May this line's mock be banked?  (The *secondary-pair* test.)

        An internal line's mock rests on the claim that the mass ``A`` the line
        is missing was lost from an **end** of the peptide -- then the cleavage
        that released it is a real breakpoint at ``fold(A)``.  That is true when
        the line's FFCs are internal-ion pairs (a terminal run is gone from
        every fragment on the line), which is the case ``reconstruct.md`` §6
        describes and the one KWK's ``A = 314.161`` -> b2 mock comes from.

        But an internal line can arise a second way, and then the claim is
        false: a **secondary** pair (Yalcin 1996; ``line_based.md`` §5) is two
        ordinary terminal ions ``b_k`` and ``y_{n-k-r}`` with an *interior* run
        of ``r`` residues between them.  ``A`` is then that interior run's mass
        and ``fold(A)`` is not a cleavage coordinate at all -- it fabricates a
        breakpoint near a terminus.  Measured: HGT's three ``A = 113.08 / 131.09``
        lines invented an N-terminal ``J`` and cost the reconstruction, even
        though HGT starts with H (137.06) and ends with K (128.09).

        The two cases separate cleanly without any ground truth.  A secondary
        pair's two sides are *both* genuine terminal fragments, so **both** of
        the TFFC's images are already known breakpoints; a true internal pair
        has at most one side that is (the other is an internal ion, which is no
        cleavage coordinate).  So a line whose TFFCs are repeatedly
        "known on both images" is a secondary line, and its mock is refused.
        """
        if not self.tffcs:
            return False
        return self.n_both_known <= secondary_gate * len(self.tffcs)


def extract_tffcs(ffc_df: pd.DataFrame, line, col_a: str, col_b: str,
                  ranking_col: str = "Ranking") -> List[Tffc]:
    """Member FFCs of ``line``, with each one's orientation resolved.

    Detection is orientation-agnostic (``line_based.md`` §4): the line stores a
    canonical ``(i, j)`` and each member's true orientation is whichever of
    ``i*mzA + j*mzB`` / ``j*mzA + i*mzB`` lands closer to the line mass.  Once
    the per-side charge is known, ``i*mz - (i-1)*MASS_H`` is that side's
    ``[M+H]+`` mass -- exactly ``coverage_table_internal``'s ``adj_a``/``adj_b``.
    """
    out: List[Tffc] = []
    idx = [k for k in np.asarray(line.member_indices).tolist() if k in ffc_df.index]
    if not idx:
        return out
    sub = ffc_df.loc[idx]
    for k, row in sub.iterrows():
        mz_a, mz_b = float(row[col_a]), float(row[col_b])
        d1 = abs(line.i * mz_a + line.j * mz_b - line.mass)
        d2 = abs(line.j * mz_a + line.i * mz_b - line.mass)
        ch_a, ch_b = (line.i, line.j) if d1 <= d2 else (line.j, line.i)
        x = ch_a * mz_a - (ch_a - 1) * MASS_H
        y = ch_b * mz_b - (ch_b - 1) * MASS_H
        rank = int(row[ranking_col]) if ranking_col in sub.columns else -1
        out.append(Tffc(x=x, y=y, ch_x=ch_a, ch_y=ch_b, ranking=rank))
    return out


def _median(vals: Sequence[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _isotope_k(a_shift: float, tol: float, max_k: int = 2) -> int:
    """``k`` if ``A ~= k * 1.00335`` for a small non-zero ``k``, else 0.

    Such a line is the **parental ladder re-appearing with one side sitting on
    its +-k-th isotope peak** -- the deconvolution does not always pick the
    monoisotopic peak (``etd.md`` §6a-i).  ``classify_line`` files it under
    ``other`` (|A| is below MIN_AA_MASS and matches no named loss), so without
    this it contributes nothing despite being the ladder.

    ⚠ Admitted as an EXTENSION candidate, never as a seed.  On a true parental
    line a TFFC's two images coincide, so seeding adds both without thinking.
    On a satellite they do not: with ``d = -A``, one side carries k isotopes, so
    the two images work out to ``u`` and ``u + d`` -- **1.003 Da apart, exactly
    one of them real**.  Seeding would bank the false one too, once per FFC.
    Routed through the ordinary two-image goodness test instead, the false image
    is ~1.003 Da from any bond -- hundreds of times the sigma window -- so it
    can only lose the test or surface as an incorrect coordinate; it can never
    be mis-attributed to a neighbouring bond. That is `etd.md` §9's argument,
    and it is why no coordinate correction is needed here.

    ⚠ A line with no parental seed to be judged against cannot be rescued this
    way at all -- there is nothing to test goodness against. Measured:
    `ETD-13.84ms` has its whole ladder on satellites and no A ~= 0 line at any
    charge sum, and correctly reports 0 coverage rather than a guess.

    ⚠ Not separable from a b/y line in EThcD (``etd.md`` §6a-i).  A plain b/y
    pair sums to ``M``, so against the c/z ParentPair (``M + H``) it sits at
    ``A = 1.00783`` -- 0.0045 Da from a +1 satellite's 1.00335, well inside the
    0.01-0.03 Da spread a line shift carries.  Feeding a b/y pair through the
    c/z basis is wrong, but benignly so: a mis-based coordinate lands on no bond
    rather than on the wrong one.  Set ``include_isotope_lines=False`` on
    EThcD-like data if that matters.

    This is NOT ``mark_isotope_satellites``'s case, and that guard stays
    untouched: there an *internal* line echoes another *internal* line, the whole
    alignment is shifted, and every coordinate it produces is 1.003 Da from
    truth.  Here the alignment is zero and only one image per FFC is wrong.
    """
    for k in range(1, max_k + 1):
        for sign in (1, -1):
            if abs(a_shift - sign * k * ISOTOPE_OFFSET) <= tol:
                return sign * k
    return 0


def _mock_sigma(total: float, n_ffc: int, tol_ppm: float, cap_da: float) -> float:
    """Uncertainty of a line's alignment shift ``A``.

    ``A = median(ParentPair - (x + y))`` over the line's FFCs, so it is a
    *difference of large masses*: its absolute error is set by ``total``, not by
    the (usually small) value of ``A`` itself.  This is precisely why ``_oa_v2``
    needed a separate, looser ``mock_tol_ppm`` -- on KWK the ``A = 314.161``
    mock sits 0.013 Da from its bond, which is 41 ppm *of the coordinate* but
    only 3 ppm of the masses it was computed from.  Here it is derived instead
    of tuned: two measured sides at ``tol_ppm`` each, then a median over ``n``
    FFCs, capped so a sparse line cannot claim an unusably wide window.
    """
    per_ffc = total * tol_ppm * 1e-6 * math.sqrt(2.0)
    return min(per_ffc / math.sqrt(max(n_ffc, 1)), cap_da)


# =============================================================================
# 4. THE PIPELINE
# =============================================================================

@dataclass
class DenovoResult:
    total: float            # coordinate-basis total (TOTAL_STAR for ETD)
    term: float             # coordinate-basis y-axis origin (negative for ETD)
    basis: "Basis"
    lines: List[LineInfo]
    master: MasterSet
    reconstructions: List[dict]
    n_paths: int
    rounds: int
    best_weight: float
    paths_raw: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


def run_denovo(
    ffc_df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    terminus: str = "H2O",
    fragmentation: str = "HCD",
    *,
    # ── line detection ──────────────────────────────────────────────────────
    delta: float = 0.015,
    delta_unit: str = "da",
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    # ── tolerance model ─────────────────────────────────────────────────────
    tol_ppm: float = 25.0,
    precursor_ppm: float = 10.0,
    sigma_floor_ppm: float = 3.0,
    mock_sigma_cap_da: float = 0.05,
    # ── master-set construction ─────────────────────────────────────────────
    master_align_threshold: int = 1,
    accept_ambiguity: bool = False,
    use_mocks: bool = True,
    include_isotope_lines: bool = True,
    isotope_tol_da: float = 0.03,
    mock_min_ffc: int = 4,
    mock_weight: float = 1.0,
    secondary_gate: float = 0.25,
    evidence_weight: float = 0.0,
    evidence_cap: float = 1.0,
    rank_floor: float = 1.0,
    rank_scale: Optional[float] = None,
    gate_mock_on_goodness: bool = True,
    max_rounds: int = 20,
    # ── reconstruction ──────────────────────────────────────────────────────
    max_paths: int = 1000,
    label_tol: float = 0.02,
    edges_path: Optional[str] = None,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    verbose: bool = False,
) -> DenovoResult:
    """Sequence a peptide de novo from one MMS map.

    Parameters
    ----------
    ffc_df
        The FFC table, already merged and cut to a top-N ranking window
        (``merge_duplicate_ffcs_charge`` then ``prepare_ffc_data``).
    parent_charge, parent_mass
        Precursor charge and **neutral** mass -- both measured quantities, not
        properties of the (unknown) sequence.
    terminus
        ``"H2O"`` (free acid) or ``"NH3"`` (amide).  This is chemistry, known
        from the sample, and it is *not* optional: it sets where the y-axis
        starts, and getting it wrong fails silently -- paths are still found,
        none of them are right (``line_based.md`` §8a).
    fragmentation
        ``"HCD"`` (b/y, the default) or ``"ETD"`` / ``"ETHCD"`` (c/z-dot).  See
        §0: it changes the coordinate basis and the ladder's charge sum, nothing
        else.  Like ``terminus``, a wrong value fails silently.
    tol_ppm
        Relative error of a deconvolved fragment mass.  Every coordinate's
        sigma is derived from this and from the mass it was measured at.
    precursor_ppm
        Relative error of ``parent_mass``; enters whenever a coordinate is
        mirrored through the precursor, and the sink edge.
    sigma_floor_ppm
        Lower bound on a merged coordinate's sigma, so corroboration cannot
        shrink the window past what the instrument can actually deliver.
    master_align_threshold
        Inactive TFFCs an internal / loss line needs before it may join the
        master set -- the ``aligned?`` gate of ``line_based.md`` §8c.
    accept_ambiguity
        Bank **both** images of an ambiguous TFFC (one is false by
        construction) instead of neither.  Off by default, as upstream.
    include_isotope_lines, isotope_tol_da
        Let a line sitting at ``A ~= k * 1.00335`` (k = +-1, +-2) contribute, as
        an *extension* candidate.  See the note above :func:`_isotope_k`.
    use_mocks, mock_min_ffc, mock_weight
        Whether an internal line contributes the breakpoint named by its own
        shift, how many FFCs it must carry first, and how much of a vote that
        breakpoint gets in the heaviest-path search.
    secondary_gate
        Fraction of a line's TFFCs that may be "already known on *both* images"
        before its mock is refused as a **secondary** (interior-gap) line --
        see :meth:`LineInfo.mock_ok`.  This is the de novo replacement for
        ``_oa_v2``'s ``_bond_of`` check on ``fold(A)``.
    evidence_weight, evidence_cap
        Extra weight per additional independent TFFC on one breakpoint (capped),
        so the search prefers the corroborated ladder when paths tie.
    rank_floor, rank_scale
        Rank-aware confidence.  A breakpoint whose best supporting FFC is ranked
        ``rank_scale`` (default: the size of the table) gets ``rank_floor`` of a
        vote in the heaviest-path search; one ranked 0 gets a full vote.
        ``rank_floor = 1.0`` (the default) ignores rankings entirely, which is
        what every stage upstream of this module does -- the ranking is used
        only to *cut* the table at top-N.  That is a real loss of information in
        de novo mode: false coordinates come overwhelmingly from the poorly
        ranked tail (measured: KWK's incorrect count goes 0 / 2 / 6 / 12 across
        top-N 200 / 500 / 1000 / 2000), and weighting lets a wider table be read
        without letting its tail out-vote its head.
    gate_mock_on_goodness
        A de novo substitute for ``_oa_v2``'s ``loose_mock`` bond check: admit a
        line's mock only if the coordinate is already in the master set or
        lengthens the heaviest path.  ``_oa_v2`` could ask whether ``fold(A)``
        lands on a real cleavage; we cannot, so we ask whether it *behaves* like
        one.
    """
    from master_reconstruction import classify_line, mark_isotope_satellites
    from master_reconstruction import LineRec as _LineRec
    from line_internal_annotation_oa_v2 import detect_lines_oa
    from edges import load_edges
    import path_finding as pf

    if edges_path is None:
        edges_path = DEFAULT_EDGES_PATH

    basis = make_basis(fragmentation, parent_mass, terminus)
    # Everything downstream works on the COORDINATE basis; only `parent_pair`
    # and the sigma scales use the real precursor mass (§0).
    total = basis.coord_total
    term = basis.coord_term
    source = (0.0, term)
    sink = pf.SINK_NODE
    parent_pair = basis.parent_pair
    precursor_sigma = basis.total * precursor_ppm * 1e-6
    allowed = sorted(load_edges(edges_path))

    def log(*a):
        if verbose:
            print(*a)

    # ── 1. Line detection (already fully de novo) ───────────────────────────
    line_charge = basis.line_charge(parent_charge)
    log(f"  basis {basis.mode}: ladder charge sum {line_charge} "
        f"(precursor {parent_charge}), coord_total {total:.4f}, "
        f"ParentPair {parent_pair:.4f}")
    line_list, master_lines = detect_lines_oa(
        ffc_df, line_charge, delta, min_ffc_number,
        col_a=col_a, col_b=col_b, line_tol=line_tol, delta_unit=delta_unit)
    log(f"  lines: {len(master_lines)} parental-charge + {len(line_list)} residual")

    # ── 2. Turn each line into TFFCs and an alignment shift ─────────────────
    # A is taken from the FFC masses (median of ParentPair - (x + y)), NOT from
    # the detection shift, which carries an extra (i + j - z)*proton -- see
    # `line_based.md` §6a: a charge sweep is a charge *assumption*, and A has no
    # proton term so it needs no guess.
    lines: List[LineInfo] = []
    for ln in list(master_lines) + list(line_list):
        tffcs = extract_tffcs(ffc_df, ln, col_a, col_b, ranking_col)
        if not tffcs:
            continue
        a_shift = _median([parent_pair - (t.x + t.y) for t in tffcs])
        key = f"({ln.i},{ln.j}) A={a_shift:+.3f}"
        info = LineInfo(key=key, i=ln.i, j=ln.j, mass=float(ln.mass),
                        a_shift=a_shift, tffcs=tffcs)
        info.category, info.detail = classify_line(a_shift, allowed)
        lines.append(info)

    # Isotope satellites are false-positive factories (`line_based.md` §8c /
    # reconstruct.md §4): a line reappearing 1.00335 Da below a stronger one
    # produces virtual TFFCs that are close enough to look plausible and far
    # enough to be wrong.  Demote them before anything uses them.
    recs = [_LineRec(key=li.key, i=li.i, j=li.j, mass=li.mass, shift=0.0,
                     a_shift=li.a_shift, a_from_detection=0.0)
            for li in lines]
    for rec, li in zip(recs, lines):
        # ``LineRec.n_ffc`` counts members with ``kind == 'observed'``; that and
        # ``category`` / ``a_shift`` / ``key`` are all the satellite test reads.
        rec.tffcs = [_Observed()] * li.n_ffc
        rec.category, rec.detail = li.category, li.detail
    mark_isotope_satellites(recs)
    for rec, li in zip(recs, lines):
        li.category, li.detail = rec.category, rec.detail

    # Parental isotope satellites (etd.md §6a-i).  Done AFTER
    # mark_isotope_satellites, which guards a different case entirely -- see
    # _isotope_k -- and only ever touches `internal` lines, so the two cannot
    # interfere.
    if include_isotope_lines:
        for li in lines:
            if li.category != "other":
                continue
            k = _isotope_k(li.a_shift, isotope_tol_da)
            if k:
                li.category = "isotope"
                li.detail = f"parental {k:+d} isotope satellite"

    parental = [li for li in lines if li.category == "parental"]
    log(f"  parental lines (|A| < 0.5): {len(parental)}")

    # ── 3. Seed the master set from the parental lines ──────────────────────
    # A parental TFFC's two sides are two independent measurements of the SAME
    # cleavage (x + y == ParentPair), so both are added and MasterSet merges
    # them -- the lighter side, being measured more precisely in absolute terms,
    # dominates the inverse-variance mean.
    master = MasterSet(total, sigma_floor_ppm, mock_weight,
                       evidence_weight, evidence_cap, rank_floor,
                       rank_scale if rank_scale else max(len(ffc_df), 1))
    for li in parental:
        for t in li.tffcs:
            for c, s in t.images(basis, tol_ppm, precursor_ppm):
                master.add(c, s, "parental", li.key, rank=t.ranking)
    log(f"  seed master set: {len(master)} breakpoints")

    scorer = Scorer(allowed, total, source, sink, precursor_sigma, max_paths)

    def score_with(ms: MasterSet, extra=None) -> float:
        return scorer.score(*ms.view(extra))

    # ── 4. Extend, in batch rounds, until the master set stops growing ───────
    # Identical control flow to `_oa_v2`'s `_tffc_analysis` (`line_based.md`
    # §8d): every TFFC in a round is judged against the set as it stood at the
    # START of the round and the winners are banked together at the end, which
    # is what keeps a round order-independent.  Only *unambiguous* TFFCs are
    # banked during the iteration; ambiguous ones are held back and re-tested,
    # since banking a forced placement is exactly what can disambiguate them.
    candidates = [li for li in lines
                  if li.category in ("internal", "loss", "isotope")]
    # Mock: the line is missing a fragment of mass A, so the cleavage that
    # produced it is a real breakpoint at fold(A).  Only `internal` lines get
    # one -- a neutral loss is not a cleavage.
    if use_mocks:
        for li in lines:
            if li.category == "internal" and li.n_ffc >= mock_min_ffc:
                s = _mock_sigma(basis.total, li.n_ffc, tol_ppm, mock_sigma_cap_da)
                li.mock = (fold(li.a_shift, total), s)

    pending: Dict[str, List[Tffc]] = {li.key: list(li.tffcs) for li in candidates}
    by_key = {li.key: li for li in lines}
    mocks_added: set = set()
    work = master.snapshot()
    rounds_used = 0

    # Most populated line first: the best-corroborated line's forced placements
    # are the safest to bank.  The key is stable, so the order is deterministic.
    order = sorted(pending, key=lambda k: (-by_key[k].n_ffc, k))

    for r in range(max_rounds):
        rounds_used = r + 1
        s0 = score_with(work)

        # Eligibility is re-derived every round: a line with no inactive TFFC in
        # round 1 can acquire one later and only then qualify (§8d).
        round_eligible = []
        for k in order:
            li = by_key[k]
            n_inactive = n_both = 0
            for t in li.tffcs:
                (hc, hs), (vc, vs) = t.images(basis, tol_ppm, precursor_ppm)
                in_h, in_v = work.contains(hc, hs), work.contains(vc, vs)
                n_inactive += bool(in_h or in_v)
                n_both += bool(in_h and in_v)
            li.n_inactive, li.n_both_known = n_inactive, n_both
            if n_inactive >= master_align_threshold:
                round_eligible.append(k)

        picks, new_mocks = [], []
        for k in round_eligible:
            li = by_key[k]
            li.eligible = True
            if li.mock is not None and k not in mocks_added:
                mocks_added.add(k)
                mc, ms_ = li.mock
                ok = (li.mock_ok(secondary_gate)
                      and (not gate_mock_on_goodness
                           or work.contains(mc, ms_)
                           or score_with(work, [(mc, ms_, "mock")]) > s0 + 1e-9))
                if not li.mock_ok(secondary_gate):
                    li.mock_refused = "secondary"
                if ok and not work.contains(mc, ms_):
                    new_mocks.append((k, mc, ms_))
            for t in pending[k]:
                (hc, hs), (vc, vs) = t.images(basis, tol_ppm, precursor_ppm)
                if work.contains(hc, hs) or work.contains(vc, vs):
                    continue                      # inactive; settled at the end
                good_h = score_with(work, [(hc, hs, "observed", t.ranking)]) > s0 + 1e-9
                good_v = score_with(work, [(vc, vs, "observed", t.ranking)]) > s0 + 1e-9
                if good_h == good_v:
                    continue                      # ambiguous / unproductive: hold
                picks.append((k, t, (hc, hs) if good_h else (vc, vs)))

        if not picks and not new_mocks:
            rounds_used = r                       # this round banked nothing
            break

        for k, mc, ms_ in new_mocks:
            work.add(mc, ms_, "mock", k, rank=by_key[k].best_rank)
            by_key[k].n_banked += 1
        for k, t, (c, s) in picks:
            work.add(c, s, "unambiguous", k, rank=t.ranking)
            pending[k].remove(t)
            by_key[k].n_unambiguous += 1
            by_key[k].n_banked += 1

    # ── 5. Settle whatever never got banked, against the converged set ──────
    s0 = score_with(work)
    for k, rest in pending.items():
        li = by_key[k]
        for t in rest:
            (hc, hs), (vc, vs) = t.images(basis, tol_ppm, precursor_ppm)
            if work.contains(hc, hs) or work.contains(vc, vs):
                li.n_unproductive += 1            # adds nothing new
                continue
            good_h = score_with(work, [(hc, hs, "observed", t.ranking)]) > s0 + 1e-9
            good_v = score_with(work, [(vc, vs, "observed", t.ranking)]) > s0 + 1e-9
            if good_h and good_v:
                li.n_ambiguous += 1
                # An ambiguous TFFC's evidence is "one of these two", never
                # "this one", so the pair is atomic: both or neither (§8e).
                if li.eligible and accept_ambiguity:
                    work.add(hc, hs, "ambiguous", k, rank=t.ranking)
                    work.add(vc, vs, "ambiguous", k, rank=t.ranking)
                    li.n_banked += 2
            elif good_h or good_v:
                li.n_unambiguous += 1
                if li.eligible:
                    c, s = (hc, hs) if good_h else (vc, vs)
                    work.add(c, s, "unambiguous", k, rank=t.ranking)
                    li.n_banked += 1
            else:
                li.n_unproductive += 1

    log(f"  master set after {rounds_used} rounds: {len(work)} breakpoints")

    # ── 6. Reconstruct from the whole master TFFC-set ───────────────────────
    coords, sigma, weights = work.view()
    best, raw_paths = scorer.paths(coords, sigma, weights)
    rows: List[dict] = []
    if best != float("-inf"):
        merge_tol = max(list(sigma.values()) + [0.0]) * 2.0
        for n, (_masses, node_seq) in enumerate(raw_paths, start=1):
            stripped = node_seq[:-1]              # drop the Sink sentinel
            rows.append({
                "#": n,
                "reconstruction": _format_gapped(stripped, total, term,
                                                 merge_tol, label_tol),
            })

    return DenovoResult(
        total=total, term=term, basis=basis, lines=lines, master=work,
        reconstructions=rows, n_paths=len(rows), rounds=rounds_used,
        best_weight=(-1.0 if best == float("-inf") else best),
        paths_raw=[p[1][:-1] for p in raw_paths],
        params=dict(delta=delta, delta_unit=delta_unit,
                    min_ffc_number=min_ffc_number, tol_ppm=tol_ppm,
                    precursor_ppm=precursor_ppm,
                    sigma_floor_ppm=sigma_floor_ppm,
                    mock_sigma_cap_da=mock_sigma_cap_da,
                    master_align_threshold=master_align_threshold,
                    accept_ambiguity=accept_ambiguity, use_mocks=use_mocks,
                    mock_min_ffc=mock_min_ffc, mock_weight=mock_weight,
                    include_isotope_lines=include_isotope_lines,
                    secondary_gate=secondary_gate,
                    evidence_weight=evidence_weight, rank_floor=rank_floor,
                    gate_mock_on_goodness=gate_mock_on_goodness,
                    n_ffc=len(ffc_df), parent_charge=parent_charge,
                    parent_mass=parent_mass, terminus=terminus,
                    fragmentation=basis.mode, line_charge=line_charge),
    )


class _Observed:
    """Stand-in with ``kind == 'observed'`` for ``LineRec.n_ffc``."""
    kind = "observed"


def _format_gapped(path, total: float, term: float, merge_tol: float,
                   label_tol: float) -> str:
    """Gapped N->C reconstruction of one path.

    Same rendering as ``master_reconstruction.format_gapped``: collapse the
    b-side and mirrored y-side breakpoints into one ladder, then print each
    segment as a residue letter or as its mass.  Naming uses the *tighter*
    tolerance -- with a wide window a measured gap can sit closer to the wrong
    residue (K and Q are 0.036 Da apart), and a confidently wrong letter is
    worse than an honest number.  ``(sink)`` marks the one gap the two ladders
    never bridged.
    """
    import path_finding as pf
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
