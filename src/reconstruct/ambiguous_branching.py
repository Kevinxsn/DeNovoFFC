#!/usr/bin/env python
"""
ambiguous_branching.py -- resolve ambiguous TFFCs by **branching** instead of
adding both of their images to one master set.

Why (the problem this exists to fix)
====================================
``master_reconstruction.py`` follows ``reconstruct_writeup.md`` §6: when a TFFC
``(X, Y)`` on a line with alignment shift ``A`` is *ambiguous* -- both of its
virtual images look good -- both are added to the master set, each at weight
1/2, on the grounds that at most one of them can be real.

That is structurally unsafe, and not just "sometimes unlucky":

  * The two images are the two breakpoints ``r`` and ``r + A`` on the same
    N->C ladder.  (Trivially: the b-side reading of X is ``X - proton = r``, and
    the b-side reading of Y's complement is ``total - (Y - proton) = r + A``,
    because ``X + Y = ParentPair - A``.)
  * ``A`` is **always** an allowed edge mass -- a line is only classified
    ``internal`` if its ``A`` is a residue-run mass present in ``edges.json``.
  * So the 2D-DAG *always* contains an edge joining the two images, and a path
    can collect both.  Collecting both scores ``2w``; collecting only the
    correct one scores ``w``.  For any weight ``w > 0`` the path that swallows
    the false image with the true one wins.

Measured on LLG 6+: both halves of the ``(2,3) -227.170`` pair are used by all
8 winning paths, and the correct reconstruction is out-scored.  Lowering
``--ambiguous_weight`` cannot help, because both halves carry the *same* weight
-- it never breaks the tie between them.

What this script does instead
=============================
At every ambiguous decision the search **forks**: one branch commits to the
horizontal image, one to the vertical, and the extension continues
independently down each.  Only one image is ever present in a given master set,
so the pathological "use both" path cannot exist anywhere.  Every leaf of the
tree is a complete, self-consistent master TFFC-set; they are then scored and
compared by heaviest-path weight.

The branching is a genuine tree search, not a post-hoc enumeration: the
classification of later TFFCs depends on what was added earlier, so each branch
re-runs the remaining extension with its own master set.  The chosen image still
carries ``--ambiguous_weight`` (default 1/2), as the write-up asks -- inside one
branch that weight is now doing something meaningful (it discounts a guessed
TFFC against a corroborated one) rather than being cancelled out.

Cost: ``B^k`` leaves for ``k`` ambiguous decisions, ``B`` = 2 by default.
LLG 6+ has k=2 -> 4 leaves.  ``--max_leaves`` caps the explosion.

  ``--branch_policy either``          {h}, {v}                 (default, B=2)
  ``--branch_policy either_or_none``  {h}, {v}, {}             (B=3)
  ``--branch_policy all``             {}, {h}, {v}, {h,v}      (B=4)

``all`` is the literal 4-way version -- it keeps the "add both" branch so you
can see it lose (or win) on its own merits, and a "drop it" branch.  The default
is the meaningful subset: ``{}`` adds nothing and ``{h,v}`` is the pathology
above, so neither is a hypothesis about *which image is real*.

Note the interaction with ``--rank_by corroborated`` (below): its tie-break is
"fewest guesses", which is only meaningful when every branch is *forced* to
commit.  Offer a ``{}`` branch and that tie-break trivially prefers guessing
nothing -- on LLG, ``--branch_policy all`` wins with "drop all three", which is
correct but two breakpoints poorer (19 bonds instead of 21).  Keep the default
``either`` unless you specifically want to watch the alternatives.

What to look for in the output
==============================
Branching is not guaranteed to *identify* the right image -- if both images
extend the path equally, their branches tie and the ambiguity is reported
honestly instead of being silently resolved the wrong way.  What it does
guarantee is that the correct reconstruction is no longer beaten by a path that
eats both halves of a pair.  The report shows, per leaf, which image each
decision took, whether it was the true bond, and whether that leaf's heaviest
paths contain the correct reconstruction.

Usage
-----
    python ambiguous_branching.py --data ... --seq ... --charge 6 --terminus h2o
    python ambiguous_branching.py --branch_policy all --max_leaves 64
    ./run_llg.sh                    # the LLG case this was written for

Every ``master_reconstruction.py`` flag is accepted and means the same thing,
except ``--ambiguous_policy``, which is replaced by ``--branch_policy``.
"""

from __future__ import annotations

import argparse
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import master_reconstruction as mr
from master_reconstruction import (           # noqa: F401  (readability)
    MasterEntry, MasterSet, Origin, PathScorer, TFFC, LineRec,
    PROTON, fold,
)
import path_finding as pf


# =============================================================================
# 1. SETUP  (mirrors master_reconstruction._run_once up to the extension step)
# =============================================================================

@dataclass
class Setup:
    pep: object
    n: int
    prefix: List[float]
    term: float
    total: float
    parent_pair: float
    allowed: List[float]
    records: List[LineRec]
    selected: List[LineRec]
    master0: MasterSet
    scorer: PathScorer
    source: Tuple[float, float]
    reference: Optional[MasterSet] = None   # fixed inactive-test reference

    def bond_of(self, coord: float, tol: float) -> Optional[int]:
        for k in range(1, self.n):
            if abs(fold(self.prefix[k], self.total) - coord) <= tol:
                return k
        return None


def setup(args, ctx, log=print) -> Setup:
    """Peptide model, line records, seeded master set -- everything the
    branching search needs before the first ambiguous decision."""
    end = True if args.terminus.lower() in ("h2o", "water", "true") else "NH3"
    pep = mr.peptide.Pep(f"[{args.seq}+{args.charge}H]{args.charge}+", end_h20=end)
    n = len(pep.AA_array)
    prefix = mr.residue_prefixes(pep)
    term = mr.terminus_mass(pep)
    total = prefix[n] + term
    parent_mass = total + args.charge * PROTON
    parent_pair = total + 2 * PROTON

    pf.SOURCE_NODE = (0.0, term)
    source = pf.SOURCE_NODE
    allowed = [m for m in ctx.allowed if m <= total]

    records = mr.build_line_records(
        ctx.ffc_df, ctx.lines, pep, parent_mass, args.charge, parent_pair,
        allowed, annotate=not args.no_annotate, iso_range=args.iso_range,
        tol_ppm=args.tol_ppm)
    if not args.keep_isotope_lines:
        mr.mark_isotope_satellites(records, tol=args.coord_tol)

    master0 = MasterSet(total, args.coord_tol)
    mr.seed_master(records, master0, pep, source=args.parental_source,
                   prefix=prefix, iso_range=args.iso_range,
                   tol_ppm=args.tol_ppm, log=log)
    reference = master0.snapshot() if args.inactive_vs == "parental" else None

    # Line selection -- same predicates as master_reconstruction._run_once.
    selected = []
    for rec in records:
        if rec.category == "parental":
            continue
        if rec.category not in mr.USABLE_CATEGORIES:
            continue
        if args.require_annot and rec.annot_type is not None and \
                mr.lba._base_category(rec.annot_type) not in ("parental", "internal"):
            continue
        rec.n_inactive = mr.count_inactive(rec, master0)
        if rec.n_inactive < args.inactive_count:
            continue
        rec.used = True
        selected.append(rec)
    selected.sort(key=lambda r: (-r.n_inactive, -r.n_ffc, r.key))
    for rec in selected:                      # mocks are part of the TFFC list
        mr.add_mock(rec, parent_pair)

    scorer = PathScorer(allowed, total, source, args.threshold, args.max_paths)
    return Setup(pep, n, prefix, term, total, parent_pair, allowed,
                 records, selected, master0, scorer, source, reference)


def clone_master(m: MasterSet) -> MasterSet:
    """Deep-enough copy: entries are rebuilt, origin records are shared."""
    new = MasterSet(m.total, m.tol)
    new.entries = [MasterEntry(e.coord, e.weight, list(e.origins))
                   for e in m.entries]
    return new


# =============================================================================
# 2. PREMISE CHECK
# =============================================================================
# Confirms, on this dataset, the three claims the whole script rests on:
#   (a) the two images of an ambiguous TFFC are r and r + A;
#   (b) A is an allowed edge mass, so the DAG always joins them;
#   (c) the winning paths really do use both.

def premise_check(st: Setup, args, log=print) -> int:
    """Report ambiguous pairs found by the *unbranched* algorithm.  Returns the
    number of ambiguous TFFCs seen."""
    master = clone_master(st.master0)
    n_amb = 0
    pairs: List[Tuple[str, float, float, float]] = []   # line, r, r+A, A

    for rec in st.selected:
        remaining = list(rec.tffcs)
        for _ in range(args.max_rounds):
            base = st.scorer.score(master.coords(), master.weight_map())
            cls_all = [(t, *mr.classify_tffc(t, master, st.scorer, base, st.reference))
                       for t in remaining]
            pick = next((c for c in cls_all if c[1] == mr.CLASS_UNAMBIGUOUS), None)
            if pick is None:
                pick = next((c for c in cls_all if c[1] == mr.CLASS_AMBIGUOUS), None)
            if pick is None:
                break
            t, kind, imgs, _ = pick
            w = 1.0 if kind == mr.CLASS_UNAMBIGUOUS else args.ambiguous_weight
            if kind == mr.CLASS_AMBIGUOUS:
                n_amb += 1
                r1 = t.x - PROTON
                r2 = st.total - (t.y - PROTON)
                pairs.append((rec.key, min(r1, r2), max(r1, r2), abs(r2 - r1)))
            for c in imgs:
                master.add(c, weight=w, origin=Origin(rec.key, t.kind, kind, t))
            remaining = _retire(remaining, t, master.tol)

    log("=" * 78)
    log("PREMISE CHECK -- why adding both images is unsafe")
    log("=" * 78)
    if not pairs:
        log("  no ambiguous TFFC arises on this dataset; branching has nothing "
            "to do\n  and will return a single leaf identical to the "
            "unbranched result.")
        return 0

    srt = sorted(st.allowed)

    def edge_exists(gap: float) -> bool:
        i = bisect_left(srt, gap)
        return any(0 <= j < len(srt) and abs(srt[j] - gap) <= args.threshold
                   for j in (i - 1, i))

    log(f"  {len(pairs)} ambiguous TFFC(s); for each, the two images are two "
        f"breakpoints A apart:\n")
    log(f"  {'line':<20}{'image 1':>11}{'image 2':>11}{'A':>10}"
        f"  {'A is an edge?':<14}bonds")
    for key, lo, hi, A in pairs:
        b1 = st.bond_of(fold(lo, st.total), args.coord_tol)
        b2 = st.bond_of(fold(hi, st.total), args.coord_tol)
        log(f"  {key:<20}{lo:>11.3f}{hi:>11.3f}{A:>10.4f}"
            f"  {str(edge_exists(A)):<14}"
            f"{('b' + str(b1)) if b1 else '--'} / "
            f"{('b' + str(b2)) if b2 else '--'}")

    best, paths = st.scorer.paths(master.coords(), master.weight_map())
    both_used = 0
    for _, ns in paths:
        used = mr._path_coords(ns, st.source)
        for _, lo, hi, _A in pairs:
            f1, f2 = fold(lo, st.total), fold(hi, st.total)
            if any(abs(c - f1) <= master.tol for c in used) and \
               any(abs(c - f2) <= master.tol for c in used):
                both_used += 1
                break
    log(f"\n  unbranched heaviest weight {best:g} over {len(paths)} tied path(s); "
        f"{both_used} of them\n  use BOTH images of some pair -- the failure "
        f"mode this script removes.")
    return n_amb


def _retire(remaining: List[TFFC], t: TFFC, tol: float) -> List[TFFC]:
    """Drop a TFFC and its mirror (Y, X) -- same rule as the unbranched loop."""
    return [o for o in remaining
            if o is not t
            and not (abs(o.x - t.y) <= tol and abs(o.y - t.x) <= tol)]


# =============================================================================
# 3. THE BRANCHING SEARCH
# =============================================================================

@dataclass
class Choice:
    """One ambiguous decision, and which image the branch committed to."""
    line_key: str
    tffc: TFFC
    taken: Tuple[float, ...]
    options: Tuple[float, float]

    def label(self, st: Setup, tol: float) -> str:
        if not self.taken:
            return f"{self.line_key}: drop"
        if len(self.taken) == 2:
            return f"{self.line_key}: both"
        c = self.taken[0]
        side = "X" if abs(c - self.options[1]) <= tol else "Y"
        k = st.bond_of(c, tol)
        return (f"{self.line_key}: trust {side} -> {c:.3f}"
                + (f" (b{k})" if k else " (false)"))


@dataclass
class Leaf:
    master: MasterSet
    choices: List[Choice]
    weight: float = 0.0
    corroborated: float = 0.0     # same path weight with guesses counted as 0
    n_guesses: int = 0            # ambiguous images this branch committed to
    n_paths: int = 0
    n_correct: int = 0
    bonds: int = 0
    n_false: int = 0
    paths: list = field(default_factory=list)


def branch_options(t: TFFC, total: float, policy: str
                   ) -> List[Tuple[float, ...]]:
    """The image sets a branch may commit to.  ``images()`` returns
    ``(horizontal, vertical)`` = (trust Y, trust X)."""
    h, v = t.images(total)
    if policy == "either":
        return [(h,), (v,)]
    if policy == "either_or_none":
        return [(h,), (v,), ()]
    if policy == "all":
        return [(), (h,), (v,), (h, v)]
    raise ValueError(policy)


def explore(st: Setup, args, master: MasterSet, line_idx: int,
            remaining: Optional[List[TFFC]], choices: List[Choice],
            leaves: List[Leaf], stats: Dict[str, int]) -> None:
    """Depth-first over ambiguous decisions.

    Unambiguous picks are committed in place (no fork -- there is nothing to
    choose).  The first ambiguous pick forks, and each fork continues the whole
    remaining extension with its own copy of the master set, because what a
    later TFFC classifies as depends on what was added earlier.
    """
    while line_idx < len(st.selected):
        if len(leaves) >= args.max_leaves:
            stats["truncated"] += 1
            return
        rec = st.selected[line_idx]
        if remaining is None:
            remaining = list(rec.tffcs)

        base = st.scorer.score(master.coords(), master.weight_map())
        cls_all = [(t, *mr.classify_tffc(t, master, st.scorer, base, st.reference))
                   for t in remaining]

        pick = next((c for c in cls_all if c[1] == mr.CLASS_UNAMBIGUOUS), None)
        if pick is not None:
            t, _, imgs, _ = pick
            for c in imgs:
                master.add(c, weight=1.0,
                           origin=Origin(rec.key, t.kind,
                                         mr.CLASS_UNAMBIGUOUS, t))
            remaining = _retire(remaining, t, master.tol)
            continue

        pick = next((c for c in cls_all if c[1] == mr.CLASS_AMBIGUOUS), None)
        if pick is not None:
            t, _, imgs, _ = pick
            h, v = t.images(st.total)
            rest = _retire(remaining, t, master.tol)
            stats["decisions"] += 1
            for opt in branch_options(t, st.total, args.branch_policy):
                child = clone_master(master)
                for c in opt:
                    child.add(c, weight=args.ambiguous_weight,
                              origin=Origin(rec.key, t.kind,
                                            mr.CLASS_AMBIGUOUS, t))
                explore(st, args, child, line_idx, list(rest),
                        choices + [Choice(rec.key, t, tuple(opt), (h, v))],
                        leaves, stats)
            return                       # every continuation is in a child

        line_idx += 1                    # nothing productive left on this line
        remaining = None

    leaves.append(Leaf(master, list(choices)))


def score_leaf(leaf: Leaf, st: Setup, args) -> None:
    """Heaviest path, correctness and ground-truth coverage for one leaf.

    Two scores are computed.  ``weight`` is the ordinary heaviest-path weight.
    ``corroborated`` re-scores with every ambiguously-placed TFFC counted as
    **0**, which is the score a branch earns from evidence it did not guess.

    The distinction matters because comparing branches by ``weight`` rewards a
    branch simply for *making more guesses*: committing to a false image can
    unlock a further ambiguous TFFC elsewhere, and each guess adds
    ``--ambiguous_weight`` regardless of whether it is right.  Measured on
    LLG 6+, the all-correct branch and the winning branch have the *same* 19
    corroborated TFFCs; the winner is ahead only by one extra guess, and no
    positive ``--ambiguous_weight`` can undo that (at w the gap is exactly w).
    """
    weights = leaf.master.weight_map()
    best, paths = st.scorer.paths(leaf.master.coords(), weights)
    leaf.weight = 0.0 if best == float("-inf") else best
    leaf.paths = paths
    leaf.n_paths = len(paths)
    leaf.n_correct = sum(
        1 for _, ns in paths
        if pf.is_reconstruction_correct(ns[:-1], st.total, args.seq,
                                        tolerance=args.threshold))
    covered, n_false, _ = mr.coverage_report(leaf.master, st.prefix,
                                             args.coord_tol)
    leaf.bonds = len(covered)
    leaf.n_false = n_false

    guessed = {e.coord for e in leaf.master.entries
               if any(o.decision == mr.CLASS_AMBIGUOUS for o in e.origins)}
    leaf.n_guesses = len(guessed)
    w0 = {e.coord: (0.0 if e.coord in guessed else e.weight)
          for e in leaf.master.entries}
    c = st.scorer.score(leaf.master.coords(), w0)
    leaf.corroborated = max(0.0, c)


def rank_key(leaf: Leaf, mode: str):
    """Branch ranking.

    ``corroborated`` (default): most evidence that was not guessed, then
    **fewest guesses** -- at equal corroborated support, prefer the branch that
    needed less speculation (plain parsimony), then the heavier total.
    ``weight``: the literal proposal, heaviest total path weight.
    """
    # Every term here must be computable without the answer.  ``leaf.bonds`` /
    # ``leaf.n_correct`` are ground truth and must never enter a ranking key.
    if mode == "weight":
        return (-leaf.weight, leaf.n_guesses)
    return (-leaf.corroborated, leaf.n_guesses, -leaf.weight)


# =============================================================================
# 4. REPORT
# =============================================================================

def report(leaves: List[Leaf], st: Setup, args, stats, log=print) -> Leaf:
    log("\n" + "=" * 78)
    log("BRANCHES")
    log("=" * 78)
    log(f"  {stats['decisions']} fork point(s) explored -> {len(leaves)} leaf "
        f"master set(s), policy '{args.branch_policy}'"
        + ("   [TRUNCATED at --max_leaves]" if stats["truncated"] else ""))

    order = sorted(range(len(leaves)),
                   key=lambda i: rank_key(leaves[i], args.rank_by) + (i,))
    log(f"  ranking by '{args.rank_by}'"
        + ("  (corroborated weight, then fewest guesses)"
           if args.rank_by == "corroborated" else "  (total path weight)"))
    log(f"\n  {'#':>3}{'corrob':>8}{'guess':>6}{'weight':>8}{'master':>8}"
        f"{'bonds':>7}{'false':>7}{'paths':>7}{'correct':>9}   choices")
    for rank, i in enumerate(order, start=1):
        lf = leaves[i]
        marks = " | ".join(c.label(st, args.coord_tol) for c in lf.choices) or "-"
        flag = "  <-- winner" if rank == 1 else ""
        log(f"  {rank:>3}{lf.corroborated:>8g}{lf.n_guesses:>6}{lf.weight:>8g}"
            f"{len(lf.master):>8}{lf.bonds:>7}{lf.n_false:>7}{lf.n_paths:>7}"
            f"{lf.n_correct:>9}   {marks}{flag}")

    other = "weight" if args.rank_by == "corroborated" else "corroborated"
    alt = min(range(len(leaves)), key=lambda i: rank_key(leaves[i], other) + (i,))
    if alt != order[0]:
        log(f"\n  NOTE: ranking by '{other}' instead would pick branch "
            f"#{order.index(alt) + 1} "
            f"({leaves[alt].n_correct}/{leaves[alt].n_paths} correct).")

    top = leaves[order[0]]
    key0 = rank_key(top, args.rank_by)[0]
    tied = [leaves[i] for i in order
            if abs(rank_key(leaves[i], args.rank_by)[0] - key0) < 1e-9]

    log("\n" + "=" * 78)
    log("WINNING BRANCH")
    log("=" * 78)
    if len(tied) > 1:
        log(f"  {len(tied)} branches tie on the primary criterion -- the data "
            f"does not fully resolve\n  the ambiguity.  Reported honestly "
            f"rather than guessed; branches carrying a correct\n  "
            f"reconstruction among them: "
            f"{sum(1 for lf in tied if lf.n_correct)}/{len(tied)}.")
    for c in top.choices:
        log(f"  decision: {c.label(st, args.coord_tol)}")
        log(f"            FFC (X, Y) = ({c.tffc.x:.4f}, {c.tffc.y:.4f})  "
            f"[{c.tffc.category or '?'}]  {c.tffc.label}")
    log(f"\n  master TFFCs {len(top.master)}   true bonds {top.bonds}/{st.n - 1}"
        f"   false {top.n_false}")
    for i, (_, ns) in enumerate(top.paths[:10], start=1):
        gapped = mr.format_gapped(ns[:-1], st.total, st.term, args.threshold,
                                  args.label_tol)
        ok = pf.is_reconstruction_correct(ns[:-1], st.total, args.seq,
                                          tolerance=args.threshold)
        log(f"    [{i}] {gapped}{'   [CORRECT]' if ok else ''}")
    if top.n_paths > 10:
        log(f"    ... {top.n_paths - 10} more")
    log(f"  correct reconstructions: {top.n_correct}/{top.n_paths}")
    return top


def build_parser() -> argparse.ArgumentParser:
    p = mr.build_parser()
    p.description = ("Branch on ambiguous TFFCs instead of adding both images "
                     "to one master set.")
    g = p.add_argument_group("ambiguous branching")
    g.add_argument("--branch_policy", default="either",
                   choices=["either", "either_or_none", "all"],
                   help="images a branch may commit to: 'either' = {h},{v} "
                        "(default); 'either_or_none' adds a drop-it branch; "
                        "'all' is the literal 4-way {},{h},{v},{h,v} -- it "
                        "keeps the add-both branch so you can watch it lose")
    g.add_argument("--rank_by", default="corroborated",
                   choices=["corroborated", "weight"],
                   help="how to compare branches: 'corroborated' (default) "
                        "scores guessed TFFCs as 0 and breaks ties by fewest "
                        "guesses, so a branch cannot win by guessing more; "
                        "'weight' is the plain heaviest-path weight")
    g.add_argument("--max_leaves", type=int, default=256,
                   help="hard cap on leaf master sets (B^k grows fast)")
    g.add_argument("--no_premise_check", action="store_true",
                   help="skip the report on why add-both is unsafe")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_annotate:
        args.parental_source = "all"
    if args.terminus == "auto":
        raise SystemExit("--terminus auto is not supported here; pick h2o or nh3")

    t0 = time.time()
    ctx = mr._prepare(args, log=print)
    st = setup(args, ctx, log=print)
    print(f"\nseeded master TFFCs: {len(st.master0)}   "
          f"lines selected for extension: {len(st.selected)} "
          f"({', '.join(r.key for r in st.selected) or '-'})")

    if not args.no_premise_check:
        print()
        premise_check(st, args, log=print)

    stats = {"decisions": 0, "truncated": 0}
    leaves: List[Leaf] = []
    explore(st, args, clone_master(st.master0), 0, None, [], leaves, stats)
    for lf in leaves:
        score_leaf(lf, st, args)
    top = report(leaves, st, args, stats, log=print)

    # ── Baseline (add-both) for comparison ────────────────────────────────
    base_args = argparse.Namespace(**vars(args))
    base_args.ambiguous_policy = "both"
    base = mr._run_once(base_args, args.terminus, ctx, log=lambda *a: None)

    print("\n" + "=" * 78)
    print("BRANCHING vs ADD-BOTH")
    print("=" * 78)
    print(f"{'':<26}{'add both (current)':>20}{'branching':>14}")
    print(f"{'master TFFCs':<26}{base['final']['master']:>20}"
          f"{len(top.master):>14}")
    print(f"{'true bonds covered':<26}{base['final']['bonds']:>20}"
          f"{top.bonds:>14}   (of {st.n - 1})")
    print(f"{'false TFFCs':<26}{base['final']['false']:>20}{top.n_false:>14}")
    print(f"{'heaviest path weight':<26}{base['final']['weight']:>20g}"
          f"{top.weight:>14g}")
    print(f"{'tied paths':<26}{base['final']['paths']:>20}{top.n_paths:>14}")
    print(f"{'correct reconstructions':<26}{base['final']['correct']:>20}"
          f"{top.n_correct:>14}")
    print(f"\ngraph builds: {st.scorer.n_builds}   "
          f"elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
