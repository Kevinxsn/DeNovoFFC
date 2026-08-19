#!/usr/bin/env python3
"""
Scoring a de novo result against a known peptide -- benchmark only
==================================================================

Everything in here uses the ground-truth sequence, and **nothing in here is
imported by ``denovo_core``**.  That separation is the point: the pipeline must
not be able to peek, so the only place a sequence appears is when we grade the
answer afterwards.

Three questions get answered:

``coverage``
    What fraction of the ``n-1`` backbone bonds did the master TFFC-set find?
    A master coordinate is credited to a bond when it agrees with that bond's
    folded ``prefix[k]`` within the coordinate's own sigma -- the same window
    the pipeline itself reasoned with, so a coordinate cannot be scored under a
    tolerance it was never allowed to use.

``#incorrect``
    Master coordinates matching no bond at all: breakpoints we added in error.
    ``line_based.md`` §8e's advice applies here too -- when the reconstruction
    is wrong, read this number first.

``correct?``
    Whether any tied reconstruction is consistent with the true sequence: every
    breakpoint the path implies must land on a real residue boundary
    (``path_finding.is_reconstruction_correct``'s criterion).  We re-implement
    it only to (a) accept the peptide's real C-terminus instead of the
    hard-coded water one and (b) use a *measurement-sized* tolerance rather than
    0.001 Da, since these coordinates are measured, not snapped onto exact
    masses.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "deiso", "python_code"),
           os.path.join(_ROOT, "reconstruct")):
    if _p not in sys.path:
        sys.path.append(_p)

from denovo_core import TERMINI, fold  # noqa: E402


def residue_prefixes(seq: str) -> List[float]:
    """``prefix[k]`` = summed residue mass of residues 1..k (``prefix[0] = 0``)."""
    import peptide  # noqa: E402
    aa = peptide.AA.amino_acid_masses
    prefix = [0.0]
    for ch in seq:
        prefix.append(prefix[-1] + aa[ch])
    return prefix


#: A gapped reconstruction is a " - "-joined list of tokens, each either a run of
#: one-letter residues (``"VEADJAGHG(sink)QEVJJR"``) or an unresolved mass
#: (``"737.251"``, possibly with a trailing ``" (sink)"``).  ``(sink)`` marks the
#: one gap the b- and y-ladders never bridged and is not part of the sequence.
def reconstruction_stats(s: str) -> dict:
    """How much sequence a gapped reconstruction actually resolves.

    ``resolved`` -- residues named by a letter; ``longest_run`` -- the longest
    contiguous stretch of them, i.e. the longest tag this answer could be
    searched with; ``gaps`` -- unresolved mass segments.
    """
    if not s:
        return dict(resolved=0, longest_run=0, gaps=0)
    resolved = longest = gaps = 0
    tokens = [t.strip() for t in s.split(" - ") if t.strip()]
    for t in tokens:
        letters = "".join(ch for ch in t.replace("(sink)", "") if ch.isalpha())
        # a mass token can carry a trailing "(sink)" but never letters
        if letters and not any(ch.isdigit() for ch in t.replace("(sink)", "")):
            resolved += len(letters)
            longest = max(longest, len(letters))
        else:
            gaps += 1
    return dict(resolved=resolved, longest_run=longest, gaps=gaps)


@dataclass
class Score:
    n_bonds: int
    bonds_found: List[int]
    coverage: float
    n_incorrect: int
    incorrect: List[tuple] = field(default_factory=list)
    n_paths: int = 0
    n_correct_paths: int = 0
    correct: bool = False
    true_mass: float = 0.0
    best_reconstruction: str = ""
    pep_len: int = 0
    resolved: int = 0
    longest_run: int = 0
    gaps: int = 0


def _bond_of(coord: float, sigma: float, folded_bonds) -> Optional[int]:
    """Backbone bond whose breakpoint matches ``coord``, or None.

    Folding is a bijection on bonds -- bond ``k``'s two masses are ``prefix[k]``
    and ``total - prefix[k]``, and ``fold`` picks exactly one -- so a folded
    coordinate names at most one cleavage.  Mis-attribution is not a risk:
    consecutive bonds are >= 57 Da (Gly) apart and the nearest one wins.
    """
    best = None
    for fk, k in folded_bonds:
        d = abs(fk - coord)
        if d <= sigma and (best is None or d < best[0]):
            best = (d, k)
    return best[1] if best else None


def score_result(result, seq: str, terminus: str,
                 path_tol_da: Optional[float] = None) -> Score:
    """Grade a :class:`~denovo_core.DenovoResult` against the true sequence.

    ⚠ The coordinate arithmetic reads its basis off the **result**, not off
    ``terminus``.  For ETD the pipeline works on ``TOTAL_STAR`` with a negative
    implied terminus (``denovo_core`` §0); recomputing ``TERMINI[terminus]``
    here would grade c/z coordinates on the b/y basis and score everything
    wrong.  ``terminus`` is used only to report the real precursor mass.
    """
    prefix = residue_prefixes(seq)
    n = len(seq)
    total = result.total                  # coordinate basis (TOTAL_STAR for ETD)
    term = result.term                    # coordinate-basis y-axis origin
    true_total = prefix[n] + TERMINI[terminus]
    folded_bonds = [(fold(prefix[k], total), k) for k in range(1, n)]

    bonds, incorrect = set(), []
    for e in result.master.entries:
        k = _bond_of(e.coord, e.sigma, folded_bonds)
        if k is None:
            incorrect.append((round(e.coord, 4), round(e.sigma, 4),
                              "/".join(sorted(set(e.kinds))),
                              ";".join(sorted(set(e.origins)))))
        else:
            bonds.add(k)

    # ── Is any tied path consistent with the true sequence? ─────────────────
    # Same criterion as path_finding.is_reconstruction_correct: every breakpoint
    # the path implies must sit on a real residue boundary.  The tolerance is
    # the widest sigma in the master set (not 0.001 Da), because these
    # coordinates are measured -- grading them at exact-mass precision would
    # fail every honest de novo answer.
    if path_tol_da is None:
        sig = [e.sigma for e in result.master.entries]
        path_tol_da = max(sig) if sig else 0.02
    import path_finding as pf
    aa = dict(pf.amino_acid_masses_merge)
    j = aa.get("J", 113.08406)
    aa["I"] = aa["L"] = j
    true_bp = [prefix[k] for k in range(n + 1)]

    m_res = total - term
    n_correct = 0
    best_str = ""
    for path, row in zip(result.paths_raw, result.reconstructions):
        b_side = [x1 for x1, _ in path]
        y_side = [m_res - (x2 - term) for _, x2 in path]
        bps = pf._merge_close([0.0, m_res] + b_side + y_side, path_tol_da)
        ok = all(any(abs(bp - t) <= path_tol_da for t in true_bp) for bp in bps)
        row["correct"] = "correct" if ok else ""
        if ok:
            n_correct += 1
            if not best_str:
                best_str = row["reconstruction"]
    if not best_str and result.reconstructions:
        best_str = result.reconstructions[0]["reconstruction"]

    return Score(
        n_bonds=n - 1,
        bonds_found=sorted(bonds),
        coverage=100.0 * len(bonds) / max(n - 1, 1),
        n_incorrect=len(incorrect),
        incorrect=incorrect,
        n_paths=result.n_paths,
        n_correct_paths=n_correct,
        correct=n_correct > 0,
        true_mass=true_total,
        best_reconstruction=best_str,
        pep_len=n,
        **reconstruction_stats(best_str),
    )
