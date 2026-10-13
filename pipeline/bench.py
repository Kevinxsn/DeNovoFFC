"""The benchmark: all seven peptides, scored against their known sequences.

⚠ This is the ONLY place in the pipeline that touches a ground-truth sequence
on the de novo route.  `denovo_core` never imports `evaluate`, and the C++
engine cannot -- scoring happens strictly *after* a result exists, here.  That
separation is the thing that makes "no ground truth in the pipeline" checkable
rather than merely claimed.

The peptides come from the LOCAL benchmark manifest (pipeline/local_bench.py;
not part of the public source).  Their sequences are used for exactly two
things, both outside the algorithm: the precursor neutral mass is
reference-derived from them (and labelled so), and the answer is graded
against them afterwards.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from . import denovo, local_bench, paths, prep
import evaluate                                                     # noqa: E402


def _sets() -> dict:
    s = local_bench.load("sets")
    return {"main": s.get("main", []), "etd": s.get("etd", []),
            "all": s.get("main", []) + s.get("etd", [])}


class _ResultAdapter:
    """Present a JSON result to `evaluate.score_result` as a `DenovoResult`.

    The scorer reaches into `.master.entries`, `.paths_raw`, `.reconstructions`,
    `.total` and `.term`.  Adapting rather than reimplementing means the C++ and
    Python engines are graded by the *same* code, so a scoring difference can
    never be mistaken for an engine difference.

    ⚠ It also MUTATES `reconstructions[i]["correct"]`, which is why those are
    dicts here and not bare strings.
    """

    class _Entry:
        __slots__ = ("coord", "sigma", "n", "kinds", "origins", "best_rank")

        def __init__(self, d):
            self.coord = d["coord"]
            self.sigma = d["sigma"]
            self.n = d.get("n", 1)
            self.kinds = d.get("kinds", [])
            self.origins = d.get("origins", [])
            self.best_rank = d.get("best_rank", -1)

    class _Master:
        def __init__(self, entries):
            self.entries = entries

    def __init__(self, res: dict):
        self.total = res["coord_total"]
        self.term = res["coord_term"]
        self.n_paths = res["n_paths"]
        self.master = self._Master([self._Entry(e) for e in res["master_set"]])
        self.paths_raw = [[tuple(node) for node in path]
                          for path in res.get("paths", [])]
        self.reconstructions = [{"#": k, "reconstruction": r}
                                for k, r in enumerate(res["reconstructions"], 1)]


def run(names=None, which: str = "main", top_list=None, engine: str = "auto",
        output_dir: str | None = None, params: dict | None = None,
        detail: bool = True, verbose: bool = False) -> pd.DataFrame:
    try:
        sets = _sets()
    except local_bench.Unavailable as exc:
        print(f"benchmark data unavailable: {exc}")
        return pd.DataFrame()
    peptides = sets[which]
    if names:
        wanted = set(names)
        peptides = [e for e in peptides if e["name"] in wanted]
        missing = wanted - {e["name"] for e in peptides}
        if missing:
            raise SystemExit(f"unknown peptide(s): {sorted(missing)}.  "
                             f"Known: {[e['name'] for e in sets['all']]}")
    top_list = list(top_list or denovo.DEFAULT_TOP_LIST)
    out_root = os.path.abspath(output_dir or "result/benchmark")
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for entry in peptides:
        try:
            path = local_bench.check_file(entry)
        except local_bench.Unavailable as exc:
            print(f"{entry['name']:<12} SKIP  ({exc})")
            continue

        seq, terminus = entry["truth"], entry["terminus"]
        mass = entry["mass"]                    # reference-derived (manifest says so)
        merged = prep.merge(prep.read_table(path), "denovo")

        for n in top_list:
            t0 = time.time()
            res = denovo.run(merged, parent_charge=entry["charge"],
                             parent_mass=mass, terminus=terminus,
                             fragmentation=entry.get("fragmentation", "HCD"),
                             top_n=n, engine=engine, params=params)
            seconds = time.time() - t0
            score = evaluate.score_result(_ResultAdapter(res), seq, terminus)

            rows.append(dict(
                peptide=entry["name"], len=len(seq),
                fragmentation=res["fragmentation"], top_n=n,
                engine=res.get("_engine", "?"), lines=res["n_lines"],
                master=len(res["master_set"]), rounds=res["rounds"],
                coverage=round(score.coverage, 1), wrong=score.n_incorrect,
                paths=score.n_paths, correct="yes" if score.correct else "no",
                resolved=f"{score.resolved}/{score.pep_len}",
                run=score.longest_run, seconds=round(seconds, 2),
                reconstruction=score.best_reconstruction))

            if verbose:
                print(f"{entry['name']:<12} top{n:<5} coverage "
                      f"{score.coverage:5.1f}%  wrong {score.n_incorrect}  "
                      f"{'CORRECT' if score.correct else 'wrong'} "
                      f"({score.n_correct_paths}/{score.n_paths})  "
                      f"[{seconds:.2f}s]\n             {score.best_reconstruction}")

            if detail:
                d = os.path.join(out_root, f"top{n}")
                os.makedirs(d, exist_ok=True)
                denovo.write_detail(
                    res, os.path.join(d, f"{entry['name']}_denovo_top{n}.xlsx"))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(out_root, "benchmark.csv"), index=False)
    return df


def summarise(df: pd.DataFrame) -> str:
    """The per-top-N totals -- the row that says whether the run is healthy."""
    if df.empty:
        return "no runs"
    out = []
    for n, sub in df.groupby("top_n"):
        n_ok = int((sub["correct"] == "yes").sum())
        resolved = sum(int(r.split("/")[0]) for r in sub["resolved"])
        total = sum(int(r.split("/")[1]) for r in sub["resolved"])
        out.append(f"  top{n}: {n_ok}/{len(sub)} correct   "
                   f"mean coverage {sub['coverage'].mean():.1f}%   "
                   f"wrong coordinates {int(sub['wrong'].sum())}   "
                   f"residues resolved {resolved}/{total}   "
                   f"{sub['seconds'].sum():.1f}s")
    # Across cutoffs: line detection is non-monotonic in top-N, so a peptide
    # can be right at one cutoff and wrong at another with neither being a
    # regression.  This is the framing denovo.md 9d reports the ETD set in.
    if df["top_n"].nunique() > 1:
        per_pep = df.groupby("peptide")["correct"].apply(lambda s: "yes" in set(s))
        out.append(f"  correct at one cutoff or the other: "
                   f"{int(per_pep.sum())}/{len(per_pep)}")
    return "\n".join(out)
