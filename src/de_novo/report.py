#!/usr/bin/env python3
"""
Benchmark report -- the 7 peptides at top200 and top500, in one readable table
=============================================================================

Runs the pipeline at the module defaults and answers three questions per run:

  * **correct?**  is any tied reconstruction consistent with the true sequence
    (every breakpoint on a real residue boundary)?
  * **candidates** how many tied heaviest paths came back, and how many of them
    are correct?
  * **length**    how much sequence the answer actually spells out -- residues
    named by a letter, the longest contiguous run of them, and how many
    unresolved mass gaps remain.

    $PY report.py                      # markdown + csv into result/
    $PY report.py --top-list 200 500 1000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_denovo import (MAIN_PEPTIDES, ETD_PEPTIDES, ALL_PEPTIDES, DATA_DIR,
                        DEFAULTS, load_map, precursor_mass)
from denovo_core import run_denovo
from evaluate import score_result


def collect(top_list, overrides: dict, peptides=None) -> pd.DataFrame:
    opts = dict(DEFAULTS)
    opts.update(overrides)
    rows = []
    for e in (peptides if peptides is not None else MAIN_PEPTIDES):
        for top_n in top_list:
            ffc = load_map(os.path.join(e.get("data_dir", DATA_DIR), e["data"]),
                           top_n)
            mass = precursor_mass(e["seq"], e["terminus"])
            t0 = time.time()
            res = run_denovo(
                ffc, parent_charge=e["charge"], parent_mass=mass,
                terminus=e["terminus"],
                fragmentation=e.get("fragmentation", "HCD"),
                delta=opts["line_delta"], delta_unit=opts["line_delta_unit"],
                min_ffc_number=opts["line_min_ffc"], tol_ppm=opts["tol_ppm"],
                precursor_ppm=opts["precursor_ppm"],
                sigma_floor_ppm=opts["sigma_floor_ppm"],
                mock_sigma_cap_da=opts["mock_sigma_cap_da"],
                master_align_threshold=opts["master_align_threshold"],
                accept_ambiguity=opts["accept_ambiguity"],
                use_mocks=opts["use_mocks"], mock_min_ffc=opts["mock_min_ffc"],
                include_isotope_lines=opts["include_isotope_lines"],
                isotope_tol_da=opts["isotope_tol_da"],
                mock_weight=opts["mock_weight"],
                secondary_gate=opts["secondary_gate"],
                evidence_weight=opts["evidence_weight"],
                evidence_cap=opts["evidence_cap"], rank_floor=opts["rank_floor"],
                gate_mock_on_goodness=opts["gate_mock_on_goodness"],
                max_rounds=opts["max_rounds"], max_paths=opts["max_paths"],
                label_tol=opts["label_tol"])
            sc = score_result(res, e["seq"], e["terminus"])
            rows.append(dict(
                peptide=e["name"], mode=e.get("fragmentation", "HCD"),
                charge=e["charge"], pep_len=sc.pep_len, top_n=top_n,
                correct="YES" if sc.correct else "no",
                candidates=sc.n_paths, n_correct=sc.n_correct_paths,
                coverage=round(sc.coverage, 1), bonds=len(sc.bonds_found),
                n_bonds=sc.n_bonds, incorrect=sc.n_incorrect,
                resolved=sc.resolved, longest_run=sc.longest_run, gaps=sc.gaps,
                seconds=round(time.time() - t0, 1),
                reconstruction=sc.best_reconstruction))
    return pd.DataFrame(rows)


def markdown(df: pd.DataFrame, top_list) -> str:
    out = []
    out.append("# De novo benchmark -- top200 vs top500\n")
    out.append("`correct?` -- some tied path puts every breakpoint on a real "
               "residue boundary.  `candidates` -- tied heaviest paths returned, "
               "of which `n correct` are right.  `resolved` -- residues the answer "
               "names by letter, out of the peptide's length.  `longest` -- "
               "longest contiguous named run.  `gaps` -- unresolved mass segments.\n")

    out.append("\n## Headline\n")
    head = ["| peptide | len | "
            + " | ".join(f"top{n}: correct? (cands)" for n in top_list)
            + " | best resolved | best run |",
            "|---|---|" + "---|" * (len(top_list) + 2)]
    for pep, g in df.groupby("peptide", sort=False):
        gi = g.set_index("top_n")
        cells = []
        for n in top_list:
            r = gi.loc[n]
            mark = "**YES**" if r.correct == "YES" else "no"
            cells.append(f"{mark} ({r.candidates})")
        head.append(f"| {pep} | {int(gi['pep_len'].iloc[0])} | "
                    + " | ".join(cells)
                    + f" | {int(gi['resolved'].max())} | {int(gi['longest_run'].max())} |")
    n_ok = df.groupby("peptide")["correct"].apply(lambda s: (s == "YES").any()).sum()
    head.append(f"\n**{n_ok}/{df['peptide'].nunique()} peptides reconstruct "
                f"correctly**; {(df.correct == 'YES').sum()}/{len(df)} individual runs.")
    out.append("\n".join(head))

    for n in top_list:
        sub = df[df.top_n == n]
        out.append(f"\n## top{n} -- full detail\n")
        cols = ["| dataset | mode | len | correct? | candidates | n correct | "
                "coverage | wrong coords | resolved | longest | gaps | s |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            mark = "**YES**" if r.correct == "YES" else "no"
            cols.append(
                f"| {r.peptide} | {r.get('mode', 'HCD')} | {r.pep_len} | "
                f"{mark} | {r.candidates} | "
                f"{r.n_correct} | {r.coverage}% ({r.bonds}/{r.n_bonds}) | "
                f"{r.incorrect} | {r.resolved}/{r.pep_len} | {r.longest_run} | "
                f"{r.gaps} | {r.seconds} |")
        cols.append(f"\n**{(sub.correct == 'YES').sum()}/{len(sub)} correct** - "
                    f"mean coverage {sub.coverage.mean():.1f}% - "
                    f"{int(sub.incorrect.sum())} wrong coordinates in total - "
                    f"{int(sub.resolved.sum())}/{int(sub.pep_len.sum())} residues resolved")
        out.append("\n".join(cols))

    out.append("\n## Reconstructions\n")
    for n in top_list:
        out.append(f"\n### top{n}\n```")
        for _, r in df[df.top_n == n].iterrows():
            tag = "OK " if r.correct == "YES" else "-- "
            out.append(f"{tag}{r.peptide:7s} {r.reconstruction}")
        out.append("```")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-list", nargs="+", type=int, default=[200, 500])
    ap.add_argument("--set", default="hcd", choices=["hcd", "etd", "all"],
                    help="which benchmark set to run (default hcd)")
    ap.add_argument("--output-dir", default=os.path.join(_HERE, "result"))
    args = ap.parse_args(argv)

    peptides = {"hcd": MAIN_PEPTIDES, "etd": ETD_PEPTIDES,
                "all": ALL_PEPTIDES}[args.set]
    df = collect(args.top_list, {}, peptides)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "benchmark.csv"), index=False)
    md = markdown(df, args.top_list)
    path = os.path.join(args.output_dir, "benchmark.md")
    with open(path, "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\nwrote {path} and benchmark.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
