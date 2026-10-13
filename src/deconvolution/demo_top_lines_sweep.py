#!/usr/bin/env python3
"""
Demo: how many FFCs should line detection see?
===============================================
Sweeps ``top_lines`` -- the number of best-ranked FFCs handed to *line
detection* -- over the Main peptide set, holding everything else fixed, and
scores each run with the charge-aware spurious-parental test.

The distinction being measured
------------------------------
Two different cutoffs, and only one of them is expensive:

``top_selected`` (15000)
    How many FFCs get **deconvolved**.  Fixed here.  Positive partial
    covariance, ``Ranking != -1``, best-ranked first -- a dataset with fewer
    qualifying FFCs simply yields fewer.

``top_lines`` (the sweep: 2000 -> 15000)
    How many of those FFCs **line detection** actually looks at.  Detection is
    roughly quadratic in its input, so this dominates the runtime; the lines it
    finds are afterwards matched against all ``top_selected`` FFCs
    (``expand_line_membership``), so a small ``top_lines`` still assigns charges
    across the full set.

The question this answers: does looking deeper into the ranking for lines find
*more real structure*, or just more junk?

Metrics per run
---------------
``lines``          detected lines (charge-sum "parental" / residual internal)
``ladder``         of those, the near-zero-shift ones -- the actual parental ladder
``assigned``       FFCs Stage 1 gave a charge to (the rest fall through to TopFD)
``pairs``          distinct parental FFC pairs the test found
``annot``/``spur`` of those, b/y-explained vs. spurious
``breakpoints``    distinct backbone bonds covered -- **the signal that matters**;
                   the others can all rise together without the result improving

Stage 2 (TopFD) is skipped: it is unaffected by ``top_lines`` and would add a
constant few minutes per run.

Usage
-----
    python demo_top_lines_sweep.py                       # full sweep, Main set
    python demo_top_lines_sweep.py --top-lines 2000 15000
    python demo_top_lines_sweep.py --only VEA3+ HAD4+
    python demo_top_lines_sweep.py --out sweep.csv
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap)

import peptide  # noqa: E402
from peptide_info import resolve as resolve_name  # noqa: E402
from spurious_parental_charge import (  # noqa: E402
    parental_ffc_annotations_from_charges,
)
from merge_ffcs import merge_duplicate_ffcs_charge  # noqa: E402

from line_detection_oa import MASTER_LADDER_SHIFT  # noqa: E402
from deconv_pipeline_oa import run_deconvolution  # noqa: E402


#: The Main reference set, as MAIN_PEPTIDES in
#: annotation_pepline/run_annotation_reconstruct.py, mapped back to the raw
#: dataset each was deconvolved from.
MAIN = [
    ("VEA3+", "VEA_15_Z3_revised"),
    ("YLE3+", "YLE_16_Z3"),
    ("HGT3+", "HGT_14_Z3"),
    ("HAD4+", "HAD_33_Z4_E16"),
    ("KWK5+", "KWK_37_Z5_E28_2"),
    ("LLG6+", "LLG_37_Z6_E30_I130"),
    ("YPS6+", "YPS_36_Z6_E25_I300"),
]

DATA_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "mms_with_intensity")
TOP_LINES_GRID = [2000, 5000, 10000, 15000]
TOP_SELECTED = 15000


def score(merged_path: str, info: dict) -> dict:
    """Run the charge-aware spurious-parental test and reduce it to counts."""
    end_h20 = True if info["RESIDUAL"] == "H2O" else "NH3"
    pep = peptide.Pep(
        f"[{info['SEQUENCE']}+{info['PARENT_CHARGE']}H]{info['PARENT_CHARGE']}+",
        end_h20=end_h20)

    df = pd.read_csv(merged_path, sep="\t")
    # The test's own dedup chokes on rows the envelope refinement could not
    # place; those carry no deconvolved m/z and are not evidence either way.
    df = df.dropna(subset=["m/z A", "m/z B"])
    if df.empty:
        return dict(pairs=0, annot=0, spur=0, breakpoints=0)
    df = merge_duplicate_ffcs_charge(df)

    with contextlib.redirect_stdout(io.StringIO()):
        annot = parental_ffc_annotations_from_charges(
            df, pep,
            parent_charge=info["PARENT_CHARGE"],
            parent_mass=pep.pep_mass,
            iso_range=4, threshold=0.1, parental_shift_threshold=0.05,
        )
    if annot.empty:
        return dict(pairs=0, annot=0, spur=0, breakpoints=0)

    spur = annot["b_ions"] == "spurious"
    bonds = set(annot.loc[~spur, "b_ions"].astype(str)) - {"spurious", ""}
    return dict(pairs=len(annot), annot=int((~spur).sum()),
                spur=int(spur.sum()), breakpoints=len(bonds))


def run_case(name: str, dataset: str, top_lines: int, out_root: str) -> dict:
    info = resolve_name(dataset)
    path = os.path.join(DATA_DIR, dataset)
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        result = run_deconvolution(
            ffc_path=path,
            parent_charge=info["PARENT_CHARGE"],
            parent_mass=info["PARENT_MASS"],
            output_dir=os.path.join(out_root, f"top_lines_{top_lines}"),
            output_prefix=dataset,
            theo_patt_path=_paths.THEO_PATT,
            top_selected=TOP_SELECTED, top_bg=TOP_SELECTED,
            top_lines=top_lines,
            run_topfd=False,
        )
    elapsed = time.time() - started
    stats = result["stats"]
    lines = result["lines"]
    ladder = sum(1 for ln in lines
                 if abs(ln.mass - info["PARENT_MASS"]) < MASTER_LADDER_SHIFT)

    row = dict(
        peptide=name, z=info["PARENT_CHARGE"], top_lines=top_lines,
        selected=stats["selected_subset"],
        lines=stats["n_lines_parental"] + stats["n_lines_residual"],
        ladder=ladder,
        assigned=stats["stage1_line_assigned"],
        seconds=round(elapsed, 1),
    )
    row.update(score(stats["merged_path"], info))
    row["log"] = buf.getvalue()
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="run only these Main entries (e.g. VEA3+ HAD4+)")
    ap.add_argument("--top-lines", nargs="+", type=int, default=TOP_LINES_GRID)
    ap.add_argument("--out-root",
                    default=os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted_new/demo_top_lines"))
    ap.add_argument("--out", help="also write the summary table here (.csv)")
    args = ap.parse_args(argv)

    cases = MAIN if not args.only else [c for c in MAIN if c[0] in set(args.only)]
    if not cases:
        ap.error("no matching Main entries")

    rows = []
    total = len(cases) * len(args.top_lines)
    k = 0
    for name, dataset in cases:
        for top_lines in args.top_lines:
            k += 1
            print(f"[{k}/{total}] {name:7} {dataset:22} top_lines={top_lines}",
                  flush=True)
            try:
                row = run_case(name, dataset, top_lines, args.out_root)
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            row.pop("log", None)
            rows.append(row)
            print(f"    lines={row['lines']:5d} (ladder {row['ladder']:3d})  "
                  f"assigned={row['assigned']:6d}  pairs={row['pairs']:3d}  "
                  f"annot={row['annot']:3d}  spur={row['spur']:3d}  "
                  f"breakpoints={row['breakpoints']:3d}  {row['seconds']:.0f}s",
                  flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        print("\nnothing ran.")
        return 1

    print("\n" + "=" * 100)
    print("TOP_LINES SWEEP -- Main dataset  (top_selected = "
          f"{TOP_SELECTED}, Stage 1 only)")
    print("=" * 100)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))

    print("\nBreakpoints covered (the number to read):")
    pivot = table.pivot_table(index="peptide", columns="top_lines",
                              values="breakpoints")
    print(pivot.to_string())
    print("\nSpurious parental pairs:")
    print(table.pivot_table(index="peptide", columns="top_lines",
                            values="spur").to_string())
    print("\nDetection seconds:")
    print(table.pivot_table(index="peptide", columns="top_lines",
                            values="seconds").to_string())

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
