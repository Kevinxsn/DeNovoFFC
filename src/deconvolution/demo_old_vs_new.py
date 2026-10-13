#!/usr/bin/env python3
"""
Demo: old pipeline vs. new, on the Main peptide set
====================================================
Scores already-written merged tables -- it deconvolves nothing, so it is
seconds, not hours. Both sides go through the identical charge-aware
spurious-parental test, so the comparison is like for like.

    old   $DENOVOFFC_DATA_ROOT/deconvoluted/merged/     (greedy lines, doubled frame)
    new   $DENOVOFFC_DATA_ROOT/deconvoluted_new/merge/  (orientation-agnostic lines)

``breakpoints`` -- distinct backbone bonds recovered -- is the column that
matters. ``pairs``/``annot`` can rise simply because a table is bigger; the bond
count cannot.

Note the two sides are *not* the same number of rows by construction: the old
pipeline doubled every FFC (X, Y) into (Y, X), so its tables are ~2x the size for
the same physical content. That is the point of the rewrite, not a confound --
but it is why ``rows`` is reported.

Usage
-----
    python demo_old_vs_new.py
    python demo_old_vs_new.py --out old_vs_new.csv
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os

import pandas as pd

import _paths  # noqa: F401  (sys.path bootstrap)

import peptide  # noqa: E402
from peptide_info import resolve as resolve_name  # noqa: E402
from spurious_parental_charge import (  # noqa: E402
    parental_ffc_annotations_from_charges,
)
from merge_ffcs import merge_duplicate_ffcs_charge  # noqa: E402

OLD_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted/merged")
NEW_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted_new/merge")

#: (label, raw dataset name, old-pipeline filename).  The old tree's names are
#: inconsistent -- some carry the _XIAOWEN infix, some are hand-named -- which is
#: itself part of why the new one is uniformly <NAME>_merged.tsv.
MAIN = [
    ("VEA3+", "VEA_15_Z3_revised",  "VEA_15_Z3_revised_merged.tsv"),
    ("YLE3+", "YLE_16_Z3",          "YLE_16_Z3_XIAOWEN_merged.tsv"),
    ("HGT3+", "HGT_14_Z3",          "HGT_merged.tsv"),
    ("HAD4+", "HAD_33_Z4_E16",      "HAD_33_Z4_E16_XIAOWEN_merged.tsv"),
    ("KWK5+", "KWK_37_Z5_E28_2",    "KWK_37_Z5_E28_2_XIAOWEN_merged.tsv"),
    ("LLG6+", "LLG_37_Z6_E30_I130", "LLG_37_Z6_E30_I130_merged.tsv"),
    ("YPS6+", "YPS_36_Z6_E25_I300", "YPS_36_Z6_E25_I300_merged.tsv"),
]


def score(path: str, info: dict) -> dict:
    end_h20 = True if info["RESIDUAL"] == "H2O" else "NH3"
    pep = peptide.Pep(
        f"[{info['SEQUENCE']}+{info['PARENT_CHARGE']}H]{info['PARENT_CHARGE']}+",
        end_h20=end_h20)

    df = pd.read_csv(path, sep="\t")
    n_raw = len(df)
    # Rows the deconvolution could not place carry no m/z; they are not evidence
    # either way, and the test's KD-tree dedup rejects them outright.
    df = df.dropna(subset=["m/z A", "m/z B"])
    n_usable = len(df)
    if df.empty:
        return dict(rows=n_raw, usable=0, pairs=0, annot=0, spur=0, breakpoints=0)
    df = merge_duplicate_ffcs_charge(df)

    with contextlib.redirect_stdout(io.StringIO()):
        annot = parental_ffc_annotations_from_charges(
            df, pep,
            parent_charge=info["PARENT_CHARGE"], parent_mass=pep.pep_mass,
            iso_range=4, threshold=0.1, parental_shift_threshold=0.05,
        )
    if annot.empty:
        return dict(rows=n_raw, usable=n_usable, pairs=0, annot=0, spur=0,
                    breakpoints=0)
    spur = annot["b_ions"] == "spurious"
    bonds = set(annot.loc[~spur, "b_ions"].astype(str)) - {"spurious", ""}
    return dict(rows=n_raw, usable=n_usable, pairs=len(annot),
                annot=int((~spur).sum()), spur=int(spur.sum()),
                breakpoints=len(bonds))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", help="write the summary table here (.csv)")
    ap.add_argument("--old-dir", default=OLD_DIR)
    ap.add_argument("--new-dir", nargs="+", default=[NEW_DIR], metavar="DIR",
                    help="one or more new-pipeline merge/ folders; each is "
                         "scored as its own column, labelled by folder name")
    args = ap.parse_args(argv)

    rows = []
    for label, dataset, old_name in MAIN:
        info = resolve_name(dataset)
        variants = [("old", os.path.join(args.old_dir, old_name))]
        for d in args.new_dir:
            tag = "new" if os.path.abspath(d) == os.path.abspath(NEW_DIR) \
                else os.path.basename(os.path.dirname(os.path.abspath(d)))
            variants.append((tag, os.path.join(d, f"{dataset}_merged.tsv")))
        for tag, path in variants:
            if not os.path.isfile(path):
                print(f"[skip] {label} {tag}: not found -- {path}")
                continue
            rows.append(dict(peptide=label, z=info["PARENT_CHARGE"],
                             pipeline=tag, **score(path, info)))
            print(f"  {label:6} {tag:3} {rows[-1]}", flush=True)

    table = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("OLD vs NEW -- Main dataset, charge-aware spurious-parental test")
    print("=" * 90)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))

    for metric in ("breakpoints", "annot", "spur", "rows"):
        print(f"\n{metric}:")
        print(table.pivot_table(index="peptide", columns="pipeline",
                                values=metric).to_string())

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
