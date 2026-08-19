#!/usr/bin/env python3
"""
Charge-aware spurious-parental test.
=====================================
Drop-in replacement for ``run_spurious_test.py``: same CLI interface, same
output format.  The only difference is that it uses the ``charge_A`` /
``charge_B`` columns already present in the deconvolution output instead of
re-trying every possible (i, j) combination.

Usage — swap TEST_SCRIPT in run_deconv_test.sh:
    TEST_SCRIPT=run_spurious_test_charge.py
"""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Charge-aware spurious-parental deconvolution performance test"
    )
    ap.add_argument(
        "ffc_file",
        help="Deconvolution output with charge_A / charge_B columns "
             "(e.g. <prefix>_merged.tsv or <prefix>_s1_replaced.tsv)",
    )
    ap.add_argument("--sequence", required=True, help="Peptide sequence")
    ap.add_argument("--charge", type=int, required=True,
                    help="Precursor charge state")
    ap.add_argument(
        "--parent-mass", type=float, default=None,
        help="Override the reconstruction reference (default: pep.pep_mass = "
             "M + Z*H). Normally leave unset.",
    )
    ap.add_argument("--iso-range", type=int, default=4,
                    help="Isotope variants for b/y matching (default: 4)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="b/y ion mass tolerance, Da (default: 0.05)")
    ap.add_argument("--parental-shift-threshold", type=float, default=0.05,
                    help="|i*mzA + j*mzB - parent_mass| cutoff, Da (default: 0.05)")
    ap.add_argument("--charge-col-a", default="charge_A",
                    help="Column name for fragment-A charge (default: charge_A)")
    ap.add_argument("--charge-col-b", default="charge_B",
                    help="Column name for fragment-B charge (default: charge_B)")
    ap.add_argument("--top-n", type=int, default=150000,
                    help="Maximum rows to load from the input file (default: 150000)")
    ap.add_argument("--end", choices=["h2o", "nh3"], default="h2o",
                    help="C-terminal group passed to peptide.Pep (default: h2o)")
    ap.add_argument("--sep", default="\t", help="Input delimiter (default: tab)")
    ap.add_argument("--show-rows", type=int, default=80,
                    help="Max table rows to print (default: 80)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    import pandas as pd
    try:
        import interpreter_modify  # noqa: F401
    except Exception:
        pass
    import peptide
    from spurious_parental import merge_parental_annotations
    from spurious_parental_charge import parental_ffc_annotations_from_charges
    from merge_ffcs import merge_duplicate_ffcs
    from merge_ffcs import merge_duplicate_ffcs_charge
    
    if not os.path.isfile(args.ffc_file):
        print(f"[error] input not found: {args.ffc_file}", file=sys.stderr)
        return 1

    end_h20 = True if args.end == "h2o" else "NH3"
    pep = peptide.Pep(
        f"[{args.sequence}+{args.charge}H]{args.charge}+", end_h20=end_h20
    )
    parent_mass = args.parent_mass if args.parent_mass is not None else pep.pep_mass

    print("=" * 60)
    print("CHARGE-AWARE SPURIOUS-PARENTAL TEST")
    print("=" * 60)
    print(f"Peptide:     {args.sequence}")
    print(f"Charge:      {args.charge}")
    print(f"parent_mass: {parent_mass}  (reconstruction reference M + Z*H)")
    print(f"Input:       {args.ffc_file}")

    ffc_df = pd.read_csv(args.ffc_file, sep=args.sep, nrows=args.top_n)
    print(f"FFC rows loaded: {len(ffc_df)}")
    #ffc_df = merge_duplicate_ffcs(ffc_df)
    ffc_df = merge_duplicate_ffcs_charge(ffc_df)
    print(f"FFC rows after merging duplicates: {len(ffc_df)}")
    print(f"Columns: {list(ffc_df.columns)}")

    # ── All parental-line FFCs annotated with assigned charges ────────────
    print("\n" + "=" * 60)
    annot_df = parental_ffc_annotations_from_charges(
        ffc_df, pep,
        parent_charge=args.charge,
        parent_mass=parent_mass,
        iso_range=args.iso_range,
        threshold=args.threshold,
        parental_shift_threshold=args.parental_shift_threshold,
        charge_col_a=args.charge_col_a,
        charge_col_b=args.charge_col_b,
    )

    n_total = len(annot_df)
    n_spurious = int((annot_df["b_ions"] == "spurious").sum()) if n_total else 0
    print(f"\nAll parental-line FFCs (charge-assigned): {n_total}")
    print(f"  Spurious : {n_spurious}")
    print(f"  Annotated: {n_total - n_spurious}")
    if n_total:
        print(f"  Spurious rate: {100.0 * n_spurious / n_total:.1f}%")
        cols = [c for c in [
            "m/z A", "m/z B", "Ranking",
            args.charge_col_a, args.charge_col_b,
            "adj_mass_A", "adj_mass_B", "b_ions", "breaking_points",
        ] if c in annot_df.columns]
        print(annot_df[cols].head(args.show_rows).to_string(index=False))

    # ── Charge-independent merged view ────────────────────────────────────
    if n_total:
        print("\n" + "=" * 60)
        print("Charge-independent merged view  (symmetric pairs collapsed)")
        print("=" * 60)

        merged = merge_parental_annotations(annot_df)
        n_merged_spurious = int((merged["b_ions"] == "spurious").sum()) if "b_ions" in merged.columns else 0
        print(f"\nUnique pairs: {len(merged)}")
        print(f"  Spurious : {n_merged_spurious}")
        print(f"  Annotated: {len(merged) - n_merged_spurious}")
        if len(merged):
            print(f"  Spurious rate: {100.0 * n_merged_spurious / len(merged):.1f}%")
        show_cols = [c for c in ["mz1_lo", "mz1_hi", "Ranking", "b_ions", "breaking_points"]
                     if c in merged.columns]
        print(merged[show_cols].head(args.show_rows).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
