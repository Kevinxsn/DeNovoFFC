#!/usr/bin/env python3
"""
Batch runner for the de novo pipeline
=====================================

Feeds an MMS map (a deconvolved FFC table with rankings) to
``denovo_core.run_denovo`` and prints / writes the reconstruction.  The
pipeline itself gets only the table, the precursor charge, the precursor
neutral mass and the C-terminal group.

The seven MAIN_PEPTIDES below are the benchmark set from
``../annotation_pepline/annotation_pipeline.md`` §5.  Their sequences are used
for **two things only**, both outside the algorithm:

  * to compute the precursor neutral mass (an experiment measures this; we
    derive it so the benchmark is not hostage to a separate mass file), and
  * to grade the answer afterwards, in ``evaluate.py``.

Pass ``--no-score`` to run with no sequence in play at all.

Usage
-----
    PY=python

    $PY run_denovo.py                          # all 7, default top-N list
    $PY run_denovo.py --only VEA3+ KWK5+
    $PY run_denovo.py --only YPS6+ --top-list 500 --verbose
    $PY run_denovo.py --tol-ppm 20 --line-delta 20 --accept-ambiguity
    $PY run_denovo.py --data my_map.tsv --charge 5 --mass 4059.361 \
                      --terminus NH3 --no-score

    $PY run_denovo.py --etd                    # the ETD / EThcD datasets
    $PY run_denovo.py --all                    # both sets
    $PY run_denovo.py --only ETD-LL6+ LL6+     # same peptide, both chemistries
    $PY run_denovo.py --data etd_map.tsv --charge 6 --mass 4490.575 \
                      --fragmentation ETD --no-score

Fragmentation
-------------
``HCD`` (b/y) is the default and nothing about it changed.  ``ETD`` / ``ETHCD``
switch the ion series to c/z-dot, which -- per ``../ETD/etd.md`` -- means a
different coordinate basis and a ladder one charge lower than the precursor.
Both are handled inside ``denovo_core`` by a :class:`~denovo_core.Basis`; every
other stage is shared.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "deiso", "python_code"),
           os.path.join(_ROOT, "reconstruct"), _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from merge_ffcs import merge_duplicate_ffcs_charge      # noqa: E402
from line_finding import prepare_ffc_data               # noqa: E402
from denovo_core import run_denovo, TERMINI             # noqa: E402

DATA_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deconvoluted/merged")
#: ETD/EThcD maps, deconvolved with the line model at z-1 (``ETD/etd.md`` §2b).
ETD_DATA_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "ETD_deconv_cz/merge")
OUT_DIR = os.path.join(_HERE, "result")

#: LL37, the peptide the ETD data was acquired on (``ETD/etd.md`` §1).
LL37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"

# The benchmark set (annotation_pipeline.md §5).  `seq` is for the precursor
# mass and for scoring; the pipeline never receives it.
MAIN_PEPTIDES = [
    dict(name="VEA3+", data="VEA_15_Z3_revised_merged.tsv",
         seq="VEADIAGHGQEVLIR", charge=3, terminus="H2O"),
    dict(name="YLE3+", data="YLE_16_Z3_XIAOWEN_merged.tsv",
         seq="YLEFISDAIIHVLHSK", charge=3, terminus="H2O"),
    dict(name="HGT3+", data="HGT_merged.tsv",
         seq="HGTVVLTALGGILK", charge=3, terminus="H2O"),
    dict(name="HAD4+", data="HAD_33_Z4_E15_MF_I270_merged.tsv",
         seq="HADGSFSDEMNTILDNLAARDFINWLIQTKITD", charge=4, terminus="H2O"),
    dict(name="KWK5+", data="KWK_37_Z5_E28_2_XIAOWEN_merged.tsv",
         seq="KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", charge=5, terminus="NH3"),
    dict(name="LL6+", data="LLG_37_Z6_E30_I130_merged.tsv",
         seq="LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", charge=6, terminus="H2O"),
    dict(name="YPS6+", data="YPS_36_Z6_E25_I300_merged.tsv",
         seq="YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY", charge=6, terminus="NH3"),
]

# ETD / EThcD on LL37 6+ (ETD/etd.md).  Same peptide as LL6+ above, so the two
# fragmentation modes are directly comparable on identical chemistry.  The
# reaction-time series is the same acquisition at five ETD reaction times.
ETD_PEPTIDES = [
    dict(name="ETD-LL6+", data="LLG_37_Z6_ETD_merged.tsv", seq=LL37,
         charge=6, terminus="H2O", fragmentation="ETD", data_dir=ETD_DATA_DIR),
    dict(name="EThcD-LL6+", data="LLG_37_Z6_ETHCD25_merged.tsv", seq=LL37,
         charge=6, terminus="H2O", fragmentation="ETHCD", data_dir=ETD_DATA_DIR),
] + [
    dict(name=f"ETD-{t.replace('p', '.')}", data=f"LLG_37_Z6_ETD_{t}_merged.tsv",
         seq=LL37, charge=6, terminus="H2O", fragmentation="ETD",
         data_dir=ETD_DATA_DIR)
    for t in ("1p54ms", "2p21ms", "3p46ms", "6p15ms", "13p84ms")
]

ALL_PEPTIDES = MAIN_PEPTIDES + ETD_PEPTIDES

DEFAULTS = dict(
    top_list=[200, 500],
    line_delta=0.015,
    line_delta_unit="da",
    line_min_ffc=3,
    tol_ppm=25.0,
    precursor_ppm=10.0,
    sigma_floor_ppm=3.0,
    mock_sigma_cap_da=0.05,
    master_align_threshold=1,
    accept_ambiguity=False,
    use_mocks=True,
    include_isotope_lines=True,
    isotope_tol_da=0.03,
    mock_min_ffc=4,
    mock_weight=1.0,
    secondary_gate=0.25,
    evidence_weight=0.0,
    evidence_cap=1.0,
    rank_floor=1.0,
    gate_mock_on_goodness=True,
    max_rounds=20,
    max_paths=1000,
    label_tol=0.02,
)


def precursor_mass(seq: str, terminus: str) -> float:
    """Neutral peptide mass -- what the instrument measures for the precursor."""
    from evaluate import residue_prefixes
    return residue_prefixes(seq)[-1] + TERMINI[terminus]


def load_map(path: str, top_n: int) -> pd.DataFrame:
    """Load, merge duplicate/mirrored FFCs, then cut to the top-N ranking."""
    df = pd.read_csv(path, sep="\t")
    merged = merge_duplicate_ffcs_charge(df, symmetric=True, reset_ranking=True)
    return prepare_ffc_data(merged, top_n=top_n)


def run_one(entry: dict, top_n: int, opts: dict, score: bool, verbose: bool):
    name = entry["name"]
    data_path = (entry.get("path")
                 or os.path.join(entry.get("data_dir", DATA_DIR), entry["data"]))
    seq = entry.get("seq")
    terminus = entry.get("terminus", "H2O")
    frag = entry.get("fragmentation", "HCD")
    mass = entry.get("mass") or precursor_mass(seq, terminus)

    print(f"\n=== {name}  (top{top_n})  {frag}, charge {entry['charge']}, "
          f"M = {mass:.4f}, C-term {terminus} ===")
    ffc_df = load_map(data_path, top_n)
    print(f"  FFCs fed to the pipeline: {len(ffc_df)}")

    t0 = time.time()
    result = run_denovo(
        ffc_df, parent_charge=entry["charge"], parent_mass=mass,
        terminus=terminus, fragmentation=frag,
        delta=opts["line_delta"], delta_unit=opts["line_delta_unit"],
        min_ffc_number=opts["line_min_ffc"],
        tol_ppm=opts["tol_ppm"], precursor_ppm=opts["precursor_ppm"],
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
        label_tol=opts["label_tol"], verbose=verbose,
    )
    dt = time.time() - t0

    print(f"  lines detected: {len(result.lines)}  "
          f"(parental {sum(1 for l in result.lines if l.category == 'parental')}, "
          f"internal {sum(1 for l in result.lines if l.category == 'internal')}, "
          f"loss {sum(1 for l in result.lines if l.category == 'loss')}, "
          f"isotope {sum(1 for l in result.lines if l.category == 'isotope')}, "
          f"other {sum(1 for l in result.lines if l.category == 'other')})")
    print(f"  master TFFC-set: {len(result.master)} breakpoints "
          f"after {result.rounds} round(s);  {result.n_paths} tied path(s), "
          f"weight {result.best_weight:.0f}   [{dt:.1f}s]")

    row = dict(peptide=name, fragmentation=frag, top_n=top_n, n_ffc=len(ffc_df),
               n_lines=len(result.lines), n_master=len(result.master),
               rounds=result.rounds, n_paths=result.n_paths, seconds=round(dt, 1))

    if score and seq:
        from evaluate import score_result
        sc = score_result(result, seq, terminus)
        print(f"  coverage {sc.coverage:.1f}% ({len(sc.bonds_found)}/{sc.n_bonds} bonds), "
              f"#incorrect {sc.n_incorrect}, "
              f"correct: {'YES' if sc.correct else 'no'} "
              f"({sc.n_correct_paths}/{sc.n_paths})")
        print(f"  -> {sc.best_reconstruction}")
        if sc.n_incorrect:
            for c, s, kinds, src in sc.incorrect[:6]:
                print(f"       incorrect {c:10.4f} +-{s:.4f}  {kinds:12s} {src}")
        row.update(coverage=round(sc.coverage, 1), n_incorrect=sc.n_incorrect,
                   correct="yes" if sc.correct else "no",
                   n_correct=sc.n_correct_paths,
                   reconstruction=sc.best_reconstruction)
    else:
        best = result.reconstructions[0]["reconstruction"] if result.reconstructions else ""
        print(f"  -> {best}")
        row.update(reconstruction=best)

    return result, row


def dump_details(result, out_path: str, name: str, top_n: int) -> None:
    """Per-line and per-coordinate detail, for inspecting a run."""
    lines = pd.DataFrame([dict(
        line=l.key, category=l.category, detail=l.detail, A=round(l.a_shift, 4),
        n_ffc=l.n_ffc, n_inactive=l.n_inactive, n_both_known=l.n_both_known,
        eligible=l.eligible, mock_refused=l.mock_refused,
        banked=l.n_banked, unambiguous=l.n_unambiguous,
        ambiguous=l.n_ambiguous, unproductive=l.n_unproductive,
        mock=None if l.mock is None else round(l.mock[0], 4),
    ) for l in result.lines])
    master = pd.DataFrame([dict(
        coord=round(e.coord, 4), sigma=round(e.sigma, 5), n=e.n,
        kinds="/".join(sorted(set(e.kinds))),
        origins=";".join(sorted(set(e.origins))),
    ) for e in sorted(result.master.entries, key=lambda e: e.coord)])
    recon = pd.DataFrame(result.reconstructions)
    with pd.ExcelWriter(out_path) as xl:
        lines.to_excel(xl, sheet_name="lines", index=False)
        master.to_excel(xl, sheet_name="master_set", index=False)
        if not recon.empty:
            recon.to_excel(xl, sheet_name="reconstructions", index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", metavar="NAME",
                   help="run only these benchmark datasets (any set)")
    p.add_argument("--etd", action="store_true",
                   help="run the ETD/EThcD datasets instead of the HCD set")
    p.add_argument("--all", action="store_true",
                   help="run both the HCD and the ETD sets")
    p.add_argument("--list", action="store_true", help="show the set and exit")
    p.add_argument("--data", help="run one arbitrary FFC table instead")
    p.add_argument("--charge", type=int, help="precursor charge (with --data)")
    p.add_argument("--mass", type=float, help="precursor neutral mass (with --data)")
    p.add_argument("--seq", help="known sequence, for scoring only (with --data)")
    p.add_argument("--terminus", default="H2O", choices=sorted(TERMINI),
                   help="C-terminal group (default H2O)")
    p.add_argument("--fragmentation", default="HCD",
                   choices=["HCD", "ETD", "ETHCD"],
                   help="ion series for --data: HCD = b/y (default), "
                        "ETD/ETHCD = c/z-dot")
    p.add_argument("--top-list", nargs="+", type=int, default=DEFAULTS["top_list"],
                   help="top-N ranking cutoffs, one run each")
    p.add_argument("--line-delta", type=float, default=DEFAULTS["line_delta"])
    p.add_argument("--line-delta-unit", default=DEFAULTS["line_delta_unit"],
                   choices=["da", "ppm"])
    p.add_argument("--line-min-ffc", type=int, default=DEFAULTS["line_min_ffc"])
    p.add_argument("--tol-ppm", type=float, default=DEFAULTS["tol_ppm"],
                   help="fragment-mass relative error (drives every sigma)")
    p.add_argument("--precursor-ppm", type=float, default=DEFAULTS["precursor_ppm"])
    p.add_argument("--sigma-floor-ppm", type=float, default=DEFAULTS["sigma_floor_ppm"])
    p.add_argument("--mock-sigma-cap-da", type=float, default=DEFAULTS["mock_sigma_cap_da"])
    p.add_argument("--master-align-threshold", type=int,
                   default=DEFAULTS["master_align_threshold"])
    p.add_argument("--accept-ambiguity", action="store_true")
    p.add_argument("--no-isotope-lines", action="store_true",
                   help="ignore parental isotope-satellite lines (etd.md 6a-i)")
    p.add_argument("--isotope-tol-da", type=float,
                   default=DEFAULTS["isotope_tol_da"])
    p.add_argument("--no-mocks", action="store_true",
                   help="internal lines contribute no mock breakpoint")
    p.add_argument("--mock-min-ffc", type=int, default=DEFAULTS["mock_min_ffc"])
    p.add_argument("--mock-weight", type=float, default=DEFAULTS["mock_weight"])
    p.add_argument("--secondary-gate", type=float, default=DEFAULTS["secondary_gate"],
                   help="fraction of a line's TFFCs allowed to be known on BOTH "
                        "images before its mock is refused (1.0 disables)")
    p.add_argument("--evidence-weight", type=float, default=DEFAULTS["evidence_weight"])
    p.add_argument("--evidence-cap", type=float, default=DEFAULTS["evidence_cap"])
    p.add_argument("--rank-floor", type=float, default=DEFAULTS["rank_floor"],
                   help="vote given to a breakpoint whose best supporting FFC is "
                        "ranked last (1.0 = ignore rankings)")
    p.add_argument("--no-mock-gate", action="store_true",
                   help="bank an internal line's mock without the goodness test")
    p.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    p.add_argument("--max-paths", type=int, default=DEFAULTS["max_paths"])
    p.add_argument("--label-tol", type=float, default=DEFAULTS["label_tol"])
    p.add_argument("--no-score", action="store_true",
                   help="do not grade against the known sequence")
    p.add_argument("--output-dir", default=OUT_DIR)
    p.add_argument("--no-detail", action="store_true", help="skip the .xlsx dumps")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    opts = dict(DEFAULTS)
    opts.update(
        line_delta=args.line_delta, line_delta_unit=args.line_delta_unit,
        line_min_ffc=args.line_min_ffc, tol_ppm=args.tol_ppm,
        precursor_ppm=args.precursor_ppm, sigma_floor_ppm=args.sigma_floor_ppm,
        mock_sigma_cap_da=args.mock_sigma_cap_da,
        master_align_threshold=args.master_align_threshold,
        accept_ambiguity=args.accept_ambiguity,
        use_mocks=not args.no_mocks, mock_min_ffc=args.mock_min_ffc,
        include_isotope_lines=not args.no_isotope_lines,
        isotope_tol_da=args.isotope_tol_da,
        mock_weight=args.mock_weight, secondary_gate=args.secondary_gate,
        evidence_weight=args.evidence_weight, evidence_cap=args.evidence_cap,
        rank_floor=args.rank_floor,
        gate_mock_on_goodness=not args.no_mock_gate,
        max_rounds=args.max_rounds, max_paths=args.max_paths,
        label_tol=args.label_tol,
    )

    if args.data:
        if args.charge is None:
            raise SystemExit("--data needs --charge")
        mass = args.mass
        if mass is None:
            if not args.seq:
                raise SystemExit("--data needs --mass (or --seq to derive it)")
            mass = precursor_mass(args.seq, args.terminus)
        entries = [dict(name=os.path.basename(args.data), path=args.data,
                        charge=args.charge, mass=mass, seq=args.seq,
                        terminus=args.terminus,
                        fragmentation=args.fragmentation)]
    else:
        if args.all or args.only:
            entries = list(ALL_PEPTIDES)      # --only selects across both sets
        elif args.etd:
            entries = list(ETD_PEPTIDES)
        else:
            entries = list(MAIN_PEPTIDES)
        if args.only:
            want = set(args.only)
            entries = [e for e in entries if e["name"] in want]
            missing = want - {e["name"] for e in entries}
            if missing:
                raise SystemExit(f"unknown dataset(s): {sorted(missing)}; "
                                 f"have {[e['name'] for e in ALL_PEPTIDES]}")

    if args.list:
        for e in entries:
            print(f"{e['name']:14s} {e.get('fragmentation', 'HCD'):6s} "
                  f"charge {e['charge']}  {e.get('terminus')}  "
                  f"{e.get('data') or e.get('path')}")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for e in entries:
        for top_n in args.top_list:
            try:
                result, row = run_one(e, top_n, opts,
                                      score=not args.no_score,
                                      verbose=args.verbose)
                rows.append(row)
                if not args.no_detail:
                    sub = os.path.join(args.output_dir, f"top{top_n}")
                    os.makedirs(sub, exist_ok=True)
                    dump_details(result,
                                 os.path.join(sub, f"{e['name']}_denovo_top{top_n}.xlsx"),
                                 e["name"], top_n)
            except Exception as exc:                    # one bad map must not
                import traceback                        # kill the batch
                traceback.print_exc()
                rows.append(dict(peptide=e["name"], top_n=top_n, error=str(exc)))

    if rows:
        summary = pd.DataFrame(rows)
        print("\n" + "=" * 100)
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
        print(f"\nwrote {os.path.join(args.output_dir, 'summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
