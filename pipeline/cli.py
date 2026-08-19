#!/usr/bin/env python3
"""DeNovoMMS -- one command, two routes.

    denovomms run --data MAP.tsv --charge 5 --mass 4001.3781 --terminus NH3
    denovomms run --data MAP.tsv --seq KWKLFKK... --charge 5 --terminus NH3

The route is chosen by whether a sequence is available:

    no sequence   ->  [deconvolution] -> DE NOVO MMS
    a sequence    ->  [deconvolution] -> ANNOTATION (de novo runs inside it)

Deconvolution is a separate axis: `--deconvolve auto` (the default) runs it
only when the input has no charge columns, and `--deconvolve yes|no` forces it.

Run `denovomms run --help` for the full flag list, `denovomms info` for what
this install can see.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import pandas as pd

from . import annotate, bench, denovo, deconvolve, paths, prep

TERMINI = ("H2O", "OH", "NH3", "NH2")
FRAGMENTATION = ("HCD", "ETD", "ETHCD")


# ── precursor mass ──────────────────────────────────────────────────────────
def precursor_neutral_mass(seq: str, terminus: str) -> float:
    """Neutral peptide mass from a sequence -- what the instrument measures.

    Only used when a sequence is available and `--mass` was not given.  On the
    de novo route the mass is an INPUT, never derived.

    ⚠ NOT `peptide.Pep(...).pep_mass`, which is the CHARGED mass `[M+zH]z+`
    (it adds z protons; at z=1 that is still 1.007 Da too high).  Derived from
    residue prefix sums instead, so the C-terminus enters exactly once --
    the same definition `src/de_novo/run_denovo.py::precursor_mass` uses.
    """
    from denovo_core import TERMINI                            # noqa: E402
    from evaluate import residue_prefixes                      # noqa: E402
    key = terminus.upper()
    if key not in TERMINI:
        raise SystemExit(f"--terminus must be one of {TERMINI}, got {terminus!r}")
    return residue_prefixes(seq)[-1] + TERMINI[key]


def charged_mass(neutral: float, charge: int) -> float:
    """[M+zH]z+ -- what the deconvolution's line model wants (its `v` carries
    no proton term, so a parental line sums to the charged mass).

    Matches `peptide.Pep.pep_mass` by construction: residue sum + terminus +
    z * proton.
    """
    # peptide.py's own proton, not denovo_core's -- they differ in the 6th
    # decimal, and this value has to agree with `Pep.pep_mass` because the
    # deconvolution compares line masses against it.
    from peptide import AA                                      # noqa: E402
    return neutral + charge * AA.element_masses["proton"]


# ── the run ─────────────────────────────────────────────────────────────────
def resolve_entry(args) -> dict:
    """Sequence / charge / mass / terminus, from the flags or a dataset name."""
    seq, charge = args.seq, args.charge
    terminus, mass = args.terminus, args.mass

    if args.name:
        # data_prep/peptide_info parses e.g. HAD_33_Z4_E16 into peptide, charge
        # and C-terminus via pep_seq.xlsx -- the dataset name is then the only
        # input needed.
        from peptide_info import resolve as resolve_name        # noqa: E402
        info = resolve_name(args.name)
        seq = seq or info["SEQUENCE"]
        charge = charge or int(info["PARENT_CHARGE"])
        terminus = terminus or info["RESIDUAL"]

    terminus = (terminus or "H2O").upper()
    if terminus not in TERMINI:
        raise SystemExit(f"--terminus must be one of {TERMINI}, got {terminus!r}")
    if not charge:
        raise SystemExit("--charge is required (or --name to resolve it)")
    if mass is None:
        if not seq:
            raise SystemExit(
                "--mass is required when no sequence is known.  The precursor "
                "neutral mass is a measured quantity; the de novo route cannot "
                "guess it.")
        mass = precursor_neutral_mass(seq, terminus)

    return dict(seq=seq, charge=int(charge), terminus=terminus,
                mass=float(mass),
                fragmentation=(args.fragmentation or "HCD").upper())


def maybe_deconvolve(args, entry: dict, out_root: str) -> tuple[str, bool]:
    """Return (path to the table to analyse, whether we deconvolved)."""
    df = prep.read_table(args.data)
    already = prep.is_deconvolved(df)

    if args.deconvolve == "auto":
        want = not already
        print(f"  deconvolution: auto -> {'yes' if want else 'no'} "
              f"({'no charge columns in the input' if want else 'input already has charge_A/charge_B'})")
    else:
        want = (args.deconvolve == "yes")
        if want and already:
            print("  deconvolution: forced ON, though the input already carries "
                  "charge columns -- it will be re-deconvolved from the raw m/z")
    if not want:
        return args.data, False

    dec_dir = os.path.join(out_root, "deconvolution")
    name = args.label or os.path.splitext(os.path.basename(args.data))[0]
    t0 = time.time()
    stats = deconvolve.run(
        args.data, parent_charge=entry["charge"],
        parent_mass_charged=charged_mass(entry["mass"], entry["charge"]),
        output_dir=dec_dir, name=name, run_topfd=not args.no_topfd,
        skiprows=args.skiprows,
        params={k: v for k, v in
                (("top_lines", args.top_lines), ("top_selected", args.top_selected))
                if v is not None})
    print(f"  deconvolution done in {time.time() - t0:.1f}s -> "
          f"{stats['merged_path']}")
    return stats["merged_path"], True


def cmd_run(args) -> int:
    entry = resolve_entry(args)
    label = args.label or os.path.splitext(os.path.basename(args.data))[0]
    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)

    known = bool(entry["seq"])
    route = "annotation (known peptide)" if known else "de novo (unknown peptide)"
    print(f"=== {label} ===")
    print(f"  route: {route}")
    print(f"  charge {entry['charge']}, neutral mass {entry['mass']:.4f}, "
          f"C-term {entry['terminus']}, fragmentation {entry['fragmentation']}")
    if known:
        print(f"  sequence: {entry['seq']}  (length {len(entry['seq'])})")

    table_path, deconvolved = maybe_deconvolve(args, entry, out_root)

    print(f"  loading {table_path}")
    raw = prep.read_table(table_path)
    route_key = "annotation" if known else "denovo"
    merge_tol = (args.merge_tolerance if args.merge_tolerance is not None
                 else prep.MERGE_TOLERANCE[route_key])
    merged = prep.merge(raw, route_key, merge_tol)
    print(f"  {len(raw)} rows -> {len(merged)} after mirror-duplicate merge "
          f"(tolerance {merge_tol})")

    manifest = dict(label=label, route=route_key, merge_tolerance=merge_tol,
                    input=os.path.abspath(args.data), table=table_path,
                    deconvolved=deconvolved, n_raw=len(raw),
                    n_merged=len(merged), **{k: v for k, v in entry.items()})

    if known:
        top_list = args.top_list or annotate.DEFAULT_TOP_LIST
        if entry["fragmentation"] != "HCD":
            print("  ⚠ the annotation route is b/y (HCD).  For a c/z map use "
                  "the de novo route (omit --seq) with --fragmentation ETD, or "
                  "src/ETD/run_etd_annotation.py.")
        rec = annotate.run(merged, seq=entry["seq"], charge=entry["charge"],
                           terminus=entry["terminus"],
                           out_root=os.path.join(out_root, "annotation"),
                           name=label, top_list=top_list,
                           params=_annotation_overrides(args))
        manifest["annotation"] = rec
        _print_annotation(rec, top_list)
    else:
        top_list = args.top_list or denovo.DEFAULT_TOP_LIST
        rows = []
        for n in top_list:
            t0 = time.time()
            res = denovo.run(merged, parent_charge=entry["charge"],
                             parent_mass=entry["mass"],
                             terminus=entry["terminus"],
                             fragmentation=entry["fragmentation"],
                             top_n=n, engine=args.engine,
                             params=_denovo_overrides(args))
            dt = time.time() - t0
            d = os.path.join(out_root, "denovo", f"top{n}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{label}_denovo_top{n}.json"), "w") as fh:
                json.dump(res, fh, indent=1)
            denovo.write_detail(res, os.path.join(d, f"{label}_denovo_top{n}.xlsx"))
            rows.append(_denovo_row(label, n, res, dt))
            _print_denovo(n, res, dt)
        manifest["denovo"] = rows
        pd.DataFrame(rows).to_csv(
            os.path.join(out_root, "denovo", "summary.csv"), index=False)

    with open(os.path.join(out_root, "run.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"\n  wrote {out_root}")
    return 0


def _denovo_overrides(args) -> dict:
    o = {}
    for flag, key in (("delta", "delta"), ("delta_unit", "delta_unit"),
                      ("min_ffc", "min_ffc_number"), ("tol_ppm", "tol_ppm"),
                      ("precursor_ppm", "precursor_ppm"),
                      ("secondary_gate", "secondary_gate"),
                      ("max_paths", "max_paths")):
        v = getattr(args, flag, None)
        if v is not None:
            o[key] = v
    if args.accept_ambiguity:
        o["accept_ambiguity"] = True
    if args.no_isotope_lines:
        o["include_isotope_lines"] = False
    if args.no_mocks:
        o["use_mocks"] = False
    return o


def _annotation_overrides(args) -> dict:
    o = {}
    if args.tol_ppm is not None:
        o["tol_ppm"] = args.tol_ppm
    if args.delta is not None:
        o["line_delta"] = args.delta
    if args.delta_unit is not None:
        o["line_delta_unit"] = args.delta_unit
    if args.accept_ambiguity:
        o["accept_ambiguity"] = True
    if args.graph:
        o["graph"] = True
    return o


def _denovo_row(label, n, res, seconds) -> dict:
    return dict(peptide=label, top_n=n, engine=res.get("_engine", "?"),
                n_ffc=res["n_ffc"], n_lines=res["n_lines"],
                n_master=len(res["master_set"]), rounds=res["rounds"],
                n_paths=res["n_paths"], seconds=round(seconds, 2),
                reconstruction=(res["reconstructions"][0]
                                if res["reconstructions"] else ""))


def _print_denovo(n, res, seconds) -> None:
    print(f"\n  --- de novo, top{n} ({res.get('_engine')}, {seconds:.2f}s) ---")
    print(f"    lines {res['n_lines']}, master TFFC-set "
          f"{len(res['master_set'])} breakpoints after {res['rounds']} round(s)")
    if not res["reconstructions"]:
        print("    no reconstruction -- no path reached the sink.  With 0 "
              "master breakpoints, look for a parental line first.")
        return
    print(f"    {res['n_paths']} tied reconstruction(s):")
    for k, r in enumerate(res["reconstructions"][:3], 1):
        print(f"      [{k}] {r}")
    if res["n_paths"] > 3:
        print(f"      ... {res['n_paths'] - 3} more")


def _print_annotation(rec, top_list) -> None:
    print("\n  --- annotation ---")
    for n in top_list:
        s = rec["per_top"].get(n, {})
        print(f"    top{n}: breakpoint coverage {s.get('breakpoint_coverage', '?')}, "
              f"master TFFC-set {s.get('master_coverage', '?')}, "
              f"#incorrect {s.get('#incorrect TFFCs', '?')}, "
              f"reconstructions {s.get('#reconstructions/correct?', '?')}")
        if s.get("reconstruction"):
            print(f"      -> {s['reconstruction']}")


# ── benchmark ───────────────────────────────────────────────────────────────
def cmd_bench(args) -> int:
    print(f"=== DeNovoMMS benchmark: {args.set} set, "
          f"top-N {args.top_list or denovo.DEFAULT_TOP_LIST} ===")
    df = bench.run(names=args.only, which=args.set, top_list=args.top_list,
                   engine=args.engine, output_dir=args.output,
                   detail=not args.no_detail, verbose=args.verbose)
    if df.empty:
        print("nothing ran -- is the benchmark data present?")
        return 1
    cols = [c for c in df.columns if c != "reconstruction"]
    print()
    print(df[cols].to_string(index=False))
    print()
    print(bench.summarise(df))
    print()
    for _, r in df[df["top_n"] == df["top_n"].max()].iterrows():
        print(f"  {r['peptide']:<12} {r['reconstruction']}")
    print(f"\n  wrote {os.path.abspath(args.output)}")
    return 0


# ── info ────────────────────────────────────────────────────────────────────
def cmd_info(args) -> int:
    print(f"DeNovoMMS  root {paths.ROOT}")
    exe = paths.cpp_engine()
    print(f"  C++ engine      {exe or 'NOT BUILT  (run cpp/build.sh)'}")
    print(f"  edges.json      {paths.EDGES_JSON if os.path.isfile(paths.EDGES_JSON) else 'not generated (built on first Python-engine run)'}")
    print(f"  theo_patt.txt   {paths.THEO_PATT if os.path.isfile(paths.THEO_PATT) else 'MISSING'}")
    tf = paths.topfd_bin()
    print(f"  TopFD           {tf or 'not found  (set $DENOVOMMS_TOPFD_BIN, or use --no-topfd)'}")
    print(f"  TopFD libs      {os.environ.get('DENOVOMMS_TOPFD_LIB', 'not set ($DENOVOMMS_TOPFD_LIB)')}")
    for mod in ("numpy", "pandas", "scipy", "openpyxl", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod:<15} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"  {mod:<15} MISSING")
    return 0


# ── argument parsing ────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="denovomms", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline on one MMS map",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("--data", required=True, help="FFC table (raw or deconvolved)")
    r.add_argument("--output", "-o", default="result", help="output directory")
    r.add_argument("--label", help="name used in output filenames "
                                   "[default: the input's basename]")

    g = r.add_argument_group("peptide (a sequence selects the annotation route)")
    g.add_argument("--seq", help="peptide sequence, if known")
    g.add_argument("--name", help="dataset name to resolve via pep_seq.xlsx, "
                                  "e.g. HAD_33_Z4_E16")
    g.add_argument("--charge", type=int, help="precursor charge")
    g.add_argument("--mass", type=float,
                   help="precursor NEUTRAL mass; required without a sequence")
    g.add_argument("--terminus", choices=TERMINI,
                   help="C-terminal group [H2O].  ⚠ fails silently when wrong")
    g.add_argument("--fragmentation", choices=FRAGMENTATION,
                   help="[HCD].  ⚠ fails silently when wrong; de novo route only")

    d = r.add_argument_group("deconvolution")
    d.add_argument("--deconvolve", choices=("auto", "yes", "no"), default="auto",
                   help="auto = only when the input has no charge columns")
    d.add_argument("--no-topfd", action="store_true",
                   help="skip Stage 2 (TopFD); Stage 1 alone still works")
    d.add_argument("--top-lines", type=int,
                   help="FFCs shown to line detection [2000]")
    d.add_argument("--top-selected", type=int,
                   help="FFCs deconvolved [15000]")
    d.add_argument("--skiprows", type=int, default=1,
                   help="header rows to skip in a RAW FFC table")

    a = r.add_argument_group("analysis")
    a.add_argument("--top-list", type=int, nargs="+",
                   help="top-N cutoffs to analyse "
                        "[de novo: 200 500; annotation: 500 2000]")
    a.add_argument("--engine", choices=("auto", "cpp", "python"), default="auto",
                   help="de novo engine")
    a.add_argument("--tol-ppm", type=float, help="fragment-mass relative error")
    a.add_argument("--delta", type=float, help="line-detection Sort-and-Split gap")
    a.add_argument("--delta-unit", choices=("da", "ppm"))
    a.add_argument("--min-ffc", type=int, help="distinct FFCs to keep a line")
    a.add_argument("--precursor-ppm", type=float)
    a.add_argument("--secondary-gate", type=float,
                   help="de novo: refuse the mock of a secondary line")
    a.add_argument("--max-paths", type=int, help="tied-path cap")
    a.add_argument("--merge-tolerance", type=float,
                   help="m/z radius for the mirror-duplicate merge "
                        "[de novo: 0.001; annotation: 0.002 -- each route's "
                        "published benchmark value]")
    a.add_argument("--accept-ambiguity", action="store_true",
                   help="bank BOTH images of an ambiguous TFFC (off by "
                        "default: net negative across 7 peptides)")
    a.add_argument("--no-isotope-lines", action="store_true",
                   help="de novo: ignore parental isotope-satellite lines")
    a.add_argument("--no-mocks", action="store_true",
                   help="de novo: internal lines contribute no mock")
    a.add_argument("--graph", action="store_true",
                   help="annotation: render the 2D reconstruction map")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser(
        "bench", help="run the benchmark peptides and score them",
        description="Run the reference peptides end to end and grade each "
                    "answer against its known sequence.  ⚠ The only place a "
                    "ground-truth sequence enters the de novo route, and it "
                    "enters strictly AFTER a result exists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    b.add_argument("--set", choices=("main", "etd", "all"), default="main",
                   help="main = the 7 HCD peptides; etd = the 7 ETD/EThcD "
                        "datasets; all = both")
    b.add_argument("--only", nargs="+", metavar="NAME",
                   help="just these, e.g. --only VEA3+ KWK5+")
    b.add_argument("--top-list", type=int, nargs="+",
                   help="top-N cutoffs [200 500]")
    b.add_argument("--engine", choices=("auto", "cpp", "python"), default="auto")
    b.add_argument("--output", "-o", default="result/benchmark")
    b.add_argument("--no-detail", action="store_true",
                   help="skip the per-run .xlsx workbooks")
    b.add_argument("--verbose", "-v", action="store_true")
    b.set_defaults(func=cmd_bench)

    i = sub.add_parser("info", help="what this install can see")
    i.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:                                    # noqa: BLE001
        print(f"\nerror: {exc}", file=sys.stderr)
        if os.environ.get("DENOVOMMS_TRACEBACK"):
            traceback.print_exc()
        else:
            print("(set DENOVOMMS_TRACEBACK=1 for the full traceback)",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
