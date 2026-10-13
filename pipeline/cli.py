#!/usr/bin/env python3
"""DeNovoFFC -- one command, two routes.

    denovoffc run --data MAP.tsv --charge 5 --mass 4001.3781 --terminus NH3
    denovoffc run --data MAP.tsv --seq KWKLFKK... --charge 5 --terminus NH3
    denovoffc prepare --format blocked-covariance --covariance C --intensity I -o T
    denovoffc combine --manifest sample.json -o result/combined
    denovoffc fill-gaps --reconstruction R.json --evidence-manifest sample.json

The route is chosen by whether a sequence is available:

    no sequence   ->  [deconvolution] -> DE NOVO FFC
    a sequence    ->  [deconvolution] -> ANNOTATION (de novo runs inside it)

Deconvolution is a separate axis: `--deconvolve auto` (the default) runs it
only when the input has no charge columns, and `--deconvolve yes|no` forces it.

Run `denovoffc <command> --help` for the full flag list, `denovoffc info` for
what this install can see.

Exit codes: 0 analysis completed (an explicit no-path / over-budget status is a
completed analysis), 1 I/O or runtime failure, 2 invalid input or a detected
parity/integrity failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import pandas as pd

from . import annotate, bench, denovo, deconvolve, options, paths, prep
from .artifacts import IntegrityError, sha256_file, write_json_atomic
from .manifest import ManifestError

TERMINI = ("H2O", "OH", "NH3", "NH2")
FRAGMENTATION = ("HCD", "ETD", "ETHCD")
EXIT_OK, EXIT_ERROR, EXIT_INVALID = 0, 1, 2


class InvalidInput(ValueError):
    """Bad flags or inputs -- exit code 2."""


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

    frag = (args.fragmentation or "HCD").upper()
    series = denovo.resolve_ion_series(frag, args.ion_series or "auto")
    denovo.validate_charge(int(charge), series)
    return dict(seq=seq, charge=int(charge), terminus=terminus,
                mass=float(mass), fragmentation=frag, ion_series=series)


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
    request = deconvolve.make_request(entry["charge"], entry["mass"], entry["ion_series"])
    print(f"  deconvolution line model: charge {request.line_model_charge}, "
          f"mass {request.line_model_charged_mass:.4f}  "
          f"(TopFD bounds: charge {request.topfd_max_charge}, "
          f"mass {request.topfd_max_mass:.4f})")
    t0 = time.time()
    stats = deconvolve.run(
        args.data, request=request, output_dir=dec_dir, name=name,
        run_topfd=not args.no_topfd, skiprows=args.skiprows,
        params={k: v for k, v in
                (("top_lines", args.top_lines), ("top_selected", args.top_selected))
                if v is not None})
    print(f"  deconvolution done in {time.time() - t0:.1f}s -> "
          f"{stats['merged_path']}")
    return stats["merged_path"], True


def _run_label(args) -> str:
    """The run label names output files and FFC ids: an explicit ``--label``
    must be a safe identifier; one derived from the file name is made safe."""
    import re
    from .validation import ArtifactError, check_identifier
    if args.label:
        try:
            return check_identifier(args.label, "--label")
        except ArtifactError as exc:
            raise InvalidInput(str(exc)) from None
    stem = os.path.splitext(os.path.basename(args.data))[0]
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", stem)[:128]
    return safe if safe[:1].isalnum() else ("map" + safe)[:128]


def cmd_run(args) -> int:
    entry = resolve_entry(args)
    label = _run_label(args)
    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)
    dn = options.overrides(args, options.ANALYSIS, "denovo")
    an = options.overrides(args, options.ANALYSIS, "annotation")
    gap = options.overrides(args, options.GAP, "gap")

    known = bool(entry["seq"])
    series = entry["ion_series"]
    if known and dn.get("reconstruction_method", "unit") != "unit":
        if series != "by":
            raise InvalidInput("RBDP is validated for HCD b/y only; the c/z annotation "
                               "route has no RBDP adapter")
        an["reconstruction_method"] = dn["reconstruction_method"]
        an["rbdp_evidence"] = dn.get("rbdp_evidence", "v2")
        print("  annotation reconstruction: RBDP adapter (reference-assisted; the "
              "coordinates are snapped to the known sequence)")
    if args.gap_fill == "ffc" and known:
        raise InvalidInput("--gap-fill scores de novo gaps; omit --seq")
    if gap and args.gap_fill != "ffc":
        raise InvalidInput(f"{options.given(args, options.GAP)} need --gap-fill ffc")
    route = (f"annotation (known peptide, {'c/z' if series == 'cz' else 'b/y'}, "
             "reference-assisted)" if known else "de novo (unknown peptide)")
    print(f"=== {label} ===")
    print(f"  route: {route}")
    print(f"  charge {entry['charge']}, neutral mass {entry['mass']:.4f}, "
          f"C-term {entry['terminus']}, fragmentation {entry['fragmentation']}, "
          f"ion series {series}")
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
    from .artifacts import runtime_info
    manifest.update(schema_version=2,
                    inputs=[dict(path=os.path.abspath(args.data),
                                 sha256=sha256_file(args.data))],
                    stages=[], runtime=runtime_info())

    if known:
        if series == "cz":
            from . import annotate_etd
            top_list = args.top_list or annotate_etd.DEFAULT_TOP_LIST
            rec = annotate_etd.run(merged, seq=entry["seq"], charge=entry["charge"],
                                   terminus=entry["terminus"],
                                   out_root=os.path.join(out_root, "annotation"),
                                   name=label, top_list=top_list, params=an,
                                   neutral_mass=entry["mass"])
        else:
            top_list = args.top_list or annotate.DEFAULT_TOP_LIST
            rec = annotate.run(merged, seq=entry["seq"], charge=entry["charge"],
                               terminus=entry["terminus"],
                               out_root=os.path.join(out_root, "annotation"),
                               name=label, top_list=top_list, params=an)
            rec["reference_assisted"] = True
        manifest["annotation"] = rec
        manifest["stages"].append(dict(stage="annotation", engine="python",
                                       series=series, status="ok"))
        _print_annotation(rec, top_list)
    else:
        top_list = args.top_list or denovo.DEFAULT_TOP_LIST
        rows = []
        for n in top_list:
            t0 = time.time()
            d = os.path.join(out_root, "denovo", f"top{n}")
            os.makedirs(d, exist_ok=True)
            ev_path = (os.path.join(d, f"{label}_denovo_top{n}.evidence.json")
                       if args.emit_evidence else None)
            res = denovo.run(merged, parent_charge=entry["charge"],
                             parent_mass=entry["mass"],
                             terminus=entry["terminus"],
                             fragmentation=entry["fragmentation"],
                             ion_series=series, top_n=n, engine=args.engine,
                             params=dn, dataset_id=label, evidence_path=ev_path)
            dt = time.time() - t0
            write_json_atomic(os.path.join(d, f"{label}_denovo_top{n}.json"), res)
            denovo.write_detail(res, os.path.join(d, f"{label}_denovo_top{n}.xlsx"))
            rows.append(_denovo_row(label, n, res, dt))
            manifest["stages"].append(dict(
                stage="reconstruction", top_n=n, engine=res.get("_engine"),
                status=res.get("status"),
                method=res.get("reconstruction", {}).get("effective_method")))
            _print_denovo(n, res, dt)
        if args.emit_evidence:
            merged.to_csv(os.path.join(out_root, "denovo", f"{label}_engine_input.tsv"),
                          sep="\t", index=False)
        manifest["denovo"] = rows
        manifest["effective_settings"] = dict(denovo=dict(denovo.DEFAULTS, **dn))
        pd.DataFrame(rows).to_csv(
            os.path.join(out_root, "denovo", "summary.csv"), index=False)
        if args.gap_fill == "ffc":
            from . import gap_fill
            n = max(top_list)
            rec_path = os.path.join(out_root, "denovo", f"top{n}",
                                    f"{label}_denovo_top{n}.json")
            res_gap = gap_fill.run_for_single_map(
                rec_path, table_path=table_path, entry=entry, label=label,
                out_dir=out_root, settings=gap, engine=args.engine)
            manifest["gap_filling"] = res_gap
            manifest["stages"].append(dict(stage="gap_filling", engine="python",
                                           status=res_gap.get("status")))

    write_json_atomic(os.path.join(out_root, "run.json"), manifest)
    print(f"\n  wrote {out_root}")
    return EXIT_OK


def _denovo_row(label, n, res, seconds) -> dict:
    return dict(peptide=label, top_n=n, engine=res.get("_engine", "?"),
                n_ffc=res["n_ffc"], n_lines=res["n_lines"],
                n_master=len(res["master_set"]), rounds=res["rounds"],
                n_paths=res["n_paths"],
                exact_optimal_paths=res.get("exact_optimal_path_count"),
                method=res.get("reconstruction", {}).get("effective_method", "unit"),
                status=res.get("status", "ok"), seconds=round(seconds, 2),
                reconstruction=(res["reconstructions"][0]
                                if res["reconstructions"] else ""))


def _print_denovo(n, res, seconds) -> None:
    print(f"\n  --- de novo, top{n} ({res.get('_engine')}, {seconds:.2f}s) ---")
    print(f"    lines {res['n_lines']}, master TFFC-set "
          f"{len(res['master_set'])} breakpoints after {res['rounds']} round(s)")
    rb = res.get("rbdp")
    if rb:
        print(f"    reconstruction {rb['requested_method']} -> "
              f"{rb['effective_method']} (status {rb['status']}"
              + (f", fallback: {rb['fallback_reason']}" if rb.get("fallback_reason") else "")
              + ")")
    if not res["reconstructions"]:
        if rb and rb.get("status") not in (None, "ok", "fallback_unit"):
            print(f"    RBDP returned no colored reconstruction: {rb['status']} "
                  "(the unit control is in the JSON's rbdp section)")
            return
        print("    no reconstruction -- no path reached the sink.  With 0 "
              "master breakpoints, look for a parental line first.")
        return
    exact = res.get("exact_optimal_path_count")
    shown = (f"{res['n_paths']} of {exact} tied reconstruction(s)"
             if exact not in (None, str(res["n_paths"]))
             else f"{res['n_paths']} tied reconstruction(s)")
    print(f"    {shown}:")
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
    print(f"=== DeNovoFFC benchmark: {args.set} set, "
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
    print(f"DeNovoFFC  root {paths.ROOT}")
    exe = paths.cpp_engine()
    print(f"  C++ engine      {exe or 'NOT BUILT  (run cpp/build.sh)'}")
    print(f"  edges.json      {paths.EDGES_JSON if os.path.isfile(paths.EDGES_JSON) else 'not generated (built on first Python-engine run)'}")
    print(f"  theo_patt.txt   {paths.THEO_PATT if os.path.isfile(paths.THEO_PATT) else 'MISSING'}")
    tf = paths.topfd_bin()
    print(f"  TopFD           {tf or 'not found  (set $DENOVOFFC_TOPFD_BIN, or use --no-topfd)'}")
    print(f"  TopFD libs      {os.environ.get('DENOVOFFC_TOPFD_LIB', 'not set ($DENOVOFFC_TOPFD_LIB)')}")
    for mod in ("numpy", "pandas", "scipy", "openpyxl", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  {mod:<15} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"  {mod:<15} MISSING")
    return 0


# ── argument parsing ────────────────────────────────────────────────────────
def cmd_prepare(args) -> int:
    from . import prepare
    meta = prepare.run(args.format, args.covariance, args.output,
                       intensity=args.intensity, mz_atol=args.mz_atol,
                       covariance_alias=not args.no_covariance_alias)
    print(f"  {meta['n_peaks']} peaks, {meta['n_ffc_rows']:,} FFC rows -> {args.output}")
    print(f"  missing intensities: {meta['missing_intensity_rows']:,}; "
          f"Covariance column: {meta['covariance_column']}")
    print(f"  wrote {prepare.sidecar_path(args.output)}")
    return EXIT_OK


def cmd_combine(args) -> int:
    from . import combine
    return combine.run_command(args)


def cmd_fill_gaps(args) -> int:
    from . import gap_fill
    return gap_fill.run_command(args)


# ── argument parsing ────────────────────────────────────────────────────────
def _add_reconstruction_args(p) -> None:
    a = p.add_argument_group("analysis")
    a.add_argument("--top-list", type=int, nargs="+",
                   help="top-N cutoffs to analyse "
                        "[de novo: 200 500; annotation: 500 2000]")
    a.add_argument("--engine", choices=("auto", "cpp", "python"), default="auto",
                   help="de novo engine")
    a.add_argument("--merge-tolerance", type=float,
                   help="m/z radius for the mirror-duplicate merge "
                        "[de novo: 0.001; annotation: 0.002 -- each route's "
                        "published benchmark value]")
    options.add(a, options.ANALYSIS)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="denovoffc", description=__doc__,
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
                   help="acquisition chemistry [HCD].  ⚠ fails silently when wrong")
    g.add_argument("--ion-series", choices=("auto", "by", "cz"), default="auto",
                   help="analysis channel: auto = by for HCD, cz for ETD/EThcD; "
                        "an explicit 'by' on EThcD is a separate b/y analysis")

    d = r.add_argument_group("deconvolution")
    d.add_argument("--deconvolve", choices=("auto", "yes", "no"), default="auto",
                   help="auto = only when the input has no charge columns")
    d.add_argument("--no-topfd", action="store_true",
                   help="skip Stage 2 (TopFD); Stage 1 alone still works")
    d.add_argument("--top-lines", type=int,
                   help="FFCs shown to line detection [2000]")
    d.add_argument("--top-selected", type=int,
                   help="FFCs deconvolved [15000]")
    d.add_argument("--skiprows", type=int, default=None,
                   help="rows to skip in a RAW FFC table [auto: read the header]")

    _add_reconstruction_args(r)
    o = r.add_argument_group("outputs and optional stages")
    o.add_argument("--emit-evidence", action="store_true",
                   help="de novo: also write stable FFC/line/node evidence ids")
    o.add_argument("--gap-fill", choices=("off", "ffc"), default="off",
                   help="de novo: rank fills of the unresolved gaps against this map")
    options.add(o, options.GAP)
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("prepare", help="convert an instrument export to an FFC table",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("--format", required=True, choices=("blocked-covariance",))
    pr.add_argument("--covariance", required=True, help="the blocked covariance export")
    pr.add_argument("--intensity", help="the matching intensity file (joined by index)")
    pr.add_argument("--output", "-o", required=True, help="the flat table to write")
    pr.add_argument("--mz-atol", type=float, default=1e-6,
                    help="allowed m/z disagreement between the two files")
    pr.add_argument("--no-covariance-alias", action="store_true",
                    help="omit the synthesized Covariance (= Partial Cov.) column")
    pr.set_defaults(func=cmd_prepare)

    c = sub.add_parser("combine", help="combine several acquisitions of one analyte",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    c.add_argument("--manifest", required=True, help="dataset manifest (JSON)")
    c.add_argument("--output", "-o", required=True, help="a new output directory")
    c.add_argument("--mode", choices=("prefix", "annotated-master"), default="prefix",
                   help="prefix = sequence-free (default); annotated-master = "
                        "reference-assisted pooling, needs --reference-sequence")
    c.add_argument("--reference-sequence",
                   help="annotated-master only; forbidden in prefix mode")
    c.add_argument("--selection", choices=("all", "best-per-series", "explicit"),
                   default="all")
    c.add_argument("--datasets", nargs="+", metavar="ID",
                   help="dataset ids for --selection explicit")
    c.add_argument("--profile", choices=("best-pair", "all"),
                   help="annotated-master settings profile [by number of datasets]")
    c.add_argument("--assume-sequence-free", action="store_true",
                   help="accept prior reconstruction artifacts that record no provenance "
                        "(legacy output) as sequence-free; the assertion is recorded")
    _add_reconstruction_args(c)
    options.add(c, options.COMBINE)
    cg = c.add_argument_group("optional gap scoring")
    cg.add_argument("--gap-fill", choices=("off", "ffc"), default="off")
    cg.add_argument("--score-datasets", choices=("all", "selected"),
                    help="evidence pool for gap scoring [selected]")
    options.add(cg, options.GAP)
    c.set_defaults(func=cmd_combine)

    f = sub.add_parser("fill-gaps", help="rank fills of reconstruction gaps by FFC evidence",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    f.add_argument("--reconstruction", required=True,
                   help="a reconstruction artifact (v2 JSON; legacy text is imported)")
    f.add_argument("--evidence-manifest", help="dataset manifest of the scoring maps")
    f.add_argument("--score-datasets", choices=("all", "selected"),
                   help="all manifest datasets, or the ones the reconstruction "
                        "recorded [selected]")
    f.add_argument("--datasets", nargs="+", metavar="ID",
                   help="an explicit evidence subset (manifest ids)")
    f.add_argument("--output", "-o", required=True, help="a new output directory")
    f.add_argument("--engine", choices=("auto", "cpp", "python"), default="auto",
                   help="fill ENUMERATION engine (the FFC matcher is --gap-engine)")
    f.add_argument("--assume-sequence-free", action="store_true",
                   help="accept a reconstruction that records no provenance (legacy "
                        "output) as sequence-free; the assertion is recorded")
    options.add(f, options.GAP)
    f.set_defaults(func=cmd_fill_gaps)

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
    except IntegrityError as exc:
        print(f"\nerror (integrity): {exc}", file=sys.stderr)
        return EXIT_INVALID
    except (InvalidInput, ManifestError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if os.environ.get("DENOVOFFC_TRACEBACK"):
            traceback.print_exc()
        return EXIT_INVALID
    except Exception as exc:                                    # noqa: BLE001
        print(f"\nerror: {exc}", file=sys.stderr)
        if os.environ.get("DENOVOFFC_TRACEBACK"):
            traceback.print_exc()
        else:
            print("(set DENOVOFFC_TRACEBACK=1 for the full traceback)",
                  file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
