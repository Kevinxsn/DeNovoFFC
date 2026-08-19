#!/usr/bin/env python3
"""
Deconvolution runner -- resolves the peptide from the dataset name
===================================================================
The dataset names encode everything the pipeline needs.  ``HAD_33_Z4_E16`` says
peptide ``HAD_33`` at charge 4; ``data_prep/peptide_info.py`` looks the sequence
and C-terminus up in ``pep_seq.xlsx`` and computes the precursor mass.  So the
only thing you ever pass is a name -- or nothing at all, in which case every
resolvable dataset in the input folder is run.

This is the same name-resolution mechanism ``annotation_pepline/
run_annotation_reconstruct.py`` uses to discover its inputs, applied one step
earlier in the chain.  Names that ``peptide_info`` cannot resolve are reported
and skipped, never guessed at.

Input   /home/sux002/data/mms_with_intensity/<NAME>
Output  /home/sux002/data/deconvoluted_new/
            merge/<NAME>_merged.tsv          <- the deliverable
            intermediate/<NAME>/...          <- per-stage tables, TopFD workdir
            log/<NAME>_log.txt               <- full console output

``merge/<NAME>_merged.tsv`` is named so that ``run_annotation_reconstruct.py``'s
discovery picks it up unchanged: point its ``--data-dir`` at the ``merge``
folder and the annotation stage runs on these tables.

Usage
-----
    python run_deconvolution.py                        # every resolvable dataset
    python run_deconvolution.py --list                 # show what would run
    python run_deconvolution.py --only HAD_33_Z4_E16
    python run_deconvolution.py --tags KWK LLG
    python run_deconvolution.py --no-test              # skip the spurious test
    python run_deconvolution.py --delta 20 --delta-unit ppm
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback

import _paths  # noqa: F401  (sys.path bootstrap)

from peptide_info import resolve as resolve_name  # noqa: E402
from deconv_pipeline_oa import run_deconvolution  # noqa: E402


DATA_DIR = "/home/sux002/data/mms_with_intensity"
OUTPUT_DIR = "/home/sux002/data/deconvoluted_new"
LOG_SUBDIR = "log"


# ---------------------------------------------------------------------------
# Shared settings.  Anything here can be overridden from the CLI.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # ── Stage 1: line finding ──
    # 0.02 Da is line_based_annotation's module default: wide enough to bridge
    # the few-mDa spread between mirror-paired FFC reconstructions, which is
    # what makes the orientation-agnostic detector pay off (line_based.md §4).
    # The previous pipeline used 0.01 Da with a literal-(i, j) detector.
    delta=0.02,
    delta_unit="da",           # "da" (fixed gap) or "ppm" (relative gap)
    min_ffc_number=3,          # a line needs this many DISTINCT FFCs
    line_tol=None,             # FFC-on-line removal tolerance; None -> delta
    top_lines=2000,            # detect lines on the best-ranked N FFCs only
                               # (None -> all of top_selected).  2000 from the
                               # demo_top_lines_sweep over the Main set: bond
                               # coverage is flat from 2000 to 15000 on all 7
                               # peptides and *better* on LLG6+, spurious pairs
                               # rise with depth, and detection is 3-6x faster.
                               # See deconvolution.md section 12.
    merge_tolerance=0.002,     # m/z proximity for the symmetric duplicate merge
    merge_over_fetch=4,        # merge this multiple of top_bg best-ranked rows
    silent_drop=False,         # keep every line membership per FFC
    fast_detector=True,        # vectorised detector; False = the reference one
    # ── Stage 1: deconvolution back end ──
    mz_tol=0.01,
    max_shift=None,
    top_selected=15000,        # now counts DISTINCT FFCs (no doubling)
    top_bg=15000,
    # ── Stage 2: TopFD (Xiaowen-tuned) ──
    run_topfd=True,
    topfd_error_tol=0.04,
    topfd_sn_ratio=0.1,
    topfd_ecscore_cutoff=0.0,
    topfd_full_pool=True,
    # -c and -m track the precursor: set per dataset from the resolved info.
    topfd_max_charge=None,
    topfd_max_mass=None,
    # The two custom-build flags, off by default so the SHIPPED v1.8.1 binary
    # works.  For Xiaowen's rebuilt debug_mms binary set 4 / True.
    topfd_max_miss_peak_num=None,
    topfd_disable_filter_by_mz=False,
)

#: Spurious-parental test (step 2 of the old run_deconv_test_xiaowen.sh).
TEST_DEFAULTS = dict(iso_range=4, threshold=0.1)


class _Tee:
    """Write to several streams at once (console + per-dataset log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def discover(data_dir: str, tags=None) -> tuple[list[dict], list[str]]:
    """Every file in ``data_dir`` whose name ``peptide_info`` can resolve.

    Returns ``(entries, skipped)``; each entry carries the resolved peptide
    parameters alongside the input path.  ``tags`` (e.g. ``["KWK", "LLG"]``)
    restricts to those peptide prefixes.
    """
    wanted = {t.upper() for t in tags} if tags else None
    entries: list[dict] = []
    skipped: list[str] = []

    for fname in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            info = resolve_name(fname)
        except (ValueError, KeyError):
            skipped.append(fname)
            continue
        if wanted and info["PEPTIDE_ID"].split("_")[0] not in wanted:
            continue
        entries.append({**info, "path": path})
    return entries, skipped


def run_one(entry: dict, output_dir: str, overrides: dict,
            run_test: bool = True, write_log: bool = True) -> dict:
    """Deconvolve one dataset (and optionally test it).  Never raises."""
    name = entry["NAME"]
    kwargs = {**DEFAULTS, **overrides}
    record = {"name": name, "status": "ok", "seconds": 0.0, "error": None,
              "merged": None}

    log_dir = os.path.join(output_dir, LOG_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_log.txt") if write_log else None

    started = time.time()
    try:
        with contextlib.ExitStack() as stack:
            if log_path:
                fh = stack.enter_context(open(log_path, "w"))
                stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, fh)))

            print(f"── {name} ─────────────────────────────────────────────")
            print(f"   sequence    {entry['SEQUENCE']}")
            print(f"   charge      {entry['PARENT_CHARGE']}")
            print(f"   parent mass {entry['PARENT_MASS']}  "
                  f"([M+{entry['PARENT_CHARGE']}H]{entry['PARENT_CHARGE']}+, "
                  f"m/z {entry['PARENT_MZ']})")
            print(f"   C-terminus  {entry['RESIDUAL']} (--end {entry['END']})")
            print()

            result = run_deconvolution(
                ffc_path=entry["path"],
                parent_charge=entry["PARENT_CHARGE"],
                parent_mass=entry["PARENT_MASS"],
                output_dir=output_dir,
                output_prefix=name,
                theo_patt_path=_paths.THEO_PATT,
                topfd_bin=_paths.TOPFD_BIN,
                topfd_env_vars={"LD_LIBRARY_PATH": _paths.TOPFD_LIB},
                **kwargs,
            )
            record["merged"] = result["stats"]["merged_path"]

            if run_test:
                print("\n" + "=" * 60)
                print("SPURIOUS-PARENTAL TEST")
                print("=" * 60)
                from run_spurious_test_charge import main as spurious_main
                spurious_main([
                    record["merged"],
                    "--sequence", entry["SEQUENCE"],
                    "--charge", str(entry["PARENT_CHARGE"]),
                    "--iso-range", str(TEST_DEFAULTS["iso_range"]),
                    "--threshold", str(TEST_DEFAULTS["threshold"]),
                    "--end", entry["END"],
                ])
    except Exception as exc:                       # keep the batch going
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    record["seconds"] = time.time() - started
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Deconvolve FFC datasets, resolving the peptide from the "
                    "dataset name.")
    ap.add_argument("--list", action="store_true",
                    help="print what would run and exit")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="run only these datasets")
    ap.add_argument("--tags", nargs="+", metavar="TAG",
                    help="restrict to these peptide prefixes (e.g. KWK LLG)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--no-log", action="store_true",
                    help="do not write per-dataset log files")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the spurious-parental test after each run")

    # Stage 1: line finding
    ap.add_argument("--delta", type=float)
    ap.add_argument("--delta-unit", choices=["da", "ppm"])
    ap.add_argument("--min-ffc", type=int, dest="min_ffc_number")
    ap.add_argument("--line-tol", type=float)
    ap.add_argument("--top-lines", type=int,
                    help="detect lines on the best-ranked N FFCs only; the "
                         "lines found are then matched against the rest")
    ap.add_argument("--merge-tolerance", type=float)
    ap.add_argument("--merge-over-fetch", type=int)
    ap.add_argument("--silent-drop", action="store_true", default=None)
    ap.add_argument("--reference-detector", action="store_true", default=None,
                    help="use line_based_annotation's reference detector "
                         "instead of the vectorised one (same lines, slower)")
    # Stage 1: back end
    ap.add_argument("--mz-tol", type=float)
    ap.add_argument("--top-selected", type=int)
    ap.add_argument("--top-bg", type=int)
    # Stage 2
    ap.add_argument("--no-topfd", action="store_true",
                    help="skip Stage 2 entirely")
    ap.add_argument("--topfd-error-tol", type=float)
    ap.add_argument("--topfd-sn-ratio", type=float)
    ap.add_argument("--topfd-ecscore-cutoff", type=float)
    ap.add_argument("--topfd-max-miss-peak-num", type=int,
                    help="custom-build flag; omit for the shipped binary")
    ap.add_argument("--topfd-disable-filter-by-mz", action="store_true",
                    default=None, help="custom-build flag")
    args = ap.parse_args(argv)

    entries, skipped = discover(args.data_dir, args.tags)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["NAME"] in wanted]
        unknown = wanted - {e["NAME"] for e in entries}
        if unknown:
            ap.error(f"unknown or unresolvable dataset(s): "
                     f"{', '.join(sorted(unknown))}")

    if args.list:
        print(f"{'name':34} {'z':>2} {'res':4} {'mass':>12}  sequence")
        for e in entries:
            print(f"{e['NAME']:34} {e['PARENT_CHARGE']:>2} {e['RESIDUAL']:4} "
                  f"{e['PARENT_MASS']:>12}  {e['SEQUENCE'][:40]}")
        print(f"\n{len(entries)} dataset(s).")
        if skipped:
            print(f"{len(skipped)} unresolvable: {', '.join(skipped)}")
        return 0

    if skipped:
        print(f"[info] skipped {len(skipped)} unresolvable file(s): "
              + ", ".join(skipped), file=sys.stderr)
    if not entries:
        print("[error] nothing to run", file=sys.stderr)
        return 1

    overrides = {k: v for k, v in {
        "delta": args.delta,
        "delta_unit": args.delta_unit,
        "min_ffc_number": args.min_ffc_number,
        "line_tol": args.line_tol,
        "top_lines": args.top_lines,
        "merge_tolerance": args.merge_tolerance,
        "merge_over_fetch": args.merge_over_fetch,
        "silent_drop": args.silent_drop,
        "fast_detector": False if args.reference_detector else None,
        "mz_tol": args.mz_tol,
        "top_selected": args.top_selected,
        "top_bg": args.top_bg,
        "run_topfd": False if args.no_topfd else None,
        "topfd_error_tol": args.topfd_error_tol,
        "topfd_sn_ratio": args.topfd_sn_ratio,
        "topfd_ecscore_cutoff": args.topfd_ecscore_cutoff,
        "topfd_max_miss_peak_num": args.topfd_max_miss_peak_num,
        "topfd_disable_filter_by_mz": args.topfd_disable_filter_by_mz,
    }.items() if v is not None}

    results = []
    for k, entry in enumerate(entries, 1):
        print("\n" + "#" * 70)
        print(f"# [{k}/{len(entries)}] {entry['NAME']}")
        print("#" * 70)
        results.append(run_one(entry, args.output_dir, overrides,
                               run_test=not args.no_test,
                               write_log=not args.no_log))

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for r in results:
        line = f"  {r['name']:34} {r['status']:10} {r['seconds']:7.1f}s"
        if r["error"]:
            line += f"   {r['error']}"
        print(line)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} succeeded.  "
          f"Merged tables in {os.path.join(args.output_dir, 'merge')}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
