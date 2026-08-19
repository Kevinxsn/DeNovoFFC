#!/usr/bin/env python3
"""
Combined batch runner: main peptides + additional discovered tables
=====================================================================
Single entry point replacing the old run_all_peptides.py /
run_additional_kwk_llg_yps.py split. Runs the FFC annotation pipeline's
individual stages (complementary test, FFC annotation, line-based annotation)
over two peptide sets, landing all outputs in one shared tree:

  1. MAIN_PEPTIDES -- the hand-picked reference set (VEA3+, YLE3+, ...).
     Output filenames are prefixed "Main_".
  2. ADDITIONAL peptides -- every KWK/LLG/YPS deconvolution table discovered
     under DATA_DIR that data_prep/peptide_info.py can resolve (same
     discovery rule the old run_additional_kwk_llg_yps.py used). Output
     filenames are unprefixed.

Main and additional share the exact same output folders -- the "Main_"
prefix is the only thing that tells them apart.

Output tree
-----------
    result/annotation/
      ffc_annotation/
        Main_VEA3+_ffc_annotation.xlsx        (top_ffc_annotation FFCs, default 2000)
        KWK_37_Z5_E28_1_ffc_annotation.xlsx
        ...
      line_based/
        top200/
          Main_VEA3+_line_based_top200.xlsx
          KWK_37_Z5_E28_1_line_based_top200.xlsx
          ...
        top500/
          ...
        top2000/
          ...
      log/
        Main_VEA3+_log.txt
        KWK_37_Z5_E28_1_log.txt
        ...

Because line-based annotation runs once per entry in ``top_list`` (default
[200, 500, 2000]), each peptide's table is loaded and merged ONCE, and the
complementary test + FFC-annotation stage each run ONCE (at top_ffc_annotation,
default 2000); only the line-based stage repeats, once per top-N.

Usage
-----
    python run_annotation.py                       # everything (main + additional)
    python run_annotation.py --list                # show what would run, run nothing
    python run_annotation.py --main-only
    python run_annotation.py --additional-only
    python run_annotation.py --only VEA3+ KWK_37_Z6_E20
    python run_annotation.py --top-list 100 300
    python run_annotation.py --tol-ppm 10 --line-delta 5 --line-delta-unit ppm
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, os.path.join(_BASE, "data_prep")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from annotation_pipeline import (
    run_complementary, run_ffc_annotation, run_line_based,
    merge_duplicate_ffcs_charge, peptide,
)
from peptide_info import resolve as resolve_name  # noqa: E402


# ---------------------------------------------------------------------------
# Where the .tsv tables live.
# ---------------------------------------------------------------------------
DATA_DIR = "/home/sux002/data/deconvoluted/merged"

# ---------------------------------------------------------------------------
# Output tree: result/annotation/{ffc_annotation, line_based/top{N}, log}
# ---------------------------------------------------------------------------
FFC_SUBDIR = "ffc_annotation"
LINE_SUBDIR = "line_based"
LOG_SUBDIR = "log"


# ---------------------------------------------------------------------------
# Main peptides (hand-picked reference set). Output filenames get "Main_".
#
# Any DEFAULTS keyword may also be given per row to override it for just that
# entry -- e.g. add  tol_ppm=10  to just one dict.
# ---------------------------------------------------------------------------
MAIN_PEPTIDES = [
    dict(name="VEA3+",  data="VEA_15_Z3_revised_merged.tsv",
         seq="VEADIAGHGQEVLIR",                        charge=3, residual="H2O"),
    dict(name="YLE3+",  data="YLE_16_Z3_XIAOWEN_merged.tsv",
         seq="YLEFISDAIIHVLHSK",                       charge=3, residual="H2O"),
    dict(name="HGT3+",  data="HGT_merged.tsv",
         seq="HGTVVLTALGGILK",                         charge=3, residual="H2O"),
    # Kept identical to run_annotation_reconstruct.py's list (see
    # annotation_pipeline.md section 5); swapped off HAD_merged.tsv on
    # 2026-08-03 -- the rationale is in that file's copy of this entry.
    dict(name="HAD4+",  data="HAD_33_Z4_E15_MF_I270_merged.tsv",
         seq="HADGSFSDEMNTILDNLAARDFINWLIQTKITD",      charge=4, residual="H2O"),
    dict(name="KWK5+",  data="KWK_37_Z5_E28_2_XIAOWEN_merged.tsv",
         seq="KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  charge=5, residual="NH3"),
    dict(name="LL6+",   data="LLG_37_Z6_E30_I130_merged.tsv",
         seq="LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  charge=6, residual="H2O"),
    dict(name="YPS6+",  data="YPS_36_Z6_E25_I300_merged.tsv",
         seq="YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY",   charge=6, residual="NH3"),
]

# ---------------------------------------------------------------------------
# Additional peptides: auto-discovered from DATA_DIR (KWK/LLG/YPS by default).
# A merged table is <NAME>_XIAOWEN_merged.tsv (Xiaowen pipeline) or
# <NAME>_merged.tsv (older pipeline); if both exist for one NAME, Xiaowen wins.
# ---------------------------------------------------------------------------
DEFAULT_ADDITIONAL_TAGS = ("KWK", "LLG", "YPS")
_MERGED_SUFFIX = "_merged.tsv"
_XIAOWEN_INFIX = "_XIAOWEN"


def _name_from_filename(fname: str) -> str | None:
    """``KWK_37_Z6_E20_XIAOWEN_merged.tsv`` -> ``KWK_37_Z6_E20`` (or None)."""
    if not fname.endswith(_MERGED_SUFFIX):
        return None
    core = fname[: -len(_MERGED_SUFFIX)]
    if core.endswith(_XIAOWEN_INFIX):
        core = core[: -len(_XIAOWEN_INFIX)]
    return core


def discover_additional(data_dir: str, tags=DEFAULT_ADDITIONAL_TAGS) -> list[dict]:
    """Scan ``data_dir`` for resolvable tables of the wanted peptide tags.

    Returns entry dicts (name/data/seq/charge/residual), deduplicated by NAME
    with the Xiaowen table preferred over the plain one.
    """
    wanted = {t.upper() for t in tags}
    picked: dict[str, tuple[dict, bool]] = {}
    skipped: list[str] = []

    for fname in sorted(os.listdir(data_dir)):
        name = _name_from_filename(fname)
        if name is None:
            continue
        try:
            info = resolve_name(name)
        except (ValueError, KeyError):
            skipped.append(fname)
            continue
        tag = info["PEPTIDE_ID"].split("_")[0]
        if tag not in wanted:
            continue

        is_xiaowen = _XIAOWEN_INFIX in fname
        prev = picked.get(name)
        if prev is not None and (prev[1] or not is_xiaowen):
            continue
        entry = dict(
            name=name, data=fname,
            seq=info["SEQUENCE"], charge=info["PARENT_CHARGE"],
            residual=info["RESIDUAL"],
        )
        picked[name] = (entry, is_xiaowen)

    entries = [e for e, _ in picked.values()]
    entries.sort(key=lambda e: e["name"])
    if skipped:
        print(f"[info] skipped {len(skipped)} non-conforming file(s): "
              + ", ".join(skipped), file=sys.stderr)
    return entries


# ---------------------------------------------------------------------------
# Shared analysis settings.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    top_ffc_complementary=15000,
    top_ffc_annotation=2000,          # single cutoff for the FFC-annotation stage
    top_list=[200, 500, 2000],        # one line-based run per entry here
    line_delta=20.0,
    line_delta_unit="ppm",             # "da" (fixed Da gap) or "ppm" (relative gap)
    line_min_ffc=3,                   # NumFFC for line detection
    align_threshold=1,                # min inactive FFCs for an internal
                                       # line to count as "aligned" to the
                                       # parental ladder
    tol_ppm=30.0,                     # shared by FFC annotator + line-based
    mock_tol_ppm=100.0,
)


class _Tee:
    """Write to several streams at once (console + per-peptide log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def _resolve_data(data: str, data_dir: str) -> str:
    """Absolute path as given; a bare filename resolves against ``data_dir``."""
    return data if os.path.isabs(data) else os.path.join(data_dir, data)


def run_one(entry: dict, data_dir: str, out_root: str, overrides: dict,
            write_log: bool = True, prefix: str = "") -> dict:
    """Run complementary + FFC-annotation (once) + line-based (once per
    top_list entry) for one peptide. Returns a result record (never raises).

    ``prefix`` is prepended to every output filename -- ``"Main_"`` for
    MAIN_PEPTIDES, ``""`` for discovered additional entries.
    """
    name = entry["name"]
    data_path = _resolve_data(entry["data"], data_dir)

    # Shared defaults <- CLI overrides <- per-row extras (most specific wins).
    row_extras = {k: v for k, v in entry.items()
                  if k not in ("name", "data", "seq", "charge", "residual")}
    kwargs = {**DEFAULTS, **overrides, **row_extras}
    top_list = kwargs["top_list"]

    record = {"name": name, "data": data_path, "status": "ok", "seconds": 0.0,
              "error": None}

    if not os.path.isfile(data_path):
        record.update(status="missing input", error=data_path)
        return record

    ffc_dir = os.path.join(out_root, FFC_SUBDIR)
    log_dir = os.path.join(out_root, LOG_SUBDIR)
    os.makedirs(ffc_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    line_dirs = {}
    for n in top_list:
        d = os.path.join(out_root, LINE_SUBDIR, f"top{n}")
        os.makedirs(d, exist_ok=True)
        line_dirs[n] = d

    log_path = os.path.join(log_dir, f"{prefix}{name}_log.txt") if write_log else None

    started = time.time()
    try:
        with contextlib.ExitStack() as stack:
            if log_path:
                fh = stack.enter_context(open(log_path, "w"))
                stack.enter_context(
                    contextlib.redirect_stdout(_Tee(sys.stdout, fh)))

            residual = entry["residual"].upper()
            end_h20 = True if residual == "H2O" else residual
            pep = peptide.Pep(
                f"[{entry['seq']}+{entry['charge']}H]{entry['charge']}+",
                end_h20=end_h20)
            parent_mass = pep.pep_mass
            print(f"Peptide: {entry['seq']}  charge={entry['charge']}  "
                  f"residual={residual}")
            print(f"  length={len(pep.AA_array)}  parent_mass={parent_mass:.4f}")

            print(f"\nLoading {data_path}")
            ffc_df = pd.read_csv(data_path, sep="\t")
            print(f"  raw rows: {len(ffc_df)}")
            merged_df = merge_duplicate_ffcs_charge(
                ffc_df, symmetric=True, reset_ranking=True, tolerance=0.002)
            print(f"  rows after merge_duplicate_ffcs_charge: {len(merged_df)}")

            # ── Stage 1: complementary test (console only) ─────────────────
            run_complementary(merged_df, pep, entry["charge"], parent_mass,
                              kwargs["top_ffc_complementary"])

            # ── Stage 2: FFC annotation (runs ONCE, at top_ffc_annotation) ──
            ffc_path = os.path.join(ffc_dir, f"{prefix}{name}_ffc_annotation.xlsx")
            run_ffc_annotation(merged_df, entry["seq"], entry["charge"], residual,
                               kwargs["top_ffc_annotation"], ffc_path,
                               tol_ppm=kwargs["tol_ppm"])

            # ── Stage 3: line-based annotation, once per top_list entry ─────
            for n in top_list:
                line_path = os.path.join(
                    line_dirs[n], f"{prefix}{name}_line_based_top{n}.xlsx")
                sheet = os.path.splitext(os.path.basename(line_path))[0][:31]
                run_line_based(merged_df, pep, entry["charge"], parent_mass, n,
                               line_path, sheet, kwargs["line_delta"],
                               delta_unit=kwargs["line_delta_unit"],
                               min_ffc=kwargs["line_min_ffc"],
                               tol_ppm=kwargs["tol_ppm"],
                               mock_tol_ppm=kwargs["mock_tol_ppm"],
                               align_threshold=kwargs["align_threshold"])
    except Exception as exc:                      # keep the batch going
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    record["seconds"] = time.time() - started
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the FFC annotation pipeline over main + additional peptides.")
    ap.add_argument("--list", action="store_true",
                    help="print what would run and exit")
    ap.add_argument("--main-only", action="store_true",
                    help="run only MAIN_PEPTIDES")
    ap.add_argument("--additional-only", action="store_true",
                    help="run only the discovered additional peptides")
    ap.add_argument("--additional-tags", nargs="+",
                    default=list(DEFAULT_ADDITIONAL_TAGS), metavar="TAG",
                    help="peptide tags to discover (default: KWK LLG YPS)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="run only these entries by name (main or additional)")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help=f"folder holding the .tsv tables (default: {DATA_DIR})")
    ap.add_argument("--output-dir", default=None,
                    help="output root (default: ./result/annotation)")
    ap.add_argument("--no-log", action="store_true",
                    help="do not write per-entry {name}_log.txt files")
    # Optional overrides of the shared settings.
    ap.add_argument("--top-ffc-complementary", type=int)
    ap.add_argument("--top-ffc-annotation", type=int)
    ap.add_argument("--top-list", type=int, nargs="+",
                    help="top-N FFC cutoffs for the line-based stage, one run "
                         "each (default: 200 500 2000)")
    ap.add_argument("--line-delta", type=float,
                    help="line-detection cluster gap, in the unit set by "
                         "--line-delta-unit (default 0.02)")
    ap.add_argument("--line-delta-unit", choices=["da", "ppm"],
                    help="unit for --line-delta: 'da' (fixed Da gap, default) "
                         "or 'ppm' (relative gap)")
    ap.add_argument("--line-min-ffc", type=int, help="NumFFC for line detection")
    ap.add_argument("--align-threshold", type=int,
                    help="min inactive FFCs for an internal line to count as "
                         "'aligned' to the parental ladder (default 1)")
    ap.add_argument("--tol-ppm", type=float)
    ap.add_argument("--mock-tol-ppm", type=float)
    args = ap.parse_args(argv)

    if args.main_only and args.additional_only:
        ap.error("--main-only and --additional-only are mutually exclusive")

    main_entries = [] if args.additional_only else list(MAIN_PEPTIDES)
    additional_entries = (
        [] if args.main_only
        else discover_additional(args.data_dir, args.additional_tags)
    )
    entries = [(e, "Main_") for e in main_entries] + \
              [(e, "") for e in additional_entries]

    if args.only:
        wanted = set(args.only)
        entries = [(e, p) for e, p in entries if e["name"] in wanted]
        unknown = wanted - {e["name"] for e, _ in entries}
        if unknown:
            ap.error(f"unknown entry name(s): {', '.join(sorted(unknown))}")

    out_root = args.output_dir or os.path.join(_HERE, "result", "annotation")

    if args.list:
        print(f"{'name':32} {'z':>2} {'res':4} {'src':4}  data")
        for e, p in entries:
            path = _resolve_data(e["data"], args.data_dir)
            mark = "" if os.path.isfile(path) else "   [MISSING]"
            src = "main" if p else "add"
            print(f"{e['name']:32} {e['charge']:>2} {e['residual']:4} "
                  f"{src:4}  {e['data']}{mark}")
        print(f"\n{len(entries)} entrie(s).")
        return 0

    overrides = {k: v for k, v in {
        "top_ffc_complementary": args.top_ffc_complementary,
        "top_ffc_annotation": args.top_ffc_annotation,
        "top_list": args.top_list,
        "line_delta": args.line_delta,
        "line_delta_unit": args.line_delta_unit,
        "line_min_ffc": args.line_min_ffc,
        "align_threshold": args.align_threshold,
        "tol_ppm": args.tol_ppm,
        "mock_tol_ppm": args.mock_tol_ppm,
    }.items() if v is not None}

    results = []
    for k, (entry, prefix) in enumerate(entries, 1):
        print("\n" + "#" * 70)
        print(f"# [{k}/{len(entries)}] {prefix}{entry['name']}  "
              f"({entry['seq'][:12]}... z={entry['charge']} {entry['residual']})")
        print("#" * 70)
        results.append(run_one(entry, args.data_dir, out_root, overrides,
                               write_log=not args.no_log, prefix=prefix))

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for r in results:
        line = f"  {r['name']:32} {r['status']:14} {r['seconds']:7.1f}s"
        if r["error"]:
            line += f"   {r['error']}"
        print(line)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} succeeded.  Outputs under {out_root}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
