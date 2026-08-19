#!/usr/bin/env python3
"""
Combined batch runner (v2): main + additional peptides, with de novo reconstruction
====================================================================================
Same two-peptide-set batch as ``run_annotation.py``, but the line-based stage
uses ``line_based_annotation/line_internal_annotation_oa_v2.py`` instead of the
v1 ``line_internal_annotation_oa.py``. v2 adds:

  * a ``Master TFFC-set`` coverage column and ``#incorrect master TFFCs``
    summary row (always on), and
  * an opt-in ``#reconstructions/correct reconstructions`` summary cell
    (``reconstruct=True``, the default here) -- the breakpoints carried by the
    parental (master-ladder) lines are handed to the de novo longest-path
    sequencer in ``../reconstruct/path_finding.py``, reporting how many tied
    reconstructions it found and how many are actually correct, and
  * an opt-in ``#unproductive/unambiguous/ambiguous`` summary row
    (``productivity=True``, off by default here since it costs several extra
    seconds per peptide) -- splits each line's *active* TFFCs by what they
    would contribute to a de novo reconstruction, judged against the frozen
    parental ladder (order- and line-independent; see line_based.md §8b), and
  * an opt-in extension of the ``Master TFFC-set`` column itself
    (``extend_master_set=True``, off by default -- needs edges.json + the de
    novo machinery, like ``productivity``) -- lets aligned internal/
    neutral-loss lines contribute to that column too, on top of the parental
    lines (their productive TFFCs plus each internal line's mock).
    ``master_align_threshold`` (default 1) is the inactive-FFC count such a
    line needs before it may join.
  * ``converge`` (default True, only matters when ``productivity`` or
    ``extend_master_set`` is on) -- builds the master TFFC-set *iteratively*
    instead of in one pass: bank each round's unambiguous TFFCs, re-derive
    aligned/active/unambiguous against the grown set, repeat until it stops
    growing (capped by ``max_convergence_rounds``, default 20). Adds a
    ``#unproductive/unambiguous/ambiguous (converged)`` row beneath the
    round-1 one, and makes ``extend_master_set`` use the converged set.
  * ``loose_mock`` (off by default) -- a charge-free mock probe alongside the
    usual charge sweep, so a loss line whose per-side charges already sum to
    the precursor charge (it lost a *neutral* run of residues, not a charged
    fragment) still gets a mock instead of none at all. Tagged
    ``[mock-loose]``; a charged reading wins any tie.
  * ``accept_ambiguity`` (off by default) -- let *ambiguous* TFFCs into the
    master TFFC-set. An ambiguous TFFC's two images are "one of these two,
    never this one"; with True BOTH are banked (one is false by
    construction), with False neither is. Off by default: measured across 7
    peptides, ambiguous TFFCs bought one usable bond but supplied 4 wrong
    coordinates, costing one peptide its correct reconstruction.
  * ``graph`` (off by default) -- when the master-set reconstruction (needs
    ``reconstruct`` AND ``extend_master_set`` both on) finds a *correct* path,
    also render the 2D reconstruction map (correct path as hero, other tied
    paths as candidates, incorrect master TFFCs marked spurious) to
    ``graph/top{N}/{name}_line_based_top{N}_graph.png``. Purely additive --
    writes one extra image file per (peptide, top-N); the coverage table is
    byte-for-byte the same with or without it.

Because the coverage-table shape differs from the v1 module's, this script
writes to its own output root (``result/annotation_reconstruct/``) rather than
sharing ``run_annotation.py``'s ``result/annotation/`` tree.

Peptide sets
------------
  1. MAIN_PEPTIDES -- the hand-picked reference set (VEA3+, YLE3+, ...).
     Output filenames are prefixed "Main_".
  2. ADDITIONAL peptides -- every KWK/LLG/YPS deconvolution table discovered
     under DATA_DIR that data_prep/peptide_info.py can resolve. Output
     filenames are unprefixed.

Main and additional share the exact same output folders -- the "Main_"
prefix is the only thing that tells them apart.

Output tree
-----------
    result/annotation_reconstruct/
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
      graph/                                    (only when --graph is on)
        top200/
          Main_VEA3+_line_based_top200_graph.png
          ...
      log/
        Main_VEA3+_log.txt
        KWK_37_Z5_E28_1_log.txt
        ...

As in run_annotation.py, each peptide's table is loaded and merged ONCE, and
the complementary test + FFC-annotation stage each run ONCE (at
top_ffc_annotation, default 2000); only the line-based (+reconstruction)
stage repeats, once per entry in top_list.

Usage
-----
    python run_annotation_reconstruct.py                  # everything (main + additional)
    python run_annotation_reconstruct.py --list           # show what would run, run nothing
    python run_annotation_reconstruct.py --main-only
    python run_annotation_reconstruct.py --additional-only
    python run_annotation_reconstruct.py --only VEA3+ KWK_37_Z6_E20
    python run_annotation_reconstruct.py --top-list 100 300
    python run_annotation_reconstruct.py --no-reconstruct           # v2 table, no de novo search
    python run_annotation_reconstruct.py --considering-spurious
    python run_annotation_reconstruct.py --productivity             # + #unproductive/unambiguous/ambiguous
    python run_annotation_reconstruct.py --productivity-tol-ppm 30 --productivity-coord-tol-ppm 30
    python run_annotation_reconstruct.py --extend-master-set        # aligned internal/loss lines join Master TFFC-set
    python run_annotation_reconstruct.py --no-converge              # single-pass master set, no convergence rounds
    python run_annotation_reconstruct.py --loose-mock               # charge-free mock probe for neutral-loss lines
    python run_annotation_reconstruct.py --accept-ambiguity         # bank both images of ambiguous TFFCs
    python run_annotation_reconstruct.py --reconstruct --extend-master-set --graph   # draw the 2D map
    python run_annotation_reconstruct.py --tol-ppm 10 --line-delta 5 --line-delta-unit ppm
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
    run_complementary, run_ffc_annotation, run_line_based_reconstruct,
    merge_duplicate_ffcs_charge, peptide,
)
from peptide_info import resolve as resolve_name  # noqa: E402


# ---------------------------------------------------------------------------
# Where the .tsv tables live.
# ---------------------------------------------------------------------------
DATA_DIR = "/home/sux002/data/deconvoluted/merged"

# ---------------------------------------------------------------------------
# Output tree: result/annotation_reconstruct/{ffc_annotation, line_based/top{N},
# graph/top{N}, log}
# ---------------------------------------------------------------------------
FFC_SUBDIR = "ffc_annotation"
LINE_SUBDIR = "line_based"
GRAPH_SUBDIR = "graph"
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
    # GLP-2 Z4 NCE15, 270 ions, from the 2026-08-03 "more fragments" re-delivery
    # (data/HAD_more_fragement -> data_prep/combine_had_more_fragments.py).  It
    # replaced HAD_merged.tsv on 2026-08-03: strictly additive over it at every
    # top-N -- same bonds plus b8, b16 and b32 -- for 81.2% vs 71.9% breakpoint
    # coverage at top500, and a 0.0% spurious-parental rate.  b16 is the bond
    # deconvolution.md section 12 had recorded as this pipeline's one regression.
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
    line_delta_unit="ppm",            # "da" (fixed Da gap) or "ppm" (relative gap)
    line_min_ffc=3,                   # NumFFC for line detection
    align_threshold=1,                # min inactive FFCs for an internal
                                       # line to count as "aligned" to the
                                       # parental ladder
    tol_ppm=30.0,                     # shared by FFC annotator + line-based
    mock_tol_ppm=100.0,
    reconstruct=True,                 # emit "#reconstructions/correct reconstructions"
    considering_spurious=False,       # also feed parental-line spurious FFCs
    edges_path=None,                  # defaults to reconstruct/edges.json
    reconstruct_threshold=0.001,      # Da tolerance for the reconstruction graph
    reconstruct_max_paths=1000,       # cap on tied longest paths collected
    productivity=False,               # emit "#unproductive/unambiguous/ambiguous"
                                       # -- off by default (several extra sec/peptide)
    productivity_tol_ppm=30.0,        # ppm tolerance for the productivity DAG's edges
    productivity_coord_tol_ppm=30.0,  # ppm tolerance vs. the parental (master) set
    extend_master_set=False,          # let aligned internal/loss lines join
                                       # the Master TFFC-set column too
                                       # -- off by default (needs edges.json +
                                       # the de novo machinery, like productivity)
    master_align_threshold=1,         # inactive FFCs such a line needs to qualify
    converge=True,                    # build the master TFFC-set iteratively
                                       # (only matters when productivity or
                                       # extend_master_set is on)
    max_convergence_rounds=20,        # safety cap on convergence rounds
    loose_mock=False,                 # charge-free mock probe alongside the
                                       # charge sweep, for neutral-loss lines
                                       # the sweep can't reach
    accept_ambiguity=False,           # let ambiguous TFFCs into the master
                                       # TFFC-set -- off by default (net
                                       # negative across 7 test peptides)
    graph=False,                      # render the 2D reconstruction map when
                                       # a correct master-set reconstruction is
                                       # found -- only matters together with
                                       # reconstruct=True + extend_master_set=True;
                                       # table output is unaffected either way
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
    """Run complementary + FFC-annotation (once) + line-based-reconstruct
    (once per top_list entry) for one peptide. Returns a result record
    (never raises).

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
    # Graph output dirs -- only created when actually requested, so a run with
    # graph=False (the default) never leaves an empty graph/ folder behind.
    graph_dirs = {}
    if kwargs["graph"]:
        for n in top_list:
            d = os.path.join(out_root, GRAPH_SUBDIR, f"top{n}")
            os.makedirs(d, exist_ok=True)
            graph_dirs[n] = d

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

            # ── Stage 3: line-based annotation (+reconstruction), once per
            # top_list entry ─────────────────────────────────────────────────
            for n in top_list:
                line_path = os.path.join(
                    line_dirs[n], f"{prefix}{name}_line_based_top{n}.xlsx")
                sheet = os.path.splitext(os.path.basename(line_path))[0][:31]
                graph_path = None
                if kwargs["graph"]:
                    graph_path = os.path.join(
                        graph_dirs[n], f"{prefix}{name}_line_based_top{n}_graph.png")
                run_line_based_reconstruct(
                    merged_df, pep, entry["charge"], parent_mass, n,
                    line_path, sheet, kwargs["line_delta"],
                    delta_unit=kwargs["line_delta_unit"],
                    min_ffc=kwargs["line_min_ffc"],
                    tol_ppm=kwargs["tol_ppm"],
                    mock_tol_ppm=kwargs["mock_tol_ppm"],
                    align_threshold=kwargs["align_threshold"],
                    reconstruct=kwargs["reconstruct"],
                    considering_spurious=kwargs["considering_spurious"],
                    edges_path=kwargs["edges_path"],
                    productivity=kwargs["productivity"],
                    productivity_tol_ppm=kwargs["productivity_tol_ppm"],
                    productivity_coord_tol_ppm=kwargs["productivity_coord_tol_ppm"],
                    reconstruct_threshold=kwargs["reconstruct_threshold"],
                    reconstruct_max_paths=kwargs["reconstruct_max_paths"],
                    extend_master_set=kwargs["extend_master_set"],
                    master_align_threshold=kwargs["master_align_threshold"],
                    converge=kwargs["converge"],
                    max_convergence_rounds=kwargs["max_convergence_rounds"],
                    loose_mock=kwargs["loose_mock"],
                    accept_ambiguity=kwargs["accept_ambiguity"],
                    graph=kwargs["graph"],
                    graph_path=graph_path,
                )
    except Exception as exc:                      # keep the batch going
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    record["seconds"] = time.time() - started
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the FFC annotation pipeline (v2, with de novo "
                    "reconstruction) over main + additional peptides.")
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
                    help="output root (default: ./result/annotation_reconstruct)")
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
                         "--line-delta-unit (default 20, ppm)")
    ap.add_argument("--line-delta-unit", choices=["da", "ppm"],
                    help="unit for --line-delta: 'da' (fixed Da gap) or "
                         "'ppm' (relative gap, default)")
    ap.add_argument("--line-min-ffc", type=int, help="NumFFC for line detection")
    ap.add_argument("--align-threshold", type=int,
                    help="min inactive FFCs for an internal line to count as "
                         "'aligned' to the parental ladder (default 1)")
    ap.add_argument("--tol-ppm", type=float)
    ap.add_argument("--mock-tol-ppm", type=float)
    # Reconstruction-specific knobs (v2 only).
    ap.add_argument("--reconstruct", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="emit the #reconstructions/correct reconstructions "
                         "summary cell via the de novo longest-path search "
                         "(default: on; use --no-reconstruct to disable)")
    ap.add_argument("--considering-spurious",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="also feed parental-line unmatched/unexplained FFCs "
                         "to the de novo search (default: off)")
    ap.add_argument("--edges-path",
                    help="path to the precomputed compound-mass table "
                         "(default: reconstruct/edges.json)")
    ap.add_argument("--reconstruct-threshold", type=float,
                    help="Da tolerance for the reconstruction graph's edges "
                         "(default 0.001)")
    ap.add_argument("--reconstruct-max-paths", type=int,
                    help="cap on tied longest paths collected (default 1000)")
    # TFFC productivity split (v2 only, off by default -- several extra sec).
    ap.add_argument("--productivity", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="emit the #unproductive/unambiguous/ambiguous row, "
                         "splitting each line's active TFFCs by what they'd "
                         "contribute to a de novo reconstruction (default: off)")
    ap.add_argument("--productivity-tol-ppm", type=float,
                    help="ppm tolerance for the productivity DAG's edges "
                         "(default 20)")
    ap.add_argument("--productivity-coord-tol-ppm", type=float,
                    help="ppm tolerance for testing whether a TFFC's "
                         "coordinate is already in the parental (master) set "
                         "(default 20)")
    # Master TFFC-set extension (v2 only, off by default -- needs edges.json +
    # the de novo machinery, like --productivity).
    ap.add_argument("--extend-master-set", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="let aligned internal/neutral-loss lines contribute "
                         "to the Master TFFC-set column too, on top of the "
                         "parental lines (default: off)")
    ap.add_argument("--master-align-threshold", type=int,
                    help="inactive FFCs an internal/loss line needs before it "
                         "may join the master TFFC-set (default 1)")
    # Iterative master-set convergence (only matters when --productivity or
    # --extend-master-set is on).
    ap.add_argument("--converge", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="build the master TFFC-set iteratively -- bank "
                         "unambiguous TFFCs, re-derive against the grown set, "
                         "repeat until it stops growing (default: on)")
    ap.add_argument("--max-convergence-rounds", type=int,
                    help="safety cap on convergence rounds (default 20)")
    # Charge-free mock probe for neutral-loss lines the charge sweep can't reach.
    ap.add_argument("--loose-mock", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="add a charge-free mock probe alongside the charge "
                         "sweep, recovering a mock for loss lines whose "
                         "charges already sum to the precursor charge "
                         "(default: off)")
    ap.add_argument("--accept-ambiguity", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="let ambiguous TFFCs into the master TFFC-set -- "
                         "banks both images of each ambiguous TFFC instead "
                         "of neither (default: off; net negative across the "
                         "7 test peptides)")
    # 2D reconstruction-graph image (v2 only; needs reconstruct + extend_master_set).
    ap.add_argument("--graph", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="render the 2D reconstruction map "
                         "(graph/top{N}/{name}_..._graph.png) whenever the "
                         "master-set reconstruction finds a correct path; "
                         "only takes effect with --reconstruct and "
                         "--extend-master-set both on (default: off)")
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

    out_root = args.output_dir or os.path.join(_HERE, "result", "annotation_reconstruct")

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
        "reconstruct": args.reconstruct,
        "considering_spurious": args.considering_spurious,
        "edges_path": args.edges_path,
        "reconstruct_threshold": args.reconstruct_threshold,
        "reconstruct_max_paths": args.reconstruct_max_paths,
        "productivity": args.productivity,
        "productivity_tol_ppm": args.productivity_tol_ppm,
        "productivity_coord_tol_ppm": args.productivity_coord_tol_ppm,
        "extend_master_set": args.extend_master_set,
        "master_align_threshold": args.master_align_threshold,
        "converge": args.converge,
        "max_convergence_rounds": args.max_convergence_rounds,
        "loose_mock": args.loose_mock,
        "accept_ambiguity": args.accept_ambiguity,
        "graph": args.graph,
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
