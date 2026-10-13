#!/usr/bin/env python3
"""
ETD annotation runner -- c/z-dot line-based annotation + de novo reconstruction
==============================================================================

The ETD counterpart of ``annotation_pepline/run_annotation_reconstruct.py``.
Same three stages, same output shape, same conventions (§6 of
``annotation_pipeline.md``); the difference is which ion series the two
annotators generate:

    --series cz   c / z-dot   (ETD; ``ffc_annotator_cz`` + ``line_internal_annotation_cz``)
    --series by   b / y       (the STOCK, unmodified modules -- for EThcD, where
                               supplemental HCD activation puts a b/y ladder in
                               the same map, and as a control on the ETD run)

Both can be run on the same table; each writes into its own subtree, so the two
ladders can be read side by side.

⚠ The charge the line model is told
-----------------------------------
An ETD c/z-dot pair carries ``parent_charge - 1`` charges (electron transfer
consumes a proton), and its reconstructed line mass is one electron above where
a b/y ladder would sit.  So the ``cz`` series runs the line model at ``z - 1``
and the ``by`` series at ``z`` -- the same distinction
``run_etd_deconvolution.py`` makes, and for the same reason.  Getting it wrong
does not error; it silently finds the ladder only via greedy peeling, split
across its isotope satellites.

Usage
-----
    python run_etd_annotation.py                          # both datasets, cz
    python run_etd_annotation.py --series cz by           # both ion series
    python run_etd_annotation.py --only LLG_37_Z6_ETD --top-list 500
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE,
           os.path.join(_ROOT, "deiso", "python_code"),
           os.path.join(_ROOT, "data_prep"),
           os.path.join(_ROOT, "line_based_annotation"),
           os.path.join(_ROOT, "theoretical_mms_annotation"),
           os.path.join(_ROOT, "annotation_pepline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cz_series  # noqa: E402
from peptide import Pep  # noqa: E402
from peptide_info import resolve as resolve_name  # noqa: E402
from merge_ffcs import merge_duplicate_ffcs_charge  # noqa: E402
from line_finding import prepare_ffc_data  # noqa: E402

# c/z-dot (this directory) and b/y (stock, untouched) annotators.
import ffc_annotator_cz  # noqa: E402
import line_internal_annotation_cz  # noqa: E402
import ffc_annotator  # noqa: E402
import line_internal_annotation_oa_v2  # noqa: E402


DATA_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "ETD_deconv_cz/merge")
OUTPUT_DIR = os.path.join(_HERE, "result")

#: Matches ``run_annot_rec.sh``, the author's working configuration for the
#: b/y pipeline, so ETD numbers are comparable with the published ones.
DEFAULTS = dict(
    top_ffc_annotation=2000,
    top_list=(500, 2000),
    tol_ppm=30.0,
    line_delta=20.0,
    line_delta_unit="ppm",
    min_ffc=3,
    iso_range=1,
    mock_tol_ppm=100.0,
    align_threshold=1,
    reconstruct=True,
    considering_spurious=False,
    productivity=True,
    productivity_tol_ppm=30.0,
    productivity_coord_tol_ppm=30.0,
    extend_master_set=True,
    master_align_threshold=1,
    converge=True,
    max_convergence_rounds=20,
    loose_mock=True,
    accept_ambiguity=False,
    # [CZ] Isotope satellites of the parental line join the master TFFC-set.
    # On this data the deconvolution leaves them behind: ETD top500 has one
    # (+1) and top2000 has +1, -1 and +2, each carrying 12-16 annotated cells.
    # Only meaningful for --series cz; the stock b/y module does not take it.
    include_isotope_lines=True,
    isotope_line_max_k=2,
    isotope_line_tol=0.05,
    # [CZ] 2D map of the master-set reconstruction, one .png beside the .xlsx.
    # Drawn only when a correct reconstruction exists.  c/z only -- the stock
    # b/y module has the same flag but is called without it here.
    graph=True,
    isomax_cutoff=5.0,
)

#: series -> (ffc-annotator module, line-based module, charge offset, label)
SERIES = {
    "cz": (ffc_annotator_cz, line_internal_annotation_cz, 1, "c/z-dot"),
    "by": (ffc_annotator, line_internal_annotation_oa_v2, 0, "b/y"),
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def discover(data_dir):
    """``<NAME>_merged.tsv`` files whose NAME ``peptide_info`` can resolve."""
    entries = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith("_merged.tsv"):
            continue
        name = fname[: -len("_merged.tsv")]
        try:
            info = resolve_name(name)
        except (ValueError, KeyError):
            continue
        entries.append({**info, "path": os.path.join(data_dir, fname)})
    return entries


def line_model_charge_mass(entry, offset):
    """(charge, mass) the LINE model should use for this ion series.

    ``offset = 1`` (c/z-dot): the pair carries one charge fewer than the
    precursor, and reconstructs to ``M + H + (z-1)*proton`` -- one electron
    above the stored charged mass.  ``offset = 0`` (b/y): unchanged.
    """
    z = entry["PARENT_CHARGE"] - offset
    mass = entry["PARENT_MASS"] + offset * (cz_series.H_ATOM - cz_series.PROTON)
    return z, mass


def run_one(entry, series, out_root, cfg, write_log=True):
    """All stages for one dataset x one ion series.  Never raises."""
    ffc_mod, line_mod, offset, label = SERIES[series]
    name = entry["NAME"]
    rec = {"name": name, "series": series, "status": "ok", "error": None,
           "seconds": 0.0, "results": {}}

    root = os.path.join(out_root, series)
    for sub in ("ffc_annotation", "log"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    log_path = os.path.join(root, "log", f"{name}_log.txt") if write_log else None

    started = time.time()
    try:
        with contextlib.ExitStack() as stack:
            if log_path:
                fh = stack.enter_context(open(log_path, "w"))
                stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, fh)))

            z_line, m_line = line_model_charge_mass(entry, offset)
            print("=" * 70)
            print(f"{name}   [{label} series]")
            print("=" * 70)
            print(f"  sequence           {entry['SEQUENCE']}")
            print(f"  precursor          {entry['PARENT_CHARGE']}+   "
                  f"charged mass {entry['PARENT_MASS']:.4f}   "
                  f"C-term {entry['RESIDUAL']}")
            print(f"  line model         charge {z_line}   "
                  f"mass {m_line:.4f}")
            if offset:
                print(f"                     (ETD: a c/z-dot pair carries "
                      f"z-1 charges and sums to M + H)")
            print()

            # ── Load + merge, once, shared by every stage ────────────────
            merged_df = pd.read_csv(entry["path"], sep="\t")
            print(f"Loaded {len(merged_df)} deconvolved FFC rows")
            merged_df = merge_duplicate_ffcs_charge(
                merged_df, symmetric=True, reset_ranking=True)
            print(f"After symmetric duplicate merge: {len(merged_df)}")

            pep = Pep(f"[{entry['SEQUENCE']}+{entry['PARENT_CHARGE']}H]"
                      f"{entry['PARENT_CHARGE']}+",
                      end_h20=True if entry["RESIDUAL"] == "H2O" else "NH3")

            # ── Stage 1: per-FFC theoretical annotation ──────────────────
            ffc_out = os.path.join(root, "ffc_annotation",
                                   f"{name}_ffc_annotation.xlsx")
            print("\n" + "=" * 70)
            print(f"[1/2] Theoretical-MMS FFC annotation ({label})   "
                  f"top_n={cfg['top_ffc_annotation']}")
            print("=" * 70)
            df_a = prepare_ffc_data(merged_df, top_n=cfg["top_ffc_annotation"])
            print(f"FFC rows fed to the FFC annotator: {len(df_a)}")
            ffc_res = ffc_mod.annotate_ffcs_with_charge(
                df_a,
                sequence=entry["SEQUENCE"],
                precursor_charge=entry["PARENT_CHARGE"],
                output_path=ffc_out,
                tol_ppm=cfg["tol_ppm"],
                residual=entry["RESIDUAL"],
                isomax_cutoff=cfg["isomax_cutoff"],
                extra_info=True,
            )
            if len(ffc_res):
                counts = ffc_res["annotation"].value_counts()
                print("\nAnnotation categories:")
                for k, v in counts.head(12).items():
                    print(f"   {str(k):24} {v}")
            rec["results"]["ffc_annotation"] = ffc_out

            # ── Stage 2: line-based + reconstruction, once per top-N ─────
            for top_n in cfg["top_list"]:
                sub = os.path.join(root, "line_based", f"top{top_n}")
                os.makedirs(sub, exist_ok=True)
                out = os.path.join(sub, f"{name}_line_based_top{top_n}.xlsx")
                print("\n" + "=" * 70)
                print(f"[2/2] Line-based annotation ({label})   top_n={top_n}")
                print("=" * 70)
                df_l = prepare_ffc_data(merged_df, top_n=top_n)
                print(f"FFC rows fed to line-based annotation: {len(df_l)}")

                cov, all_lines = line_mod.run_internal_annotation(
                    df_l, pep,
                    parent_charge=z_line,
                    parent_mass=m_line,
                    delta=cfg["line_delta"],
                    delta_unit=cfg["line_delta_unit"],
                    min_ffc_number=cfg["min_ffc"],
                    iso_range=cfg["iso_range"],
                    tol_ppm=cfg["tol_ppm"],
                    mock_tol_ppm=cfg["mock_tol_ppm"],
                    align_threshold=cfg["align_threshold"],
                    reconstruct=cfg["reconstruct"],
                    considering_spurious=cfg["considering_spurious"],
                    productivity=cfg["productivity"],
                    productivity_tol_ppm=cfg["productivity_tol_ppm"],
                    productivity_coord_tol_ppm=cfg["productivity_coord_tol_ppm"],
                    extend_master_set=cfg["extend_master_set"],
                    master_align_threshold=cfg["master_align_threshold"],
                    converge=cfg["converge"],
                    max_convergence_rounds=cfg["max_convergence_rounds"],
                    loose_mock=cfg["loose_mock"],
                    accept_ambiguity=cfg["accept_ambiguity"],
                    # [CZ] c/z module only -- the stock b/y module has no such
                    # parameter, so it is passed only for --series cz.
                    # `graph`/`graph_path` exist in BOTH modules; `graph_title`
                    # is c/z-only -- the stock b/y module has no such parameter,
                    # so passing it unconditionally would TypeError.
                    graph=cfg["graph"],
                    graph_path=os.path.join(
                        sub, f"{name}_line_based_top{top_n}_graph.png"),
                    **({"include_isotope_lines": cfg["include_isotope_lines"],
                        "isotope_line_max_k": cfg["isotope_line_max_k"],
                        "isotope_line_tol": cfg["isotope_line_tol"],
                        "graph_title": (
                            f"{entry['SEQUENCE']}  {entry['PARENT_CHARGE']}+   "
                            f"{name}  top{top_n}   "
                            f"{label} master TFFC-set reconstruction")}
                       if series == "cz" else {}),
                )
                print(f"Detected lines: {len(all_lines)}")
                data_rows = cov.iloc[: len(pep.AA_array) - 1]
                with pd.option_context("display.max_colwidth", None,
                                       "display.width", 260):
                    print(data_rows.to_string())

                recon = cov.attrs.get("reconstructions")
                if recon is not None and not recon.empty:
                    n_ok = int((recon["correct"] != "").sum())
                    print(f"\nMaster-set reconstructions: {len(recon)}  "
                          f"({n_ok} correct)   incorrect TFFCs: "
                          f"{len(cov.attrs.get('incorrect_tffcs', []))}")
                    for _, r in recon.head(5).iterrows():
                        print(f"  {r['#']:>3}. {r['reconstruction']}"
                              f"{'   <-- CORRECT' if r['correct'] else ''}")

                sheet = f"top{top_n}"
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    cov.to_excel(writer, sheet_name=sheet,
                                 index_label=f"#FFC={top_n}")
                    ws = writer.sheets[sheet]
                    for col_cells in ws.columns:
                        ws.column_dimensions[
                            col_cells[0].column_letter].width = 32
                    if recon is not None:
                        recon.to_excel(writer, sheet_name="reconstructions",
                                       index=False)
                        ws2 = writer.sheets["reconstructions"]
                        ws2.column_dimensions["A"].width = 6
                        ws2.column_dimensions["B"].width = 140
                        ws2.column_dimensions["C"].width = 12
                print(f"Saved {out}")

                rec["results"][f"top{top_n}"] = _summarise(cov, len(pep.AA_array))
    except Exception as exc:
        rec.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    rec["seconds"] = time.time() - started
    return rec


def _summarise(cov, n_res):
    """Pull the headline numbers out of a coverage table for the batch report."""
    out = {}
    for col in cov.columns:
        if isinstance(col, str) and col.startswith("Breakpoint Coverage"):
            out["breakpoint_coverage"] = col.split("=", 1)[-1]
        if isinstance(col, str) and col.startswith("Master TFFC-set"):
            out["master_coverage"] = col.split("=", 1)[-1].rstrip(")")
            master_col = col
    summary = cov.iloc[n_res - 1:]
    for row_name in ("#reconstructions/correct reconstructions",
                     "#reconstructions/correct?", "#incorrect TFFCs"):
        if row_name in summary.index:
            vals = [v for v in summary.loc[row_name] if isinstance(v, str) and v]
            if vals:
                out[row_name] = vals[0]
    try:
        out["master_recon"] = summary.loc["#reconstructions/correct?", master_col]
        out["incorrect_tffcs"] = summary.loc["#incorrect TFFCs", master_col]
    except Exception:
        pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--series", nargs="+", default=["cz"], choices=["cz", "by"])
    ap.add_argument("--top-list", nargs="+", type=int)
    ap.add_argument("--tol-ppm", type=float)
    ap.add_argument("--line-delta", type=float)
    ap.add_argument("--line-delta-unit", choices=["da", "ppm"])
    ap.add_argument("--min-ffc", type=int)
    ap.add_argument("--include-isotope-lines", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="[cz only] let isotope satellites of the parental "
                         "line join the master TFFC-set (default on)")
    ap.add_argument("--isotope-line-max-k", type=int)
    ap.add_argument("--isotope-line-tol", type=float)
    ap.add_argument("--graph", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="draw the 2D reconstruction map (default on)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    for key in ("top_list", "tol_ppm", "line_delta", "line_delta_unit",
                "min_ffc", "include_isotope_lines", "isotope_line_max_k",
                "isotope_line_tol", "graph"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    entries = discover(args.data_dir)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["NAME"] in wanted]
    if args.list:
        for e in entries:
            print(f"{e['NAME']:24} z={e['PARENT_CHARGE']} {e['SEQUENCE']}")
        return 0
    if not entries:
        print(f"[error] nothing to run in {args.data_dir}", file=sys.stderr)
        return 1

    results = []
    for entry in entries:
        for series in args.series:
            print("\n" + "#" * 70)
            print(f"# {entry['NAME']}  [{series}]")
            print("#" * 70)
            results.append(run_one(entry, series, args.output_dir, cfg,
                                   write_log=not args.no_log))

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    for r in results:
        head = f"  {r['name']:22} {r['series']:3} {r['status']:8} {r['seconds']:7.1f}s"
        print(head + (f"   {r['error']}" if r["error"] else ""))
        for key, val in r["results"].items():
            if isinstance(val, dict):
                bits = "  ".join(f"{k}={v}" for k, v in val.items())
                print(f"      {key:10} {bits}")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
