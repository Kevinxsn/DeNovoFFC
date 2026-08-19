"""The annotation stage -- the known-peptide route.

When the sequence is known there is no reason to run de novo separately: the
line-based module's v2 path already builds the master TFFC-set and reconstructs
from it (`reconstruct` + `extend_master_set`), and additionally tells you which
theoretical ion every FFC is and which backbone bonds are covered.  The de novo
search is a stage inside it, not an alternative to it.

Three analyses share one loaded-and-merged table:

  1. complementary / spurious-parental test   console only
  2. theoretical-MMS FFC annotation           one .xlsx, run once
  3. line-based internal annotation           one .xlsx per top-N, and it is
     + de novo reconstruction                 this one that reconstructs

⚠ Stage 3's `reconstruct` + `extend_master_set` BOTH being True is what turns
on the master-set reconstruction -- there is no third flag for it.  It arrives
via `cov_table.attrs`, which is easy to miss when skimming for a parameter.
"""
from __future__ import annotations

import os

import pandas as pd

from . import paths
import peptide                                                  # noqa: E402
from annotation_pipeline import (run_complementary,             # noqa: E402
                                 run_ffc_annotation,
                                 run_line_based_reconstruct)

#: The author's own working configuration (annotation_pipeline.md 6,
#: run_annot_rec.sh) minus `--accept-ambiguity`, which is left OFF: measured
#: across 7 peptides it is net negative -- it buys a bond or two but injects
#: wrong coordinates that can cost a correct reconstruction.
DEFAULTS = dict(
    top_ffc_complementary=15000,
    top_ffc_annotation=2000,
    line_delta=20.0,
    line_delta_unit="ppm",
    line_min_ffc=3,
    line_iso_range=1,
    align_threshold=1,
    tol_ppm=30.0,
    mock_tol_ppm=100.0,
    reconstruct=True,
    considering_spurious=False,
    reconstruct_threshold=0.001,
    reconstruct_max_paths=1000,
    productivity=True,
    productivity_tol_ppm=30.0,
    productivity_coord_tol_ppm=30.0,
    extend_master_set=True,
    master_align_threshold=1,
    converge=True,
    max_convergence_rounds=20,
    loose_mock=True,
    accept_ambiguity=False,
    graph=False,
)

DEFAULT_TOP_LIST = [500, 2000]

FFC_SUBDIR, LINE_SUBDIR, GRAPH_SUBDIR = "ffc_annotation", "line_based", "graph"


def make_peptide(seq: str, charge: int, terminus: str):
    """`peptide.Pep` for one peptide.

    ⚠ `end_h20` is True for a free acid and the group NAME ("NH3") for an
    amide.  A C-terminal amide is 0.984 Da lighter and getting it wrong fails
    SILENTLY -- every y ion is off, paths are still found, none are right.
    """
    residual = terminus.upper()
    end_h20 = True if residual in ("H2O", "OH") else residual
    return peptide.Pep(f"[{seq}+{charge}H]{charge}+", end_h20=end_h20)


def run(merged_df: pd.DataFrame, *, seq: str, charge: int, terminus: str,
        out_root: str, name: str, top_list=None,
        params: dict | None = None) -> dict:
    """Run all three analyses for one known peptide.

    Returns a record with the output paths and the headline numbers scraped
    back out of the coverage table.
    """
    p = dict(DEFAULTS, **(params or {}))
    top_list = list(top_list or DEFAULT_TOP_LIST)
    residual = terminus.upper()

    pep = make_peptide(seq, charge, residual)
    parent_mass = pep.pep_mass

    ffc_dir = os.path.join(out_root, FFC_SUBDIR)
    os.makedirs(ffc_dir, exist_ok=True)

    record = {"name": name, "seq": seq, "charge": charge, "terminus": residual,
              "parent_mass": parent_mass, "length": len(pep.AA_array),
              "outputs": {}, "per_top": {}}

    # ── 1. complementary / spurious-parental test (console only) ──────────
    run_complementary(merged_df, pep, charge, parent_mass,
                      p["top_ffc_complementary"])

    # ── 2. theoretical-MMS FFC annotation (runs ONCE) ─────────────────────
    ffc_path = os.path.join(ffc_dir, f"{name}_ffc_annotation.xlsx")
    run_ffc_annotation(merged_df, seq, charge, residual,
                       p["top_ffc_annotation"], ffc_path, tol_ppm=p["tol_ppm"])
    record["outputs"]["ffc_annotation"] = ffc_path

    # ── 3. line-based annotation + reconstruction, once per top-N ─────────
    for n in top_list:
        line_dir = os.path.join(out_root, LINE_SUBDIR, f"top{n}")
        os.makedirs(line_dir, exist_ok=True)
        line_path = os.path.join(line_dir, f"{name}_line_based_top{n}.xlsx")

        graph_path = None
        if p["graph"]:
            graph_dir = os.path.join(out_root, GRAPH_SUBDIR, f"top{n}")
            os.makedirs(graph_dir, exist_ok=True)
            graph_path = os.path.join(
                graph_dir, f"{name}_line_based_top{n}_graph.png")

        # returns (coverage_table, all_lines)
        table, all_lines = run_line_based_reconstruct(
            merged_df, pep, charge, parent_mass, n, line_path,
            sheet_name=f"top{n}", delta=p["line_delta"],
            delta_unit=p["line_delta_unit"], min_ffc=p["line_min_ffc"],
            iso_range=p["line_iso_range"], tol_ppm=p["tol_ppm"],
            mock_tol_ppm=p["mock_tol_ppm"], align_threshold=p["align_threshold"],
            reconstruct=p["reconstruct"],
            considering_spurious=p["considering_spurious"],
            edges_path=paths.ensure_edges_json(),
            reconstruct_threshold=p["reconstruct_threshold"],
            reconstruct_max_paths=p["reconstruct_max_paths"],
            productivity=p["productivity"],
            productivity_tol_ppm=p["productivity_tol_ppm"],
            productivity_coord_tol_ppm=p["productivity_coord_tol_ppm"],
            extend_master_set=p["extend_master_set"],
            master_align_threshold=p["master_align_threshold"],
            converge=p["converge"],
            max_convergence_rounds=p["max_convergence_rounds"],
            loose_mock=p["loose_mock"], accept_ambiguity=p["accept_ambiguity"],
            graph=p["graph"], graph_path=graph_path)

        record["per_top"][n] = _summarise(table, all_lines)
        record["outputs"][f"line_based_top{n}"] = line_path
        if graph_path and os.path.isfile(graph_path):
            record["outputs"][f"graph_top{n}"] = graph_path

    return record


def _summarise(table, all_lines=None) -> dict:
    """Pull the headline numbers out of a coverage table.

    The interesting cells live in the `Master TFFC-set` column, whose header
    carries the coverage percentage: `#incorrect TFFCs` (master TFFCs matching
    no backbone bond -- read this FIRST when the reconstruction is wrong) and
    `#reconstructions/correct?`.
    """
    out = {"n_lines": len(all_lines) if all_lines is not None else None}
    if table is None:
        return out
    master_col = next((c for c in table.columns
                       if str(c).startswith("Master TFFC-set")), None)
    cov_col = next((c for c in table.columns
                    if str(c).startswith("Breakpoint Coverage")), None)
    if master_col:
        out["master_coverage"] = str(master_col).split("coverage=")[-1].rstrip(")")
        for row in ("#incorrect TFFCs", "#reconstructions/correct?"):
            if row in table.index:
                out[row] = table.at[row, master_col]
    if cov_col:
        out["breakpoint_coverage"] = str(cov_col).split("=")[-1]
    recon = table.attrs.get("reconstructions")
    if recon is not None and len(recon):
        out["n_reconstructions"] = len(recon)
        correct = recon[recon["correct"].astype(str).str.lower().isin(
            ("yes", "true", "1"))] if "correct" in recon.columns else recon.iloc[0:0]
        out["reconstruction"] = (correct.iloc[0]["reconstruction"]
                                 if len(correct)
                                 else recon.iloc[0]["reconstruction"])
        out["correct"] = "yes" if len(correct) else "no"
    return out
