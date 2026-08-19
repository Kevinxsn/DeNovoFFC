"""
FFC Annotation Pipeline
=======================
Single entry point that runs the three post-deconvolution FFC analyses on one
data table, given only the basic peptide information (sequence, charge,
terminal residual) and the deconvolution output path.

The three analyses share the *same* input table, so this script loads and
merges it once and then feeds each method its own view:

  1. Complementary / spurious-parental test
     (deiso/python_code/spurious_parental_charge.parental_ffc_annotations_from_charges)
     -- trusts the deconvolution charge assignment and asks, for every
     parental-line FFC, whether it can be annotated as a complementary b/y
     pair or is spurious.  Console output only.

  2. Theoretical-MMS FFC annotation
     (theoretical_mms_annotation/ffc_annotator.annotate_ffcs_with_charge)
     -- annotates each FFC against the theoretical FFC map (Note 11) with a
     single-m/z fallback.  Writes an .xlsx.

  3. Line-based internal annotation
     (line_based_annotation/line_internal_annotation_oa.run_internal_annotation)
     -- orientation-agnostic greedy-line coverage table with explicit
     internal-ion labels (charge recovered from line detection rather than
     trusted from the deconvolution's per-side assignment).  Writes an .xlsx.

Analyses 2 and 3 both match fragments with a relative **ppm** window, so they
share a single ``tol_ppm`` setting rather than each carrying its own tolerance.

Data flow
---------
    read table
        -> merge_duplicate_ffcs_charge(symmetric=True, reset_ranking=True)   [once]
        -> prepare_ffc_data(top_n=TOP_FFC_COMPLEMENTARY)  -> analysis 1
        -> prepare_ffc_data(top_n=TOP_FFC_ANNOTATION)     -> analysis 2  (.xlsx)
        -> prepare_ffc_data(top_n=TOP_FFC_LINE)           -> analysis 3  (.xlsx)

Because the three analyses need very different amounts of data, each gets its
own top-FFC cutoff (default 15000 / 2000 / 100).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Locate the sibling analysis packages and put them on the import path.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, ".."))          # .../2D_spec_dict
_DEISO = os.path.join(_BASE, "deiso", "python_code")
_TMMS = os.path.join(_BASE, "theoretical_mms_annotation")
_LINE = os.path.join(_BASE, "line_based_annotation")

for _p in (_DEISO, _TMMS, _LINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared deiso helpers.
import interpreter_modify  # noqa: F401  (registers modification handling)
import peptide
from merge_ffcs import merge_duplicate_ffcs_charge
from line_finding import prepare_ffc_data

# Analysis 1: complementary / spurious-parental test.
from spurious_parental_charge import parental_ffc_annotations_from_charges
from spurious_parental import merge_parental_annotations

# Analysis 2: theoretical-MMS FFC annotation.
from ffc_annotator import annotate_ffcs_with_charge

# Analysis 3: line-based internal annotation (orientation-agnostic variant --
# the newest implementation; shares the same relative ppm mass-tolerance
# model as the FFC annotator above).
from line_internal_annotation_oa import run_internal_annotation

# Analysis 3 (v2): line-based internal annotation with de novo reconstruction
# reporting. Imported under a distinct name -- only run_line_based_reconstruct
# uses it, so the plain (v1) run_line_based above is unaffected.
from line_internal_annotation_oa_v2 import run_internal_annotation as run_internal_annotation_v2


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------

def run_complementary(merged_df, pep, charge, parent_mass, top_n,
                      iso_range=4, threshold=0.05,
                      parental_shift_threshold=0.05):
    """Analysis 1 -- charge-assigned parental FFCs with spurious/annotated
    status.  Prints a summary and the charge-independent merged view; returns
    the per-row annotation DataFrame."""
    print("\n" + "=" * 70)
    print(f"[1/3] Complementary (spurious-parental) test   top_n={top_n}")
    print("=" * 70)

    df = prepare_ffc_data(merged_df, top_n=top_n)
    print(f"FFC rows fed to complementary test: {len(df)}")

    annot_df = parental_ffc_annotations_from_charges(
        df,
        pep,
        parent_charge=charge,
        parent_mass=parent_mass,
        iso_range=iso_range,
        threshold=threshold,
        parental_shift_threshold=parental_shift_threshold,
    )

    n_total = len(annot_df)
    if not n_total:
        print("\nNo parental-line FFCs found.")
        return annot_df

    # Charge-independent merged view (symmetric pairs collapsed).
    merged = merge_parental_annotations(annot_df)
    n_merged = len(merged)
    n_spurious = (
        int((merged["b_ions"] == "spurious").sum())
        if "b_ions" in merged.columns else 0
    )
    spurious_rate = (n_spurious / n_merged * 100) if n_merged else 0.0

    print(f"\nUnique pairs: {n_merged}")
    print(f"  Spurious : {n_spurious}")
    print(f"  Annotated: {n_merged - n_spurious}")
    print(f"  Spurious rate: {spurious_rate:.1f}%")

    show_cols = [
        c for c in ["mz1_lo", "mz1_hi", "Ranking", "b_ions", "breaking_points"]
        if c in merged.columns
    ]
    print(merged[show_cols].to_string(index=False))

    return annot_df


def run_ffc_annotation(merged_df, sequence, charge, residual, top_n,
                       output_path, tol_ppm=20, isomax_cutoff=5.0,
                       extra_info=True):
    """Analysis 2 -- theoretical-MMS FFC annotation.  Writes ``output_path``
    (.xlsx) and returns the result DataFrame."""
    print("\n" + "=" * 70)
    print(f"[2/3] Theoretical-MMS FFC annotation   top_n={top_n}")
    print("=" * 70)

    df = prepare_ffc_data(merged_df, top_n=top_n)
    print(f"FFC rows fed to FFC annotator: {len(df)}")

    result = annotate_ffcs_with_charge(
        df,
        sequence=sequence,
        precursor_charge=charge,
        output_path=output_path,
        tol_ppm=tol_ppm,
        residual=residual,
        isomax_cutoff=isomax_cutoff,
        extra_info=extra_info,
    )

    # Top-10 annotations (result is already ordered by rank).
    if len(result):
        print("\nTop 10 annotations:")
        with pd.option_context("display.max_colwidth", None, "display.width", 250):
            print(result.head(10).to_string(index=False))

    return result


def run_line_based(merged_df, pep, charge, parent_mass, top_n,
                   output_path, sheet_name, delta, delta_unit="da",
                   min_ffc=3, iso_range=1, tol_ppm=20.0, mock_tol_ppm=100.0,
                   align_threshold=1):
    """Analysis 3 -- line-based internal annotation.  Writes ``output_path``
    (.xlsx) and returns ``(coverage_table, all_lines)``.

    ``delta`` (line-detection cluster gap) is the most important knob to
    retune between runs -- it has no default here and must be supplied by the
    caller (see ``run_pipeline``'s ``line_delta``). ``delta_unit`` selects
    whether ``delta`` is a fixed Da gap ("da", default) or a relative ppm gap
    ("ppm"). ``align_threshold`` is the minimum number of inactive FFCs an
    internal line needs to count as "aligned" to the parental ladder."""
    print("\n" + "=" * 70)
    print(f"[3/3] Line-based internal annotation   top_n={top_n}")
    print("=" * 70)

    df = prepare_ffc_data(merged_df, top_n=top_n)
    print(f"FFC rows fed to line-based annotation: {len(df)}")

    cov_table, all_lines = run_internal_annotation(
        df,
        pep,
        parent_charge=charge,
        parent_mass=parent_mass,
        delta=delta,
        delta_unit=delta_unit,
        min_ffc_number=min_ffc,
        iso_range=iso_range,
        tol_ppm=tol_ppm,
        mock_tol_ppm=mock_tol_ppm,
        align_threshold=align_threshold,
    )
    print(f"Detected lines: {len(all_lines)}")

    # Same view the standalone line_internal_annotation prints: the coverage
    # rows for the backbone cleavage sites (the first n-1 rows), before the
    # appended Row/Column-count and summary rows.
    data_rows = cov_table.iloc[: len(pep.AA_array) - 1]
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(data_rows.to_string())

    # Write with a fixed column width of 32 (same as the standalone
    # line_internal_annotation.py) so the wide annotation cells stay readable.
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cov_table.to_excel(writer, sheet_name=sheet_name,
                           index_label=f"#FFC={top_n}")
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            ws.column_dimensions[col_cells[0].column_letter].width = 32
    print(f"Saved line-based table to {output_path} [{sheet_name}]")
    return cov_table, all_lines


def run_line_based_reconstruct(merged_df, pep, charge, parent_mass, top_n,
                               output_path, sheet_name, delta, delta_unit="da",
                               min_ffc=3, iso_range=1, tol_ppm=20.0,
                               mock_tol_ppm=100.0, align_threshold=1,
                               reconstruct=True, considering_spurious=False,
                               edges_path=None, reconstruct_threshold=0.001,
                               reconstruct_max_paths=1000,
                               productivity=False, productivity_tol_ppm=30.0,
                               productivity_coord_tol_ppm=30.0,
                               extend_master_set=False,
                               master_align_threshold=1,
                               converge=True, max_convergence_rounds=20,
                               loose_mock=False, accept_ambiguity=False,
                               graph=False, graph_path=None):
    """Analysis 3 (v2) -- line-based internal annotation with de novo
    reconstruction reporting (line_internal_annotation_oa_v2). Writes
    ``output_path`` (.xlsx) and returns ``(coverage_table, all_lines)``.

    Same knobs as ``run_line_based``, plus:
      reconstruct : bool, default True
          Emit the ``#reconstructions/correct reconstructions`` summary cell
          by running the de novo longest-path search over the parental
          ladder's breakpoints. True here (unlike the v2 module's own
          default) since reconstruction is the point of this stage.
      considering_spurious : bool, default False
          Also feed the parental lines' unmatched/unexplained FFCs to the
          de novo search.
      edges_path : str, optional
          Path to the precomputed compound-mass table; defaults to the v2
          module's own ``DEFAULT_EDGES_PATH`` (``reconstruct/edges.json``).
      reconstruct_threshold, reconstruct_max_paths
          Passed straight to the de novo longest-path search.
      productivity : bool, default False
          Emit the ``#unproductive/unambiguous/ambiguous`` row, splitting each
          line's active TFFCs by what they would contribute to a de novo
          reconstruction (judged against the frozen parental ladder). Off by
          default -- costs several extra seconds per peptide (DAG builds).
      productivity_tol_ppm, productivity_coord_tol_ppm : float, default 30
          Relative (ppm) tolerance for the productivity DAG's edges, and for
          testing whether a TFFC's coordinate is already in the parental
          (master) set. ppm, not a fixed Da window, for the same reason
          fragment matching uses ppm -- deconvolved masses carry a roughly
          relative error, so a single Da window is too loose for light
          coordinates and too tight for heavy ones.
      extend_master_set : bool, default False
          Let aligned internal/neutral-loss lines contribute to the
          ``Master TFFC-set`` column too, on top of the parental lines
          (their productive TFFCs plus each internal line's mock). Off by
          default -- like ``productivity``, it needs ``edges.json`` and the
          de novo machinery, so the column stays parental-only unless asked.
          When BOTH this and ``reconstruct`` are True, the v2 module also runs
          a de novo search over the *whole* master TFFC-set (not just the
          parental ladder) and attaches it as ``cov_table.attrs
          ["reconstructions"]`` -- this function writes that, when present, to
          a second ``reconstructions`` sheet in ``output_path`` (columns
          ``#``, ``reconstruction``, ``correct``), and prints up to 5 of them.
      master_align_threshold : int, default 1
          Inactive FFCs an internal/loss line must have (breakpoint already
          on the master TFFC-set) before it may join that set. Re-derived
          every round against the *grown* set when ``converge=True``.
      converge : bool, default True
          Only takes effect when ``productivity`` or ``extend_master_set`` is
          True. Builds the master TFFC-set iteratively: bank the unambiguous
          TFFCs, re-derive aligned/active/unambiguous against the grown set,
          repeat until the set stops increasing. Adds the
          ``#unproductive/unambiguous/ambiguous (converged)`` row beneath the
          round-1 row, and makes ``extend_master_set`` use the converged set
          rather than the round-1 one.
      max_convergence_rounds : int, default 20
          Safety cap on those rounds (convergence is typically 2-4).
      loose_mock : bool, default False
          Add a charge-free mock probe alongside the usual charge sweep, so a
          loss line whose charges already sum to the precursor charge (it
          lost a *neutral* run of residues, so the sweep's
          ``z = parent_charge - (i+j) >= 1`` assumption never fires) still
          gets a mock. Tagged ``[mock-loose]``; a charged reading wins any
          tie.
      accept_ambiguity : bool, default False
          Let ambiguous TFFCs into the master TFFC-set. An ambiguous TFFC's
          two images are "one of these two, never this one" -- with True BOTH
          are banked (one is false by construction), with False neither is,
          for the column, the reconstruction, and the incorrect-TFFC tally
          alike. Default False: measured across 7 peptides, ambiguous TFFCs
          bought one usable bond but supplied 4 wrong coordinates, costing
          one peptide its correct reconstruction.
      graph : bool, default False
          Only takes effect when a master-set reconstruction runs (both
          ``reconstruct`` and ``extend_master_set`` True) AND that search
          finds a correct path. When it does, also render the 2D map
          (``../reconstruct/path_graph_revised.visualize_all_paths``) with
          the correct reconstruction as the hero, the other tied paths as
          candidates, and the incorrect master TFFCs marked spurious. Writes
          one extra image file; changes nothing in the coverage table.
      graph_path : str, optional
          Where to write the image. When omitted, this function derives one
          from ``output_path`` (same basename, ``_graph.png`` suffix) rather
          than falling back to the v2 module's ``master_set_reconstruction.png``,
          so consecutive top_list runs never overwrite each other's graph.
    """
    if graph and graph_path is None:
        graph_path = os.path.splitext(output_path)[0] + "_graph.png"
    print("\n" + "=" * 70)
    print(f"[3/3] Line-based internal annotation (reconstruct)   top_n={top_n}")
    print("=" * 70)

    df = prepare_ffc_data(merged_df, top_n=top_n)
    print(f"FFC rows fed to line-based annotation: {len(df)}")

    cov_table, all_lines = run_internal_annotation_v2(
        df,
        pep,
        parent_charge=charge,
        parent_mass=parent_mass,
        delta=delta,
        delta_unit=delta_unit,
        min_ffc_number=min_ffc,
        iso_range=iso_range,
        tol_ppm=tol_ppm,
        mock_tol_ppm=mock_tol_ppm,
        align_threshold=align_threshold,
        reconstruct=reconstruct,
        considering_spurious=considering_spurious,
        edges_path=edges_path,
        reconstruct_threshold=reconstruct_threshold,
        reconstruct_max_paths=reconstruct_max_paths,
        productivity=productivity,
        productivity_tol_ppm=productivity_tol_ppm,
        productivity_coord_tol_ppm=productivity_coord_tol_ppm,
        extend_master_set=extend_master_set,
        master_align_threshold=master_align_threshold,
        converge=converge,
        max_convergence_rounds=max_convergence_rounds,
        loose_mock=loose_mock,
        accept_ambiguity=accept_ambiguity,
        graph=graph,
        graph_path=graph_path,
    )
    print(f"Detected lines: {len(all_lines)}")

    data_rows = cov_table.iloc[: len(pep.AA_array) - 1]
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(data_rows.to_string())

    # De novo reconstruction from the WHOLE master TFFC-set (parental +
    # aligned internal/loss lines) -- only populated when both reconstruct=True
    # and extend_master_set=True; see line_internal_annotation_oa_v2's
    # _master_set_reconstruction. Carried out-of-band via cov_table.attrs
    # rather than a coverage-table cell, so it's written to its own sheet.
    recon = cov_table.attrs.get("reconstructions")
    if recon is not None and not recon.empty:
        n_ok = int((recon["correct"] != "").sum())
        print(f"\nMaster-set reconstructions: {len(recon)}  ({n_ok} correct)   "
              f"incorrect TFFCs: {len(cov_table.attrs.get('incorrect_tffcs', []))}")
        for _, r in recon.head(5).iterrows():
            print(f"  {r['#']:>3}. {r['reconstruction']}"
                  f"{'   <-- CORRECT' if r['correct'] else ''}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cov_table.to_excel(writer, sheet_name=sheet_name,
                           index_label=f"#FFC={top_n}")
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            ws.column_dimensions[col_cells[0].column_letter].width = 32
        if recon is not None:
            recon.to_excel(writer, sheet_name="reconstructions", index=False)
            ws2 = writer.sheets["reconstructions"]
            ws2.column_dimensions["A"].width = 6
            ws2.column_dimensions["B"].width = 140
            ws2.column_dimensions["C"].width = 12
    print(f"Saved line-based table to {output_path} [{sheet_name}]"
          + ("  + [reconstructions]" if recon is not None else ""))
    if graph:
        if graph_path is not None and os.path.isfile(graph_path):
            print(f"Saved reconstruction graph to {graph_path}")
        else:
            print("No reconstruction graph written (no correct reconstruction found)")
    return cov_table, all_lines


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str,
    sequence: str,
    charge: int,
    residual: str = "H2O",
    *,
    top_ffc_complementary: int = 15000,
    top_ffc_annotation: int = 2000,
    top_ffc_line: int = 100,
    line_delta: float = 0.02,
    line_delta_unit: str = "da",
    line_min_ffc: int = 3,
    line_align_threshold: int = 1,
    tol_ppm: float = 20.0,
    mock_tol_ppm: float = 100.0,
    ffc_annotation_output: str | None = None,
    line_based_output: str | None = None,
    output_dir: str | None = None,
    output_prefix: str | None = None,
    sep: str = "\t",
):
    """Load one deconvolution table, merge it once, and run all three analyses.

    Parameters
    ----------
    data_path : str
        Path to the deconvolution output table (TSV by default).
    sequence : str
        Peptide sequence (single-letter amino-acid codes, no modifications).
    charge : int
        Precursor charge state.
    residual : str
        Terminal residual group: 'H2O' (default) or 'NH3'.
    top_ffc_complementary, top_ffc_annotation, top_ffc_line : int
        Per-analysis top-FFC cutoffs (defaults 15000 / 2000 / 100).
    line_delta : float
        Line-detection cluster gap for the orientation-agnostic line-based
        annotation -- the main knob to retune between runs (wider merges more
        FFCs onto one line; narrower splits them). Default 0.02.
    line_delta_unit : str
        Unit for ``line_delta``: 'da' (default, a fixed 0.02 Da gap) or 'ppm'
        (a relative gap, evaluated against each cluster's reconstructed mass).
    line_min_ffc : int
        NumFFC for line detection: the minimum number of FFCs a greedy line
        must contain to be kept (passed to run_internal_annotation's
        min_ffc_number). Default 3.
    line_align_threshold : int
        Minimum number of inactive FFCs (FFCs whose breakpoint is already on
        a parental line) an internal line needs to count as "aligned" to the
        parental ladder in the line-based coverage table. Default 1.
    tol_ppm : float
        Shared relative mass tolerance (ppm) for fragment matching. Both the
        FFC annotator and the line-based annotation use it, so the two
        reports are built on the same tolerance model. Default 20.
    mock_tol_ppm : float
        Looser, independent ppm tolerance used only by the line-based
        annotation when probing a line's shift for a mock (virtual) FFC.
        Default 100.
    ffc_annotation_output, line_based_output : str, optional
        Explicit output paths for the FFC-annotation and line-based .xlsx
        files. When given, they override the auto-generated names (a relative
        name is placed under ``output_dir``). When omitted, the names are
        derived from ``output_prefix``.
    output_dir : str, optional
        Directory for the two .xlsx outputs (defaults to the ``result/``
        subfolder next to this script).
    output_prefix : str, optional
        Basename prefix for the outputs (defaults to ``{sequence[:8]}_{charge}+``).
    sep : str
        Field separator for ``pd.read_csv`` (default tab).

    Returns
    -------
    dict with keys 'complementary', 'ffc_annotation', 'line_based' holding the
    result of each stage.
    """
    residual = residual.upper()
    if residual not in ("H2O", "NH3"):
        raise ValueError(f"residual must be 'H2O' or 'NH3', got {residual!r}")
    line_delta_unit = line_delta_unit.lower()
    if line_delta_unit not in ("da", "ppm"):
        raise ValueError(f"line_delta_unit must be 'da' or 'ppm', got {line_delta_unit!r}")

    if output_dir is None:
        output_dir = os.path.join(_HERE, "result")
    os.makedirs(output_dir, exist_ok=True)
    if output_prefix is None:
        output_prefix = f"{sequence[:8]}_{charge}+"

    def _resolve(path, default_name):
        """Absolute path as given; a bare name is placed under output_dir."""
        if path is None:
            path = default_name
        return path if os.path.isabs(path) else os.path.join(output_dir, path)

    ffc_xlsx = _resolve(ffc_annotation_output, f"{output_prefix}_ffc_annotation.xlsx")
    line_xlsx = _resolve(line_based_output, f"{output_prefix}_line_based.xlsx")
    # Excel sheet name: from the line-based file's basename, <=31 chars.
    line_sheet = os.path.splitext(os.path.basename(line_xlsx))[0][:31]

    # ── Peptide model (H2O -> end_h20=True; NH3 -> end_h20='NH3') ──────────
    end_h20 = True if residual == "H2O" else residual
    pep = peptide.Pep(f"[{sequence}+{charge}H]{charge}+", end_h20=end_h20)
    parent_mass = pep.pep_mass
    print(f"Peptide: {sequence}  charge={charge}  residual={residual}")
    print(f"  length={len(pep.AA_array)}  parent_mass={parent_mass:.4f}")

    # ── Load + merge once ──────────────────────────────────────────────────
    print(f"\nLoading {data_path}")
    ffc_df = pd.read_csv(data_path, sep=sep)
    print(f"  raw rows: {len(ffc_df)}")
    merged_df = merge_duplicate_ffcs_charge(
        ffc_df, symmetric=True, reset_ranking=True, tolerance = 0.002
    )
    print(f"  rows after merge_duplicate_ffcs_charge: {len(merged_df)}")

    # ── Run the three analyses on the shared merged table ──────────────────
    complementary = run_complementary(
        merged_df, pep, charge, parent_mass, top_ffc_complementary
    )
    ffc_annotation = run_ffc_annotation(
        merged_df, sequence, charge, residual, top_ffc_annotation, ffc_xlsx,
        tol_ppm=tol_ppm
    )
    line_based = run_line_based(
        merged_df, pep, charge, parent_mass, top_ffc_line,
        line_xlsx, line_sheet, line_delta, delta_unit=line_delta_unit,
        min_ffc=line_min_ffc, tol_ppm=tol_ppm, mock_tol_ppm=mock_tol_ppm,
        align_threshold=line_align_threshold
    )

    print("\n" + "=" * 70)
    print("Pipeline complete. Excel outputs:")
    print(f"  FFC annotation : {ffc_xlsx}")
    print(f"  Line-based     : {line_xlsx}")
    print("=" * 70)

    return {
        "complementary": complementary,
        "ffc_annotation": ffc_annotation,
        "line_based": line_based,
    }


# ---------------------------------------------------------------------------
# Configuration / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Just set NAME ──────────────────────────────────────────────────────
    # NAME is a dataset name like HAD_33_Z6_E20; peptide_info.py (in data_prep)
    # turns it into the sequence, charge and C-terminal residual by reading
    # pep_seq.xlsx.  The parent mass is NOT needed here -- peptide.Pep computes
    # it from sequence+charge+residual inside run_pipeline.
    #   NAME=... python annotation_pipeline.py   also works (env overrides).
    # List the peptides it knows: python ../data_prep/peptide_info.py --list
    NAME = os.environ.get("NAME", "VEA_15_Z3_revised")

    # Import the resolver from the sibling data_prep package.
    _DATA_PREP = os.path.join(_BASE, "data_prep")
    if _DATA_PREP not in sys.path:
        sys.path.insert(0, _DATA_PREP)
    from peptide_info import resolve as _resolve_name  # noqa: E402
    

    try:
        info = _resolve_name(NAME)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"[error] cannot resolve NAME={NAME}: {exc.args[0]}")
    PEP_SEQ = info["SEQUENCE"]
    CHARGE = info["PARENT_CHARGE"]
    RESIDUAL = info["RESIDUAL"]            # 'H2O' or 'NH3', ready for run_pipeline

    # Deconvolution output for this NAME, under .../deconvoluted/merged/.  The
    # Xiaowen pipeline writes <NAME>_XIAOWEN_merged.tsv; the older run_deconv_test
    # wrote <NAME>_merged.tsv.  Prefer the Xiaowen one, fall back to the plain
    # one.  Override with DATA_PATH=... in the environment for anything else.
    _MERGED_DIR = "/home/sux002/data/deconvoluted/merged"
    DATA_PATH = os.environ.get("DATA_PATH")
    if DATA_PATH is None:
        candidates = [
            #os.path.join(_MERGED_DIR, f"{NAME}_XIAOWEN_merged.tsv"),
            os.path.join(_MERGED_DIR, f"{NAME}_merged.tsv"),
        ]
        DATA_PATH = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    if not os.path.isfile(DATA_PATH):
        tried = "\n        ".join(candidates) if not os.environ.get("DATA_PATH") else DATA_PATH
        raise SystemExit(
            f"[error] deconvolution table not found for NAME={NAME}. Tried:\n"
            f"        {tried}\n"
            f"        set DATA_PATH=... if this run used a different filename."
        )

    # Short label for the output .xlsx names, e.g. HAD_33 + 6 -> 'HAD6+'.
    #name = f"{info['PEPTIDE_ID'].split('_')[0]}{CHARGE}+"
    name=NAME

    print(f"NAME={NAME}  ->  seq={PEP_SEQ[:12]}...  charge={CHARGE}  "
          f"residual={RESIDUAL}  label={name}")

    # ── Per-analysis top-FFC cutoffs ───────────────────────────────────────
    TOP_FFC_COMPLEMENTARY = 15000
    TOP_FFC_ANNOTATION = 2000
    TOP_FFC_LINE = 50

    # Line-detection cluster gap -- the main knob to retune between runs.
    LINE_DELTA = 0.02
    LINE_DELTA_UNIT = "da"   # "da" (fixed Da gap) or "ppm" (relative gap)

    # NumFFC for line detection: min FFCs a greedy line must contain to be kept.
    LINE_MIN_FFC = 3

    # Min inactive FFCs for an internal line to count as "aligned" to the
    # parental ladder.
    LINE_ALIGN_THRESHOLD = 1

    # Shared mass tolerance (ppm) for both the FFC annotator and the
    # line-based annotation.
    TOL_PPM = 20.0
    MOCK_TOL_PPM = 100.0   # looser, independent tolerance for mock FFCs

    # ── Output Excel file names (set these to whatever you like) ────────────
    # Absolute paths are used as-is; a bare filename lands next to this script.
    FFC_ANNOTATION_OUTPUT = f"{name}_ffc_annotation.xlsx"   # e.g. "VEA3+_ffc_annotation.xlsx"
    LINE_BASED_OUTPUT = f"{name}_line_based.xlsx"       # e.g. "VEA3+_line_based.xlsx"

    run_pipeline(
        DATA_PATH,
        PEP_SEQ,
        CHARGE,
        residual=RESIDUAL,
        top_ffc_complementary=TOP_FFC_COMPLEMENTARY,
        top_ffc_annotation=TOP_FFC_ANNOTATION,
        top_ffc_line=TOP_FFC_LINE,
        line_delta=LINE_DELTA,
        line_delta_unit=LINE_DELTA_UNIT,
        line_min_ffc=LINE_MIN_FFC,
        line_align_threshold=LINE_ALIGN_THRESHOLD,
        tol_ppm=TOL_PPM,
        mock_tol_ppm=MOCK_TOL_PPM,
        ffc_annotation_output=FFC_ANNOTATION_OUTPUT,
        line_based_output=LINE_BASED_OUTPUT,
    )
