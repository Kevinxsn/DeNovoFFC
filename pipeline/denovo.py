"""The de novo stage -- the unknown-peptide route.

Two interchangeable engines behind one function:

  ``cpp``     bin/denovo_ffc, a port of the Python reference.  ~25x faster
              end to end and the default whenever it has been built.
  ``python``  src/de_novo/denovo_core.py, the reference implementation.

They are checked against each other on every reported quantity -- line
categories, master-set coordinates to 1e-6 Da, convergence rounds, and the
full set of tied reconstructions -- by ``tests/local/parity.py`` on the
measured benchmark set and by the portable synthetic tests.  Agreement is
established for the inputs those cover, not proven in general.  If they ever
disagree, the Python one is right by definition and the port has a bug.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pandas as pd

from . import paths
from . import prep

#: denovo.md 7.  Chosen by sweep over all 7 benchmark peptides x {200, 500};
#: it is a plateau, not a spike -- eight neighbouring configurations score the
#: same 14/14.
DEFAULTS = dict(
    delta=0.015,
    delta_unit="da",
    min_ffc_number=3,
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
    secondary_gate=0.25,
    gate_mock_on_goodness=True,
    max_rounds=20,
    max_paths=1000,
    label_tol=0.02,
    # Final-graph and reconstruction settings.  Defaults reproduce the
    # original behaviour exactly; the research RBDP benchmark used
    # recon_sigma_scale=0.4 + presentation_order="informative", and that must
    # be asked for explicitly -- enabling RBDP never tightens tolerances.
    recon_sigma_scale=1.0,
    presentation_order="enumeration",
    reconstruction_method="unit",
    rbdp_evidence="v2",
)

#: Settings that belong to the reconstruction stage, not to denovo_core's
#: discovery keywords.
_RECON_ONLY = ("reconstruction_method", "rbdp_evidence")

METHODS = ("unit", "rbdp", "rbdp-guarded")
ION_SERIES = ("auto", "by", "cz")


def resolve_ion_series(fragmentation: str, ion_series: str = "auto") -> str:
    """``auto`` -> ``by`` for HCD, ``cz`` for ETD/EThcD.  An explicit ``by`` on
    an ETD/EThcD acquisition is a separate b/y analysis of the same table, not a
    pooled reading of both ladders."""
    frag = (fragmentation or "HCD").upper()
    if ion_series not in ION_SERIES:
        raise ValueError(f"ion series must be one of {ION_SERIES}, got {ion_series!r}")
    series = ("by" if frag == "HCD" else "cz") if ion_series == "auto" else ion_series
    if frag == "HCD" and series == "cz":
        raise ValueError("an HCD acquisition has no c/z ladder; use ion series 'by'")
    return series


def validate_charge(precursor_charge: int, ion_series: str) -> None:
    """A complementary pair is two detected fragments, each >= 1 charge: the
    ladder budget is z (b/y) or z - 1 (c/z), and must be >= 2."""
    need = 3 if ion_series == "cz" else 2
    if int(precursor_charge) < need:
        raise ValueError(
            f"a {'c/z pair carries z-1 charges' if ion_series == 'cz' else 'b/y pair'}"
            f"; the precursor charge must be >= {need}, got {precursor_charge}")

#: ⚠ top-N is the DOMINANT hyper-parameter here and lower is cleaner
#: (denovo.md 6): coverage keeps rising past 500 but wrong coordinates rise
#: ~10x faster, and one wrong coordinate usually costs the reconstruction.
#: 7/7 correct at both 200 and 500; 4/7 at 1000; 2/7 at 2000.
DEFAULT_TOP_LIST = [200, 500]


def _cpp_flags(params: dict) -> list[str]:
    """Map the shared parameter dict onto the C++ CLI."""
    f = ["--delta", str(params["delta"]),
         "--delta-unit", params["delta_unit"],
         "--min-ffc", str(params["min_ffc_number"]),
         "--tol-ppm", str(params["tol_ppm"]),
         "--precursor-ppm", str(params["precursor_ppm"]),
         "--sigma-floor-ppm", str(params["sigma_floor_ppm"]),
         "--mock-sigma-cap", str(params["mock_sigma_cap_da"]),
         "--align-threshold", str(params["master_align_threshold"]),
         "--mock-min-ffc", str(params["mock_min_ffc"]),
         "--secondary-gate", str(params["secondary_gate"]),
         "--isotope-tol", str(params["isotope_tol_da"]),
         "--max-rounds", str(params["max_rounds"]),
         "--max-paths", str(params["max_paths"]),
         "--label-tol", str(params["label_tol"]),
         "--recon-sigma-scale", repr(float(params["recon_sigma_scale"])),
         "--presentation-order", params["presentation_order"],
         "--reconstruction-method", params["reconstruction_method"],
         "--rbdp-evidence", params["rbdp_evidence"]]
    if params["accept_ambiguity"]:          f.append("--accept-ambiguity")
    if not params["use_mocks"]:             f.append("--no-mocks")
    if not params["include_isotope_lines"]: f.append("--no-isotope-lines")
    if not params["gate_mock_on_goodness"]: f.append("--no-mock-gate")
    return f


def run_cpp(merged_df: pd.DataFrame, *, parent_charge: int, parent_mass: float,
            terminus: str, fragmentation: str, ion_series: str, top_n: int,
            params: dict, engine: str, dataset_id: str = "data",
            evidence_path: str | None = None) -> dict:
    """Run the C++ engine on an already-merged table."""
    with tempfile.TemporaryDirectory() as tmp:
        tsv = os.path.join(tmp, "map.tsv")
        out = os.path.join(tmp, "result.json")
        merged_df.to_csv(tsv, sep="\t", index=False)
        cmd = [engine, "--data", tsv, "--charge", str(parent_charge),
               "--mass", f"{parent_mass:.10f}", "--terminus", terminus,
               "--fragmentation", fragmentation, "--ion-series", ion_series,
               "--top-n", str(top_n), "--dataset-id", dataset_id,
               "--json", out, "--quiet"] + _cpp_flags(params)
        if evidence_path:
            cmd += ["--emit-evidence", evidence_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 2:
            # the engine's own convention: invalid input or a parity failure
            from .artifacts import IntegrityError
            msg = proc.stderr.strip()
            if "parity" in msg:
                raise IntegrityError(f"denovo_ffc: {msg}")
            raise ValueError(f"denovo_ffc refused the input: {msg}")
        if proc.returncode != 0:
            raise RuntimeError(f"denovo_ffc failed (exit {proc.returncode}):\n"
                               f"{proc.stderr.strip()}")
        with open(out) as fh:
            return json.load(fh)


def run_python(merged_df: pd.DataFrame, *, parent_charge: int,
               parent_mass: float, terminus: str, fragmentation: str,
               ion_series: str, top_n: int, params: dict) -> dict:
    """Run the Python reference and shape its result like the C++ JSON."""
    from denovo_core import run_denovo                          # noqa: E402
    from . import artifacts
    paths.ensure_edges_json()
    core = {k: v for k, v in params.items() if k not in _RECON_ONLY}
    basis_frag = "ETD" if ion_series == "cz" else "HCD"
    res = run_denovo(prep.top_n(merged_df, top_n),
                     parent_charge=parent_charge, parent_mass=parent_mass,
                     terminus=terminus, fragmentation=basis_frag, **core)
    method = params["reconstruction_method"]
    report = override = None
    if method != "unit":
        from rbdp.runner import run as rbdp_run                 # noqa: E402
        from rbdp.snapshot import freeze                        # noqa: E402
        report, outcome = rbdp_run(freeze(res, params), method,
                                   evidence=params["rbdp_evidence"],
                                   max_paths=params["max_paths"])
        if outcome["replace_paths"]:
            ps = _order_paths(outcome["paths"], outcome["hypotheses"], res, params)
            override = dict(paths=ps[0], hypotheses=ps[1], exact=outcome["exact"],
                            optimum=outcome["optimum"],
                            score_method=outcome["score_method"])
        elif outcome["withhold_paths"]:
            override = dict(paths=[], hypotheses=[], exact=0, optimum=-1.0,
                            score_method=outcome["score_method"])
    out = artifacts.python_result_to_dict(
        res, top_n=top_n, acquisition_fragmentation=fragmentation.upper(),
        ion_series=ion_series, precursor_charge=parent_charge,
        neutral_mass=parent_mass, terminus=terminus, params=params, method=method,
        rbdp_report=report, override_paths=override)
    out["_engine"] = "python"
    return out


def _order_paths(paths_, hyps, res, params):
    """Presentation order for an RBDP path set (same rule as the engine)."""
    from denovo_core import _format_gapped_counts                # noqa: E402
    if params["presentation_order"] != "informative":
        return paths_, hyps
    scored = []
    for k, p in enumerate(paths_):
        text, resolved, longest, gaps = _format_gapped_counts(
            p, res.total, res.term, res.merge_tol, params["label_tol"])
        scored.append((-resolved, -longest, gaps, text, [tuple(n) for n in p], k))
    scored.sort()
    return [paths_[s[5]] for s in scored], [hyps[s[5]] for s in scored]


def run(merged_df: pd.DataFrame, *, parent_charge: int, parent_mass: float,
        terminus: str = "H2O", fragmentation: str = "HCD", top_n: int = 500,
        engine: str = "auto", params: dict | None = None,
        ion_series: str = "auto", dataset_id: str = "data",
        evidence_path: str | None = None) -> dict:
    """Sequence one map de novo.  ``merged_df`` must already be merged."""
    p = dict(DEFAULTS, **(params or {}))
    series = resolve_ion_series(fragmentation, ion_series)
    validate_charge(parent_charge, series)
    if p["reconstruction_method"] not in METHODS:
        raise ValueError(f"reconstruction method must be one of {METHODS}")
    if p["reconstruction_method"] != "unit" and series != "by":
        raise ValueError("RBDP is validated for single-series HCD b/y only; it is "
                         "not available for the c/z series (use 'unit')")
    if not (p["recon_sigma_scale"] > 0):
        raise ValueError("recon_sigma_scale must be positive")
    if engine in ("auto", "cpp"):
        exe = paths.cpp_engine(required=(engine == "cpp"))
        if exe:
            out = run_cpp(merged_df, parent_charge=parent_charge,
                          parent_mass=parent_mass, terminus=terminus,
                          fragmentation=fragmentation, ion_series=series,
                          top_n=top_n, params=p, engine=exe,
                          dataset_id=dataset_id, evidence_path=evidence_path)
            out["_engine"] = "cpp"
            return out
    if evidence_path:
        raise ValueError("evidence export (--emit-evidence) needs the C++ engine")
    return run_python(merged_df, parent_charge=parent_charge,
                      parent_mass=parent_mass, terminus=terminus,
                      fragmentation=fragmentation, ion_series=series,
                      top_n=top_n, params=p)


def write_detail(result: dict, path: str) -> None:
    """Per-run detail workbook: `lines`, `master_set`, `reconstructions`."""
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(result["lines"]).to_excel(xl, sheet_name="lines", index=False)
        ms = pd.DataFrame(result["master_set"])
        if not ms.empty:
            ms["origins"] = ms["origins"].apply(lambda v: "; ".join(v))
        ms.to_excel(xl, sheet_name="master_set", index=False)
        pd.DataFrame({"#": range(1, len(result["reconstructions"]) + 1),
                      "reconstruction": result["reconstructions"]}
                     ).to_excel(xl, sheet_name="reconstructions", index=False)
