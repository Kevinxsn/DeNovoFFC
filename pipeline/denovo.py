"""The de novo stage -- the unknown-peptide route.

Two interchangeable engines behind one function:

  ``cpp``     bin/denovo_mms, a port of the Python reference.  ~25x faster
              end to end and the default whenever it has been built.
  ``python``  src/de_novo/denovo_core.py, the reference implementation.

They are verified to agree on every reported quantity -- line categories,
master-set coordinates to 1e-6 Da, convergence rounds, and the full set of
tied reconstructions -- by ``tests/parity.py``.  If they ever disagree, the
Python one is right by definition and the port has a bug.
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
)

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
         "--label-tol", str(params["label_tol"])]
    if params["accept_ambiguity"]:          f.append("--accept-ambiguity")
    if not params["use_mocks"]:             f.append("--no-mocks")
    if not params["include_isotope_lines"]: f.append("--no-isotope-lines")
    if not params["gate_mock_on_goodness"]: f.append("--no-mock-gate")
    return f


def run_cpp(merged_df: pd.DataFrame, *, parent_charge: int, parent_mass: float,
            terminus: str, fragmentation: str, top_n: int,
            params: dict, engine: str) -> dict:
    """Run the C++ engine on an already-merged table."""
    with tempfile.TemporaryDirectory() as tmp:
        tsv = os.path.join(tmp, "map.tsv")
        out = os.path.join(tmp, "result.json")
        merged_df.to_csv(tsv, sep="\t", index=False)
        cmd = [engine, "--data", tsv, "--charge", str(parent_charge),
               "--mass", f"{parent_mass:.10f}", "--terminus", terminus,
               "--fragmentation", fragmentation, "--top-n", str(top_n),
               "--json", out, "--quiet"] + _cpp_flags(params)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"denovo_mms failed:\n{proc.stderr.strip()}")
        with open(out) as fh:
            return json.load(fh)


def run_python(merged_df: pd.DataFrame, *, parent_charge: int,
               parent_mass: float, terminus: str, fragmentation: str,
               top_n: int, params: dict) -> dict:
    """Run the Python reference and shape its result like the C++ JSON."""
    from denovo_core import run_denovo                          # noqa: E402
    paths.ensure_edges_json()
    res = run_denovo(prep.top_n(merged_df, top_n),
                     parent_charge=parent_charge, parent_mass=parent_mass,
                     terminus=terminus, fragmentation=fragmentation, **params)
    return {
        "n_ffc": res.params["n_ffc"],
        "fragmentation": res.basis.mode,
        "line_charge": res.params["line_charge"],
        "coord_total": res.total,
        "coord_term": res.term,
        "rounds": res.rounds,
        "n_paths": res.n_paths,
        "best_weight": res.best_weight,
        "n_lines": len(res.lines),
        "reconstructions": [r["reconstruction"] for r in res.reconstructions],
        "master_set": [dict(coord=e.coord, sigma=e.sigma, n=e.n,
                            best_rank=e.best_rank, kinds=e.kinds,
                            origins=e.origins)
                       for e in res.master.entries],
        "paths": [[list(node) for node in path] for path in res.paths_raw],
        "lines": [dict(key=li.key, i=li.i, j=li.j, mass=li.mass,
                       a_shift=li.a_shift, category=li.category,
                       detail=li.detail, n_ffc=li.n_ffc,
                       n_inactive=li.n_inactive, eligible=li.eligible,
                       n_banked=li.n_banked, unproductive=li.n_unproductive,
                       unambiguous=li.n_unambiguous, ambiguous=li.n_ambiguous,
                       mock=li.mock is not None, mock_refused=li.mock_refused)
                  for li in res.lines],
        "_engine": "python",
    }


def run(merged_df: pd.DataFrame, *, parent_charge: int, parent_mass: float,
        terminus: str = "H2O", fragmentation: str = "HCD", top_n: int = 500,
        engine: str = "auto", params: dict | None = None) -> dict:
    """Sequence one map de novo.  ``merged_df`` must already be merged."""
    p = dict(DEFAULTS, **(params or {}))
    if engine in ("auto", "cpp"):
        exe = paths.cpp_engine(required=(engine == "cpp"))
        if exe:
            out = run_cpp(merged_df, parent_charge=parent_charge,
                          parent_mass=parent_mass, terminus=terminus,
                          fragmentation=fragmentation, top_n=top_n,
                          params=p, engine=exe)
            out["_engine"] = "cpp"
            return out
    return run_python(merged_df, parent_charge=parent_charge,
                      parent_mass=parent_mass, terminus=terminus,
                      fragmentation=fragmentation, top_n=top_n, params=p)


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
