"""The deconvolution stage -- optional on both routes.

Turns a raw MMS map (m/z pairs, covariance, ranking, intensities -- no charges)
into a table with a charge and a monoisotopic mass on every fragment.  Two
stages plus a merge:

  Stage 1  FFC-line deconvolution.  Detect lines, read each FFC's per-side
           charges off the line it lands on, recover the monoisotopic mass
           with the isotope-envelope / theo_patt machinery.
  Stage 2  TopFD fallback for everything Stage 1 could not place.
  Merge    both into one table, `deconv_method` saying which produced each row.

⚠ Stage 2 needs the TopFD binary, which is NOT vendored (paths.topfd_bin).
Without it, ``run(..., run_topfd=False)`` still produces a usable Stage-1-only
table; it just places fewer rows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import paths
from .artifacts import sha256_file, write_json_atomic
from deconv_pipeline_oa import run_deconvolution               # noqa: E402

#: deconvolution.md 7.  `top_lines=2000` is tuned: bond coverage is FLAT from
#: 2000 to 15000 on all 7 benchmark peptides while spurious parental pairs rise
#: in 5 of 7 and detection takes 1.5-5x longer.  Looking deeper into the
#: ranking for lines finds more lines, not more peptide.
DEFAULTS = dict(
    delta=0.02,
    delta_unit="da",
    min_ffc_number=3,
    top_lines=2000,
    top_selected=15000,
    top_bg=15000,
    merge_tolerance=0.002,
    merge_over_fetch=4,
    mz_tol=0.01,
    fast_detector=True,
    topfd_error_tol=0.04,
    topfd_sn_ratio=0.1,
    topfd_ecscore_cutoff=0.0,
    topfd_full_pool=True,
)


#: peptide.py's proton -- the constant the line model and TopFD bounds use.
LINE_PROTON = 1.00725
#: The hydrogen atom electron transfer delivers (cz_series.H_ATOM).
H_ATOM = 1.00783


@dataclass(frozen=True)
class DeconvolutionRequest:
    """What the deconvolution is told, with the two meanings kept apart.

    ``line_model_*`` describe the LINE GEOMETRY: the charge sum a complementary
    pair carries and the mass ``i*mzA + j*mzB`` it reconstructs to.
    ``topfd_*`` describe the PRECURSOR: a fragment cannot out-charge or
    out-weigh the intact ion.  For b/y they coincide; for c/z they do not:

        b/y   line charge z      line mass M + z*p
        c/z   line charge z - 1  line mass M + H + (z-1)*p    (one electron above)
              TopFD still gets z and M + z*p

    The original wrapper passed the precursor charge to both places, so an ETD
    map was deconvolved on the b/y ladder; changing only that argument to z - 1
    would have wrongly lowered TopFD's bound.
    """
    precursor_charge: int
    precursor_neutral_mass: float
    ion_series: str
    line_model_charge: int
    line_model_charged_mass: float
    topfd_max_charge: int
    topfd_max_mass: float

    def as_dict(self) -> dict:
        return dict(precursor_charge=self.precursor_charge,
                    precursor_neutral_mass=self.precursor_neutral_mass,
                    ion_series=self.ion_series,
                    line_model_charge=self.line_model_charge,
                    line_model_charged_mass=self.line_model_charged_mass,
                    topfd_max_charge=self.topfd_max_charge,
                    topfd_max_mass=self.topfd_max_mass,
                    proton=LINE_PROTON, h_atom=H_ATOM)


def make_request(precursor_charge: int, neutral_mass: float,
                 ion_series: str) -> DeconvolutionRequest:
    z = int(precursor_charge)
    if ion_series not in ("by", "cz"):
        raise ValueError(f"ion series must be 'by' or 'cz', got {ion_series!r}")
    charged = neutral_mass + z * LINE_PROTON
    if ion_series == "cz":
        if z < 3:
            raise ValueError(f"a c/z pair carries z-1 charges; z must be >= 3, got {z}")
        line_z, line_m = z - 1, neutral_mass + H_ATOM + (z - 1) * LINE_PROTON
    else:
        if z < 2:
            raise ValueError(f"a b/y pair needs z >= 2, got {z}")
        line_z, line_m = z, charged
    return DeconvolutionRequest(z, float(neutral_mass), ion_series, line_z, line_m,
                                topfd_max_charge=z, topfd_max_mass=charged)


def input_layout(ffc_path: str) -> dict:
    """How the deconvolution should read ``ffc_path``.

    It reads with ``header=None`` and assigns column NAMES by position, so the
    first line must be skipped and the names must be the file's own.  A
    prepared or exported table names its columns in its header; the legacy
    flat export starts with a single row-count line instead and uses the
    default column order.
    """
    with open(ffc_path) as fh:
        first = fh.readline().rstrip("\n").rstrip("\r")
    fields = first.split("\t")
    if len(fields) == 1 and fields[0].strip().lstrip("-").isdigit():
        return dict(skiprows=1, col_names=None, header="row-count line (legacy flat)")
    if {"m/z A", "m/z B"}.issubset(fields):
        need = {"Ranking", "Partial Cov."}
        missing = need - set(fields)
        if missing:
            raise ValueError(f"{ffc_path}: header lacks {sorted(missing)}")
        return dict(skiprows=1, col_names=fields, header="named columns")
    raise ValueError(f"{ffc_path}: cannot tell its layout -- expected a header naming "
                     "'m/z A' / 'm/z B' or a single row-count line; pass skiprows "
                     "explicitly to override")


def run(ffc_path: str, *, request: DeconvolutionRequest, output_dir: str, name: str,
        run_topfd: bool = True, params: dict | None = None,
        skiprows: int | None = None, col_names: list | None = None) -> dict:
    """Deconvolve one raw FFC table.

    Returns ``run_deconvolution``'s stats dict (the big DataFrames it also
    returns are dropped -- they are already on disk), with ``merged_path``
    naming the deliverable.  A metadata sidecar next to it records the line
    model that produced it, so a later run can check its chemistry.
    """
    p = dict(DEFAULTS, **(params or {}))

    topfd = paths.topfd_bin() if run_topfd else None
    if run_topfd and topfd is None:
        raise FileNotFoundError(
            "TopFD not found.  Set $DENOVOFFC_TOPFD_BIN (and "
            "$DENOVOFFC_TOPFD_LIB to its TopPIC shared libraries), put `topfd` "
            "on PATH, or pass --no-topfd to run Stage 1 alone.")

    layout = input_layout(ffc_path) if skiprows is None else dict(
        skiprows=skiprows, col_names=col_names, header="explicit skiprows")
    os.makedirs(output_dir, exist_ok=True)
    out = run_deconvolution(
        ffc_path=ffc_path,
        parent_charge=request.line_model_charge,
        parent_mass=request.line_model_charged_mass,
        output_dir=output_dir,
        output_prefix=name,
        theo_patt_path=paths.THEO_PATT,
        run_topfd=bool(topfd),
        topfd_bin=topfd or "topfd",
        topfd_max_charge=request.topfd_max_charge,
        topfd_max_mass=request.topfd_max_mass,
        topfd_env_vars=paths.topfd_env() or None,
        skiprows=layout["skiprows"],
        col_names=layout["col_names"],
        **p,
    )
    stats = dict(out["stats"])
    merged = stats.get("merged_path")
    if merged:
        write_json_atomic(merged + ".meta.json", dict(
            kind="deconvolved_ffc_table", source=os.path.basename(ffc_path),
            source_sha256=sha256_file(ffc_path), input_layout=layout["header"],
            line_model=dict(request.as_dict(), precursor_charge=request.precursor_charge),
            topfd=bool(topfd), parameters=p))
    stats["request"] = request.as_dict()
    return stats
