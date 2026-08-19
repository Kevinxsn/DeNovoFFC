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

from . import paths
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


def run(ffc_path: str, *, parent_charge: int, parent_mass_charged: float,
        output_dir: str, name: str, run_topfd: bool = True,
        params: dict | None = None, skiprows: int = 1,
        col_names: list | None = None) -> dict:
    """Deconvolve one raw FFC table.

    ⚠ ``parent_mass_charged`` is the CHARGED mass [M+zH]z+, not the neutral
    mass.  The line model has no proton term (v = i*mzA + j*mzB), so a parental
    line sums to the charged mass; handing it the neutral one puts every line's
    shift off by z * proton.

    Returns ``run_deconvolution``'s stats dict (the big DataFrames it also
    returns are dropped -- they are already on disk), with ``merged_path``
    naming the deliverable.
    """
    p = dict(DEFAULTS, **(params or {}))

    topfd = paths.topfd_bin() if run_topfd else None
    if run_topfd and topfd is None:
        raise FileNotFoundError(
            "TopFD not found.  Set $DENOVOMMS_TOPFD_BIN (and "
            "$DENOVOMMS_TOPFD_LIB to its TopPIC shared libraries), put `topfd` "
            "on PATH, or pass --no-topfd to run Stage 1 alone.")

    os.makedirs(output_dir, exist_ok=True)
    out = run_deconvolution(
        ffc_path=ffc_path,
        parent_charge=parent_charge,
        parent_mass=parent_mass_charged,
        output_dir=output_dir,
        output_prefix=name,
        theo_patt_path=paths.THEO_PATT,
        run_topfd=bool(topfd),
        topfd_bin=topfd or "topfd",
        topfd_max_charge=parent_charge,
        topfd_max_mass=parent_mass_charged,
        topfd_env_vars=paths.topfd_env() or None,
        skiprows=skiprows,
        col_names=col_names,
        **p,
    )
    return dict(out["stats"])
