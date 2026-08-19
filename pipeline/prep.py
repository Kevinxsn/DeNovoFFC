"""Loading an FFC table, and the one preprocessing pass every route shares.

``merge_duplicate_ffcs_charge`` collapses mirror duplicates -- FFC(X, Y) and
FFC(Y, X) are one physical correlation, and the deconvolution's A/B ordering
is arbitrary -- then ``prepare_ffc_data`` drops the unranked rows and takes the
best-ranked top-N.

Both stay in Python.  They run once per peptide on a table of ~15 k rows, so
they are not a bottleneck, and the merge's KD-tree proximity pass is the one
piece with a real third-party dependency (scipy).  See DESIGN.md 4.2.
"""
from __future__ import annotations

import os

import pandas as pd

from . import paths  # noqa: F401  (sys.path bootstrap)
from merge_ffcs import merge_duplicate_ffcs_charge      # noqa: E402
from line_finding import prepare_ffc_data               # noqa: E402

#: ⚠ The merge's m/z proximity radius is NOT the same on the two routes, and
#: it is not cosmetic -- it changes how many rows survive, hence which FFCs the
#: top-N cut admits, hence which lines are detected (line detection is
#: non-monotonic in top-N).  On KWK 5+ the two give 13 731 vs 13 684 rows and a
#: master set of 22 vs 23 breakpoints.
#:
#: Each value is the one its published benchmark was measured at, so both stay:
#:   de novo     0.001  (denovo.md 5 -- run_denovo.py leaves the function default)
#:   annotation  0.002  (annotation_pipeline.md -- run_annotation_reconstruct.py)
MERGE_TOLERANCE = {"denovo": 0.001, "annotation": 0.002}
DEFAULT_MERGE_TOLERANCE = MERGE_TOLERANCE["denovo"]


def read_table(path: str) -> pd.DataFrame:
    """Read an FFC table.  Separator is taken from the extension."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FFC table not found: {path}")
    sep = "," if path.endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep)


def is_deconvolved(df: pd.DataFrame) -> bool:
    """Does this table already carry per-side charges?

    A raw MMS map is m/z pairs with covariance and a ranking and nothing else;
    the deconvolution is what adds ``charge_A`` / ``charge_B``.  This is what
    ``--deconvolve auto`` keys on.
    """
    return {"charge_A", "charge_B"}.issubset(df.columns)


def merge(df: pd.DataFrame, route: str = "denovo",
          tolerance: float | None = None) -> pd.DataFrame:
    """Collapse mirror duplicates.  See MERGE_TOLERANCE for why `route` matters."""
    if tolerance is None:
        tolerance = MERGE_TOLERANCE.get(route, DEFAULT_MERGE_TOLERANCE)
    return merge_duplicate_ffcs_charge(df, symmetric=True, reset_ranking=True,
                                       tolerance=tolerance)


def top_n(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return prepare_ffc_data(df, top_n=n)
