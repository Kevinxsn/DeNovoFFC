"""Import-path bootstrap for the ``deconvolution/`` package.

Every module here starts with ``import _paths  # noqa: F401`` so the sibling
project folders it borrows code from are importable regardless of the cwd the
pipeline was launched from.

Nothing in those folders is modified -- this package only *imports* from them,
so the existing ``deiso/python_code`` and ``line_based_annotation`` pipelines
keep working exactly as before.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

#: Stage-1 deconvolution back end, TopFD runner, greedy_line primitives.
DEISO = os.path.join(BASE, "deiso", "python_code")
#: The orientation-agnostic line detector (single source of truth -- see
#: ``line_based_annotation/line_based.md`` section 4).
LINE_BASED = os.path.join(BASE, "line_based_annotation")
#: ``peptide_info.resolve`` -- NAME -> sequence / charge / mass / C-terminus.
DATA_PREP = os.path.join(BASE, "data_prep")

for _p in (HERE, DEISO, LINE_BASED, DATA_PREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: TopPIC's theoretical isotope-pattern table, used by the Stage-1 back end for
#: monoisotopic refinement.  Absolute, so the cwd never matters.
THEO_PATT = os.path.join(DEISO, "theo_patt.txt")

#: Shipped TopFD binary (v1.8.1) and the shared libraries it needs.
TOPFD_BIN = os.path.join(BASE, "deiso", "bin", "topfd")
TOPFD_LIB = os.path.join(BASE, "deiso", "lib", "toppic", "toppic")
