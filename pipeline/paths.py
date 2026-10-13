"""Where everything lives, and the sys.path bootstrap the vendored code needs.

``src/`` mirrors the directory layout the vendored modules were written
against, so every one of them imports its siblings unchanged.  Importing this
module makes that layout importable regardless of the cwd.
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "src")

DEISO       = os.path.join(SRC, "deiso", "python_code")
LINE_BASED  = os.path.join(SRC, "line_based_annotation")
THEORETICAL = os.path.join(SRC, "theoretical_mms_annotation")
RECONSTRUCT = os.path.join(SRC, "reconstruct")
DATA_PREP   = os.path.join(SRC, "data_prep")
DECONV      = os.path.join(SRC, "deconvolution")
ANNOTATION  = os.path.join(SRC, "annotation_pepline")
DE_NOVO     = os.path.join(SRC, "de_novo")
ETD         = os.path.join(SRC, "ETD")

for _p in (DEISO, LINE_BASED, THEORETICAL, RECONSTRUCT, DATA_PREP,
           DECONV, ANNOTATION, DE_NOVO, ETD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The packages written for this repository (``rbdp``, ``combination``,
# ``gap_scoring``) are imported as packages from ``src/``.  Appended LAST, so a
# flat vendored module always wins over a directory of the same name.
if SRC not in sys.path:
    sys.path.append(SRC)

#: The C++ de novo engine.  Built by ``cpp/build.sh``; absent until then, in
#: which case the pipeline falls back to the Python reference.
CPP_ENGINE = os.path.join(ROOT, "bin", "denovo_ffc")

#: The compound-residue-mass table the Python reference needs.  The C++ engine
#: generates its own in ~80 ms and never reads this file (DESIGN.md 6.1).
EDGES_JSON = os.path.join(RECONSTRUCT, "edges.json")

#: TopPIC's theoretical isotope-pattern table (Stage-1 monoisotopic refinement).
THEO_PATT = os.path.join(DEISO, "theo_patt.txt")


def cpp_engine(required: bool = False) -> str | None:
    """Path to the C++ engine, or None when it has not been built."""
    if os.path.isfile(CPP_ENGINE) and os.access(CPP_ENGINE, os.X_OK):
        return CPP_ENGINE
    if required:
        raise FileNotFoundError(
            f"{CPP_ENGINE} not found -- build it with:  cpp/build.sh")
    return None


def topfd_bin() -> str | None:
    """TopFD (Stage 2 of the deconvolution).

    Deliberately NOT vendored: it is a ~38 MB binary plus shared libraries
    belonging to TopPIC, and pinning a copy here would silently fork someone
    else's release.  Resolution order:

        $DENOVOFFC_TOPFD_BIN  ->  `topfd` on PATH  ->  None

    None is not fatal -- ``--no-topfd`` runs Stage 1 alone.
    """
    env = os.environ.get("DENOVOFFC_TOPFD_BIN")
    if env and os.path.isfile(env):
        return env
    return shutil.which("topfd")


def topfd_env() -> dict:
    """Extra environment TopFD needs (its TopPIC shared libraries).

    Set ``$DENOVOFFC_TOPFD_LIB`` to the directory holding them; TopFD exits at
    load time without it.
    """
    lib = os.environ.get("DENOVOFFC_TOPFD_LIB")
    if not lib:
        return {}
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": f"{lib}:{existing}" if existing else lib}


def ensure_edges_json(verbose: bool = True) -> str:
    """Build ``reconstruct/edges.json`` if it is missing (the Python path
    needs it; it is 114 MB, so it is generated rather than committed)."""
    if os.path.isfile(EDGES_JSON):
        return EDGES_JSON
    if verbose:
        print(f"  edges.json missing -- generating {EDGES_JSON} (~30 s, one time)")
    import edges as _edges                                    # noqa: E402
    _edges.build_and_save(max_n=20, path=EDGES_JSON)
    return EDGES_JSON
