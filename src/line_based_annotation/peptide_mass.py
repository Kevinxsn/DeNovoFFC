"""
Quick peptide-mass calculator
=============================
Give it a sequence (modifications allowed, e.g. ``VTIMPKAcDIQLAR``), the
C-terminal *residual* group, and a charge; it prints the masses.

The heavy lifting is already done by ``deiso/python_code/peptide.py``:
``Pep`` parses the modified sequence into an ``AA`` array and sums residue +
PTM masses.  This module only wraps it in a one-call / CLI interface.

Usage
-----
    python peptide_mass.py YLEFISDAIIHVLHSK 2
    python peptide_mass.py EQFDDYGHMRF 1 --residual NH2

    >>> from peptide_mass import peptide_mass
    >>> peptide_mass("YLEFISDAIIHVLHSK", charge=2)["mz"]
"""

from __future__ import annotations

import argparse
import os
import sys

_DEISO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "deiso", "python_code")
sys.path.insert(0, os.path.abspath(_DEISO))

import peptide            # noqa: E402
from peptide import AA    # noqa: E402

PROTON = AA.element_masses["proton"]

#: C-terminal residual groups understood by ``Pep`` (``end_h20`` argument).
#: True == free acid (-OH, +H2O); a key of ``AA.element_masses`` == that group.
RESIDUALS = {
    "H2O": True,        # free acid, the default
    "OH": True,
    "free": True,
    "NH2": "NH2",       # C-terminal amide
    "NH3": "NH3",
}


def peptide_mass(seq: str, charge: int = 1, residual: str = "H2O") -> dict:
    """Masses of ``seq`` at charge ``charge`` with C-terminal ``residual``.

    Returns a dict with:
        neutral  monoisotopic neutral peptide mass M
        mh       singly-protonated [M+H]+ mass
        mz       observed m/z of [M + z H]z+
        charged  total mass of the z-protonated species (z * mz)
        length   number of residues
    """
    if charge < 1:
        raise ValueError("charge must be >= 1")
    try:
        end = RESIDUALS[residual]
    except KeyError:
        raise ValueError(
            f"unknown residual {residual!r}; choose from {sorted(RESIDUALS)}"
        ) from None

    # Pep's raw-sequence grammar is "[SEQ+nH]n+"; it adds n protons itself, so
    # ask for the singly-protonated form and derive the rest arithmetically.
    pep = peptide.Pep(f"[{seq}+1H]1+", end_h20=end)
    mh = pep.pep_mass                    # [M+H]+
    neutral = mh - PROTON
    return {
        "neutral": neutral,
        "mh": mh,
        "mz": (neutral + charge * PROTON) / charge,
        "charged": neutral + charge * PROTON,
        "length": len(pep.AA_array),
        "pep": pep,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="peptide sequence, modifications allowed")
    ap.add_argument("charge", nargs="?", type=int, default=1,
                    help="precursor charge state (default 1)")
    ap.add_argument("--residual", "-r", default="H2O",
                    choices=sorted(RESIDUALS),
                    help="C-terminal residual group (default H2O = free acid)")
    args = ap.parse_args(argv)

    r = peptide_mass(args.sequence, args.charge, args.residual)
    z = args.charge
    print(f"sequence     : {args.sequence}  ({r['length']} residues, "
          f"C-term {args.residual})")
    print(f"neutral  M   : {r['neutral']:.4f}")
    print(f"[M+H]+       : {r['mh']:.4f}")
    print(f"[M+{z}H]{z}+ mass : {r['charged']:.4f}")
    print(f"[M+{z}H]{z}+ m/z  : {r['mz']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
