"""Resolve a dataset NAME into the peptide parameters the pipelines need.

The dataset names (``HAD_33_Z6_E20``, ``VEA_15_Z3_revised``, ...) already encode
everything: the first two tokens identify the peptide and the ``Z`` token is the
parent charge.  ``pep_seq.xlsx`` maps that identifier to a sequence and a
C-terminal group, and ``peptide_mass.py`` turns those into a mass.

    $ python peptide_info.py HAD_33_Z6_E20
    SEQUENCE='HADGSFSDEMNTILDNLAARDFINWLIQTKITD'
    PARENT_CHARGE=6
    PARENT_MASS=3769.8586
    END=h2o
    RESIDUAL=H2O
    PEPTIDE_ID=HAD_33

Output is shell-eval-able, so a runner script can do::

    eval "$(python peptide_info.py "$NAME")"

A note on the C-terminus: ``pep_seq.xlsx`` writes a C-terminal amide as
``-NH2``, but in this codebase's vocabulary an amide is ``NH3`` (the residual
*added* to the residue sum).  ``peptide.AA.element_masses['NH2']`` is a further
proton lighter and is NOT the amide -- so ``-NH2`` in the sheet maps to
``NH3`` here.  Getting this backwards shifts the parent mass by 1.00725.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "line_based_annotation")))

from peptide_mass import peptide_mass  # noqa: E402

PEP_SEQ_XLSX = os.path.join(_HERE, "pep_seq.xlsx")

#: How pep_seq.xlsx spells a C-terminus -> (peptide_mass residual, --end flag).
#: "-NH2" in the sheet means a C-terminal amide, which is `NH3` here.
CTERM = {
    "OH": ("H2O", "h2o"),
    "H2O": ("H2O", "h2o"),
    "NH2": ("NH3", "nh3"),
    "NH3": ("NH3", "nh3"),
}

_NAME_RE = re.compile(r"^([A-Za-z]{3})_(\d+)_[Zz](\d+)")
_PEPTIDE_RE = re.compile(r"^([A-Za-z]+)-(NH2|NH3|OH|H2O)(\d+)\+$")


def parse_name(name):
    """``HAD_33_Z6_E20`` -> ("HAD33", "HAD_33", 6)."""
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(
            f"cannot parse dataset name {name!r}; expected e.g. HAD_33_Z6_E20"
        )
    tag, length, charge = m.group(1).upper(), m.group(2), int(m.group(3))
    return tag + length, f"{tag}_{length}", charge


def load_sequences(xlsx=PEP_SEQ_XLSX):
    """Shorthand-prefix -> (sequence, cterm), e.g. "HAD33" -> (..., "OH")."""
    df = pd.read_excel(xlsx)
    out = {}
    for shorthand, peptide in zip(df["Shorthand"], df["Peptide"]):
        shorthand, peptide = str(shorthand).strip(), str(peptide).strip()
        m = _PEPTIDE_RE.match(peptide)
        if not m:
            raise ValueError(f"cannot parse peptide cell {peptide!r}")
        seq, cterm, charge = m.group(1), m.group(2).upper(), m.group(3)
        # The shorthand is <tag><length><charge>; strip the trailing charge to
        # get the peptide key, so HGT142 and HGT143 collapse to HGT14.
        if not shorthand.endswith(charge):
            raise ValueError(
                f"shorthand {shorthand!r} does not end with its charge {charge!r}"
            )
        key = shorthand[: -len(charge)]
        if key in out and out[key] != (seq, cterm):
            raise ValueError(
                f"conflicting sequences for {key!r}: {out[key]} vs {(seq, cterm)}"
            )
        out[key] = (seq, cterm)
    return out


def resolve(name, xlsx=PEP_SEQ_XLSX):
    key, peptide_id, charge = parse_name(name)
    table = load_sequences(xlsx)
    if key not in table:
        raise KeyError(
            f"no sequence for {key!r} (from {name!r}); "
            f"known: {', '.join(sorted(table))}"
        )
    seq, cterm = table[key]
    residual, end_flag = CTERM[cterm]
    masses = peptide_mass(seq, charge=charge, residual=residual)
    return {
        "NAME": name,
        "PEPTIDE_ID": peptide_id,
        "SEQUENCE": seq,
        "PARENT_CHARGE": charge,
        # The pipelines want the charged mass [M+zH]z+, not the neutral mass.
        "PARENT_MASS": round(masses["charged"], 4),
        "NEUTRAL_MASS": round(masses["neutral"], 4),
        "PARENT_MZ": round(masses["mz"], 4),
        "END": end_flag,
        "RESIDUAL": residual,
        "CTERM": cterm,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", help="Dataset name, e.g. HAD_33_Z6_E20")
    ap.add_argument("--field", help="Print just this field's value")
    ap.add_argument("--xlsx", default=PEP_SEQ_XLSX)
    ap.add_argument("--list", action="store_true",
                    help="List the peptides known to the sheet and exit")
    args = ap.parse_args(argv)

    if args.list:
        for key, (seq, cterm) in sorted(load_sequences(args.xlsx).items()):
            print(f"{key:10s} {cterm:4s} {seq}")
        return 0

    if not args.name:
        ap.error("a dataset name is required (or use --list)")

    try:
        info = resolve(args.name, args.xlsx)
    except (ValueError, KeyError) as exc:
        print(f"peptide_info: {exc.args[0]}", file=sys.stderr)
        return 2
    if args.field:
        if args.field not in info:
            ap.error(f"unknown field {args.field!r}; have {', '.join(info)}")
        print(info[args.field])
        return 0

    for key in ("SEQUENCE", "PARENT_CHARGE", "PARENT_MASS", "END", "RESIDUAL",
                "PEPTIDE_ID", "NEUTRAL_MASS", "PARENT_MZ"):
        value = info[key]
        print(f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
