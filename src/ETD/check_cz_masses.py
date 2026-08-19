#!/usr/bin/env python3
"""Validate the c/z-dot fragment index against the collaborator's own 1D annotation.

``Intensity_Sum.LL37_ETD_Z6_1ms`` was annotated independently (by the group that
produced the data), so its ``c9 (+2)`` / ``z.9 (+2)`` labels are a ground truth
for the mass model in ``cz_series.py``.  This script re-derives each labelled
peak from our own peptide model and reports the deviation, which is the only
thing that can catch a wrong constant -- a 1.008 Da slip in the internal offset
or a c/c-dot mix-up produces a perfectly plausible-looking pipeline that
annotates nothing.

    python check_cz_masses.py
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "deiso", "python_code")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "data_prep")))

import cz_series  # noqa: E402
from peptide import Pep  # noqa: E402
from peptide_info import resolve as resolve_name  # noqa: E402

INTENSITY = "/home/sux002/data/ETD_LLG/Intensity_Sum.LL37_ETD_Z6_1ms"
NAME = "LLG_37_Z6_ETD"

#: "c9 (+2)", "z.9 (+2)", "b2 (+1)", "y17 (+5) (1C13)" -- only the *first*
#: alternative of a '/'-separated list is checked, and only unambiguous
#: monoisotopic terminal ions (no C13 tag) are used.
_TERMINAL = re.compile(r"^(c|c\.|z|z\.|z'|b|y)(\d+)\s+\(\+(\d+)\)$")


def theoretical(pep, series: str, num: int, charge: int) -> float:
    """m/z of the named terminal ion at ``charge``, from our own model."""
    n = len(pep.AA_array)
    if series in ("b", "c", "c."):
        neutral = pep.ion_mass(f"b{num}") - cz_series.PROTON
        if series == "c":
            neutral += cz_series.NTERM_OFFSET
        elif series == "c.":                       # c-dot = c - H
            neutral += cz_series.NTERM_OFFSET - cz_series.H_ATOM
    else:
        neutral = pep.ion_mass(f"y{num}") - cz_series.PROTON
        if series == "z":                          # Biemann z = y - NH3
            neutral -= cz_series.NH3
        elif series == "z.":                       # z-dot = z + H  (the ETD ion)
            neutral += cz_series.CTERM_OFFSET
        elif series == "z'":                       # z' = z + 2H
            neutral += cz_series.CTERM_OFFSET + cz_series.H_ATOM
    if num >= n:
        return float("nan")
    return (neutral + charge * cz_series.PROTON) / charge


def main() -> int:
    info = resolve_name(NAME)
    seq, z = info["SEQUENCE"], info["PARENT_CHARGE"]
    pep = Pep(f"[{seq}+{z}H]{z}+", end_h20=True)
    print(f"{seq}  ({len(seq)} residues, {z}+)\n")

    by_series: dict[str, list[float]] = {}
    with open(INTENSITY) as fh:
        for line in fh:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            mz = float(parts[1])
            label = parts[3].strip().split(" / ")[0]
            m = _TERMINAL.match(label)
            if not m:
                continue
            series, num, charge = m.group(1), int(m.group(2)), int(m.group(3))
            theo = theoretical(pep, series, num, charge)
            if not np.isfinite(theo):
                continue
            by_series.setdefault(series, []).append((mz - theo) * 1e6 / theo)

    print(f"{'series':8} {'n':>5}  {'median ppm':>11} {'max |ppm|':>10}")
    ok = True
    for series in ("b", "y", "c", "c.", "z", "z.", "z'"):
        devs = by_series.get(series)
        if not devs:
            continue
        med, mx = float(np.median(devs)), float(np.max(np.abs(devs)))
        flag = "" if abs(med) < 20 else "   <-- MISMATCH"
        if abs(med) >= 20:
            ok = False
        print(f"{series:8} {len(devs):>5}  {med:>11.2f} {mx:>10.2f}{flag}")

    print("\nc/z-dot offsets in force:")
    print(f"   b -> c        {cz_series.NTERM_OFFSET:+.5f}")
    print(f"   y -> z-dot    {cz_series.CTERM_OFFSET:+.5f}")
    print(f"   internal      {cz_series.INTERNAL_OFFSET:+.5f}")
    print(f"   c + z. - M    {cz_series.PARENT_OFFSET:+.5f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
