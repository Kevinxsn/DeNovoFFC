"""The ion series an ExD spectrum can contain, and how to swap between them.

Motivation
----------
A pure-ETD map of LL37 6+ carries strong `-27.99 Da` lines on **both** the
`(2,3)` and `(1,4)` charge states -- far too strong to be spurious, and matching
no named neutral loss. They are **a/y pairs**, and the a fragment carries the
hydrogen atom that electron transfer delivered:

    857.757 (4+) // 1037.604 (1+)   ==  y28        // a9  (+1H)
    711.921 (2+) // 1014.929 (3+)   ==  a12 (+1H)  // y25

Verified against our own peptide model at **+1.5 / +2.3 ppm** and **+0.4 /
+1.3 ppm**.  (Mechanism: J. Am. Soc. Mass Spectrom. 22(3) 480, "Unusual
Fragmentation of Linked Peptides by ExD" -- a/y formation in ExD is a genuine
channel, not CO loss from a b ion, which is never formed.)

⚠ **A plain b/y map cannot find these.** Without the transferred hydrogen,
`a9` sits 1.008 Da away -- **975 ppm**, ~30x the matching window:

    a9 (+1H)   theo 1037.6016   obs 1037.6040    +2.3 ppm    <- match
    a9 (plain) theo 1036.5938   obs 1037.6040   +975.6 ppm   <- miss

So "annotate the leftovers with b/y" needs *two* b/y-family maps, not one: the
ETD a/y family (`by_h`, N-terminal fragment + H) and the ordinary HCD one
(`by`, no H anywhere).

What each series sums to
------------------------
The pair sum is what shows up as a line, so it is the thing to reason about:

| series | N-term frag | C-term frag | pair sum | line `A` |
|---|---|---|---|---|
| `cz`   | c  = b + NH3   | z. = y + H - NH3 | M + H      | 0 |
| `by_h` | b + H          | y                | M + H      | 0 |
| `by`   | b              | y                | M          | H = 1.008 |

`by_h` needs no separate a-ion: the existing `-CO` neutral loss turns
`b + H` into `a + H`, so the observed a/y pairs come out as `(b9+H)-CO, y28`
and classify as `parental*`.

⚠ **`by` and an isotope satellite are 0.0045 Da apart.** A plain b/y pair sits
at `A = 1.00783` and a `+1` C13 satellite of the c/z ladder at `A = 1.00335`.
Line detection here carries 0.01-0.03 Da of spread, so a `-1`-ish line
(`(2,3) -0.977` in ETD, `(2,3) -0.996` in EThcD) **cannot be told apart from the
two by its shift alone** -- it has to be decided by annotating its FFCs.  Both
`etd.md` §6a-i (which currently reads them as isotope satellites) and this file
are describing the same lines; which reading is right is an open question the
dual-series run is the right experiment for.

Usage
-----
    from series_maps import use_series
    with use_series("by_h"):
        result = ffc_annotator_cz.annotate_ffcs_with_charge(...)
"""

from __future__ import annotations

import contextlib

import cz_series

import ffc_annotator_cz

#: name -> the five knobs ``ffc_annotator_cz.ACTIVE`` reads.
#:   nterm/cterm : added to the b-ion / y-ion neutral mass
#:   internal    : added to an internal fragment's residue sum
#:   parent      : what a complementary pair sums to, minus M
#:   n_label/c_label/suffix : how the labels print
SERIES = {
    # The ETD backbone series.  This is the default state of the annotator.
    "cz": dict(
        nterm=cz_series.NTERM_OFFSET,       # c  = b + NH3
        cterm=cz_series.CTERM_OFFSET,       # z. = y + H - NH3
        internal=cz_series.INTERNAL_OFFSET,  # both cut ends -> +H
        parent=cz_series.PARENT_OFFSET,     # c + z. = M + H
        n_label="c", c_label="z", suffix="",
        note="c / z-dot -- the ETD backbone series",
    ),
    # The ExD a/y channel.  The transferred hydrogen sits on the N-terminal
    # fragment; `-CO` then gives the a ion, so no separate a series is needed.
    "by_h": dict(
        nterm=cz_series.H_ATOM,             # b + H   (and a + H via -CO)
        cterm=0.0,                          # y unchanged
        internal=cz_series.H_ATOM,          # the H is on the fragment
        parent=cz_series.H_ATOM,            # (b+H) + y = M + H
        n_label="b", c_label="y", suffix="+H",
        note="ETD a/y family -- N-terminal fragment carries the transferred H",
    ),
    # Ordinary collision-induced b/y, no hydrogen transfer.  Expected in EThcD
    # from the supplemental HCD; possible in 'pure' ETD wherever collisional
    # activation is not fully avoided.
    "by": dict(
        nterm=0.0, cterm=0.0, internal=0.0,
        parent=0.0,                         # b + y = M
        n_label="b", c_label="y", suffix="",
        note="plain b / y -- collision-induced, no H transfer",
    ),
}

#: Order the fallback passes are tried in when the caller does not say.
#: `by_h` first because it is the one the -27.99 lines actually need.
DEFAULT_FALLBACK = ("by_h", "by")


@contextlib.contextmanager
def use_series(name: str, mod=ffc_annotator_cz):
    """Run ``mod`` (the annotator) with ion series ``name`` installed.

    Restores the previous series on exit, including on exception, so a caller
    that forgets to reset cannot poison a later pass.
    """
    if name not in SERIES:
        raise KeyError(f"unknown series {name!r}; have {', '.join(SERIES)}")
    saved = dict(mod.ACTIVE)
    saved_re, saved_role = mod._BY_LABEL_RE, mod._SERIES_ROLE
    try:
        spec = SERIES[name]
        mod.ACTIVE.update({k: spec[k] for k in
                           ("nterm", "cterm", "internal", "parent",
                            "n_label", "c_label", "suffix")})
        mod.rebuild_label_rules()
        yield spec
    finally:
        mod.ACTIVE.clear()
        mod.ACTIVE.update(saved)
        mod._BY_LABEL_RE, mod._SERIES_ROLE = saved_re, saved_role


def describe() -> str:
    """One line per series, for a run header."""
    out = []
    for name, spec in SERIES.items():
        out.append(f"  {name:6} pair sum = M {spec['parent']:+.5f}   "
                   f"N{spec['nterm']:+.5f} / C{spec['cterm']:+.5f}   "
                   f"{spec['note']}")
    return "\n".join(out)
