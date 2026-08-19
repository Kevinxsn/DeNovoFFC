"""
=============================================================================
ETD VARIANT -- c / z-dot ion series
=============================================================================
**COPY of ``../theoretical_mms_annotation/ffc_annotator.py`` (2026-08-03) with
the b/y ion series replaced by c/z-dot.**  Nothing in
``theoretical_mms_annotation/`` is modified.

Four edits, each marked ``# [CZ]``:
  1. ``generate_theoretical_ions`` -- c = b + NH3, z-dot = y + H - NH3,
     internal = sum(residues) + H (an ETD internal is cut at both ends, so it
     carries the z-dot end and the c end, netting one hydrogen).
  2. ``compute_peptide_mass`` -- ``ParentMass`` for the ``offset`` column
     becomes M + H, because a c/z-dot pair sums to M + H rather than M.
  3. Labels ``b{i}`` / ``y{i}`` print as ``c{i}`` / ``z{i}``.  ``ion_type``
     keeps the b/y vocabulary so every compatibility and classification rule
     (``is_compatible``, ``classify_ffc_relationship``) is untouched -- the
     *geometry* of a c/z pair is that of a b/y pair.
  4. ``_classify_ion_side`` matches ``^([cz])\\d`` instead of ``^([by])\\d``,
     and ``raw_breakpoints`` reads the parental breakpoint off the c ion by
     ``ion_type`` instead of assuming ion_a.

⚠ Edit 4 is the one to understand before touching this file again.  Almost
everything here keys on ``ion_type``, which edit 3 deliberately leaves as
'b'/'y' -- but ``_classify_ion_side`` is the single place that keys on the
printed **label**, and edit 3 changed those.  It was missed initially, and it
took out two things at once: the b-in-X / y-in-Y placement stopped firing (so a
z ion could sit in the X column), and ``raw_breakpoints`` -- which documents a
dependency on that placement -- then read a parental breakpoint off the z ion,
reporting the peptide length n for every affected row.  Neither raised.
**If you add another label-derived rule, it needs the c/z spelling too.**

The column convention this produces: **c ions in X, z ions in Y**, and the
``breaking point`` column is the c-ion position, i.e. the residue after which
the backbone cleaved -- the same bond index as the ``c{k}, z{n-k}`` rows in
``line_internal_annotation_cz.py``'s coverage table.

See ``cz_series.py`` for the derivation and the validation against the
collaborator's own 1D annotation.

Original docstring follows.
-----------------------------------------------------------------------------

FFC (Fragment-Fragment Correlation) Annotator

Implements the theoretical MS/MS spectrum (Supplemental Note 10) and the
theoretical FFC map (Supplemental Note 11) for a peptide, and uses them to
annotate observed FFC pairs (2D) or single peaks (1D).

Differences from the earlier naive version (deiso/python_code/simple_annot.py):
  - Neutral losses restricted to H2O, NH3, CO only (the first three of the
    ordered set H2O, NH3, CO, CO2, HCOOH). Applies to b, y, and internal
    ions alike.
  - No double neutral losses: each fragment carries at most one loss.
  - isomax(FragmentMass) is looked up from the Averagine-based table
    (james_code/Averagine_Envelopes) via averagine_isomax.py, rather than a
    fixed constant.
  - FFC annotation (2D) does not just match m/z A and m/z B independently:
    it requires the two matched fragments to be a *compatible* pair per
    Note 11 (non-overlapping, charge sum <= precursor charge, at most one
    shared neutral loss, and either adjacent or both terminal).
  - After matching, a b ion is always placed in Interpretation A and a y
    ion in Interpretation B (m/z, label, and error swapped together if
    needed). Internal-ion, "No Match", and ambiguous (mixed b/y candidate
    list) labels are left wherever they landed, unless swapping is needed
    to correctly place a b or y ion found on the other side.

annotate_ffcs_with_charge() extends annotate_ffcs() with two more behaviors,
for input that also carries per-side charge assignments from line-finding /
deconvolution (columns 'charge_A', 'charge_B'):
  - Charge-constrained matching: when charge_A/charge_B is not NaN for a
    row, only theoretical ions of that exact charge are considered for
    that side (both in the joint pair match and the fallback below).
  - Single-m/z fallback: if no compatible FFC pair is found jointly, each
    side is still independently annotated against the theoretical
    spectrum (Note 10, no pair-compatibility requirement) instead of both
    sides reporting "No Match". If that side's charge is known, the single
    best match is reported; if unknown, every matching candidate is listed,
    one per line (see AMBIGUOUS_SEPARATOR), since charge is the only way to
    disambiguate here.

Usage (2D - FFC pairs):
    import pandas as pd
    from ffc_annotator import annotate_ffcs

    df_pairs = pd.DataFrame({'m/z A': [...], 'm/z B': [...], 'Ranking': [...]})
    result = annotate_ffcs(
        df_pairs,
        sequence="PEPTIDESEQUENCE",
        precursor_charge=3,
        output_path="annotated_ffcs.xlsx",
    )

Usage (2D - FFC pairs, charge-aware with single-m/z fallback):
    from ffc_annotator import annotate_ffcs_with_charge

    df_pairs = pd.DataFrame({
        'm/z A': [...], 'm/z B': [...], 'Ranking': [...],
        'charge_A': [...], 'charge_B': [...],  # may contain NaN
    })
    result = annotate_ffcs_with_charge(
        df_pairs,
        sequence="PEPTIDESEQUENCE",
        precursor_charge=3,
        output_path="annotated_ffcs_charge.xlsx",
        extra_info=True,  # adds TFFC A/B, ParentMass - TotalMass, Annotation
    )

Usage (1D - single peak list, Note 10 spectrum only, no FFC compatibility):
    from ffc_annotator import annotate_peaks

    df_peaks = pd.DataFrame({'m/z': [175.119, 300.156, ...]})
    result = annotate_peaks(
        df_peaks,
        sequence="PEPTIDESEQUENCE",
        precursor_charge=3,
        output_path="annotated_peaks.xlsx",
    )
"""

import bisect
import os
import re
import sys
from collections import namedtuple

# [CZ] This copy lives outside theoretical_mms_annotation/, so point at that
# directory for the shared Averagine table and at this one for cz_series.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "theoretical_mms_annotation")))

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

from averagine_isomax import AveragineIsoMaxTable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTON = 1.00727646677

AA_MASSES = {
    'G': 57.02146, 'A': 71.03711, 'V': 99.06841, 'L': 113.08406,
    'I': 113.08406, 'P': 97.05276, 'F': 147.06841, 'W': 186.07931,
    'M': 131.04049, 'S': 87.03203, 'T': 101.04768, 'C': 103.00919,
    'Y': 163.06333, 'H': 137.05891, 'D': 115.02694, 'E': 129.04259,
    'N': 114.04293, 'Q': 128.05858, 'K': 128.09496, 'R': 156.10111,
}

# Neutral losses for b/y ions: only H2O, NH3, CO (first three of the ordered
# set H2O, NH3, CO, CO2, HCOOH from Supplemental Note 10), single loss only.
NEUTRAL_LOSSES = {
    '': 0.0,
    '-H2O': 18.01056,
    '-NH3': 17.02655,
    '-CO': 27.99491,
}

C13_SHIFT = 1.003355

WATER = 18.01056
NH3 = 17.02655

RESIDUAL_MASSES = {
    'H2O': WATER,
    'NH3': NH3,
}

DEFAULT_ISOMAX_CUTOFF_PCT = 5.0
DEFAULT_TOL_PPM = 20

# Separator between candidate labels in an ambiguous multi-candidate cell.
# A newline (rather than " / ") so each candidate renders on its own line
# in a spreadsheet cell once wrap_text is enabled (see _write_xlsx()).
AMBIGUOUS_SEPARATOR = "\n"


# ---------------------------------------------------------------------------
# Peptide parsing helpers
# ---------------------------------------------------------------------------

def parse_sequence(sequence):
    """Parse a peptide sequence into a list of (residue, mass) tuples."""
    residues = []
    seq = sequence.upper()
    for i, aa in enumerate(seq):
        if aa not in AA_MASSES:
            raise ValueError(f"Unknown amino acid: {aa} at position {i} in {sequence}")
        residues.append((aa, AA_MASSES[aa]))
    return residues


# [CZ] The c / z-dot offsets.  ``cz_series`` lives beside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cz_series import (  # noqa: E402
    NTERM_OFFSET as CZ_NTERM, CTERM_OFFSET as CZ_CTERM,
    INTERNAL_OFFSET as CZ_INTERNAL, PARENT_OFFSET as CZ_PARENT,
)

# [CZ] ---- The ACTIVE ion series ------------------------------------------
# Everything series-dependent in this file reads these five values, so ONE
# annotator can serve c/z-dot, plain b/y, and the ETD a/y family instead of
# three drifting copies.  ``series_maps.use_series()`` swaps them under a
# context manager; the defaults below reproduce the c/z behaviour EXACTLY, so
# any caller that does not touch ACTIVE is unaffected.
#
#   nterm/cterm  offsets added to the b-ion / y-ion neutral mass
#   internal     offset added to an internal fragment's residue sum
#   parent       what a complementary pair sums to, minus M
#   n_label/c_label   printed series letters (b/y -> c/z, etc.)
#   suffix       appended to a terminal label, e.g. "+H"
ACTIVE = {
    "nterm": CZ_NTERM, "cterm": CZ_CTERM, "internal": CZ_INTERNAL,
    "parent": CZ_PARENT, "n_label": "c", "c_label": "z", "suffix": "",
}


def compute_prefix_sums(residues):
    """prefix[i] = sum of residue masses [0..i-1] (so b_i = prefix[i])."""
    n = len(residues)
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + residues[i][1]
    return prefix, prefix[n]


def _resolve_residual_mass(residual):
    """Resolve a residual spec ('H2O', 'NH3', or a numeric mass) to a float."""
    if isinstance(residual, str):
        residual_mass = RESIDUAL_MASSES.get(residual.upper())
        if residual_mass is None:
            raise ValueError(f"Unknown residual '{residual}'. Use 'H2O', 'NH3', or a numeric value.")
        return residual_mass
    return float(residual)


def compute_peptide_mass(sequence, residual='H2O'):
    """ParentMass -- what a complementary FFC pair's two TFFCs sum to.

    [CZ] For b/y that is the neutral peptide mass M.  For c/z-dot it is M + H:
    electron transfer delivers one hydrogen atom that ends up split across the
    pair.  The ``offset`` column is ``ParentMass - (TFFC X + TFFC Y)``, so
    using M here would report every clean ETD parental pair at offset -1.008
    instead of 0.
    """
    residues = parse_sequence(sequence)
    _prefix, total_residue_mass = compute_prefix_sums(residues)
    return (total_residue_mass + _resolve_residual_mass(residual)
            + ACTIVE["parent"])                                     # [CZ]


# ---------------------------------------------------------------------------
# Ion representation
# ---------------------------------------------------------------------------

# k, m: 1-indexed inclusive residue positions spanned by the fragment.
# ion_type: 'b', 'y', or 'internal'.
# loss: '' or one of the NEUTRAL_LOSSES keys (single loss only, any ion type).
# priority: tuple used to rank plausibility, lower = more plausible.
Ion = namedtuple('Ion', ['k', 'm', 'ion_type', 'charge', 'loss', 'isotope', 'mz', 'label', 'priority'])


def _iso_tag(iso):
    if iso == 0:
        return ""
    if iso == 1:
        return " (C13)"
    return f" ({iso}C13)"


def _priority(ion_type, loss_name, isotope, charge):
    """Lower = more plausible. Ordered by:

      1. modification distance: a neutral loss plus each isotope step each
         count 1, so the cleanest explanation (monoisotopic, no loss) wins.
         This is deliberately ahead of ion type: a bare internal ion is
         preferred over a b/y ion that only matches after a neutral loss
         and/or an isotope shift -- e.g. i(3:13) (clean) beats
         b11-H2O (2C13) (1 loss + 2 isotopes) for the same observed m/z.
      2. ion type: b/y preferred over internal, but only as a tiebreak
         among equally-clean candidates.
      3. isotope: prefer the monoisotopic (lower-isotope) explanation.
      4. charge: lower preferred.
    """
    type_score = 0 if ion_type in ('b', 'y') else 1
    loss_score = 1 if loss_name else 0
    modification_distance = loss_score + isotope
    return (modification_distance, type_score, isotope, charge)


def is_terminal(ion_or_km, n, m=None):
    """True if the fragment touches either end of the peptide (k==1 or m==n).
    Accepts either an Ion (or any object with .k/.m) or explicit (k, m)."""
    if m is None:
        k, m = ion_or_km.k, ion_or_km.m
    else:
        k = ion_or_km
    return k == 1 or m == n


# ---------------------------------------------------------------------------
# Theoretical MS/MS spectrum (Supplemental Note 10)
# ---------------------------------------------------------------------------

def generate_theoretical_ions(sequence, max_charge, residual='H2O',
                               isomax_cutoff=DEFAULT_ISOMAX_CUTOFF_PCT,
                               isomax_table=None):
    """Generate the theoretical MS/MS spectrum of a peptide.

    For each charge 1..(max_charge-1), generates b-, y-, and internal ions
    with neutral losses (H2O, NH3, CO; single loss only, on any ion type)
    and isotopic variants (+1..+isomax, isomax looked up per-fragment from
    the Averagine table).

    Returns (ions, n) where ions is a list of Ion namedtuples and n is the
    peptide length (needed downstream for terminal/compatibility checks).
    """
    residual_mass = _resolve_residual_mass(residual)

    residues = parse_sequence(sequence)
    n = len(residues)
    prefix, total_residue_mass = compute_prefix_sums(residues)
    peptide_mass = total_residue_mass + residual_mass

    table = isomax_table if isomax_table is not None else AveragineIsoMaxTable(cutoff_pct=isomax_cutoff)

    # --- Base fragments (charge-independent): (k, m, ion_type, loss, label_base, neutral_base) ---
    base_fragments = []

    for i in range(1, n):  # [CZ] c-ions: (1, i);  c = b + NH3
        b_neutral = prefix[i] + ACTIVE["nterm"]                  # [CZ]
        label_base = f'{ACTIVE["n_label"]}{i}{ACTIVE["suffix"]}'    # [CZ]
        for loss_name, loss_mass in NEUTRAL_LOSSES.items():
            neutral = b_neutral - loss_mass
            if neutral <= 0:
                continue
            base_fragments.append((1, i, 'b', loss_name, label_base, neutral))

    for i in range(1, n):  # [CZ] z-dot ions: (n-i+1, n);  z. = y + H - NH3
        k = n - i + 1
        # NOTE ``peptide_mass`` here is the *local* one computed a few lines up
        # (residue sum + residual = M), NOT compute_peptide_mass's M + H -- so
        # ``peptide_mass - prefix[n-i]`` is the plain y-ion neutral and the
        # z-dot offset applies to it directly.
        y_neutral = peptide_mass - prefix[n - i] + ACTIVE["cterm"]   # [CZ]
        label_base = f'{ACTIVE["c_label"]}{i}'                          # [CZ]
        for loss_name, loss_mass in NEUTRAL_LOSSES.items():
            neutral = y_neutral - loss_mass
            if neutral <= 0:
                continue
            base_fragments.append((k, n, 'y', loss_name, label_base, neutral))

    for i in range(1, n - 1):  # internal ions: (i+1, j)
        for j in range(i + 1, n):
            # [CZ] An ETD internal is cut twice -- a z-dot end (-NH3 + H) and
            # a c end (+NH3) -- so it nets one hydrogen.  This +H is why c/z
            # cannot be emulated by modifying the peptide: an internal mass is a
            # difference of prefixes, which cancels any N-terminal modification.
            frag_mass = prefix[j] - prefix[i] + ACTIVE["internal"]   # [CZ]
            if frag_mass <= 0:
                continue
            label_base = f"i({i + 1}:{j})"
            for loss_name, loss_mass in NEUTRAL_LOSSES.items():
                neutral = frag_mass - loss_mass
                if neutral <= 0:
                    continue
                base_fragments.append((i + 1, j, 'internal', loss_name, label_base, neutral))

    # --- Expand isotopes (charge-independent) ---
    iso_fragments = []
    for k, m, ion_type, loss_name, label_base, neutral_base in base_fragments:
        isomax = table.get_isomax(neutral_base)
        for iso in range(isomax + 1):
            neutral_mass = neutral_base + iso * C13_SHIFT
            iso_fragments.append((k, m, ion_type, loss_name, label_base, iso, neutral_mass))

    # --- Expand charge states 1..max_charge-1 ---
    ions = []
    for z in range(1, max_charge):
        for k, m, ion_type, loss_name, label_base, iso, neutral_mass in iso_fragments:
            mz = (neutral_mass + z * PROTON) / z
            label = f"{label_base}{loss_name} (+{z}){_iso_tag(iso)}"
            ions.append(Ion(
                k=k, m=m, ion_type=ion_type, charge=z, loss=loss_name, isotope=iso,
                mz=mz, label=label, priority=_priority(ion_type, loss_name, iso, z),
            ))

    return ions, n


# ---------------------------------------------------------------------------
# Theoretical FFC map (Supplemental Note 11)
# ---------------------------------------------------------------------------

def is_compatible(ion_a, ion_b, n, precursor_charge, enforce_charge_sum=False):
    """Compatibility of two substrings per Supplemental Note 11:
      (i)   non-overlapping
      (ii)  charge-sum <= precursor charge -- only checked if enforce_charge_sum=True
      (iii) at most one neutral loss between them
      (iv)  adjacent or both terminal
    """
    # (i) non-overlapping
    if not (ion_a.m < ion_b.k or ion_b.m < ion_a.k):
        return False

    # (ii) sum of charges does not exceed the precursor charge
    if enforce_charge_sum and ion_a.charge + ion_b.charge > precursor_charge:
        return False

    # (iii) at most one neutral loss total between the pair
    loss_count = (1 if ion_a.loss else 0) + (1 if ion_b.loss else 0)
    if loss_count > 1:
        return False

    # (iv) adjacent or both terminal
    adjacent = (ion_a.m + 1 == ion_b.k) or (ion_b.m + 1 == ion_a.k)
    both_terminal = is_terminal(ion_a, n) and is_terminal(ion_b, n)
    if not (adjacent or both_terminal):
        return False

    return True


def generate_theoretical_ffc_map(ions, n, precursor_charge, enforce_charge_sum=False):
    """Enumerate all compatible ion pairs (the theoretical FFC map).

    This is provided for inspection/completeness; annotate_ffcs() does not
    call this directly since for a given peptide length it is much cheaper
    to look up compatible candidates only for the specific observed m/z
    pairs being annotated (see match_ffc_pair()).
    """
    for i, ion_a in enumerate(ions):
        for ion_b in ions[i + 1:]:
            if is_compatible(ion_a, ion_b, n, precursor_charge, enforce_charge_sum):
                yield (ion_a, ion_b)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def build_mz_index(ions):
    """Sort ions by m/z for fast tolerance-window lookup."""
    sorted_ions = sorted(ions, key=lambda ion: ion.mz)
    mzs = [ion.mz for ion in sorted_ions]
    return sorted_ions, mzs


def candidates_for(mz_query, sorted_ions, mzs, tol_ppm):
    """All ions within tol_ppm of mz_query."""
    tol_da = mz_query * tol_ppm * 1e-6
    lo = bisect.bisect_left(mzs, mz_query - tol_da)
    hi = bisect.bisect_right(mzs, mz_query + tol_da)
    return sorted_ions[lo:hi]


def match_mz(mz_query, sorted_ions, mzs, tol_ppm=DEFAULT_TOL_PPM):
    """Best single-ion match (Note 10 spectrum only, no FFC compatibility).

    Returns (label, error_da) or ("No Match", -1000).
    """
    cands = candidates_for(mz_query, sorted_ions, mzs, tol_ppm)
    if not cands:
        return "No Match", -1000
    best = min(cands, key=lambda ion: (ion.priority, abs(mz_query - ion.mz)))
    return best.label, mz_query - best.mz


def pair_priority(ion_a, ion_b):
    return tuple(pa + pb for pa, pb in zip(ion_a.priority, ion_b.priority))


def match_ffc_pair(mz_a, mz_b, sorted_ions, mzs, n, precursor_charge, tol_ppm=DEFAULT_TOL_PPM,
                    enforce_charge_sum=False):
    """Find the most plausible compatible ion pair matching (mz_a, mz_b).

    Fragment assigned to the "A" side must match mz_a; fragment assigned to
    the "B" side must match mz_b (both within tol_ppm), and the two
    fragments must be compatible per Note 11 (is_compatible()). If
    enforce_charge_sum is True, the pair's charge sum must also not exceed
    precursor_charge.

    Returns (label_a, err_a, label_b, err_b), or ("No Match", -1000,
    "No Match", -1000) if no compatible pair is found.
    """
    cands_a = candidates_for(mz_a, sorted_ions, mzs, tol_ppm)
    cands_b = candidates_for(mz_b, sorted_ions, mzs, tol_ppm)

    best = None
    best_priority = None
    for ion_a in cands_a:
        for ion_b in cands_b:
            if not is_compatible(ion_a, ion_b, n, precursor_charge, enforce_charge_sum):
                continue
            prio = pair_priority(ion_a, ion_b)
            if best is None or prio < best_priority:
                best = (ion_a, ion_b)
                best_priority = prio

    if best is None:
        return "No Match", -1000, "No Match", -1000

    ion_a, ion_b = best
    return ion_a.label, mz_a - ion_a.mz, ion_b.label, mz_b - ion_b.mz


# ---------------------------------------------------------------------------
# Charge-constrained matching (with single-m/z fallback)
# ---------------------------------------------------------------------------

def _has_known_charge(charge):
    """True if `charge` is a usable (non-null) charge value."""
    return charge is not None and not pd.isna(charge)


def filter_by_charge(ions, required_charge):
    """Restrict a list of Ion candidates to `required_charge`.

    If required_charge is None/NaN (unknown), returns ions unchanged.
    """
    if not _has_known_charge(required_charge):
        return ions
    required_charge = int(round(required_charge))
    return [ion for ion in ions if ion.charge == required_charge]


def filter_by_charge_soft(ions, required_charge):
    """Like filter_by_charge, but if the charge constraint would remove ALL
    candidates -- i.e. the assigned charge matches no theoretical ion at this
    m/z -- ignore it and return `ions` unchanged.

    A charge that matches nothing is treated as spurious (a mis-assignment
    from deconvolution), so it can't turn an otherwise-matchable peak into a
    No Match or block an otherwise-valid compatible pair. This only ever
    relaxes an empty result; when the constraint leaves any candidate, that
    (correctly-charged) subset is used as-is.
    """
    filtered = filter_by_charge(ions, required_charge)
    return filtered if filtered else ions


def match_ffc_pair_with_charge(mz_a, mz_b, sorted_ions, mzs, n, precursor_charge,
                                charge_a=None, charge_b=None, tol_ppm=DEFAULT_TOL_PPM,
                                enforce_charge_sum=False):
    """Like match_ffc_pair(), but restricts side A to charge_a and side B to
    charge_b whenever those are known (not NaN).

    The charge constraint is applied softly (see filter_by_charge_soft): a
    charge that matches nothing at that m/z is treated as spurious and
    ignored for that side, so a mis-charged peak (e.g. a y3 the deconvolution
    labelled +2) can't block an otherwise-valid complementary pair.

    Returns (ion_a, ion_b) -- the matched Ion objects for the best compatible
    pair -- or None if no compatible pair exists (signalling the caller
    should fall back to single-m/z matching).
    """
    cands_a = filter_by_charge_soft(candidates_for(mz_a, sorted_ions, mzs, tol_ppm), charge_a)
    cands_b = filter_by_charge_soft(candidates_for(mz_b, sorted_ions, mzs, tol_ppm), charge_b)

    best = None
    best_priority = None
    for ion_a in cands_a:
        for ion_b in cands_b:
            if not is_compatible(ion_a, ion_b, n, precursor_charge, enforce_charge_sum):
                continue
            prio = pair_priority(ion_a, ion_b)
            if best is None or prio < best_priority:
                best = (ion_a, ion_b)
                best_priority = prio

    return best


def match_mz_single_with_charge(mz_query, sorted_ions, mzs, required_charge=None,
                                 tol_ppm=DEFAULT_TOL_PPM, max_charge=None):
    """Match a single m/z value independently (no FFC-pair compatibility
    requirement), optionally restricted to required_charge.

    The charge constraint is applied softly (see filter_by_charge_soft): if
    the assigned charge matches nothing at this m/z it is treated as spurious
    and ignored, so the peak is annotated at whatever charge(s) it does match
    (e.g. a y3 the deconvolution mis-labelled +2 is still reported as y3),
    and it's then treated as charge-unknown for the single-vs-list decision
    below.

    max_charge caps the charge considered when this side's own charge is NOT
    pinned down (unknown, or known-but-spurious/relaxed). It encodes charge
    conservation across the FFC: if the OTHER side of the pair has a known
    charge k and the precursor has charge z, this side can carry at most
    z - k (so both fragments' charges sum to <= z). Candidates above
    max_charge are dropped; if that leaves nothing, the peak is "No Match".
    A pinned-down known charge is trusted as-is and ignores max_charge.

    Monoisotopic prioritization: if the peak can be explained as a
    monoisotopic (isotope 0) ion at all, isotopic (isotope > 0) explanations
    are dropped to keep the candidate list short -- EXCEPT isotopic terminal
    (b/y) ions with no neutral loss, which are always kept (an isotopic
    terminal ion that also carries a neutral loss is dropped like the
    internals). (If no monoisotopic explanation exists, isotopic candidates,
    including internal ones, are kept.) This can collapse a would-be
    ambiguous list down to a single definitive match.

    - required_charge known (not NaN) AND actually matches, or exactly one
      candidate matches regardless of charge: returns the single best match
      as (label, error_da, ion, confirmed_charge), same label/error
      convention as match_mz(). ("No Match", -1000, None, None) if nothing
      matches at any charge.
    - required_charge unknown/spurious and more than one candidate matches:
      charge can't disambiguate which specific ion it is, so every matching
      candidate (across all charges) is returned, sorted best-first and
      joined into one label with a newline between each (AMBIGUOUS_SEPARATOR)
      -- so each candidate lands on its own line when viewed in a
      spreadsheet cell (wrap_text is enabled in _write_xlsx). The numeric
      error is that of the single best candidate. `ion` is None here since
      there's no single definitive match -- but confirmed_charge is still
      that shared charge (not None) if every candidate happens to agree on
      charge, since TFFC only depends on charge, not on which specific
      ion/isotope/loss it turned out to be. confirmed_charge is None if the
      candidates disagree on charge, or nothing matched at all.

    Returns (label, error_da, ion_or_None, confirmed_charge_or_None).
    """
    all_cands = candidates_for(mz_query, sorted_ions, mzs, tol_ppm)
    if not all_cands:
        return "No Match", -1000, None, None

    constrained = filter_by_charge(all_cands, required_charge)
    if constrained:
        cands = constrained
        # A known charge that actually matched pins down the ion's charge.
        charge_pins_down = _has_known_charge(required_charge)
    else:
        # Assigned charge matched nothing -> spurious; ignore it and treat
        # the peak as charge-unknown (annotate at whatever it does match).
        cands = all_cands
        charge_pins_down = False

    # Charge-sum cap: when this side's own charge isn't pinned down and the
    # OTHER side has a known charge, this side's charge can't exceed
    # max_charge = precursor_charge - other_known_charge (charge is conserved
    # in a single fragmentation). A pinned-down known charge is trusted as-is.
    if not charge_pins_down and max_charge is not None:
        cands = [ion for ion in cands if ion.charge <= max_charge]
        if not cands:
            return "No Match", -1000, None, None

    # Monoisotopic prioritization: if the peak can be explained as a
    # monoisotopic (isotope 0) ion, drop the isotopic (isotope > 0)
    # explanations to keep the candidate list short -- EXCEPT isotopic
    # *terminal* (b/y) ions with NO neutral loss, which are kept (bare
    # terminal-ion isotope peaks are common and worth surfacing; an isotopic
    # terminal ion that also needs a neutral loss is too speculative to
    # list). If no monoisotopic explanation exists, all candidates
    # (including isotopic internals) are kept. This can also collapse a
    # would-be ambiguous list down to a single definitive match.
    if any(ion.isotope == 0 for ion in cands):
        cands = [ion for ion in cands
                 if ion.isotope == 0 or (ion.ion_type in ('b', 'y') and not ion.loss)]

    cands_sorted = sorted(cands, key=lambda ion: (ion.priority, abs(mz_query - ion.mz)))

    if charge_pins_down or len(cands_sorted) == 1:
        best = cands_sorted[0]
        return best.label, mz_query - best.mz, best, best.charge

    label = AMBIGUOUS_SEPARATOR.join(ion.label for ion in cands_sorted)
    err = mz_query - cands_sorted[0].mz
    charges = {ion.charge for ion in cands_sorted}
    confirmed_charge = charges.pop() if len(charges) == 1 else None
    return label, err, None, confirmed_charge


# ---------------------------------------------------------------------------
# b-in-A / y-in-B placement
# ---------------------------------------------------------------------------

# [CZ] This classifier reads the printed LABEL, not ``ion_type`` -- and the
# labels this module emits are ``c12`` / ``z25``, not ``b12`` / ``y25``.  Left
# as ``[by]`` it matched nothing, every side came back 'other', the swap in
# ``_enforce_by_placement`` never fired, and BOTH visible symptoms followed
# from that one line:
#   * X / Y stayed in deconvolution order, so a z ion could sit in the X column;
#   * ``raw_breakpoints`` takes a parental breakpoint from ``ion_a.m`` on the
#     documented guarantee that placement already put the N-terminal ion there,
#     so an unswapped row reported the *z* ion's ``.m`` -- which is always the
#     peptide length n, since a z ion ends at the C-terminus.  That is where
#     the spurious "37" on a 37-mer came from.
# The c/z series is mapped back onto the b/y ROLE names, so ``_by_swap_needed``
# and every caller stay untouched: c plays b (N-terminal), z plays y.
# Built from ACTIVE, so a series swap re-derives it (series_maps.use_series).
_BY_LABEL_RE = re.compile(r'^([cz])\d')

#: Printed series letter -> the b/y role it plays in the placement rule.
_SERIES_ROLE = {'c': 'b', 'z': 'y'}


def rebuild_label_rules():
    """[CZ] Re-derive the placement regex/roles from ``ACTIVE``.

    Must be called after any change to ``ACTIVE['n_label']`` /
    ``['c_label']`` -- ``series_maps.use_series`` does it for you.  Forgetting
    it is the exact failure documented above: the regex stops matching, the
    b-in-X / y-in-Y swap silently stops firing, and parental breakpoints come
    back as the peptide length.
    """
    global _BY_LABEL_RE, _SERIES_ROLE
    n, c = ACTIVE["n_label"], ACTIVE["c_label"]
    _BY_LABEL_RE = re.compile(rf'^({re.escape(n)}|{re.escape(c)})\d')
    _SERIES_ROLE = {n: 'b', c: 'y'}


def _classify_ion_side(label):
    """Classify an Interpretation label as 'b', 'y', or 'other'.

    [CZ] Reads the c/z labels this module emits and returns the b/y *role*:
    ``c`` -> 'b' (N-terminal side), ``z`` -> 'y' (C-terminal side).

    'other' covers "No Match", internal-ion labels, and ambiguous
    multi-candidate labels (joined by AMBIGUOUS_SEPARATOR) that don't
    unanimously agree on c or z.
    """
    if not label or label == "No Match":
        return 'other'
    types = set()
    for part in label.split(AMBIGUOUS_SEPARATOR):
        m = _BY_LABEL_RE.match(part.strip())
        types.add(_SERIES_ROLE[m.group(1)] if m else 'other')
    if types == {'b'}:
        return 'b'
    if types == {'y'}:
        return 'y'
    return 'other'


def _by_swap_needed(type_a, type_b):
    """True if swapping A<->B strictly reduces b/y-placement violations
    (a b ion belongs in Interpretation A, a y ion in Interpretation B)."""
    def violations(t_a, t_b):
        return (1 if t_a == 'y' else 0) + (1 if t_b == 'b' else 0)
    return violations(type_b, type_a) < violations(type_a, type_b)


def _enforce_by_placement(mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a=None, ion_b=None,
                           charge_a=None, charge_b=None, charge_a_known=None, charge_b_known=None):
    """Swap (m/z, label, error, ion, charge, charge_known) across A/B so a y
    ion ends up in Interpretation B and a b ion in Interpretation A.
    Internal-ion, No-Match, and ambiguous (mixed b/y) labels are left in
    place unless a swap is needed to correctly place a b or y ion found on
    the other side.

    ion_a/ion_b are the matched Ion objects backing label_a/label_b (or None
    if there isn't a single definitive match for that side); charge_a/
    charge_b are the confirmed charge for that side (possibly known even
    when ion_a/ion_b is None, see match_mz_single_with_charge);
    charge_a_known/charge_b_known are whether that side's *input*
    charge_A/charge_B (from the observed-data row, before any matching) was
    known. All are carried along and swapped in lockstep so callers that
    need them (e.g. for TFFC/classification/the unknown-charge "*" marker)
    stay consistent with the displayed labels.
    """
    type_a = _classify_ion_side(label_a)
    type_b = _classify_ion_side(label_b)
    if _by_swap_needed(type_a, type_b):
        return (mz_b, label_b, err_b, mz_a, label_a, err_a, ion_b, ion_a,
                charge_b, charge_a, charge_b_known, charge_a_known)
    return (mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a, ion_b,
            charge_a, charge_b, charge_a_known, charge_b_known)


# ---------------------------------------------------------------------------
# TFFC and parental / internal / secondary classification
# ---------------------------------------------------------------------------

def tffc_mass(mz, charge):
    """Transformed FFC mass: mass_i(X) = i * (X - ProtonMass)."""
    return charge * (mz - PROTON)


def _tffc_for_side(mz, ion, confirmed_charge, extend_ffc_values):
    """TFFC mass for one side of a row, or None if it can't be determined.

    If there's a single definitive ion, its charge is always used. If not
    (ambiguous multi-candidate list or "No Match"), confirmed_charge (the
    shared charge across every ambiguous candidate, if any) is only used
    when extend_ffc_values is True -- with it False, only a genuinely
    definitive single-ion match yields a TFFC value.
    """
    if ion is not None:
        return tffc_mass(mz, ion.charge)
    if extend_ffc_values and confirmed_charge is not None:
        return tffc_mass(mz, confirmed_charge)
    return None


def _mod_marker(ion):
    """Per-side modification marker for a matched ion:
      '*'  if it carries a neutral loss (takes precedence);
      '~'  if it carries an isotopic shift but no loss;
      ''   if it is a bare monoisotopic ion.
    An ion with BOTH a neutral loss and an isotope is marked as a neutral
    loss only ('*'), so a side never contributes more than one character.
    """
    if ion.loss:
        return '*'
    if ion.isotope > 0:
        return '~'
    return ''


def classify_ffc_relationship(ion_a, ion_b, n):
    """Classify the structural relationship between two matched fragments,
    into one of 5 base categories:

      parental     both terminal AND adjacent (complementary b/y pair
                   reconstructing the intact peptide)
      internal     exactly one side is an internal ion (the other terminal),
                   adjacent
      secondary    two NON-OVERLAPPING terminal fragments, one b and one y,
                   that are not adjacent (e.g. b5/y7 rather than the true
                   complementary b5/y_(n-5) -- Yalcin et al. 1996 secondary
                   fragmentation)
      unexplained  neither adjacent nor both terminal -- ion_a and ion_b
                   were each individually matched, but don't form a valid
                   theoretical FFC relationship
      unmatched    a pair that can't be a real FFC:
                     - both terminal but not a valid secondary: two
                       same-type terminal fragments (both b or both y), or a
                       one-b-one-y terminal pair that overlaps; or
                     - two internal ions (internal-internal), even when they
                       abut complementarily -- they leave residues
                       unaccounted for at both termini.

    'parental', 'internal' and 'secondary' get a per-side modification
    suffix, one character per side in X (ion_a) then Y (ion_b) order:
      '*'  for a neutral loss on that side (takes precedence);
      '~'  for an isotopic shift (no loss) on that side;
      ''   for a bare monoisotopic side.
    A side carrying BOTH a loss and an isotope is marked '*' only, so each
    side contributes at most one character and the suffix is one of '', '*',
    '~', '**', '~~', '*~', '~*'. E.g. (b5-H2O, y10 C13) -> 'parental*~'.

    Only ever called with two definitively-matched Ion objects. (The
    annotate_ffcs_with_charge wrapper also emits 'unmatched' for rows where
    either side is an outright "No Match", and 'unexplained' for rows with
    an ambiguous multi-candidate list on a side.)
    """
    both_terminal = is_terminal(ion_a, n) and is_terminal(ion_b, n)
    adjacent = (ion_a.m + 1 == ion_b.k) or (ion_b.m + 1 == ion_a.k)

    if both_terminal and not adjacent:
        # A true secondary is two non-overlapping terminal fragments, one b
        # and one y. Two same-type terminal fragments (both b or both y)
        # always overlap, and a one-b-one-y terminal pair that overlaps
        # likewise can't coexist in one precursor -- neither is a real FFC.
        one_b_one_y = {ion_a.ion_type, ion_b.ion_type} == {'b', 'y'}
        overlapping = not (ion_a.m < ion_b.k or ion_b.m < ion_a.k)
        if one_b_one_y and not overlapping:
            base = 'secondary'
        else:
            return 'unmatched'
    elif both_terminal:      # both terminal AND adjacent -> complementary
        base = 'parental'
    elif adjacent:
        # Adjacent, not both terminal -> at least one internal ion. Two
        # internal ions (internal-internal), even when they abut
        # complementarily, are treated as unmatched: a real FFC pairs a
        # fragment with the complementary remainder of the precursor, but
        # two internal ions leave residues unaccounted for at both termini.
        if ion_a.ion_type == 'internal' and ion_b.ion_type == 'internal':
            return 'unmatched'
        base = 'internal'
    else:
        return 'unexplained'

    # Per-side modification markers, in X (ion_a) then Y (ion_b) order: '*'
    # for a neutral loss, '~' for an isotopic shift (a side with both is
    # marked '*' only). So the suffix is one of '', '*', '~', '**', '~~',
    # '*~', '~*'. E.g. (b5-H2O, y10 C13) -> 'parental*~'.
    return f'{base}{_mod_marker(ion_a)}{_mod_marker(ion_b)}'


def raw_breakpoints(ion_a, ion_b, n, annotation):
    """Un-deduplicated breakpoint data for one row.

    Every breakpoint is reported as a b-ion position -- the residue count
    from the peptide's N-terminus after which the backbone was cleaved
    (i.e. a break after residue p corresponds to a b_p ion). This is a
    fixed, orientation-independent convention: it does not matter whether a
    fragment is expressed as a b-ion, a y-ion, or an internal ion, nor
    which of the two observed peaks it landed on.

    Returns:
      - None for annotations with no well-defined breaking point
        ('secondary', 'ambiguous', 'unmatched').
      - an int for a parental pair (annotation starting with 'parental'):
        the b-ion position of the cleavage, ion_a.m (ion_a is guaranteed to be the
        b-ion after b-in-A/y-in-B placement, so its length IS the cleavage
        position) -- e.g. b3/y12 at peptide length 15 -> 3 (not 12; the
        complementary y12 describes the same physical break at position 3).
      - a sorted tuple of ints for an internal pair (annotation starting
        with 'internal'): the cleavage position(s) bounding the adjacent
        (left, right) fragments -- the junction between them (always
        present), plus the outer edge of whichever fragment doesn't
        already sit at the peptide's own N-/C-terminus. All are absolute
        positions from the N-terminus (= b-ion positions). E.g.
        (i(3:14), y1) at peptide length 15: left=i(3:14) contributes its
        own left edge 3-1=2 (since k=3>1, not the peptide's N-terminus)
        and the junction 14 (right=y1 already reaches the C-terminus, so
        contributes nothing) -> (2, 14). The mirror orientation -- a
        terminal b-ion complementary to an internal ion, e.g. b3/i(4:14)
        -> (3, 14) -- yields the same b-position convention automatically.
    """
    if annotation.startswith('parental'):
        # [CZ] Take the breakpoint from the **c ion** explicitly, rather than
        # trusting ion_a to be it.  Placement is what puts the N-terminal ion
        # in A, but that is an invariant maintained elsewhere -- and when it
        # silently stopped holding (see _classify_ion_side above) this returned
        # the z ion's ``.m``, i.e. the peptide length, on every affected row.
        # Reading ``ion_type`` cannot fail that way, and it is exactly the
        # requested convention: the breaking point is the c-ion position.
        #
        # ``ion_type`` stays 'b'/'y' by design (a c/z pair has the geometry of
        # a b/y pair), so 'b' here means "the c ion".  c_k spans residues 1..k,
        # so its length IS the cleavage position -- bond k, matching the
        # ``c{k}, z{n-k}`` row labels of the line-based coverage table.
        c_ion = ion_a if ion_a.ion_type == 'b' else ion_b
        return c_ion.m
    if annotation.startswith('internal'):
        if ion_a.m + 1 == ion_b.k:
            left, right = ion_a, ion_b
        else:
            left, right = ion_b, ion_a
        bps = []
        if left.k > 1:
            bps.append(left.k - 1)
        bps.append(left.m)
        if right.m < n:
            bps.append(right.m)
        return tuple(sorted(bps))
    return None


def _dedup_breakpoint(raw, seen):
    """Format one row's raw_breakpoints() value, replacing repeats already
    seen in an earlier row (processed in row order) with '-'.

    `seen` is a single set shared across parental AND internal rows, since
    the same physical cleavage position can show up in either kind of row
    -- e.g. breakpoint 6 from an internal (2,6) pair should also suppress a
    later parental b6/y9 pair's own breakpoint of 6. A repeated position is
    replaced by '-' (never blanked), so a row that genuinely HAS a breaking
    point never renders as empty: a repeated parental breakpoint becomes
    '-', and a repeated number within an internal tuple becomes '-' while
    any other, not-yet-seen number in the same tuple is kept. Only rows
    with no breaking point at all (raw is None -- secondary/ambiguous/
    unmatched) render as ''.
    """
    if raw is None:
        return ''
    if isinstance(raw, tuple):
        parts = []
        for val in raw:
            if val in seen:
                parts.append('-')
            else:
                seen.add(val)
                parts.append(str(val))
        return '(' + ','.join(parts) + ')'
    if raw in seen:
        return '-'
    seen.add(raw)
    return str(raw)


# ---------------------------------------------------------------------------
# Main annotation functions
# ---------------------------------------------------------------------------

def annotate_ffcs(df_observed, sequence, precursor_charge, output_path="annotated_ffcs.xlsx",
                   tol_ppm=DEFAULT_TOL_PPM, residual='H2O', isomax_cutoff=DEFAULT_ISOMAX_CUTOFF_PCT,
                   enforce_charge_sum=False):
    """Annotate FFC pairs against the theoretical FFC map and write to xlsx.

    If no compatible FFC pair is found jointly (Note 11), each side is
    still annotated independently against the theoretical spectrum (Note
    10, no pair-compatibility requirement) instead of both sides reporting
    "No Match". Since this function has no charge information, the
    fallback always lists every matching candidate for that side, one per
    line (see AMBIGUOUS_SEPARATOR) -- see annotate_ffcs_with_charge() for
    the charge-aware version, where a known charge collapses this to a
    single best match.

    Parameters
    ----------
    df_observed : pd.DataFrame
        Must have columns 'm/z A', 'm/z B', and 'Ranking'.
    sequence : str
        Peptide sequence (single-letter amino acid codes).
    precursor_charge : int
        Precursor ion charge state; fragment ions are generated for charges
        1..(precursor_charge-1).
    output_path : str
        Path for the output xlsx file.
    tol_ppm : float
        Mass tolerance in ppm (default 20).
    residual : str or float
        Terminal residual group: 'H2O' (default) or 'NH3', or a numeric mass.
    isomax_cutoff : float
        Relative-intensity cutoff (percent) used to determine isomax per
        fragment from the Averagine table (default 5, matching
        james_code/IsoMax_example.f90).
    enforce_charge_sum : bool
        If True, enforce the FFC charge-sum compatibility check (Note 11,
        criterion ii): a matched pair is only accepted if the sum of the
        two fragments' charges does not exceed precursor_charge. Default
        False (not enforced).

    Returns
    -------
    pd.DataFrame with annotation results.
    """
    res_label = residual if isinstance(residual, str) else f'{residual:.5f}'
    print(f"Generating theoretical ions for {sequence} (charge {precursor_charge}, "
          f"residual={res_label}, isomax_cutoff={isomax_cutoff}%)...")
    ions, n = generate_theoretical_ions(sequence, precursor_charge, residual=residual,
                                         isomax_cutoff=isomax_cutoff)
    print(f"  -> {len(ions)} theoretical m/z values generated.")
    sorted_ions, mzs = build_mz_index(ions)

    rows = []
    for _, row in df_observed.iterrows():
        mz_a = row['m/z A']
        mz_b = row['m/z B']
        ranking = row['Ranking']

        label_a, err_a, label_b, err_b = match_ffc_pair(mz_a, mz_b, sorted_ions, mzs, n,
                                                         precursor_charge, tol_ppm,
                                                         enforce_charge_sum)

        if label_a == "No Match" and label_b == "No Match":
            # No compatible FFC pair -- fall back to annotating each side
            # independently (Note 10 spectrum only, no pair-compatibility
            # requirement). There's no charge information in this function,
            # so every matching candidate is listed, one per line, same as
            # the unknown-charge case in annotate_ffcs_with_charge.
            label_a, err_a, _ion_a, _charge_a = match_mz_single_with_charge(mz_a, sorted_ions, mzs, None, tol_ppm)
            label_b, err_b, _ion_b, _charge_b = match_mz_single_with_charge(mz_b, sorted_ions, mzs, None, tol_ppm)

        (mz_a, label_a, err_a, mz_b, label_b, err_b, _ion_a, _ion_b,
         _charge_a, _charge_b, _charge_a_known, _charge_b_known) = _enforce_by_placement(
            mz_a, label_a, err_a, mz_b, label_b, err_b
        )

        rows.append({
            'Ranking': ranking,
            'm/z A': round(mz_a, 3),
            'Interpretation A': label_a,
            'error A': round(err_a, 3) if err_a != -1000 else -1000,
            'm/z B': round(mz_b, 3),
            'Interpretation B': label_b,
            'error B': round(err_b, 3) if err_b != -1000 else -1000,
        })

    df_result = pd.DataFrame(rows)
    _write_xlsx(df_result, output_path)
    print(f"Output written to {output_path}")
    return df_result


def annotate_ffcs_with_charge(df_observed, sequence, precursor_charge,
                               output_path="annotated_ffcs_charge.xlsx",
                               tol_ppm=DEFAULT_TOL_PPM, residual='H2O',
                               isomax_cutoff=DEFAULT_ISOMAX_CUTOFF_PCT,
                               enforce_charge_sum=False, extra_info=True,
                               extend_ffc_values=True):
    """Charge-aware variant of annotate_ffcs(), with a single-m/z fallback.

    Same theoretical FFC map matching as annotate_ffcs(), plus:

    1. Charge-constrained matching: df_observed must also carry 'charge_A'
       and 'charge_B' columns (e.g. from line-finding/deconvolution). When a
       row's charge_A/charge_B is not NaN, only theoretical ions of that
       exact charge are considered for that side -- both in the joint
       FFC-pair match and the single-m/z fallback below.
    2. Single-m/z fallback: if no compatible FFC pair is found jointly (as
       would happen in annotate_ffcs()), each side is still independently
       annotated against the theoretical spectrum (Note 10, no pair
       compatibility requirement) instead of both sides reporting
       "No Match". If that side's charge is known, the single best match is
       reported (same convention as annotate_ffcs()); if unknown, every
       matching candidate is listed, one per line (see AMBIGUOUS_SEPARATOR),
       since charge is the only way to disambiguate here. Whenever this
       fallback is used AND the side's input charge_A/charge_B was NaN,
       the displayed label
       (including "No Match") is prefixed with '*' (e.g. "*b5 (+2)",
       "*No Match") to flag that its charge isn't actually confirmed --
       jointly-matched labels never get this prefix, since those were
       verified as a compatible pair regardless of whether charge_A/
       charge_B was supplied.
    3. 'error X'/'error Y' are '-' (not a number) wherever that side is
       "No Match" (with or without the '*' prefix), instead of the -1000
       sentinel used elsewhere in this module.

    Output columns (X/Y replace the old A/B suffix convention): 'rank', 'X',
    'Y', then (if extra_info) 'annotation', 'breaking point', then 'm/z X',
    'm/z Y', 'error X', 'error Y', then (if extra_info) 'TFFC X', 'TFFC Y',
    'offset', 'deviation'.

    If extra_info=True, six additional columns are added:
      'TFFC X', 'TFFC Y'  -- transformed FFC masses, mass_i(X) =
                             i * (X - ProtonMass), using each side's
                             observed m/z and matched charge.
      'offset'            -- the peptide's theoretical neutral mass minus
                             (TFFC X + TFFC Y).
      'deviation'         -- for a 'parental'/'secondary'/'internal' row
                             (any * / ~ variant), the observed offset minus
                             the *theoretical* offset that a perfectly-
                             measured FFC of the two matched ions would show
                             (0 for a clean parental; a neutral loss adds its
                             mass, an isotope subtracts a C13 step, and
                             internal/secondary include the gap of
                             unaccounted residues). Sign is kept (not abs).
                             '' for 'unexplained'/'unmatched' rows.
      'annotation'        -- one of 5 base categories: 'parental',
                             'internal', 'secondary', 'unexplained', or
                             'unmatched' (see classify_ffc_relationship()).
                             'parental', 'internal' and 'secondary' get a
                             per-side modification suffix (X side then Y
                             side): '*' for a neutral loss, '~' for an
                             isotopic shift; a side with both is marked '*'.
                             So the suffix is one of '', '*', '~', '**',
                             '~~', '*~', '~*' -- e.g. (b5-H2O, y10 C13) ->
                             'parental*~'.
      'breaking point'    -- the cleavage position(s), always as b-ion
                             positions (residue count from the N-terminus
                             after which the backbone was cleaved), see
                             raw_breakpoints():
                               parental(*): a single integer, the b-ion
                                 position -- e.g. b3/y12 at peptide
                                 length 15 -> "3" (the complementary y12
                                 describes the same break at position 3).
                               internal(*): "(p, q)" -- up to two integers:
                                 the junction between the adjacent
                                 fragments, plus the outer edge of
                                 whichever fragment isn't already at the
                                 peptide's own terminus -- e.g. (i(3:14),
                                 y1) at peptide length 15 -> "(2,14)".
                               anything else: ''.
                             To avoid restating the same physical cleavage
                             repeatedly, each individual breakpoint number
                             is only shown the first time it's seen (across
                             all rows, in row order) -- parental and
                             internal numbers share the same "seen" pool,
                             since the same cleavage can appear in either
                             kind of row. A repeated position is replaced by
                             '-' (never blanked): a repeated parental
                             breakpoint renders as "-", and a repeated
                             number within an internal pair as e.g. "(-,10)"
                             while any new number in the same pair is kept.
                             Only rows with no breaking point at all render
                             as ''.
    TFFC X/Y require a confirmed charge on their respective side. If
    extend_ffc_values=True (default), that includes not just a single
    definitive match but also an ambiguous multi-candidate list where every
    candidate happens to share the same charge (TFFC only depends on
    charge, not on which specific ion/isotope/loss it is). With
    extend_ffc_values=False, only a single definitive match counts, same as
    before this option existed. 'offset' requires both TFFC X and TFFC Y to
    be available. Otherwise these are -1000 (consistent with the -1000
    sentinel used elsewhere for unmatched errors). annotation is:
      'unmatched'    if either side is an outright "No Match", or the two
                     matched terminal fragments can't coexist in one
                     precursor (same-type or overlapping -- see
                     classify_ffc_relationship());
      'unexplained'  if both sides matched something, but at least one is an
                     ambiguous multi-candidate list, or the two matched
                     fragments don't form a valid theoretical FFC
                     relationship (neither adjacent nor both terminal).

    Parameters
    ----------
    df_observed : pd.DataFrame
        Must have columns 'm/z A', 'm/z B', 'Ranking', 'charge_A', 'charge_B'.
        charge_A/charge_B may be NaN per row (charge unknown for that side).
    sequence : str
        Peptide sequence (single-letter amino acid codes).
    precursor_charge : int
        Precursor ion charge state; fragment ions are generated for charges
        1..(precursor_charge-1).
    output_path : str
        Path for the output xlsx file.
    tol_ppm : float
        Mass tolerance in ppm (default 20).
    residual : str or float
        Terminal residual group: 'H2O' (default) or 'NH3', or a numeric mass.
    isomax_cutoff : float
        Relative-intensity cutoff (percent) used to determine isomax per
        fragment from the Averagine table (default 5, matching
        james_code/IsoMax_example.f90).
    enforce_charge_sum : bool
        If True, enforce the FFC charge-sum compatibility check (Note 11,
        criterion ii) on the joint pair match. Default False (not enforced).
    extra_info : bool
        If True (default), add the 'annotation', 'breaking point',
        'TFFC X', 'TFFC Y', 'offset', and 'deviation' columns described
        above. If False, just 'rank', 'X', 'Y', 'm/z X', 'm/z Y',
        'error X', 'error Y'.
    extend_ffc_values : bool
        Only relevant when extra_info=True. If True (default), TFFC X/Y
        also get a value when a side is an ambiguous multi-candidate list
        as long as every candidate shares the same charge. If False, TFFC
        X/Y are only computed from a single definitive match (the
        behavior before this option existed).

    Returns
    -------
    pd.DataFrame with annotation results.
    """
    res_label = residual if isinstance(residual, str) else f'{residual:.5f}'
    print(f"Generating theoretical ions for {sequence} (charge {precursor_charge}, "
          f"residual={res_label}, isomax_cutoff={isomax_cutoff}%)...")
    ions, n = generate_theoretical_ions(sequence, precursor_charge, residual=residual,
                                         isomax_cutoff=isomax_cutoff)
    print(f"  -> {len(ions)} theoretical m/z values generated.")
    sorted_ions, mzs = build_mz_index(ions)
    parent_mass = compute_peptide_mass(sequence, residual=residual) if extra_info else None

    rows = []
    for _, row in df_observed.iterrows():
        mz_a = row['m/z A']
        mz_b = row['m/z B']
        ranking = row['Ranking']
        charge_a = row['charge_A']
        charge_b = row['charge_B']

        joint = match_ffc_pair_with_charge(mz_a, mz_b, sorted_ions, mzs, n, precursor_charge,
                                            charge_a=charge_a, charge_b=charge_b,
                                            tol_ppm=tol_ppm, enforce_charge_sum=enforce_charge_sum)
        fallback_used = joint is None

        if joint is not None:
            ion_a, ion_b = joint
            label_a, err_a = ion_a.label, mz_a - ion_a.mz
            label_b, err_b = ion_b.label, mz_b - ion_b.mz
            charge_a_confirmed, charge_b_confirmed = ion_a.charge, ion_b.charge
        else:
            # Charge conservation: an unknown-charge side can carry at most
            # precursor_charge minus the OTHER side's known charge (if known),
            # so the two fragments' charges can't sum past the precursor.
            max_a = precursor_charge - int(round(charge_b)) if _has_known_charge(charge_b) else None
            max_b = precursor_charge - int(round(charge_a)) if _has_known_charge(charge_a) else None
            label_a, err_a, ion_a, charge_a_confirmed = match_mz_single_with_charge(mz_a, sorted_ions, mzs, charge_a, tol_ppm, max_charge=max_a)
            label_b, err_b, ion_b, charge_b_confirmed = match_mz_single_with_charge(mz_b, sorted_ions, mzs, charge_b, tol_ppm, max_charge=max_b)

        # Input charge_A/charge_B known, before any matching -- used below to
        # mark independently-assigned (fallback) labels whose charge wasn't
        # actually confirmed by the input.
        charge_a_known = _has_known_charge(charge_a)
        charge_b_known = _has_known_charge(charge_b)

        (mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a, ion_b,
         charge_a_confirmed, charge_b_confirmed, charge_a_known, charge_b_known) = _enforce_by_placement(
            mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a, ion_b,
            charge_a_confirmed, charge_b_confirmed, charge_a_known, charge_b_known
        )

        # Fallback (independently-assigned, not verified as a compatible
        # pair) labels get a leading '*' on whichever side's input charge
        # wasn't actually known, flagging that its charge is unconfirmed.
        display_label_a = f'*{label_a}' if fallback_used and not charge_a_known else label_a
        display_label_b = f'*{label_b}' if fallback_used and not charge_b_known else label_b

        row_raw_bp = None
        if extra_info:
            # Each side's TFFC is reported independently, whenever THAT side
            # has a confirmed charge -- a single definitive ion always
            # counts; an ambiguous multi-candidate list only counts if
            # extend_ffc_values is True (see _tffc_for_side()) -- regardless
            # of whether the other side is resolved at all.
            tffc_a = _tffc_for_side(mz_a, ion_a, charge_a_confirmed, extend_ffc_values)
            tffc_b = _tffc_for_side(mz_b, ion_b, charge_b_confirmed, extend_ffc_values)

            if ion_a is not None and ion_b is not None:
                annotation = classify_ffc_relationship(ion_a, ion_b, n)
                row_raw_bp = raw_breakpoints(ion_a, ion_b, n, annotation)
            elif label_a == "No Match" or label_b == "No Match":
                # 'unmatched' only for an outright "No Match" on either
                # side; if both sides matched *something* but at least one
                # is an ambiguous multi-candidate list, it's 'unexplained'.
                annotation = 'unmatched'
            else:
                annotation = 'unexplained'

        # Column order: rank, X, Y, annotation, breaking point (when
        # present), then m/z, error, and (when present) TFFC/offset.
        row_dict = {
            'rank': ranking,
            'X': display_label_a,
            'Y': display_label_b,
        }
        if extra_info:
            row_dict['annotation'] = annotation
            row_dict['breaking point'] = None  # filled in the dedup pass below
        row_dict['m/z X'] = round(mz_a, 3)
        row_dict['m/z Y'] = round(mz_b, 3)
        row_dict['error X'] = round(err_a, 3) if err_a != -1000 else '-'
        row_dict['error Y'] = round(err_b, 3) if err_b != -1000 else '-'
        if extra_info:
            row_dict['TFFC X'] = round(tffc_a, 3) if tffc_a is not None else -1000
            row_dict['TFFC Y'] = round(tffc_b, 3) if tffc_b is not None else -1000
            row_dict['offset'] = (
                round(parent_mass - (tffc_a + tffc_b), 3) if tffc_a is not None and tffc_b is not None else -1000
            )
            # Deviation of the observed shift from the theoretical shift, for
            # classified FFCs only (parental/secondary/internal, incl. their
            # * / ~ modification variants). The 'theoretical shift' is the
            # offset a perfectly-measured FFC of these two ions would show:
            # 0 for a clean parental; a neutral loss adds its mass (e.g.
            # +18.011 for H2O), an isotope subtracts a C13 step, and
            # internal/secondary also include the unaccounted-gap mass.
            # deviation = observed offset - theoretical offset (sign kept);
            # '' for unexplained/unmatched rows.
            if (ion_a is not None and ion_b is not None
                    and annotation.startswith(('parental', 'secondary', 'internal'))):
                theo_offset = parent_mass - (tffc_mass(ion_a.mz, ion_a.charge)
                                             + tffc_mass(ion_b.mz, ion_b.charge))
                obs_offset = parent_mass - (tffc_a + tffc_b)
                row_dict['deviation'] = round(obs_offset - theo_offset, 4)
            else:
                row_dict['deviation'] = ''

        rows.append((row_dict, row_raw_bp))

    # Second pass: fill in 'breaking point', deduplicating repeats across
    # rows (in row/rank order) -- see _dedup_breakpoint(). One shared `seen`
    # set spans both parental and internal rows, since the same physical
    # cleavage position can appear in either.
    if extra_info:
        seen = set()
        for row_dict, row_raw_bp in rows:
            row_dict['breaking point'] = _dedup_breakpoint(row_raw_bp, seen)

    df_result = pd.DataFrame([row_dict for row_dict, _ in rows])
    _write_xlsx(df_result, output_path)
    print(f"Output written to {output_path}")
    return df_result


def annotate_peaks(df_observed, sequence, precursor_charge, output_path="annotated_peaks.xlsx",
                    tol_ppm=DEFAULT_TOL_PPM, residual='H2O', isomax_cutoff=DEFAULT_ISOMAX_CUTOFF_PCT,
                    mz_column='m/z'):
    """Annotate a single column of observed m/z values against the
    theoretical MS/MS spectrum (Note 10). No FFC compatibility check is
    applied since there is only one m/z per row.

    Returns
    -------
    pd.DataFrame with annotation results.
    """
    if isinstance(df_observed, pd.Series):
        df_observed = df_observed.to_frame(name='m/z')
        mz_column = 'm/z'
    elif mz_column not in df_observed.columns:
        numeric_cols = df_observed.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) == 1:
            mz_column = numeric_cols[0]
            print(f"  Auto-detected m/z column: '{mz_column}'")
        else:
            raise ValueError(f"Column '{mz_column}' not found. Available columns: {list(df_observed.columns)}")

    res_label = residual if isinstance(residual, str) else f'{residual:.5f}'
    print(f"Generating theoretical ions for {sequence} (charge {precursor_charge}, "
          f"residual={res_label}, isomax_cutoff={isomax_cutoff}%)...")
    ions, _n = generate_theoretical_ions(sequence, precursor_charge, residual=residual,
                                          isomax_cutoff=isomax_cutoff)
    print(f"  -> {len(ions)} theoretical m/z values generated.")
    sorted_ions, mzs = build_mz_index(ions)

    rows = []
    for rank, (_, row) in enumerate(df_observed.iterrows(), 1):
        observed = row[mz_column]
        label, err_da = match_mz(observed, sorted_ions, mzs, tol_ppm)
        err_ppm = (err_da / observed) * 1e6 if err_da != -1000 else -1000

        rows.append({
            'Ranking': rank,
            'm/z': round(observed, 3),
            'Interpretation': label,
            'error (Da)': round(err_da, 3) if err_da != -1000 else -1000,
            'error (ppm)': round(err_ppm, 2) if err_ppm != -1000 else -1000,
        })

    df_result = pd.DataFrame(rows)
    _write_xlsx(df_result, output_path)
    print(f"Output written to {output_path}")
    return df_result


# ---------------------------------------------------------------------------
# xlsx output
# ---------------------------------------------------------------------------

def _write_xlsx(df, path):
    """Write annotated DataFrame to a formatted xlsx file.

    Cells containing AMBIGUOUS_SEPARATOR ("\\n"), i.e. an ambiguous
    multi-candidate label, get wrap_text=True so each candidate renders on
    its own line, and the row height is grown to fit the tallest cell in
    that row.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Sheet1'

    header_font = Font(bold=True, name='Arial', size=11, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='4472C4')
    data_font = Font(name='Arial', size=11)
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    DEFAULT_ROW_HEIGHT = 15

    headers = list(df.columns)
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
        cell.border = thin_border

    for r in range(len(df)):
        row_max_lines = 1
        for c in range(len(headers)):
            val = df.iloc[r, c]
            cell = ws.cell(row=r + 2, column=c + 1, value=val)
            cell.font = data_font
            cell.border = thin_border
            n_lines = str(val).count('\n') + 1 if isinstance(val, str) else 1
            row_max_lines = max(row_max_lines, n_lines)
            cell.alignment = Alignment(
                horizontal='right' if isinstance(val, (int, float)) else 'left',
                vertical='top', wrap_text=(n_lines > 1),
            )
        if row_max_lines > 1:
            ws.row_dimensions[r + 2].height = DEFAULT_ROW_HEIGHT * row_max_lines

    for c in range(1, len(headers) + 1):
        max_len = len(str(ws.cell(row=1, column=c).value))
        for r in range(2, len(df) + 2):
            val = ws.cell(row=r, column=c).value
            # Width from the longest individual line, not the raw (possibly
            # multi-line) string length.
            longest_line = max((len(line) for line in str(val or '').split('\n')), default=0)
            max_len = max(max_len, longest_line)
        ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = max_len + 3

    wb.save(path)


# ---------------------------------------------------------------------------
# Example / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_ffc = pd.DataFrame({
        'm/z A': [300.156, 565.275, 1008.564],
        'm/z B': [654.359, 814.482, 371.194],
        'Ranking': [1, 2, 3],
    })

    print("=== 2D FFC Annotation ===")
    result_ffc = annotate_ffcs(
        demo_ffc,
        sequence="SAMPLER",
        precursor_charge=3,
        output_path="/tmp/demo_ffc.xlsx",
    )
    print(result_ffc.to_string(index=False))

    demo_peaks = pd.DataFrame({'m/z': [175.119, 300.156, 565.275, 814.482, 1008.564]})

    print("\n=== 1D Peak Annotation ===")
    result_1d = annotate_peaks(
        demo_peaks,
        sequence="SAMPLER",
        precursor_charge=3,
        output_path="/tmp/demo_peaks.xlsx",
    )
    print(result_1d.to_string(index=False))
