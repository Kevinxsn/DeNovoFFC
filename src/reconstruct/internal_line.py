"""
line_alignment.py — Step 1 of internal-line incorporation for DeNovoFFC.

Goal of this first step
-----------------------
Given (a) a *gapped parental reconstruction* (e.g. the path read off the
parental TFFC line, "299.15 - DIA - 194.08 - GQEV - 382.27") and (b) a single
internal / neutral-line TFFC ``(X, Y)`` together with its line ``shift``,
decide how that TFFC relates to the parental line:

    inactive       — a shift-image of (X, Y) coincides with a breakpoint that
                     is *already* on the parental line, so the TFFC generates
                     no genuinely new virtual TFFC.
    non_productive — active, but neither virtual candidate (X+shift, Y) nor
                     (X, Y+shift) lengthens the heaviest path (neither is a
                     "good" TFFC).
    unambiguous    — active, and exactly one virtual candidate is good.
    ambiguous      — active, and both virtual candidates are good.

A virtual TFFC is *good* (per DeNovoFFC) iff adding it to the spectrum
strictly increases the weight of the heaviest source->sink path in the
2D-spectrum graph.  Here every graph edge has weight 1, so "weight" is the
edge count returned by ``path_finding.find_all_longest_paths``.

Mass conventions
----------------
``X``, ``Y`` and ``shift`` are MONOISOTOPIC, singly-protonated ion masses —
the same convention in which b1(V)=100.076, b2(VE)=229.118, and a
complementary (b_i, y_{n-i}) pair sums to

    ParentMass = M + 2*proton          (M = neutral peptide mass).

A neutral line with offset ``shift`` is the locus  X + Y = ParentMass - shift;
shifting a point on it by (shift, 0) or (0, shift) lands it on the parental
line  X + Y = ParentMass.

The graph itself is built in path_finding's *neutral residue-sum* coordinates
(b-axis -> residue sum, y-axis -> water + complementary residue sum), so each
ion-mass coordinate is converted by subtracting one proton before insertion.

This module reuses path_finding.{construct_graph_nodes, build_adjacency_graph,
find_all_longest_paths} unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from path_finding import (
    amino_acid_masses_merge,
    construct_graph_nodes,
    build_adjacency_graph,
    find_all_longest_paths,
    SOURCE_NODE,
)
from edges import load_edges

# ── Physical constants (match path_finding's SOURCE_NODE water exactly) ──────
PROTON = 1.007276
WATER = 18.01056            # == SOURCE_NODE[1]; keep identical to avoid dup coords


# ── Parsing the gapped reconstruction ────────────────────────────────────────
def parse_gapped_reconstruction(recon, aa_masses=None, sep="-"):
    """
    Parse a gapped reconstruction into ordered segment masses and the
    cumulative breakpoint residue-sums it implies.

    Accepts either a string  "299.15 - DIA - 194.08 - GQEV - 382.27"
    or a list                [299.15, "DIA", 194.08, "GQEV", 382.27].

    A numeric token is one gap segment.  A letter token expands to one resolved
    residue segment per character (each producing its own breakpoint).

    Returns
    -------
    (segments, breakpoints, M_res)
        segments    : list[float]  ordered segment masses
        breakpoints : list[float]  cumulative residue-sum at every *internal*
                                   cleavage site (the trailing C-terminus,
                                   i.e. the full residue sum, is excluded)
        M_res       : float        total residue mass (sum of all segments)
    """
    if aa_masses is None:
        aa_masses = amino_acid_masses_merge
    # I and L are isobaric (113.08406); path_finding stores this mass under "J".
    aa_masses = {**aa_masses,
                 "I": aa_masses.get("J", 113.08406),
                 "L": aa_masses.get("J", 113.08406)}
    tokens = ([t.strip() for t in recon.split(sep) if t.strip()]
              if isinstance(recon, str) else list(recon))

    segments = []
    for tok in tokens:
        if isinstance(tok, (int, float)):
            segments.append(float(tok))
            continue
        s = str(tok).strip()
        try:
            segments.append(float(s))                       # numeric gap
        except ValueError:
            for ch in s:                                    # resolved residues
                if ch not in aa_masses:
                    raise ValueError(f"Unknown residue '{ch}' in token '{s}'")
                segments.append(aa_masses[ch])

    cum, breakpoints = 0.0, []
    for seg in segments:
        cum += seg
        breakpoints.append(cum)
    M_res = breakpoints[-1] if breakpoints else 0.0
    return segments, breakpoints[:-1], M_res                # drop C-terminus


# ── Spectrum / heaviest-path helpers ─────────────────────────────────────────
def parental_spectrum(breakpoints, M_res):
    """
    1D spectrum-coordinate list (neutral residue-sum convention) for the
    parental reconstruction, mirroring path_finding's `spec` construction:
    every breakpoint contributes a b-axis coordinate (residue sum) and its
    complementary y-axis coordinate (water + complementary residue sum).
    """
    coords = {0.0, WATER, M_res + WATER}
    for s in breakpoints:
        coords.add(s)                                       # b-side
        coords.add(WATER + (M_res - s))                     # complementary y-side
    return sorted(coords)


def _lower_half(spec):
    """Lower half of the (symmetric) spectrum — same convention as path_finding."""
    s = sorted(spec)
    mid = len(s) // 2 + 1
    return sorted(s[:mid])


def heaviest_path_len(spec, allowed_masses, total, threshold=0.02):
    """Edge count of the heaviest source->sink path for a given spectrum."""
    lower = _lower_half(spec)
    nodes = construct_graph_nodes(lower)
    graph = build_adjacency_graph(
        nodes=nodes,
        allowed_masses=allowed_masses,
        total_peptide_mass=total,
        threshold=threshold,
    )
    max_len, _ = find_all_longest_paths(graph, start_node=SOURCE_NODE, max_paths=1)
    return max_len


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class TFFCClass:
    classification: str          # inactive | non_productive | unambiguous | ambiguous
    tffc: tuple                  # (X, Y) on the neutral line
    shift: float
    cand_h: tuple                # (X+shift, Y)  horizontal-shift virtual TFFC
    cand_v: tuple                # (X, Y+shift)  vertical-shift virtual TFFC
    good_h: bool = False
    good_v: bool = False
    L0: int = None               # baseline heaviest-path length
    Lh: int = None               # length with cand_h added
    Lv: int = None               # length with cand_v added
    inactive_reason: str = ""
    add: list = field(default_factory=list)   # virtual TFFC(s) to add to grandmaster

    def __str__(self):
        head = (f"TFFC (X,Y) = ({self.tffc[0]:.3f}, {self.tffc[1]:.3f})   "
                f"shift = {self.shift:.3f}   ->  {self.classification.upper()}")
        ch = f"  horizontal (X+shift, Y) = ({self.cand_h[0]:.3f}, {self.cand_h[1]:.3f})"
        cv = f"  vertical   (X, Y+shift) = ({self.cand_v[0]:.3f}, {self.cand_v[1]:.3f})"
        if self.classification == "inactive":
            return f"{head}\n{ch}\n{cv}\n  reason: {self.inactive_reason}"
        lines = [head, ch + f"   good={self.good_h}  (L0={self.L0} -> {self.Lh})",
                 cv + f"   good={self.good_v}  (L0={self.L0} -> {self.Lv})"]
        if self.add:
            adds = ", ".join(f"({a[0]:.3f}, {a[1]:.3f})" for a in self.add)
            w = "1/2 each" if self.classification == "ambiguous" else "1"
            lines.append(f"  -> add to grandmaster: {adds}   (weight {w})")
        else:
            lines.append("  -> add nothing")
        return "\n".join(lines)


# ── The Step-1 classifier ────────────────────────────────────────────────────
def classify_tffc(recon, tffc, shift, allowed_masses,
                  total=None, threshold=0.02, coord_tol=0.01,
                  aa_masses=None):
    """
    Classify one neutral-line TFFC ``(X, Y)`` against a gapped parental
    reconstruction.  See module docstring for the four possible outcomes.

    Parameters
    ----------
    recon          : str | list   gapped reconstruction
    tffc           : (X, Y)       monoisotopic, protonated ion masses
    shift          : float        neutral-line offset (monoisotopic)
    allowed_masses : list[float]  compound AA masses (from edges.json)
    total          : float        neutral peptide mass M; defaults to M_res+water
    threshold      : float        mass tolerance for graph edges (Da)
    coord_tol      : float        tolerance for the inactive coincidence test (Da)

    Returns
    -------
    TFFCClass
    """
    X, Y = float(tffc[0]), float(tffc[1])
    shift = float(shift)

    segments, breakpoints, M_res = parse_gapped_reconstruction(recon, aa_masses)
    if total is None:
        total = M_res + WATER

    # Existing parental b-axis coordinates (monoisotopic), incl. the two termini.
    existing_b = {PROTON, M_res + PROTON} | {s + PROTON for s in breakpoints}

    cand_h = (X + shift, Y)             # horizontal shift -> new b-coord X+shift
    cand_v = (X, Y + shift)             # vertical   shift -> b-coord stays X

    def lands_on_existing(bcoord):
        return any(abs(bcoord - e) <= coord_tol for e in existing_b)

    h_on, v_on = lands_on_existing(cand_h[0]), lands_on_existing(cand_v[0])
    if h_on or v_on:
        why = []
        if h_on:
            why.append(f"(X+shift)={cand_h[0]:.3f} coincides with an existing breakpoint")
        if v_on:
            why.append(f"X={cand_v[0]:.3f} coincides with an existing breakpoint")
        return TFFCClass("inactive", (X, Y), shift, cand_h, cand_v,
                         inactive_reason="; ".join(why))

    # Active: test goodness of each virtual candidate via the heaviest-path delta.
    base_spec = parental_spectrum(breakpoints, M_res)
    L0 = heaviest_path_len(base_spec, allowed_masses, total, threshold)

    def goodness(cand):
        # a virtual TFFC = one breakpoint: add its b-coord and complementary y-coord
        spec = base_spec + [cand[0] - PROTON, cand[1] - PROTON]
        L = heaviest_path_len(spec, allowed_masses, total, threshold)
        return (L > L0), L

    good_h, Lh = goodness(cand_h)
    good_v, Lv = goodness(cand_v)

    if good_h and good_v:
        cls, add = "ambiguous", [cand_h, cand_v]
    elif good_h or good_v:
        cls, add = "unambiguous", ([cand_h] if good_h else [cand_v])
    else:
        cls, add = "non_productive", []

    return TFFCClass(cls, (X, Y), shift, cand_h, cand_v,
                     good_h, good_v, L0, Lh, Lv, "", add)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    EDGE_FILE = "edges.json"
    if not Path(EDGE_FILE).exists():
        from edges import build_and_save
        print(f"Edge file '{EDGE_FILE}' not found — generating (max_n=4) ...")
        build_and_save(max_n=4, path=EDGE_FILE)
    allowed = load_edges(EDGE_FILE)
    print(f"Loaded {len(allowed):,} allowed masses\n")

    # Parental reconstruction of VEADIAGHGQEVLIR (3+), missing breakpoints b1,b2,b7,b13,b14
    recon = "299.15 - DIA - 194.08 - GQEV - 382.27"

    # A neutral-line TFFC whose VERTICAL image (X, Y+shift) inserts breakpoint b7
    # (splits the 194.08 = G+H gap), while its horizontal image does not help.
    #tffc, shift = (656.325, 851.461), 100.076      # shift = b1(V) loss
    tffc, shift = (1203.631, 175.119),  229.112 
    print(classify_tffc(recon, tffc, shift, allowed))