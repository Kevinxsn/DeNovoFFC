"""c / z-dot ion series -- the three constants that turn b/y code into ETD code.

Everything in this directory that differs from the b/y pipeline reduces to the
numbers below.  They are collected here so the two copied annotators
(``ffc_annotator_cz.py``, ``line_internal_annotation_cz.py``) state the same
chemistry, and so the derivation is written down once.

The chemistry
-------------
ETD transfers an electron to ``[M+zH]z+``; the charge-reduced radical cleaves the
N--Ca bond, giving an even-electron **c** fragment and a radical **z-dot**
fragment.  Writing ``prefix[k]`` for the sum of residues 1..k and ``M`` for the
neutral peptide mass (``prefix[n] + terminus``):

    b_k  = prefix[k]                     c_k   = prefix[k] + NH3
    y_j  = M - prefix[n-j]               z._j  = M - prefix[n-j] - NH3 + H

so ``c + z. = M + H`` -- the electron plus the proton it neutralised arrive as
one hydrogen atom, which is why an ETD pair is *not* mass-balanced against the
precursor the way a b/y pair is.

An **internal** ETD fragment is cut twice: its N-terminal end is a z-dot-type end
(``-NH3 + H``) and its C-terminal end is a c-type end (``+NH3``), so

    internal_cz(s..e) = sum(residues s..e) + H

This ``+H`` is the reason c/z cannot be faked with a "virtual peptide".  Any
re-parametrisation of the residue masses that puts ``+NH3`` on every prefix
necessarily cancels it out of every internal fragment (an internal's mass is a
*difference* of prefixes), leaving internals 1.008 Da wrong -- ~1000 ppm, two
orders of magnitude past the 20 ppm matching window.  Hence the copies.

Verified against the collaborator's own 1D annotation of
``Intensity_Sum.LL37_ETD_Z6_1ms`` (LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES):

    ion            annotated m/z    theory here     |dev|
    b2 (1+)          227.17562       227.17537      0.0003
    c2 (1+)          244.20190       244.20192      0.0000
    y9 (2+)          536.29941       536.29890      0.0005
    z.9 (2+)         528.28979       528.28952      0.0003

The file also carries ``z`` (= y - NH3) and ``c.`` (= c - H) labels.  Those are
the +/-1.008 Da satellites of the main series and sit 0.0045 Da from a C13 step,
which is inside the matching window for any fragment above ~230 Da -- generating
them as separate series would multiply the candidate lists without adding
resolvable information, so only the dominant **c / z-dot** pair is generated
here.  The isotope expansion already covers where those satellites land.

The coordinate basis (why ``TOTAL_STAR``)
-----------------------------------------
The master-TFFC-set and reconstruction machinery works in *breakpoint
coordinates*: a bond ``k`` is the pair ``(prefix[k], total - prefix[k])``, and
``fold()`` canonicalises the two onto one node.  That works because in b/y a
measured neutral ``N`` reads either as ``prefix[k]`` (b) or ``total - prefix[k]``
(y) -- an involution.

For c/z, subtract ``NH3`` from every measured ``[M+H]+`` mass and the same
structure reappears:

    u = x - NH3 - PROTON      c ion  ->  u = prefix[k]
                              z. ion ->  u = TOTAL_STAR - prefix[k]

with ``TOTAL_STAR = total + H - 2*NH3``.  So the whole coordinate layer is reused
unchanged; only the peptide "total" it is told about changes.  ``TOTAL_STAR`` is
a bookkeeping constant, not a mass -- the implied terminus
(``TOTAL_STAR - prefix[n]``) is negative, which is harmless: ``path_finding``
only ever uses it as the y-axis origin, and ``total_star - term_star`` is still
exactly the real residue sum, so ``is_reconstruction_correct`` still compares
against the true sequence.
"""

#: Proton mass, as used by ``deiso/python_code/peptide.py`` (AA.element_masses).
#: Kept identical so c/z masses and b/y masses are on the same scale.
PROTON = 1.00725

#: Ammonia -- the c-ion increment over b.
NH3 = 17.02655

#: Water -- the standard C-terminal residual.
H2O = 18.01056

#: Hydrogen *atom* (not a proton): the electron-plus-proton that ETD delivers.
H_ATOM = 1.00783

# ── The three series offsets ────────────────────────────────────────────────
#: b_k -> c_k
NTERM_OFFSET = NH3                      # +17.02655
#: y_j -> z-dot_j
CTERM_OFFSET = H_ATOM - NH3             # -16.01872
#: internal(s..e) -> internal_cz(s..e)
INTERNAL_OFFSET = H_ATOM                # +1.00783
#: c + z-dot = M + PARENT_OFFSET  (a b/y pair sums to M exactly)
PARENT_OFFSET = H_ATOM                  # +1.00783

#: Display names, so a coverage table reads ``c12, z25`` rather than ``b12, y25``.
KIND_LABEL = {"b": "c", "y": "z", "internal": "i"}


def total_star(total: float) -> float:
    """Peptide total on the ``u = x - NH3`` coordinate basis (see module docs)."""
    return total + H_ATOM - 2.0 * NH3


def to_coord(mzH: float) -> float:
    """A measured ``[M+H]+`` fragment mass -> its breakpoint coordinate.

    ``c`` ions land on ``prefix[k]``; ``z-dot`` ions land on
    ``total_star - prefix[k]``.  Which of the two a given mass is stays
    undecided -- that is exactly the ambiguity ``fold()`` canonicalises away.
    """
    return mzH - NH3 - PROTON


def parent_charge_for_lines(precursor_charge: int) -> int:
    """Charge sum of a complementary ETD pair.

    Electron transfer consumes one of the precursor's protons, so the c/z-dot
    pair from ``[M+zH]z+`` carries ``z - 1`` charges between them, not ``z``.
    The reconstructed line mass is nearly unchanged --

        v = c + a*p + z. + b*p = (M + H) + (z-1)*p = M + z*p + (H - p)

    -- i.e. the ETD parental line sits only ``H_ATOM - PROTON`` = 0.00058 Da (one
    electron) above where a b/y parental line would sit, but on charge sum
    ``z - 1``.  A detector that identifies parental lines by ``i + j == z`` will
    therefore miss it; one that identifies them by near-zero shift will not.
    """
    return precursor_charge - 1
