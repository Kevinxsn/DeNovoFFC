"""DeNovoMMS -- MMS (2D fragment-fragment correlation) peptide sequencing.

Two routes through one set of stages, chosen by whether the peptide is known:

    unknown peptide   [deconvolution] -> de novo MMS
    known peptide     [deconvolution] -> annotation pipeline
                                         (which runs de novo internally)

See DESIGN.md for why, and README.md for how.
"""
__version__ = "1.0.0"
