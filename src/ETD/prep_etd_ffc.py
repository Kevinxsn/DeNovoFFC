#!/usr/bin/env python3
"""
ETD/EThcD data prep -- block-format covariance file -> flat FFC table + intensity
================================================================================

The ETD_LLG dataset arrives in a *different* file layout from everything under
``$DENOVOFFC_DATA_ROOT/original_mms/``.  This script converts it to the flat table
the deconvolution expects, so nothing downstream has to know about the
difference.

The two layouts
---------------

**What the existing pipeline reads** (``data_prep/combine_ffc_intensity.py``):
one header line with the row count, then one line per FFC ::

    2078
    157.10868359  157.10868359  0.92348487E+04  0.92314971E+04  0.0  -1
    157.10868359  171.11264824  0.39839657E+03  0.39233931E+03  63.7  53616
    ...                                             ^Cov  ^PartialCov ^Score ^Rank

**What ETD_LLG gives us** -- a *blocked, upper-triangular* layout.  One header
line with the peak count, then, for every peak ``i``, a 3-field block header
followed by one 5-field line per partner ``j > i`` ::

    10133
       1   150.04605800   2160            <- peak 1, m/z, 2160 partners follow
       2   150.04633900   0.10671960E+02  0.00000000E+00  18333253
       3   150.04665100   0.17090605E+01  0.00000000E+00  18333252
      12   163.05433400   0.13485838E-01  0.60806321E-01   3382948
       ...
       2   150.04633900   2673            <- peak 2 starts

Three consequences, all handled here:

* **Only ``j > i`` is stored.**  Each physical FFC appears exactly once, and the
  self-pair ``(i, i)`` is absent.  The old format stored the self-pair with
  ``Ranking = -1``; nothing downstream needs it (``prepare_ffc_data`` drops
  ``Ranking == -1`` anyway), so we simply do not emit one.
* **There are two numeric columns, not three.**  The old format carries
  ``Covariance``, ``Partial Cov.`` and ``Score``; this one carries two values
  plus the ranking.  The five fields are ``idx, mz, Partial Cov., Score,
  Ranking`` -- **the dropped column is `Covariance`, not `Partial Cov.`**
  Getting that backwards is silent, because the pipeline's first pre-processing
  step filters on ``Partial Cov. > 0``.

  This was settled empirically by the parallel ``HAD_more_fragement`` work,
  which had the luxury of a matching *old-format* flat file for the same run to
  correlate against (field 3 vs old Partial Cov. r = 0.9915; field 4 vs old
  Score r = 0.7483 and monotone in Ranking).  It is consistent with what this
  file shows on its own: column 4 is strictly monotonic with ``Ranking`` (the
  top-scoring pair is rank 2, and every pair whose column 4 is exactly 0 is
  dumped past rank 18 000 000), so column 4 is ``Score``.

  ``Covariance`` is therefore absent.  It is filled with a **copy of column 3**
  rather than left empty, so downstream code that expects the column to exist
  keeps working; nothing reads it for a decision.  Note also that every value in
  these exports is strictly positive -- they are already sign-filtered -- so the
  ``Partial Cov. > 0`` filter drops nothing.
* **Ranking is global over the full peak matrix** (max 51 331 449 =
  10133x10132/2), not over the stored rows, and no row carries ``-1``.  It is
  passed through untouched; ``prepare_ffc_data`` only ever sorts on it.

Intensities
-----------
``Intensity_Sum.<name>`` has exactly one row per peak, in the same order as the
covariance file's blocks (10133 / 13528 rows, matching the peak counts).  So the
intensity join is an **index** lookup, not the m/z-tolerance ``merge_asof`` that
``combine_ffc_intensity.py`` has to do for the older datasets -- the m/z values
are byte-identical in the two files, which this script asserts.  That removes
the incomplete-intensity failure mode ``deconvolution.md`` section 10 describes
for the ``HAD_33_Z4_E15_I*`` datasets: here every peak has an intensity.

Usage
-----
    python prep_etd_ffc.py                                   # everything in --raw-dir
    python prep_etd_ffc.py --list                            # show what would run
    python prep_etd_ffc.py --raw-dir $DENOVOFFC_DATA_ROOT/ETD_LLG_more_data
    python prep_etd_ffc.py --only LLG_37_Z6_ETD

Datasets are **auto-discovered**: every ``Covariance_Data.X`` with a matching
``Intensity_Sum.X`` is converted, and the dataset NAME is derived from ``X`` by
``derive_name`` (``LL37_ETD_Z6_13.84ms`` -> ``LLG_37_Z6_ETD_13p84ms``) so
``data_prep/peptide_info.py`` can resolve it.  ``DATASETS`` pins the two
original names, whose results are already published under the short spelling.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd


# [DeNovoFFC] The research defaults pointed at one author's data directories;
# a public installation has none, so both directories are now required
# arguments.  The generic, single-file entry point is ``denovoffc prepare``
# (pipeline/prepare.py -> ``prepare_blocked`` below).
RAW_DIR = None
OUT_DIR = None

#: Explicit NAME -> (covariance file, intensity file) overrides.  Anything not
#: listed here is auto-discovered by ``discover()`` below; these two are pinned
#: because their generated names would otherwise be ``..._ETD_1MS`` /
#: ``..._ETHCD_25_1MS`` and the published results are already filed under the
#: shorter names.
DATASETS = {
    "LLG_37_Z6_ETD": (
        "Covariance_Data.LL37_ETD_Z6_1ms",
        "Intensity_Sum.LL37_ETD_Z6_1ms",
    ),
    "LLG_37_Z6_ETHCD25": (
        "Covariance_Data.LL37_EThcD-25_Z6_1ms",
        "Intensity_Sum.LL37_EThcD-25_Z6_1ms",
    ),
}

OUT_COLUMNS = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score",
               "Ranking", "intensity A", "intensity B"]

#: ``LL37`` -> the ``LLG_37`` spelling ``data_prep/peptide_info.py`` parses.
#: Its regex is ``^([A-Za-z]{3})_(\d+)_[Zz](\d+)`` -- exactly three letters,
#: underscore, length, underscore, Z<charge> -- everything after that is free
#: text.  The collaborator writes the *common* name (LL-37), which is two
#: letters, so ``PEPTIDE_ALIASES`` maps it onto the sheet's three-residue key.
_TAG_RE = re.compile(r"^([A-Za-z]{2,4})(\d+)$")

#: Collaborator's peptide tag -> the three-letter key in ``pep_seq.xlsx``
#: (which is the peptide's first three residues).  LL-37 starts ``LLG...``.
PEPTIDE_ALIASES = {"LL": "LLG"}


def derive_name(stem: str) -> str | None:
    """``LL37_ETD_Z6_13.84ms`` -> ``LLG_37_Z6_ETD_13p84ms``, or None.

    The collaborator's stems are ``<peptide><len>_<method>_Z<charge>_<extra>``;
    ``peptide_info`` wants ``<PEP>_<len>_Z<charge>_<free text>``, so the peptide
    token is split and the method moved after the charge.  Dots become ``p`` --
    a NAME becomes a filename and later an Excel sheet/column key, and a dotted
    name reads like an extension in half the places it appears.
    """
    parts = stem.split("_")
    m = _TAG_RE.match(parts[0]) if parts else None
    if not m:
        return None
    tag, length = m.group(1).upper(), m.group(2)
    tag = PEPTIDE_ALIASES.get(tag, tag)
    if len(tag) != 3:                       # peptide_info wants exactly three
        return None
    charge = next((p for p in parts[1:] if re.fullmatch(r"[Zz]\d+", p)), None)
    if charge is None:
        return None
    rest = [p for p in parts[1:] if p != charge]
    suffix = "_".join(rest).replace(".", "p").replace("-", "")
    return f"{tag}_{length}_{charge.upper()}_{suffix}".rstrip("_")


def discover(raw_dir: str) -> dict:
    """Every ``Covariance_Data.X`` in ``raw_dir`` with a matching
    ``Intensity_Sum.X``, keyed by the NAME ``derive_name`` gives it.

    ``DATASETS`` wins where a stem is pinned there, so re-running over the
    original folder keeps the published names.
    """
    pinned = {cov: name for name, (cov, _i) in DATASETS.items()}
    out = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.startswith("Covariance_Data."):
            continue
        stem = fname[len("Covariance_Data."):]
        inten = f"Intensity_Sum.{stem}"
        if not os.path.exists(os.path.join(raw_dir, inten)):
            print(f"[skip] {fname}: no matching {inten}", file=sys.stderr)
            continue
        name = pinned.get(fname) or derive_name(stem)
        if name is None:
            print(f"[skip] {fname}: cannot derive a peptide_info NAME from "
                  f"{stem!r}", file=sys.stderr)
            continue
        out[name] = (fname, inten)
    return out


def read_intensity(path: str) -> pd.DataFrame:
    """``Intensity_Sum.*`` -> DataFrame(mz, intensity), in file order.

    Columns are ``intensity, m/z, error, annotation``; the annotation may
    contain spaces and ``/``-separated alternatives, so only the first three
    fields are parsed.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            rows.append((float(parts[1]), float(parts[0])))
    return pd.DataFrame(rows, columns=["mz", "intensity"])


def read_blocked_covariance(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand the blocked upper-triangular covariance file.

    Returns ``(ffc_frame, peak_mz)`` where ``ffc_frame`` has one row per stored
    FFC and ``peak_mz`` is the peak list in block order (index 0 = peak 1).
    """
    # Ragged rows: 1 field (the peak count), 3 (a block header) or 5 (an FFC).
    # Reading all five slots and letting the short rows come back NaN is both
    # simpler and faster than any line-by-line dispatch.
    raw = pd.read_csv(
        path, sep=r"\s+", header=None, names=["c0", "c1", "c2", "c3", "c4"],
        engine="c", dtype=np.float64,
    )

    n_peaks = int(raw["c0"].iloc[0])
    body = raw.iloc[1:]

    # A block header is the only row with exactly three fields, i.e. the only
    # one whose 4th slot is empty.  Everything else is an FFC.
    is_header = body["c3"].isna().to_numpy()

    peak_idx = body["c0"].to_numpy()[is_header].astype(np.int64)
    peak_mz = body["c1"].to_numpy()[is_header]
    if len(peak_mz) != n_peaks:
        raise ValueError(
            f"{os.path.basename(path)}: header says {n_peaks} peaks, "
            f"found {len(peak_mz)} block headers")
    if not np.array_equal(peak_idx, np.arange(1, n_peaks + 1)):
        raise ValueError(f"{os.path.basename(path)}: block headers are not "
                         f"peaks 1..{n_peaks} in order")

    # Each FFC's "m/z A" is the m/z of the block it sits in: broadcast the
    # header m/z down over the rows that follow it.
    block_of_row = np.cumsum(is_header) - 1          # 0-based block index
    ffc_mask = ~is_header

    a_idx = block_of_row[ffc_mask]
    out = pd.DataFrame({
        "m/z A": peak_mz[a_idx],
        "m/z B": body["c1"].to_numpy()[ffc_mask],
        # Column 3 is Partial Cov. -- the DROPPED column is Covariance (see the
        # module docstring; settled against a matching old-format file by the
        # parallel HAD_more_fragement work).  Covariance is filled with a copy
        # so downstream code that expects the column still works; nothing reads
        # it for a decision.
        "Covariance": body["c2"].to_numpy()[ffc_mask],
        "Partial Cov.": body["c2"].to_numpy()[ffc_mask],
        "Score": body["c3"].to_numpy()[ffc_mask],
        "Ranking": body["c4"].to_numpy()[ffc_mask].astype(np.int64),
    })
    # The partner index (column 0 of an FFC row) is dropped: "m/z B" carries the
    # same information and is what the pipeline keys on.  Keep it for the
    # intensity join, though -- it is an exact index, unlike the m/z.
    out.attrs["b_index"] = body["c0"].to_numpy()[ffc_mask].astype(np.int64)
    out.attrs["a_index"] = a_idx + 1
    return out, peak_mz


def convert(name: str, cov_path: str, int_path: str, out_path: str,
            mz_atol: float = 1e-6) -> dict:
    t0 = time.time()
    print(f"── {name} " + "─" * (60 - len(name)))
    print(f"   covariance  {cov_path}")
    print(f"   intensity   {int_path}")

    ffc, peak_mz = read_blocked_covariance(cov_path)
    print(f"   peaks       {len(peak_mz)}")
    print(f"   FFC rows    {len(ffc):,}  (upper triangular, self-pairs absent)")

    inten = read_intensity(int_path)
    if len(inten) != len(peak_mz):
        raise ValueError(
            f"{name}: intensity file has {len(inten)} peaks but the covariance "
            f"file has {len(peak_mz)}; the index join is not valid")

    # The two files are supposed to describe the same peak list in the same
    # order.  Verify before trusting the index join -- a silent misalignment
    # here would attach every intensity to the wrong peak.
    dev = np.abs(inten["mz"].to_numpy() - peak_mz)
    if dev.max() > mz_atol:
        raise ValueError(
            f"{name}: peak m/z disagree between the two files "
            f"(max |Δ| = {dev.max():.3e} > {mz_atol:.0e}); "
            f"the index join is not valid")
    print(f"   peak m/z    agree to {dev.max():.1e} Da -> index join is exact")

    intensity = inten["intensity"].to_numpy()
    ffc["intensity A"] = intensity[ffc.attrs["a_index"] - 1]
    ffc["intensity B"] = intensity[ffc.attrs["b_index"] - 1]

    n_missing = int(ffc[["intensity A", "intensity B"]].isna().any(axis=1).sum())
    print(f"   intensities {len(ffc) - n_missing:,} complete, {n_missing:,} "
          f"incomplete")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffc[OUT_COLUMNS].to_csv(out_path, sep="\t", index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"   wrote       {out_path}  ({size_mb:,.0f} MB, "
          f"{time.time() - t0:.1f}s)\n")

    return {"name": name, "peaks": len(peak_mz), "ffcs": len(ffc),
            "missing_intensity": n_missing, "path": out_path}


# =============================================================================
# [DeNovoFFC] Generic streaming preparation of ONE explicit file pair
# =============================================================================
# ``read_blocked_covariance`` above loads the whole export into a DataFrame,
# which some inputs (tens of millions of rows) cannot afford, and it needs the
# dataset-name machinery of ``discover``.  ``prepare_blocked`` streams blocks
# straight to the output, validates everything the layout promises, and never
# makes a lossy top-N cut: deconvolution and scoring want different row sets.

PREP_FORMAT_VERSION = 1


class BlockedFormatError(ValueError):
    """The covariance export does not have the documented blocked layout."""


def read_peak_table(path: str) -> list:
    """``Intensity_Sum.*`` -> ``[(mz, mz_token, intensity_token), ...]`` in file
    order; row k is peak k+1.  Only the first three fields are parsed (the
    annotation may contain spaces)."""
    out = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split(None, 3)
            if not parts:
                continue
            if len(parts) < 2:
                raise BlockedFormatError(f"{path}:{lineno}: expected 'intensity m/z ...'")
            try:
                mz = float(parts[1])
                float(parts[0])
            except ValueError:
                raise BlockedFormatError(f"{path}:{lineno}: non-numeric intensity/m/z") from None
            out.append((mz, parts[1], parts[0]))
    return out


def _num(tok: str, path: str, lineno: int, what: str) -> float:
    try:
        v = float(tok)
    except ValueError:
        raise BlockedFormatError(f"{path}:{lineno}: {what} is not a number: {tok!r}") from None
    if v != v:
        raise BlockedFormatError(f"{path}:{lineno}: {what} is NaN")
    return v


def prepare_blocked(cov_path: str, out_path: str, int_path: str = None,
                    mz_atol: float = 1e-6, covariance_alias: bool = True) -> dict:
    """Stream ``cov_path`` (blocked, upper-triangular) to a flat FFC table.

    Layout: a peak count, then for every peak ``i`` a block header
    ``(i, m/z, n_partners)`` followed by ``n_partners`` rows
    ``(j, m/z, Partial Cov., Score, Ranking)`` with ``j > i``.

    * The third partner field is PARTIAL covariance.  The export has no
      ordinary covariance; with ``covariance_alias`` a ``Covariance`` column
      copying Partial Cov. is written for downstream code that expects the
      column, and the summary records it as a synthesized alias.
    * Intensities are joined by validated peak INDEX (row k of the intensity
      file is peak k+1) and the m/z of both files must agree -- never a
      nearest-m/z join.
    * Block sizes, index order (1..N, no gaps, no duplicates), triangular
      orientation, numeric fields and partner m/z are all checked; any failure
      raises ``BlockedFormatError`` and the partial output is removed.
    * Original rankings are passed through untouched, gaps and all.

    Returns a summary dict (row counts, missing values, ranges).
    """
    peaks = read_peak_table(int_path) if int_path else None
    header_mz = {}                      # index -> m/z (only without a peak table)
    pending_partner = {}                # j -> partner m/z first seen (no peak table)
    cols = ["m/z A", "m/z B"] + (["Covariance"] if covariance_alias else []) + \
           ["Partial Cov.", "Score", "Ranking", "intensity A", "intensity B"]
    n_rows = n_blocks = n_missing_int = n_nonpos_pcov = 0
    rank_min = rank_max = None
    tmp = out_path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        with open(cov_path) as fin, open(tmp, "w") as fout:
            fout.write("\t".join(cols) + "\n")
            first = fin.readline()
            try:
                n_peaks = int(first.split()[0])
            except (ValueError, IndexError):
                raise BlockedFormatError(f"{cov_path}:1: expected the peak count") from None
            if peaks is not None and len(peaks) != n_peaks:
                raise BlockedFormatError(
                    f"intensity file has {len(peaks)} peaks but the covariance file "
                    f"declares {n_peaks}; the index join is not valid")
            cur_i, cur_mz_tok, cur_int, expected, got = 0, None, "", 0, 0
            lineno = 1
            for line in fin:
                lineno += 1
                f = line.split()
                if not f:
                    continue
                if len(f) == 3:                                  # block header
                    if got != expected:
                        raise BlockedFormatError(
                            f"{cov_path}:{lineno}: block {cur_i} declared {expected} "
                            f"partners but has {got}")
                    try:
                        i = int(f[0])
                        n_part = int(f[2])
                    except ValueError:
                        raise BlockedFormatError(f"{cov_path}:{lineno}: bad block header") from None
                    if i != cur_i + 1:
                        raise BlockedFormatError(
                            f"{cov_path}:{lineno}: block {i} follows block {cur_i}; "
                            "headers must be peaks 1..N in order, without gaps or repeats")
                    if i > n_peaks:
                        raise BlockedFormatError(f"{cov_path}:{lineno}: block {i} > {n_peaks} peaks")
                    mz = _num(f[1], cov_path, lineno, "block m/z")
                    if peaks is not None:
                        if abs(peaks[i - 1][0] - mz) > mz_atol:
                            raise BlockedFormatError(
                                f"{cov_path}:{lineno}: peak {i} m/z {mz} != intensity file "
                                f"{peaks[i - 1][0]}")
                        cur_int = peaks[i - 1][2]
                    else:
                        header_mz[i] = mz
                        if i in pending_partner and abs(pending_partner.pop(i) - mz) > mz_atol:
                            raise BlockedFormatError(
                                f"{cov_path}:{lineno}: partner m/z of peak {i} disagrees "
                                "with its block header")
                    cur_i, cur_mz_tok, expected, got = i, f[1], n_part, 0
                    n_blocks += 1
                    continue
                if len(f) != 5:
                    raise BlockedFormatError(f"{cov_path}:{lineno}: expected 3 or 5 fields, "
                                             f"got {len(f)}")
                if cur_i == 0:
                    raise BlockedFormatError(f"{cov_path}:{lineno}: partner row before any block")
                try:
                    j = int(f[0])
                    rank = int(f[4])
                except ValueError:
                    raise BlockedFormatError(f"{cov_path}:{lineno}: bad partner index/ranking") from None
                if not (cur_i < j <= n_peaks):
                    raise BlockedFormatError(
                        f"{cov_path}:{lineno}: partner {j} of peak {cur_i} breaks the "
                        "upper-triangular layout (need i < j <= N)")
                mzb = _num(f[1], cov_path, lineno, "partner m/z")
                pcov = _num(f[2], cov_path, lineno, "Partial Cov.")
                _num(f[3], cov_path, lineno, "Score")
                if peaks is not None:
                    if abs(peaks[j - 1][0] - mzb) > mz_atol:
                        raise BlockedFormatError(
                            f"{cov_path}:{lineno}: partner {j} m/z {mzb} != intensity file "
                            f"{peaks[j - 1][0]}")
                    int_b = peaks[j - 1][2]
                else:
                    prev = pending_partner.setdefault(j, mzb)
                    if abs(prev - mzb) > mz_atol:
                        raise BlockedFormatError(f"{cov_path}:{lineno}: peak {j} listed "
                                                 "with two different m/z values")
                    int_b = ""
                if peaks is None:
                    n_missing_int += 1
                n_nonpos_pcov += pcov <= 0
                rank_min = rank if rank_min is None else min(rank_min, rank)
                rank_max = rank if rank_max is None else max(rank_max, rank)
                row = [cur_mz_tok, f[1]] + ([f[2]] if covariance_alias else []) + \
                      [f[2], f[3], f[4], cur_int, int_b]
                fout.write("\t".join(row) + "\n")
                got += 1
                n_rows += 1
            if got != expected:
                raise BlockedFormatError(f"{cov_path}: last block {cur_i} declared {expected} "
                                         f"partners but has {got}")
            if n_blocks != n_peaks:
                raise BlockedFormatError(f"{cov_path}: header declares {n_peaks} peaks, "
                                         f"found {n_blocks} blocks")
            if pending_partner:
                raise BlockedFormatError(f"{cov_path}: partner peak(s) never declared: "
                                         f"{sorted(pending_partner)[:5]}")
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dict(format_version=PREP_FORMAT_VERSION, n_peaks=n_peaks, n_ffc_rows=n_rows,
                columns=cols, covariance_column=("synthesized alias of Partial Cov."
                                                 if covariance_alias else "absent"),
                partial_cov_column="Partial Cov. (third partner field)",
                missing_intensity_rows=n_missing_int,
                nonpositive_partial_cov_rows=int(n_nonpos_pcov),
                ranking_min=rank_min, ranking_max=rank_max,
                ranking_note="original rankings, passed through untouched",
                intensity_join="peak index (validated m/z)" if peaks is not None else "none")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw-dir", required=True)                      # [DeNovoFFC]
    ap.add_argument("--out-dir", required=True)                      # [DeNovoFFC]
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="convert only these")
    ap.add_argument("--list", action="store_true",
                    help="show what would be converted, convert nothing")
    args = ap.parse_args(argv)

    datasets = discover(args.raw_dir)
    if not datasets:
        print(f"[error] nothing convertible in {args.raw_dir}", file=sys.stderr)
        return 1
    names = args.only or sorted(datasets)
    unknown = [n for n in names if n not in datasets]
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(unknown)}; "
                 f"have {', '.join(sorted(datasets))}")

    if args.list:
        for name in names:
            print(f"{name:26} <- {datasets[name][0]}")
        return 0

    summary = []
    for name in names:
        cov, inten = datasets[name]
        summary.append(convert(
            name,
            os.path.join(args.raw_dir, cov),
            os.path.join(args.raw_dir, inten),
            os.path.join(args.out_dir, name),
        ))

    print("=" * 70)
    print(f"{'dataset':22} {'peaks':>8} {'FFCs':>12} {'no-intensity':>13}")
    for s in summary:
        print(f"{s['name']:22} {s['peaks']:>8,} {s['ffcs']:>12,} "
              f"{s['missing_intensity']:>13,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
