"""Tabulate the headline numbers out of a tree of line-based `.xlsx` outputs.

Reads nothing but the files ``run_annotation_reconstruct.py`` already wrote, so
it is seconds rather than a re-run -- the same idea as
``../deconvolution/demo_old_vs_new.py``, one stage later in the chain.

For each ``line_based/top{N}/*_line_based_top{N}.xlsx`` it pulls:

  * ``#FFC``                    -- the top-N actually fed in (column-0 header)
  * ``lines``                   -- detected line columns
  * ``bp cov`` / ``master cov`` -- the two coverage headers (§6 of line_based.md)
  * ``#incorrect``              -- master TFFCs matching no backbone bond
  * ``parental recon``          -- ``#reconstructions/correct reconstructions``
                                  from the parental ladder alone (§8a)
  * ``master recon``            -- ``#reconstructions/correct?`` from the whole
                                  master TFFC-set (§8e)
  * the reconstruction strings from the second sheet

and, when a matching ``log/*_log.txt`` exists, the complementary stage's
spurious rate.

Usage
-----
    python summarize_line_based.py result/annotation_had_more_fragments
    python summarize_line_based.py <root> --top 500 --recon
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

_COV_RE = re.compile(r"=\s*([\d.]+)%")
_SPUR_RE = re.compile(r"Spurious rate:\s*([\d.]+)%")


def _pct(header: str):
    m = _COV_RE.search(str(header))
    return float(m.group(1)) if m else float("nan")


def _row(df: pd.DataFrame, label: str) -> pd.Series | None:
    key = df.columns[0]
    hit = df[df[key].astype(str).str.strip() == label]
    return hit.iloc[0] if len(hit) else None


def _cell(df: pd.DataFrame, label: str, column: str):
    r = _row(df, label)
    if r is None or column not in df.columns:
        return None
    v = r[column]
    return None if pd.isna(v) else v


def _first_non_null(df: pd.DataFrame, label: str):
    """The `#reconstructions/...` cell sits on whichever column carries it."""
    r = _row(df, label)
    if r is None:
        return None
    for v in r[1:]:
        if not pd.isna(v):
            return v
    return None


def summarize_one(path: str, log_dir: str | None = None) -> dict:
    xl = pd.ExcelFile(path)
    df = xl.parse(xl.sheet_names[0])
    key = df.columns[0]

    bp_col = next((c for c in df.columns if str(c).startswith("Breakpoint Coverage")), None)
    ms_col = next((c for c in df.columns if str(c).startswith("Master TFFC-set")), None)

    # Bond rows are the ones before the first summary row ("line type").
    labels = df[key].astype(str).tolist()
    n_bonds = labels.index("line type") if "line type" in labels else len(df)
    bonds = df.iloc[:n_bonds]

    def _covered(col):
        if col is None:
            return None
        s = pd.to_numeric(bonds[col].astype(str).str.rstrip("*"), errors="coerce")
        return int((s.fillna(0) > 0).sum())

    # Line columns: everything between the master column and the trailing
    # "theoretical mass" column.
    cols = list(df.columns)
    start = cols.index(ms_col) + 1 if ms_col in cols else 2
    end = cols.index("theoretical mass") if "theoretical mass" in cols else len(cols)
    n_lines = end - start

    name = os.path.basename(path)
    name = name[: name.rfind("_line_based_top")]

    recons = []
    if "reconstructions" in xl.sheet_names:
        rec = xl.parse("reconstructions")
        for _, r in rec.iterrows():
            recons.append((str(r.get("reconstruction", "")).strip(),
                           str(r.get("correct", "")).strip()))

    out = {
        "name": name,
        "#FFC": str(key).replace("#FFC=", ""),
        "bonds": n_bonds,
        "lines": n_lines,
        "bp cov %": _pct(bp_col),
        "bp bonds": _covered(bp_col),
        "master cov %": _pct(ms_col),
        "master bonds": _covered(ms_col),
        "#incorrect": _cell(df, "#incorrect TFFCs", ms_col),
        "parental recon": _first_non_null(
            df, "#reconstructions/correct reconstructions"),
        "master recon": _cell(df, "#reconstructions/correct?", ms_col),
        "spurious %": float("nan"),
        "_recons": recons,
    }

    if log_dir:
        log = os.path.join(log_dir, f"{name}_log.txt")
        if os.path.isfile(log):
            with open(log) as fh:
                m = _SPUR_RE.search(fh.read())
            if m:
                out["spurious %"] = float(m.group(1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="an annotation output root (holds line_based/)")
    ap.add_argument("--top", type=int, help="only this top-N folder")
    ap.add_argument("--recon", action="store_true",
                    help="also print each entry's reconstruction strings")
    args = ap.parse_args(argv)

    line_root = os.path.join(args.root, "line_based")
    log_dir = os.path.join(args.root, "log")
    if not os.path.isdir(line_root):
        ap.error(f"no line_based/ under {args.root}")

    tops = sorted(d for d in os.listdir(line_root) if d.startswith("top"))
    if args.top:
        tops = [t for t in tops if t == f"top{args.top}"]

    for top in tops:
        folder = os.path.join(line_root, top)
        rows = [summarize_one(os.path.join(folder, f), log_dir)
                for f in sorted(os.listdir(folder)) if f.endswith(".xlsx")]
        if not rows:
            continue
        print(f"\n=== {top} " + "=" * 60)
        table = pd.DataFrame([{k: v for k, v in r.items() if k != "_recons"}
                              for r in rows])
        print(table.to_string(index=False))
        if args.recon:
            print()
            for r in rows:
                print(f"  {r['name']}")
                for text, correct in r["_recons"] or [("(none)", "")]:
                    flag = "  <-- CORRECT" if correct == "correct" else ""
                    print(f"      {text}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
