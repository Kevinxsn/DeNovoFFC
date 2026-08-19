#!/usr/bin/env python3
"""Parity check: the C++ engine against the Python reference.

The C++ engine (``cpp/``) is a port, not a reimplementation -- it must agree
with ``src/de_novo/denovo_core.py`` on every reported quantity, not merely
"reconstruct the same peptide".  This script runs both on the same prepared
table and diffs:

    line count and per-line category, master-set size and coordinates,
    convergence rounds, tied-path count, and the full set of reconstructions.

Run it after ANY change to either implementation.

    PY=/home/sux002/miniforge3/envs/2d_spec/bin/python
    $PY tests/parity.py                 # the 7 HCD benchmark peptides
    $PY tests/parity.py --top-list 200 500
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
BIN = os.path.join(ROOT, "bin", "denovo_mms")
for _p in (os.path.join(SRC, "deiso", "python_code"),
           os.path.join(SRC, "reconstruct"),
           os.path.join(SRC, "de_novo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from merge_ffcs import merge_duplicate_ffcs_charge          # noqa: E402
from line_finding import prepare_ffc_data                   # noqa: E402
from denovo_core import run_denovo                          # noqa: E402
from run_denovo import (MAIN_PEPTIDES, ETD_PEPTIDES, DATA_DIR,   # noqa: E402
                        precursor_mass)

#: Coordinates are floats travelling through a JSON round-trip and two
#: different formatters; anything under this is not a disagreement.
COORD_TOL = 1e-6


def load_merged(entry):
    data_dir = entry.get("data_dir", DATA_DIR)
    df = pd.read_csv(os.path.join(data_dir, entry["data"]), sep="\t")
    return merge_duplicate_ffcs_charge(df, symmetric=True, reset_ranking=True)


def run_cpp(table_path, entry, mass, top_n):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    cmd = [BIN, "--data", table_path, "--charge", str(entry["charge"]),
           "--mass", f"{mass:.10f}", "--terminus", entry["terminus"],
           "--fragmentation", entry.get("fragmentation", "HCD"),
           "--top-n", str(top_n), "--json", out, "--quiet"]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    dt = time.time() - t0
    with open(out) as fh:
        res = json.load(fh)
    os.unlink(out)
    res["_wall"] = dt
    return res


def compare(name, py, cpp):
    """Return a list of human-readable disagreements (empty == parity)."""
    bad = []

    def eq(label, a, b):
        if a != b:
            bad.append(f"{label}: python={a!r} cpp={b!r}")

    eq("n_lines", len(py.lines), cpp["n_lines"])
    eq("rounds", py.rounds, cpp["rounds"])
    eq("n_master", len(py.master), len(cpp["master_set"]))
    eq("n_paths", py.n_paths, cpp["n_paths"])
    eq("best_weight", round(py.best_weight, 6), round(cpp["best_weight"], 6))
    eq("line_charge", py.params["line_charge"], cpp["line_charge"])

    py_cat = sorted((li.key, li.category) for li in py.lines)
    cpp_cat = sorted((li["key"], li["category"]) for li in cpp["lines"])
    if py_cat != cpp_cat:
        only_py = [x for x in py_cat if x not in cpp_cat]
        only_cpp = [x for x in cpp_cat if x not in py_cat]
        bad.append(f"line categories differ: only-python={only_py[:4]} "
                   f"only-cpp={only_cpp[:4]}")

    py_c = sorted(e.coord for e in py.master.entries)
    cpp_c = sorted(e["coord"] for e in cpp["master_set"])
    if len(py_c) == len(cpp_c):
        worst = max((abs(a - b) for a, b in zip(py_c, cpp_c)), default=0.0)
        if worst > COORD_TOL:
            bad.append(f"master coordinates differ by up to {worst:.3e} Da")

    py_r = sorted(r["reconstruction"] for r in py.reconstructions)
    cpp_r = sorted(cpp["reconstructions"])
    if py_r != cpp_r:
        only_py = [x for x in py_r if x not in cpp_r][:2]
        only_cpp = [x for x in cpp_r if x not in py_r][:2]
        bad.append(f"reconstructions differ: only-python={only_py} "
                   f"only-cpp={only_cpp}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-list", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--etd", action="store_true", help="the ETD/EThcD set instead")
    args = ap.parse_args()

    if not os.path.exists(BIN):
        raise SystemExit(f"{BIN} not found -- run cpp/build.sh first")

    peptides = ETD_PEPTIDES if args.etd else MAIN_PEPTIDES
    if args.only:
        peptides = [e for e in peptides if e["name"] in set(args.only)]

    rows, failures = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for entry in peptides:
            merged = load_merged(entry)
            path = os.path.join(tmp, entry["name"].replace("+", "p") + ".tsv")
            merged.to_csv(path, sep="\t", index=False)
            mass = entry.get("mass") or precursor_mass(entry["seq"], entry["terminus"])

            for top_n in args.top_list:
                t0 = time.time()
                py = run_denovo(prepare_ffc_data(merged, top_n=top_n),
                                parent_charge=entry["charge"], parent_mass=mass,
                                terminus=entry["terminus"],
                                fragmentation=entry.get("fragmentation", "HCD"))
                py_wall = time.time() - t0
                cpp = run_cpp(path, entry, mass, top_n)

                bad = compare(entry["name"], py, cpp)
                failures += bool(bad)
                rows.append(dict(peptide=entry["name"], top_n=top_n,
                                 lines=len(py.lines), master=len(py.master),
                                 paths=py.n_paths,
                                 py_s=round(py_wall, 2),
                                 cpp_s=round(cpp["_wall"], 3),
                                 speedup=round(py_wall / max(cpp["_wall"], 1e-6), 1),
                                 status="OK" if not bad else "MISMATCH"))
                for line in bad:
                    print(f"  !! {entry['name']} top{top_n}: {line}")

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    if failures:
        print(f"FAIL -- {failures} of {len(rows)} runs disagree")
        return 1
    print(f"PASS -- all {len(rows)} runs agree "
          f"(mean speedup x{sum(r['speedup'] for r in rows) / len(rows):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
