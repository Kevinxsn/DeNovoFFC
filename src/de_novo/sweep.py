#!/usr/bin/env python3
"""
Hyper-parameter sweep over the 7 benchmark peptides
===================================================

Runs ``denovo_core.run_denovo`` under several configurations and prints one
row per (config, peptide) plus a per-config summary, so a default can be picked
on evidence rather than taste.  Ground truth is used only by ``evaluate``.

    $PY sweep.py                       # the built-in config list
    $PY sweep.py --configs base no-mocks tol20
    $PY sweep.py --top-list 500 2000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_denovo import MAIN_PEPTIDES, DATA_DIR, DEFAULTS, load_map, precursor_mass
from denovo_core import run_denovo
from evaluate import score_result

#: name -> overrides on DEFAULTS.  One knob at a time around the current
#: defaults (delta=0.02 da, line_min_ffc=3, mock_min_ffc=4, tol_ppm=30,
#: secondary_gate=0.5, align=1), plus a few combinations of whatever wins.
CONFIGS = {
    "base":            {},
    # ── fragment-mass tolerance (drives every sigma) ────────────────────────
    "tol20":           dict(tol_ppm=20.0),
    "tol25":           dict(tol_ppm=25.0),
    "tol35":           dict(tol_ppm=35.0),
    "tol40":           dict(tol_ppm=40.0),
    # ── line detection ──────────────────────────────────────────────────────
    "delta0.015":      dict(line_delta=0.015),
    "delta0.025":      dict(line_delta=0.025),
    "delta0.03":       dict(line_delta=0.03),
    "delta-ppm20":     dict(line_delta=20.0, line_delta_unit="ppm"),
    "minffc4":         dict(line_min_ffc=4),
    # ── mocks ───────────────────────────────────────────────────────────────
    "mock3":           dict(mock_min_ffc=3),
    "mock5":           dict(mock_min_ffc=5),
    "mock6":           dict(mock_min_ffc=6),
    "no-mocks":        dict(use_mocks=False),
    "no-mock-gate":    dict(gate_mock_on_goodness=False),
    "sec0.0":          dict(secondary_gate=0.0),
    "sec0.25":         dict(secondary_gate=0.25),
    "sec-off":         dict(secondary_gate=1.0),
    "mock-cap0.03":    dict(mock_sigma_cap_da=0.03),
    "mock-cap0.08":    dict(mock_sigma_cap_da=0.08),
    # ── master-set construction ─────────────────────────────────────────────
    "align2":          dict(master_align_threshold=2),
    "ambiguity":       dict(accept_ambiguity=True),
    # ── the sigma model itself ──────────────────────────────────────────────
    "prec5":           dict(precursor_ppm=5.0),
    "prec20":          dict(precursor_ppm=20.0),
    "floor1":          dict(sigma_floor_ppm=1.0),
    "floor6":          dict(sigma_floor_ppm=6.0),
    "evidence0.25":    dict(evidence_weight=0.25),
    "mock-w0.5":       dict(mock_weight=0.5),
    "rank0.5":         dict(rank_floor=0.5),
    # ── refinement around delta = 0.015 Da (the sweep's clear winner) ───────
    "d0.010":          dict(line_delta=0.010),
    "d0.012":          dict(line_delta=0.012),
    "d0.018":          dict(line_delta=0.018),
    "d15":             dict(line_delta=0.015),
    "d15-tol20":       dict(line_delta=0.015, tol_ppm=20.0),
    "d15-tol25":       dict(line_delta=0.015, tol_ppm=25.0),
    "d15-tol40":       dict(line_delta=0.015, tol_ppm=40.0),
    "d15-mock3":       dict(line_delta=0.015, mock_min_ffc=3),
    "d15-mock5":       dict(line_delta=0.015, mock_min_ffc=5),
    "d15-sec0.25":     dict(line_delta=0.015, secondary_gate=0.25),
    "d15-sec-off":     dict(line_delta=0.015, secondary_gate=1.0),
    "d15-align2":      dict(line_delta=0.015, master_align_threshold=2),
    "d15-minffc4":     dict(line_delta=0.015, line_min_ffc=4),
    "d15-nomock":      dict(line_delta=0.015, use_mocks=False),
    "d15-amb":         dict(line_delta=0.015, accept_ambiguity=True),
    "d12-mock5":       dict(line_delta=0.012, mock_min_ffc=5),
    # ── robustness check around the winner (delta 0.015 + secondary_gate .25) ─
    "d15-sec0.0":      dict(line_delta=0.015, secondary_gate=0.0),
    "d12-sec0.25":     dict(line_delta=0.012, secondary_gate=0.25),
    "d10-sec0.25":     dict(line_delta=0.010, secondary_gate=0.25),
    "d18-sec0.25":     dict(line_delta=0.018, secondary_gate=0.25),
    "d20-sec0.25":     dict(line_delta=0.020, secondary_gate=0.25),
    "W-tol20":         dict(line_delta=0.015, secondary_gate=0.25, tol_ppm=20.0),
    "W-tol25":         dict(line_delta=0.015, secondary_gate=0.25, tol_ppm=25.0),
    "W-tol40":         dict(line_delta=0.015, secondary_gate=0.25, tol_ppm=40.0),
    "W-minffc4":       dict(line_delta=0.015, secondary_gate=0.25, line_min_ffc=4),
    "W-align2":        dict(line_delta=0.015, secondary_gate=0.25,
                            master_align_threshold=2),
}

def run(cfg_name: str, overrides: dict, top_list, cache: dict):
    opts = dict(DEFAULTS)
    opts.update(overrides)
    rows = []
    for e in MAIN_PEPTIDES:
        for top_n in top_list:
            key = (e["name"], top_n)
            if key not in cache:
                cache[key] = load_map(os.path.join(DATA_DIR, e["data"]), top_n)
            ffc = cache[key]
            mass = precursor_mass(e["seq"], e["terminus"])
            t0 = time.time()
            try:
                res = run_denovo(
                    ffc, parent_charge=e["charge"], parent_mass=mass,
                    terminus=e["terminus"],
                    delta=opts["line_delta"], delta_unit=opts["line_delta_unit"],
                    min_ffc_number=opts["line_min_ffc"], tol_ppm=opts["tol_ppm"],
                    precursor_ppm=opts["precursor_ppm"],
                    sigma_floor_ppm=opts["sigma_floor_ppm"],
                    mock_sigma_cap_da=opts["mock_sigma_cap_da"],
                    master_align_threshold=opts["master_align_threshold"],
                    accept_ambiguity=opts["accept_ambiguity"],
                    use_mocks=opts["use_mocks"], mock_min_ffc=opts["mock_min_ffc"],
                    mock_weight=opts["mock_weight"],
                    secondary_gate=opts["secondary_gate"],
                    evidence_weight=opts["evidence_weight"],
                    evidence_cap=opts["evidence_cap"],
                    rank_floor=opts["rank_floor"],
                    gate_mock_on_goodness=opts["gate_mock_on_goodness"],
                    max_rounds=opts["max_rounds"], max_paths=opts["max_paths"],
                    label_tol=opts["label_tol"])
                sc = score_result(res, e["seq"], e["terminus"])
                rows.append(dict(config=cfg_name, peptide=e["name"], top_n=top_n,
                                 coverage=round(sc.coverage, 1),
                                 incorrect=sc.n_incorrect, paths=sc.n_paths,
                                 correct=int(sc.correct), pep_len=sc.pep_len,
                                 resolved=sc.resolved, longest=sc.longest_run,
                                 gaps=sc.gaps,
                                 reconstruction=sc.best_reconstruction,
                                 secs=round(time.time() - t0, 1)))
            except Exception as exc:
                rows.append(dict(config=cfg_name, peptide=e["name"], top_n=top_n,
                                 coverage=float("nan"), incorrect=-1, paths=-1,
                                 correct=0, pep_len=0, resolved=0, longest=0,
                                 gaps=0, reconstruction="",
                                 secs=round(time.time() - t0, 1),
                                 error=str(exc)[:60]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--top-list", nargs="+", type=int, default=[200, 500])
    ap.add_argument("--out", default=os.path.join(_HERE, "result", "sweep.csv"))
    args = ap.parse_args(argv)

    cache: dict = {}
    all_rows = []
    for name in args.configs:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name!r}; have {list(CONFIGS)}")
        rows = run(name, CONFIGS[name], args.top_list, cache)
        all_rows += rows
        df = pd.DataFrame(rows)
        print(f"\n### {name}   {CONFIGS[name] or '(defaults)'}")
        print(df.drop(columns=["reconstruction"], errors="ignore").to_string(index=False))
        print(f"    -> correct {int(df['correct'].sum())}/{len(df)}   "
              f"mean coverage {df['coverage'].mean():.1f}%   "
              f"total incorrect {int(df['incorrect'].clip(lower=0).sum())}")

    out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 80)
    summary = (out.groupby("config")
                  .agg(correct=("correct", "sum"), n=("correct", "size"),
                       coverage=("coverage", "mean"),
                       incorrect=("incorrect", lambda s: s.clip(lower=0).sum()),
                       resolved=("resolved", "sum"),
                       longest=("longest", "mean"),
                       paths=("paths", "median"), secs=("secs", "sum"))
                  .sort_values(["correct", "coverage"], ascending=False))
    summary["coverage"] = summary["coverage"].round(1)
    summary["longest"] = summary["longest"].round(1)
    print(summary.to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
