#!/usr/bin/env python3
"""
Deconvolution for ETD/EThcD -- the line model runs on ``parent_charge - 1``
==========================================================================

Thin driver around the **unmodified** ``deconvolution/deconv_pipeline_oa.py``.
The only thing it changes is the charge the *line model* is told about, and it
does so for one concrete reason.

Why the charge differs
----------------------
Electron transfer neutralises one of the precursor's protons, so an ETD
c / z-dot pair from ``[M+6H]6+`` carries **five** charges between the two
fragments, not six.  The reconstructed line mass barely moves --

    v = c + a*p + z. + b*p = (M + H) + 5p = (M + 6p) + (H - p)

-- i.e. one electron, 0.0006 Da, above where a b/y parental line sits.  So the
ETD ladder is at the *same place* as an HCD ladder but on a **different charge
sum**, and ``detect_lines_oa``'s parental pass, which selects on
``i + j == parent_charge``, cannot see it.

Measured on ``LLG_37_Z6_ETD`` with ``parent_charge = 6`` (the naive run):

    parental pass (i+j = 6):  48 lines, every one junk -- shifts of +291 to
                              +1086 Da, 3-7 FFCs each, `ladder` False throughout
    greedy residual:          the real ladder, at charge sum 5 --
                              (2,3) 4496.626 shift +0.007  25 FFCs
                              (1,4) 4496.625 shift +0.006  15 FFCs
                              plus its +1..+5 isotope satellites, each with MORE
                              members (61, 55, 50, 43, 37) than the
                              monoisotopic line itself

That last point is the real damage.  The parental pass exists to clear the
ladder **together with its isotope satellites** (``satellite_lines``); left to
greedy peeling, the satellites are peeled first (they are more populated) and
each one takes a share of the ladder's FFCs with it.  The ladder's evidence ends
up split six ways.  Stage 1 placed only 761 of 15000 FFCs.

Passing ``parent_charge = 5`` puts the ladder back in the parental pass where it
belongs.  TopFD still gets ``-c 6`` and the true precursor mass: those describe
the *precursor*, not the line geometry, and a fragment's charge is bounded by
the precursor's.

Usage
-----
    python run_etd_deconvolution.py                     # both datasets
    python run_etd_deconvolution.py --only LLG_37_Z6_ETD
    python run_etd_deconvolution.py --parent-charge-offset 0   # the naive run
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_DECONV = os.path.abspath(os.path.join(_HERE, "..", "deconvolution"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _DECONV)

import _paths  # noqa: F401,E402  (sys.path bootstrap from deconvolution/)

import cz_series  # noqa: E402
from peptide_info import resolve as resolve_name  # noqa: E402
from deconv_pipeline_oa import run_deconvolution  # noqa: E402


DATA_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "ETD_prep")
OUTPUT_DIR = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "ETD_deconv_cz")

#: Same defaults as ``deconvolution/run_deconvolution.py``; see deconvolution.md
#: section 7 for why each one is what it is.  Nothing here is ETD-specific --
#: the ETD adjustment is the charge, applied in ``run_one``.
DEFAULTS = dict(
    delta=0.02,
    delta_unit="da",
    min_ffc_number=3,
    line_tol=None,
    top_lines=2000,
    merge_tolerance=0.002,
    merge_over_fetch=4,
    silent_drop=False,
    fast_detector=True,
    mz_tol=0.01,
    max_shift=None,
    top_selected=15000,
    top_bg=15000,
    run_topfd=True,
    topfd_error_tol=0.04,
    topfd_sn_ratio=0.1,
    topfd_ecscore_cutoff=0.0,
    topfd_full_pool=True,
    topfd_max_miss_peak_num=None,
    topfd_disable_filter_by_mz=False,
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def discover(data_dir: str):
    entries, skipped = [], []
    for fname in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            info = resolve_name(fname)
        except (ValueError, KeyError):
            skipped.append(fname)
            continue
        entries.append({**info, "path": path})
    return entries, skipped


def run_one(entry, output_dir, overrides, charge_offset=1, write_log=True):
    """Deconvolve one ETD dataset.  Never raises."""
    name = entry["NAME"]
    z_precursor = entry["PARENT_CHARGE"]
    # The line model's charge: z - 1 for ETD (see the module docstring).
    z_line = max(1, z_precursor - charge_offset)
    # ...and the mass that charge sum reconstructs to: the c/z-dot pair sums to
    # M + H, and carries z_line protons, so v = M + H + z_line*p.  Expressed
    # against the stored charged mass (M + z*p) that is one electron up.
    parent_mass = (entry["PARENT_MASS"]
                   + cz_series.H_ATOM - charge_offset * cz_series.PROTON)

    kwargs = {**DEFAULTS, **overrides}
    record = {"name": name, "status": "ok", "seconds": 0.0, "error": None,
              "merged": None}

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_log.txt") if write_log else None

    started = time.time()
    try:
        with contextlib.ExitStack() as stack:
            if log_path:
                fh = stack.enter_context(open(log_path, "w"))
                stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, fh)))

            print(f"── {name} " + "─" * 50)
            print(f"   sequence         {entry['SEQUENCE']}")
            print(f"   precursor charge {z_precursor}   "
                  f"(TopFD -c {z_precursor})")
            print(f"   LINE-MODEL charge {z_line}   "
                  f"<- ETD: a c/z-dot pair carries z-1 charges")
            print(f"   line-model mass  {parent_mass:.4f}   "
                  f"(stored charged mass {entry['PARENT_MASS']:.4f} "
                  f"{parent_mass - entry['PARENT_MASS']:+.5f})")
            print()

            result = run_deconvolution(
                ffc_path=entry["path"],
                parent_charge=z_line,
                parent_mass=parent_mass,
                output_dir=output_dir,
                output_prefix=name,
                theo_patt_path=_paths.THEO_PATT,
                topfd_bin=_paths.TOPFD_BIN,
                topfd_env_vars={"LD_LIBRARY_PATH": _paths.TOPFD_LIB},
                # TopFD bounds describe the PRECURSOR, not the line geometry:
                # a fragment cannot out-charge or out-weigh the intact ion.
                topfd_max_charge=z_precursor,
                topfd_max_mass=entry["PARENT_MASS"],
                **kwargs,
            )
            record["merged"] = result["stats"]["merged_path"]
            record["stats"] = result["stats"]
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    record["seconds"] = time.time() - started
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--parent-charge-offset", type=int, default=1,
                    help="charges the line model drops relative to the "
                         "precursor: 1 for ETD (default), 0 to reproduce the "
                         "b/y convention")
    ap.add_argument("--delta", type=float)
    ap.add_argument("--delta-unit", choices=["da", "ppm"])
    ap.add_argument("--min-ffc", type=int, dest="min_ffc_number")
    ap.add_argument("--top-lines", type=int)
    ap.add_argument("--top-selected", type=int)
    ap.add_argument("--no-topfd", action="store_true")
    args = ap.parse_args(argv)

    entries, skipped = discover(args.data_dir)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["NAME"] in wanted]

    if args.list:
        for e in entries:
            print(f"{e['NAME']:24} z={e['PARENT_CHARGE']} "
                  f"line z={e['PARENT_CHARGE'] - args.parent_charge_offset} "
                  f"{e['SEQUENCE']}")
        return 0
    if not entries:
        print("[error] nothing to run", file=sys.stderr)
        return 1

    overrides = {k: v for k, v in {
        "delta": args.delta,
        "delta_unit": args.delta_unit,
        "min_ffc_number": args.min_ffc_number,
        "top_lines": args.top_lines,
        "top_selected": args.top_selected,
        "run_topfd": False if args.no_topfd else None,
    }.items() if v is not None}

    results = []
    for k, entry in enumerate(entries, 1):
        print("\n" + "#" * 70)
        print(f"# [{k}/{len(entries)}] {entry['NAME']}")
        print("#" * 70)
        results.append(run_one(entry, args.output_dir, overrides,
                               charge_offset=args.parent_charge_offset,
                               write_log=not args.no_log))

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for r in results:
        line = f"  {r['name']:24} {r['status']:8} {r['seconds']:7.1f}s"
        st = r.get("stats") or {}
        if st:
            line += (f"   stage1={st.get('stage1_output_rows')}"
                     f" stage2={st.get('stage2_output_rows')}"
                     f" lines={st.get('n_lines_parental')}"
                     f"/{st.get('n_lines_residual')}")
        if r["error"]:
            line += f"   {r['error']}"
        print(line)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
