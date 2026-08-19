#!/usr/bin/env python3
"""
Xiaowen-tuned TopFD runner (Stage-2 replacement)
================================================
This module holds ONLY the TopFD command builder for the fine-tuned
parameter set that Xiaowen Liu (author of TopFD) provided in
``xiaowen_result/``.  It is a drop-in replacement for
``deconv_all_greedy._run_topfd`` and produces the same ``deconv_ms2.env``
file, with the same column layout, so every downstream step of the
pipeline (dedup-by-PEAK_IDX, residual annotation, merge) is reused
unchanged.

What Xiaowen changed relative to our current Stage-2 call
--------------------------------------------------------
Our current call (deconv_all_greedy._run_topfd):

    topfd -T -e 0.005 -c <max_charge>  <peaks>

Xiaowen's fine-tuned call (from xiaowen_result/report.md):

    topfd -T --gene-sql -c 3 -m 1620 -s 0.1 -d -t 0 -e 0.04 \
          --output-dp-envs --max-miss-peak-num 4 --disable-filter-by-mz  <peaks>

Flag-by-flag:

    -c 3                      max charge state              (was effectively 2)
    -m 1620                   max monoisotopic mass (Da)    (NEW; ~precursor mass)
    -s 0.1                    MS/MS signal-to-noise ratio   (NEW; very permissive)
    -d                        disable frag-number filtering (NEW)
    -t 0                      ECScore cutoff = 0            (NEW; keep everything)
    -e 0.04                   m/z error tolerance           (was 0.005)
    --max-miss-peak-num 4     max missing peaks per envelope  (custom-build flag)
    --disable-filter-by-mz    skip the filter-by-mz step       (custom-build flag)
    --gene-sql                also write deconv.sqlite         (optional / off here)
    --output-dp-envs          dump candidate/DP envelopes      (optional / off here)

plus a source-level change (min_match_peak_num / min_consecutive_peak_num
for the high-mass group lowered 3 -> 2) that is baked into HIS binary and
has no command-line flag.

IMPORTANT — which binary you need
---------------------------------
The three custom flags ``--max-miss-peak-num``, ``--disable-filter-by-mz``,
``--output-dp-envs`` (and the min_match 3->2 change) exist ONLY in Xiaowen's
rebuilt binary (branch ``debug_mms`` @ ``4c9d4f1``).  Our shipped binary
(``bin/topfd``, v1.8.1) does NOT recognise them and will exit with
``unrecognised option`` if they are passed.

Therefore:

* To reproduce Xiaowen's run EXACTLY: point ``topfd_bin`` at his rebuilt
  binary and keep the defaults below.
* To approximate it with our current binary: pass
  ``max_miss_peak_num=None`` and ``disable_filter_by_mz=False`` so only the
  supported subset (-c -m -s -d -t -e) is used.  The env output columns are
  identical either way.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


# ── Xiaowen's fine-tuned defaults (from xiaowen_result/report.md) ───────────
XIAOWEN_MAX_CHARGE = 3
XIAOWEN_MAX_MASS = 1620.0          # peptide-specific: ~ precursor neutral mass
XIAOWEN_SN_RATIO = 0.1
XIAOWEN_ECSCORE_CUTOFF = 0.0
XIAOWEN_ERROR_TOL = 0.04
XIAOWEN_MAX_MISS_PEAK_NUM = 4      # custom-build flag
XIAOWEN_DISABLE_FRAG_NUM_FILTER = True   # -d
XIAOWEN_DISABLE_FILTER_BY_MZ = True      # custom-build flag


def build_topfd_xiaowen_cmd(
    topfd_bin: str,
    input_path: str,
    *,
    max_charge: int = XIAOWEN_MAX_CHARGE,
    max_mass: float = XIAOWEN_MAX_MASS,
    sn_ratio: float = XIAOWEN_SN_RATIO,
    ecscore_cutoff: float = XIAOWEN_ECSCORE_CUTOFF,
    error_tol: float = XIAOWEN_ERROR_TOL,
    disable_frag_num_filter: bool = XIAOWEN_DISABLE_FRAG_NUM_FILTER,
    max_miss_peak_num: Optional[int] = XIAOWEN_MAX_MISS_PEAK_NUM,
    disable_filter_by_mz: bool = XIAOWEN_DISABLE_FILTER_BY_MZ,
    gene_sql: bool = False,
    output_dp_envs: bool = False,
) -> List[str]:
    """
    Assemble Xiaowen's fine-tuned TopFD command line.

    Set ``max_miss_peak_num=None`` and ``disable_filter_by_mz=False`` (and
    keep ``gene_sql``/``output_dp_envs`` off) to stay within the flag set
    supported by our shipped v1.8.1 binary.
    """
    cmd: List[str] = [topfd_bin, "-T"]          # -T: single text peak list
    if gene_sql:
        cmd.append("--gene-sql")
    cmd += ["-c", str(max_charge)]
    cmd += ["-m", str(max_mass)]
    cmd += ["-s", str(sn_ratio)]
    if disable_frag_num_filter:
        cmd.append("-d")
    cmd += ["-t", str(ecscore_cutoff)]
    cmd += ["-e", str(error_tol)]
    if output_dp_envs:
        cmd.append("--output-dp-envs")
    if max_miss_peak_num is not None:
        cmd += ["--max-miss-peak-num", str(max_miss_peak_num)]
    if disable_filter_by_mz:
        cmd.append("--disable-filter-by-mz")
    cmd.append(input_path)
    return cmd


def print_topfd_xiaowen_params(
    topfd_bin: str,
    *,
    max_charge: int = XIAOWEN_MAX_CHARGE,
    max_mass: float = XIAOWEN_MAX_MASS,
    sn_ratio: float = XIAOWEN_SN_RATIO,
    ecscore_cutoff: float = XIAOWEN_ECSCORE_CUTOFF,
    error_tol: float = XIAOWEN_ERROR_TOL,
    disable_frag_num_filter: bool = XIAOWEN_DISABLE_FRAG_NUM_FILTER,
    max_miss_peak_num: Optional[int] = XIAOWEN_MAX_MISS_PEAK_NUM,
    disable_filter_by_mz: bool = XIAOWEN_DISABLE_FILTER_BY_MZ,
    gene_sql: bool = False,
    output_dp_envs: bool = False,
) -> None:
    """Print a labelled summary of the TopFD parameters used for this run."""
    custom = (max_miss_peak_num is not None) or disable_filter_by_mz \
        or output_dp_envs or gene_sql
    rows = [
        ("binary",                     topfd_bin),
        ("-c  max charge",             max_charge),
        ("-m  max mass (Da)",          max_mass),
        ("-s  MS/MS S/N ratio",        sn_ratio),
        ("-t  ECScore cutoff",         ecscore_cutoff),
        ("-e  m/z error tol",          error_tol),
        ("-d  disable frag-num filter", disable_frag_num_filter),
        ("--max-miss-peak-num",
         "(omitted)" if max_miss_peak_num is None else max_miss_peak_num),
        ("--disable-filter-by-mz",     disable_filter_by_mz),
        ("--gene-sql",                 gene_sql),
        ("--output-dp-envs",           output_dp_envs),
    ]
    print("[TopFD/Xiaowen] Parameters:")
    for label, val in rows:
        print(f"    {label:<28} {val}")
    if custom:
        print("    NOTE: run uses custom-build-only flags "
              "(--max-miss-peak-num / --disable-filter-by-mz / "
              "--gene-sql / --output-dp-envs);")
        print("          these require Xiaowen's rebuilt (debug_mms) binary; "
              "shipped v1.8.1 will reject them.")


def run_topfd_xiaowen(
    input_path: str,
    topfd_bin: str = "topfd",
    max_charge: int = XIAOWEN_MAX_CHARGE,
    error_tol: float = XIAOWEN_ERROR_TOL,
    env_extra_args: Optional[Dict[str, str]] = None,
    *,
    # extra Xiaowen knobs (not part of the base _run_topfd signature; passed
    # in by the pipeline wrapper via functools.partial / module config)
    max_mass: float = XIAOWEN_MAX_MASS,
    sn_ratio: float = XIAOWEN_SN_RATIO,
    ecscore_cutoff: float = XIAOWEN_ECSCORE_CUTOFF,
    disable_frag_num_filter: bool = XIAOWEN_DISABLE_FRAG_NUM_FILTER,
    max_miss_peak_num: Optional[int] = XIAOWEN_MAX_MISS_PEAK_NUM,
    disable_filter_by_mz: bool = XIAOWEN_DISABLE_FILTER_BY_MZ,
    gene_sql: bool = False,
    output_dp_envs: bool = False,
) -> str:
    """
    Run Xiaowen's fine-tuned TopFD on a peak list and return the path to the
    ``deconv_ms2.env`` output.

    Signature-compatible with ``deconv_all_greedy._run_topfd`` for the first
    five arguments, so it can be swapped in directly; the ``*``-only
    arguments carry Xiaowen's additional parameters.
    """
    cmd = build_topfd_xiaowen_cmd(
        topfd_bin,
        input_path,
        max_charge=max_charge,
        max_mass=max_mass,
        sn_ratio=sn_ratio,
        ecscore_cutoff=ecscore_cutoff,
        error_tol=error_tol,
        disable_frag_num_filter=disable_frag_num_filter,
        max_miss_peak_num=max_miss_peak_num,
        disable_filter_by_mz=disable_filter_by_mz,
        gene_sql=gene_sql,
        output_dp_envs=output_dp_envs,
    )

    print_topfd_xiaowen_params(
        topfd_bin,
        max_charge=max_charge,
        max_mass=max_mass,
        sn_ratio=sn_ratio,
        ecscore_cutoff=ecscore_cutoff,
        error_tol=error_tol,
        disable_frag_num_filter=disable_frag_num_filter,
        max_miss_peak_num=max_miss_peak_num,
        disable_filter_by_mz=disable_filter_by_mz,
        gene_sql=gene_sql,
        output_dp_envs=output_dp_envs,
    )

    env = os.environ.copy()
    if env_extra_args:
        for key, val in env_extra_args.items():
            env[key] = val + ":" + env[key] if key in env else val

    # Run with cwd = input file's directory so deconv_ms2.env lands there.
    input_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    print(f"[TopFD/Xiaowen] Running: {' '.join(cmd)}")
    print(f"[TopFD/Xiaowen] Working directory: {input_dir}")
    subprocess.run(cmd, check=True, env=env, cwd=input_dir)

    env_path = os.path.join(input_dir, "deconv_ms2.env")
    if not os.path.exists(env_path):
        stem = Path(input_path).stem
        alt = os.path.join(input_dir, f"{stem}_ms2.env")
        if os.path.exists(alt):
            env_path = alt
        else:
            raise FileNotFoundError(
                f"TopFD did not produce expected .env file. "
                f"Looked for: {env_path} and {alt}"
            )
    return env_path
