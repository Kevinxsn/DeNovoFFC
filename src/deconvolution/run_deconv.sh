#!/usr/bin/env bash
#
# Deconvolution entry point.
#
# Sets up the environment TopFD needs and hands every argument through to
# run_deconvolution.py, which resolves each peptide from its dataset name.
#
#   ./run_deconv.sh                        # every resolvable dataset
#   ./run_deconv.sh --list                 # show what would run, run nothing
#   ./run_deconv.sh --only HAD_33_Z4_E16
#   ./run_deconv.sh --tags KWK LLG
#   ./run_deconv.sh --no-test              # skip the spurious-parental test
#   ./run_deconv.sh --delta 20 --delta-unit ppm
#
# See deconvolution.md for what the pipeline does and which knobs matter.
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# The 2d_spec conda env; the system python may lack numpy/scipy.
PYTHON=${PYTHON:-/home/sux002/miniforge3/envs/2d_spec/bin/python}

# TopFD's shared libraries.  run_deconvolution.py also passes this per
# invocation, so either entry point works -- this is for running topfd by hand.
export LD_LIBRARY_PATH="${BASE_DIR}/deiso/lib/toppic/toppic:${ORT_LIB_DIR:-}:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

exec "$PYTHON" -u "${SCRIPT_DIR}/run_deconvolution.py" "$@"
