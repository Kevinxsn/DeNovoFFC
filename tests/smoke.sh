#!/usr/bin/env bash
# End-to-end smoke test: every route through the pipeline, on real data.
#
#   tests/smoke.sh [OUTDIR]
#
# Needs the benchmark data under /home/sux002/data.  Set $DENOVOMMS_TOPFD_BIN
# and $DENOVOMMS_TOPFD_LIB for the deconvolution routes; without them those
# cases are skipped rather than failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${1:-/tmp/denovomms-smoke}"
DECONV=/home/sux002/data/deconvoluted/merged
ETD=/home/sux002/data/ETD_deconv_cz/merge
RAW=/home/sux002/data/mms_with_intensity
rm -rf "$OUT"; mkdir -p "$OUT"

pass=0; fail=0; skip=0
run() {  # run <name> <args...>
  local name="$1"; shift
  printf '%-42s ' "$name"
  if "$ROOT/denovomms" run "$@" -o "$OUT/$name" >"$OUT/$name.log" 2>&1; then
    printf 'PASS\n'; pass=$((pass+1))
  else
    printf 'FAIL  (see %s)\n' "$OUT/$name.log"; tail -3 "$OUT/$name.log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
}
skip() { printf '%-42s SKIP  (%s)\n' "$1" "$2"; skip=$((skip+1)); }

echo "=== DeNovoMMS smoke test -> $OUT ==="

# 1. de novo, deconvolved input, C++ engine
run denovo-cpp --data "$DECONV/KWK_37_Z5_E28_2_XIAOWEN_merged.tsv" \
  --charge 5 --mass 4001.3781 --terminus NH3 --label KWK5+ --top-list 500

# 2. de novo, Python reference engine (must agree with the above)
run denovo-python --data "$DECONV/KWK_37_Z5_E28_2_XIAOWEN_merged.tsv" \
  --charge 5 --mass 4001.3781 --terminus NH3 --label KWK5+ --top-list 500 \
  --engine python

# 3. de novo, ETD (c/z ladder, ladder charge sum z-1)
if [[ -d "$ETD" ]]; then
  run denovo-etd --data "$ETD/LLG_37_Z6_ETD_merged.tsv" \
    --charge 6 --mass 4490.5753 --terminus H2O --fragmentation ETD \
    --label ETD-LL6+ --top-list 500
else
  skip denovo-etd "no $ETD"
fi

# 4. annotation, known peptide
run annotation --data "$DECONV/VEA_15_Z3_revised_merged.tsv" \
  --seq VEADIAGHGQEVLIR --charge 3 --terminus H2O --label VEA3+ --top-list 500

# 5. raw -> deconvolution -> de novo
if [[ -n "${DENOVOMMS_TOPFD_BIN:-}" && -f "$RAW/VEA_15_Z3_revised" ]]; then
  run deconv-denovo --data "$RAW/VEA_15_Z3_revised" \
    --charge 3 --mass 1605.8474 --terminus H2O --label VEA3+ \
    --deconvolve yes --top-list 500
  run deconv-annotation --data "$RAW/VEA_15_Z3_revised" \
    --seq VEADIAGHGQEVLIR --charge 3 --terminus H2O --label VEA3+ \
    --deconvolve yes --top-list 500
else
  skip deconv-denovo "\$DENOVOMMS_TOPFD_BIN unset or no raw table"
  skip deconv-annotation "\$DENOVOMMS_TOPFD_BIN unset or no raw table"
fi

# The two engines must produce the same answer.  `engine` and `seconds` are
# the only fields allowed to differ; everything else is compared verbatim.
# (tests/parity.py is the thorough version -- it also diffs the master-set
# coordinates and every tied path.)
if [[ -f "$OUT/denovo-cpp/run.json" && -f "$OUT/denovo-python/run.json" ]]; then
  printf '%-42s ' "engines-agree"
  read -r same msg < <(python3 - "$OUT/denovo-cpp/run.json" "$OUT/denovo-python/run.json" <<'PYEOF'
import json, sys
def rows(p):
    return [{k: v for k, v in r.items() if k not in ("engine", "seconds")}
            for r in json.load(open(p))["denovo"]]
a, b = rows(sys.argv[1]), rows(sys.argv[2])
print("yes" if a == b else "no", "" if a == b else f"{a} != {b}")
PYEOF
)
  if [[ "$same" == "yes" ]]; then printf 'PASS\n'; pass=$((pass+1))
  else printf 'FAIL\n    %s\n' "$msg"; fail=$((fail+1)); fi
fi

echo
echo "pass $pass   fail $fail   skip $skip"
[[ $fail -eq 0 ]]
