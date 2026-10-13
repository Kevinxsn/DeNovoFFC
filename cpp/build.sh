#!/usr/bin/env bash
# Build the C++ de novo engine and its native tests.  No dependencies beyond a
# C++17 compiler -- CMake is used when available, plain g++ otherwise.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../bin"
mkdir -p "$OUT"

# DENOVOFFC_NO_CMAKE=1 forces the plain-compiler path (also how it is tested).
if [[ -z "${DENOVOFFC_NO_CMAKE:-}" ]] && command -v cmake >/dev/null 2>&1; then
  cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$HERE/build" -j "$(nproc 2>/dev/null || echo 4)"
else
  CORE=$(ls "$HERE"/src/*.cpp | grep -v '/main\.cpp$')
  "${CXX:-g++}" -O2 -std=c++17 -I"$HERE/include" $CORE "$HERE/src/main.cpp" \
    -o "$OUT/denovo_ffc"
  "${CXX:-g++}" -O2 -std=c++17 -I"$HERE/include" $CORE "$HERE/tests/test_native.cpp" \
    -o "$OUT/denovo_tests"
fi
echo "built $OUT/denovo_ffc and $OUT/denovo_tests"
