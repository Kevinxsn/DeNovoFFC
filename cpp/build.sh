#!/usr/bin/env bash
# Build the C++ de novo engine.  No dependencies beyond a C++17 compiler --
# CMake is used when available, a single g++ invocation otherwise.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../bin"
mkdir -p "$OUT"

if command -v cmake >/dev/null 2>&1; then
  cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$HERE/build" -j "$(nproc 2>/dev/null || echo 4)"
else
  "${CXX:-g++}" -O2 -std=c++17 -I"$HERE/include" \
    "$HERE"/src/*.cpp -o "$OUT/denovo_mms"
fi
echo "built $OUT/denovo_mms"
