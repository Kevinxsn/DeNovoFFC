"""
edges.py — Precompute compound amino-acid mass sets and export to disk.

Generates List_n = set of distinct compound masses of n amino acids
(rounded to a configurable precision) and writes them to a JSON file
so that downstream scripts (path_finding, etc.) can load them instantly
without recomputing every time.

Usage
-----
  # From the command line — build up to 4-AA combos, save to edges.json:
  python edges.py --max_n 4 --output edges.json

  # From another script:
  from edges import load_edges, build_and_save

  # One-time generation
  build_and_save(max_n=4, path="edges.json")

  # Fast loading later
  allowed_masses = load_edges("edges.json")   # flat sorted list of Da values
"""

import json
import argparse
from pathlib import Path

# ── Monoisotopic residue masses (Da) ─────────────────────────────────────
AA_MASSES = {
    'A':  71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G':  57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P':  97.05276, 'S':  87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V':  99.06841,
}

PRECISION = 3          # 0.001 Da


# ── Core helpers ─────────────────────────────────────────────────────────
def quantize(x, precision=PRECISION):
    """Round to fixed precision and return as int (for exact set ops)."""
    return round(x * 10**precision)


def dequantize(q, precision=PRECISION):
    """Convert quantized int back to float Da."""
    return q / 10**precision


def minkowski_sum(A, B):
    """A ⊕ B = {a+b : a∈A, b∈B}, duplicates removed."""
    return {a + b for a in A for b in B}


def build_lists(max_n, precision=PRECISION):
    """
    Build List_1 … List_{max_n}.

    Returns
    -------
    dict  {n: set of quantized-int masses}
    """
    list1 = {quantize(m, precision) for m in AA_MASSES.values()}
    lists = {1: list1}
    current = list1
    for n in range(2, max_n + 1):
        current = minkowski_sum(current, list1)
        lists[n] = current
    return lists


# ── Build, save, load ────────────────────────────────────────────────────
def build_and_save(max_n=4, path="edges.json", precision=PRECISION):
    """
    Compute compound masses up to *max_n* amino acids and write to *path*.

    The JSON structure:
    {
      "precision": 3,
      "max_n": 4,
      "lists": {
        "1": [57.021, 71.037, ...],
        "2": [114.042, 128.058, ...],
        ...
      },
      "all_masses": [57.021, 71.037, ..., 744.xxx, ...]
    }
    """
    lists = build_lists(max_n, precision)

    # Convert quantized ints → float Da, sorted
    lists_float = {}
    cumulative = set()
    for n in range(1, max_n + 1):
        vals = sorted(dequantize(q, precision) for q in lists[n])
        lists_float[str(n)] = vals
        cumulative |= lists[n]

    all_masses = sorted(dequantize(q, precision) for q in cumulative)

    payload = {
        "precision": precision,
        "max_n": max_n,
        "lists": lists_float,
        "all_masses": all_masses,
    }

    out = Path(path)
    with open(out, "w") as f:
        json.dump(payload, f)

    print(f"Wrote {len(all_masses):,} distinct masses (n=1..{max_n}) → {out}")
    for n in range(1, max_n + 1):
        print(f"  List_{n}: {len(lists_float[str(n)]):>8,} entries")

    return payload


def load_edges(path="edges.json"):
    """
    Load the precomputed edge file and return the flat sorted mass list.

    Returns
    -------
    list[float]   All distinct compound masses (Da), sorted ascending.
    """
    with open(path) as f:
        payload = json.load(f)
    return payload["all_masses"]


def load_edges_by_n(path="edges.json"):
    """
    Load the precomputed edge file and return per-n mass lists.

    Returns
    -------
    dict  {int n: list[float]}
    """
    with open(path) as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload["lists"].items()}


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Precompute compound amino-acid masses for de novo sequencing."
    )
    parser.add_argument(
        "--max_n", type=int, default=4,
        help="Maximum number of AAs per compound mass (default: 4)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="edges.json",
        help="Output JSON file path (default: edges.json)."
    )
    parser.add_argument(
        "--precision", type=int, default=PRECISION,
        help=f"Decimal digits for deduplication (default: {PRECISION})."
    )
    args = parser.parse_args()

    payload = build_and_save(
        max_n=args.max_n,
        path=args.output,
        precision=args.precision,
    )

    # Quick summary
    print()
    print(f"{'n':>4} | {'|List_n|':>12} | {'cumulative':>12}")
    print("-" * 36)
    cumulative = 0
    seen = set()
    for n in range(1, args.max_n + 1):
        size = len(payload["lists"][str(n)])
        seen.update(payload["lists"][str(n)])
        print(f"{n:>4} | {size:>12,} | {len(seen):>12,}")


if __name__ == "__main__":
    main()