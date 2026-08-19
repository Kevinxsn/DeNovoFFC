# DeNovoMMS

Peptide sequencing from **MMS maps** — 2D fragment–fragment correlation data,
where each row is a pair of m/z values that co-vary across scans because both
fragments came from the same precursor.

One command, two routes:

```
peptide unknown  →  [deconvolution]  →  de novo sequencing
peptide known    →  [deconvolution]  →  annotation  (de novo runs inside it)
```

Deconvolution is optional on both, and skipped automatically when the input is
already deconvolved.

The de novo search is implemented in C++ (`cpp/`) and runs about **25–30×
faster** than the Python reference, which is kept alongside it and checked
against it on every reported quantity (`tests/parity.py`).

---

## Install

```bash
git clone <this repo> && cd DeNovoMMS

# 1. Python side
pip install -r requirements.txt          # numpy, pandas, scipy, openpyxl, matplotlib

# 2. C++ de novo engine  (needs only a C++17 compiler)
cpp/build.sh                             # -> bin/denovo_mms

# 3. Check what the install can see
./denovomms info
```

`./denovomms` picks an interpreter that can import pandas. To pin one:

```bash
export DENOVOMMS_PYTHON=/path/to/python
```

### Optional: TopFD (Stage 2 of the deconvolution)

TopFD is part of [TopPIC](https://www.toppic.org/) and is **not** bundled, so
it is not pinned to a stale release. Only needed if you deconvolve.

```bash
export DENOVOMMS_TOPFD_BIN=/path/to/topfd
export DENOVOMMS_TOPFD_LIB=/path/to/toppic/lib     # its shared libraries
```

Without it, add `--no-topfd`: Stage 1 alone still produces a usable table, it
just places fewer fragments.

---

## Quick start

### The peptide is unknown — de novo

```bash
./denovomms run \
  --data map.tsv \
  --charge 5 \
  --mass 4001.3781 \
  --terminus NH3 \
  -o result/
```

`--mass` is the precursor **neutral** mass, measured. Output:

```
result/denovo/top200/<label>_denovo_top200.json    the full result
                     <label>_denovo_top200.xlsx    sheets: lines, master_set,
                                                   reconstructions
result/denovo/top500/...
result/denovo/summary.csv                          one row per top-N
result/run.json                                    what was run
```

### The peptide is known — annotation

Add `--seq`. The route switches automatically.

```bash
./denovomms run \
  --data map.tsv \
  --seq KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK \
  --charge 5 --terminus NH3 \
  -o result/
```

There is no need to also run de novo: the annotation's line-based stage builds
the same master TFFC-set and reconstructs from it, *and* tells you which
theoretical ion every FFC is.

```
result/annotation/ffc_annotation/<label>_ffc_annotation.xlsx
result/annotation/line_based/top500/<label>_line_based_top500.xlsx
result/annotation/line_based/top2000/...
result/annotation/graph/top500/...png     (with --graph)
```

If your dataset name is in `src/data_prep/pep_seq.xlsx`, `--name` resolves the
sequence, charge and C-terminus for you:

```bash
./denovomms run --data map.tsv --name HAD_33_Z4_E16 -o result/
```

### Starting from a raw map — deconvolve first

```bash
./denovomms run \
  --data raw_ffc_table \
  --charge 3 --mass 1605.8474 --terminus H2O \
  --deconvolve yes \
  -o result/
```

Writes `result/deconvolution/merge/<label>_merged.tsv` and analyses it.

`--deconvolve` takes `auto` (default), `yes` or `no`. `auto` deconvolves only
when the input has no `charge_A` / `charge_B` columns — i.e. when it is raw.
The decision is printed.

### ETD / EThcD

```bash
./denovomms run --data etd_map.tsv --charge 6 --mass 4490.5753 \
                --terminus H2O --fragmentation ETD -o result/
```

ETD cleaves a different bond, so the ladder is c/z-dot rather than b/y. De novo
route only; for annotated c/z work see `src/ETD/run_etd_annotation.py`.

---

## Input format

A tab- or comma-separated table with a header. Required:

| column | meaning |
|---|---|
| `m/z A`, `m/z B` | the two correlated fragment m/z values |
| `Ranking` | correlation rank, 1 = best; `-1` = unranked (dropped) |

A **deconvolved** table also has `charge_A` / `charge_B`; a **raw** one
typically has `Covariance`, `Partial Cov.`, `Score`, `intensity A`,
`intensity B` instead. Use `--col-a` / `--col-b` / `--ranking-col` on the C++
engine if your columns are named differently; for a raw table headed by extra
lines, `--skiprows`.

---

## The three things you must get right

These are chemistry, not tuning. **Each of them fails silently when wrong** —
the pipeline still returns a reconstruction, it is just never the right one.

| flag | what it is | wrong value gives you |
|---|---|---|
| `--mass` | precursor **neutral** mass | every alignment shift off; no parental line |
| `--terminus` | `H2O` (free acid) or `NH3` (amide) | every C-terminal fragment 0.984 Da off |
| `--fragmentation` | `HCD` (b/y) or `ETD`/`ETHCD` (c/z·) | ladder on the wrong charge sum → 0 parental lines |

If a run reports **0 parental lines** or **0 master breakpoints**, check these
three before anything else.

---

## Tuning

Defaults are the values the published benchmarks were measured at. The two
worth knowing:

**`--top-list` — how deep into the ranking to read.** On the de novo route this
is the dominant knob and **lower is cleaner**: across seven benchmark peptides,
7/7 reconstruct correctly at top200 and top500, 4/7 at top1000, 2/7 at top2000.
Coverage does keep rising past 500, but wrong coordinates rise about ten times
faster, and one wrong coordinate usually costs the whole reconstruction.
Defaults: `200 500` de novo, `500 2000` annotation.

**`--tol-ppm` — fragment-mass relative error.** Drives every uncertainty in the
de novo model. Default 25 (de novo) / 30 (annotation); flat across 20–40.

Others: `--delta` (line-detection gap, default 0.015 Da — it breaks above
0.018), `--secondary-gate`, `--accept-ambiguity`, `--no-mocks`,
`--merge-tolerance`, `--max-paths`. `denovomms run --help` lists them all.

### Engine

```bash
--engine auto      # C++ if built, else Python  (default)
--engine cpp       # require the C++ engine
--engine python    # the reference implementation
```

Both produce identical results; `python` is there so a suspected port bug can
be settled in one command.

---

## Reading the output

**De novo.** A *gapped* reconstruction: residues the data resolves appear as
letters, unresolved runs as their mass. `I` and `L` are indistinguishable by
mass and print as `J`. `(sink)` marks the one gap the N- and C-terminal ladders
never bridged.

```
830.515 - 241.177 - EKVG - 511.281 - D(sink) - 170.107 - JKAGPAVAVVGQA - 541.324
```

Several reconstructions may tie. That is the size of the tie, not a shortlist
to sift — with no ground truth there is nothing to rank them by.

The `.xlsx` has three sheets: `lines` (each detected line, its category, its
alignment shift, what it contributed), `master_set` (every breakpoint with its
uncertainty, evidence count and which line proposed it) and `reconstructions`.

**Annotation.** A coverage table — rows are backbone bonds, columns are
detected lines, cells are the annotation. The summary line the CLI prints:

```
top500: breakpoint coverage 100.0%, master TFFC-set 100.0%, #incorrect 4,
        reconstructions 1/yes
```

`#incorrect` counts master TFFCs matching no backbone bond — ones added in
error. **When the reconstruction is wrong, read that number first.**

---

## The benchmark

Seven reference peptides, run end to end and graded against their known
sequences:

```bash
./denovomms bench                        # the 7 HCD peptides, top200 + top500
./denovomms bench --set etd              # the 7 ETD / EThcD datasets
./denovomms bench --set all              # both
./denovomms bench --only KWK5+ LL6+ -v   # a subset, with per-run detail
./denovomms bench --top-list 500         # one cutoff
./denovomms bench --engine python        # the reference engine instead
```

Takes ~20 s for all seven (about 4 s of that in the engine; the rest is
loading and merging the tables). Prints a per-run table, the totals, and each
peptide's reconstruction, and writes `result/benchmark/benchmark.csv` plus a
per-run `.xlsx`.

```
  top200: 7/7 correct   mean coverage 68.5%   wrong coordinates 0   residues resolved 98/188
  top500: 7/7 correct   mean coverage 71.1%   wrong coordinates 1   residues resolved 107/188
```

⚠ This is the **only** place a ground-truth sequence enters the de novo route,
and it enters strictly *after* a result exists. `--set etd` reports 5/7 per
cutoff and 6/7 across the two — line detection is non-monotonic in top-N, so a
peptide can be right at one cutoff and wrong at another without either being a
regression.

The benchmark needs the reference data under `/home/sux002/data`.

## Tests

```bash
python tests/parity.py            # C++ engine vs Python reference (28 runs)
python tests/parity.py --etd
tests/smoke.sh                    # every route, end to end
```

`parity.py` diffs line categories, master-set coordinates, convergence rounds
and the full set of tied reconstructions — not just "same peptide". Run it
after any change to either implementation.

Both need the benchmark data under `/home/sux002/data`.

---

## Using the C++ engine directly

```bash
bin/denovo_mms --data merged_map.tsv --charge 5 --mass 4001.3781 \
               --terminus NH3 --top-n 500 --json out.json
bin/denovo_mms --help
```

⚠ It expects an **already-merged** table (mirror duplicates collapsed). The
merge stays in Python; `./denovomms run` does it for you. `--top-n` here only
applies the ranking cut.

---

## Layout

```
denovomms          entry point
pipeline/          route selection and stage wrappers
cpp/               the C++ de novo engine
bin/denovo_mms     built binary
src/               the analysis modules, with their own design docs
tests/             parity + smoke
DESIGN.md          why everything is the way it is -- read this before changing anything
```

`src/` keeps the directory layout the modules were written against, so each of
them imports its siblings unchanged. Their design docs travel with them:
`src/deconvolution/deconvolution.md`, `src/line_based_annotation/line_based.md`,
`src/annotation_pepline/annotation_pipeline.md`, `src/de_novo/denovo.md`,
`src/ETD/etd.md`.
