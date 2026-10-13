# DeNovoFFC

DeNovoFFC reconstructs peptide sequences from two-dimensional fragment–fragment
correlation (FFC) maps. It supports HCD b/y and ETD c/z· analysis, with optional
combination of complementary acquisitions, orientation-constrained reconstruction
(RBDP), and FFC-based ranking of unresolved mass gaps.

The de novo route uses a measured precursor mass and does not require a sequence.
Supplying a sequence selects a separate, reference-assisted annotation route.
Results retain tied reconstructions and unresolved gaps so that ambiguity remains
visible.

The C++17 engine performs de novo reconstruction, RBDP, prefix combination, and
gap-candidate enumeration. Python handles preprocessing, deconvolution,
reference-assisted annotation, and gap evidence matching. A Python reference
implementation of the reconstruction engine is also included.

## Installation

From the repository root, use Python 3.10 or later and a C++17 compiler:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DENOVOFFC_PYTHON="$VIRTUAL_ENV/bin/python"

./cpp/build.sh
./denovoffc info
python tests/run_tests.py
```

The build uses CMake when available and falls back to `g++`. On Linux,
`DENOVOFFC_NO_CMAKE=1 ./cpp/build.sh` explicitly selects the compiler-only build.
`--engine auto` uses C++ when built; `--engine cpp` requires it, and
`--engine python` selects the reference implementation. The Python engine generates
its compound-mass table on first use, which takes additional time and disk space.

The source includes the modules, lookup resources, and synthetic data needed for
the examples and portable tests. No separate research repository is needed.

TopFD is an optional external dependency for the second deconvolution stage.
Set `DENOVOFFC_TOPFD_BIN` to its executable and, if needed,
`DENOVOFFC_TOPFD_LIB` to its library directory. Use `--no-topfd` when running only
the first deconvolution stage. Already-deconvolved maps do not require TopFD.

## Try the included example

The [synthetic example](examples/synthetic/README.md) contains complementary HCD
and ETD maps. These are generated data intended to demonstrate the workflow.
Run the following commands from the repository root:

```bash
# Reconstruct the HCD acquisition.
./denovoffc run --data examples/synthetic/maps/hcd.tsv \
    --charge 3 --mass 1936.888995 --terminus H2O --top-list 500 \
    --engine cpp -o result/example_hcd

# Reconstruct the ETD acquisition.
./denovoffc run --data examples/synthetic/maps/etd.tsv \
    --charge 4 --mass 1936.888995 --terminus H2O --fragmentation ETD \
    --top-list 500 --engine cpp -o result/example_etd

# Combine both acquisitions and rank fills for the remaining gaps.
./denovoffc combine --manifest examples/synthetic/sample.json \
    --engine cpp --gap-fill ffc -o result/example_combined
```

The combined result is `result/example_combined/reconstruction.json`; gap rankings
are in `result/example_combined/gap_filling/`. The example retains alternative
reconstructions. A completed run does not imply a unique sequence.

`combine` and `fill-gaps` require an empty or new output directory. Choose a new
output name when repeating these commands.

## Use your own data

### Input tables and precursor chemistry

Use a tab-separated table (`.tsv`) or comma-separated table (`.csv`). The included
[maps](examples/synthetic/maps/) illustrate the column names and layout:

| Columns | Meaning |
| --- | --- |
| `m/z A`, `m/z B` | The two fragment m/z values in each FFC |
| `Ranking` | FFC ranking used for top-N selection |
| `Covariance`, `Partial Cov.`, `Score` | Correlation measurements and score |
| `intensity A`, `intensity B` | Fragment intensities, used during deconvolution |
| `charge_A`, `charge_B` | Assigned fragment charges in deconvolved input |

With `--deconvolve auto`, the presence of both charge columns selects the
already-deconvolved route. Set `--deconvolve yes` to deconvolve raw input;
`--deconvolve no` skips this stage.

Specify `--mass` as the **neutral precursor mass in Da**, not precursor m/z or
charged mass. Set `--terminus H2O` for a free carboxyl terminus or `--terminus NH3`
for a C-terminal amide. Use the measured precursor charge for `--charge`.

`--fragmentation HCD` selects b/y by default. `--fragmentation ETD` or `ETHCD`
selects c/z·, whose complementary fragment pair carries z−1 charges; this route
requires precursor charge ≥3. An explicit `--ion-series by` selects a separate
b/y analysis of an EThcD map. Both ladders are not analyzed jointly in one run.

To convert a blocked covariance export before deconvolution, replace these input
filenames with your files:

```bash
./denovoffc prepare --format blocked-covariance \
    --covariance Covariance_Data.x --intensity Intensity_Sum.x \
    -o prepared/map.tsv
```

Preparation writes a flat FFC table and a `.meta.json` sidecar. Preserve sidecars
with their maps: they record preparation and, for deconvolved output, line-model
settings.

### Optional RBDP reconstruction

The default reconstruction method is `unit`. RBDP applies orientation constraints
from internal-line evidence during the final graph search. It does not change
line discovery or provide a calibrated probability of sequence correctness.

```bash
./denovoffc run --data examples/synthetic/maps/hcd.tsv \
    --charge 3 --mass 1936.888995 --terminus H2O --top-list 500 \
    --reconstruction-method rbdp-guarded --engine cpp -o result/example_rbdp
```

| `--reconstruction-method` | De novo behavior |
| --- | --- |
| `unit` | Ordinary reconstruction, without RBDP constraints |
| `rbdp` | Constrained reconstruction; may return no candidates when evidence is insufficient or constraints are infeasible |
| `rbdp-guarded` | Falls back to the unit result on abstention, infeasibility, or a shorter constrained optimum; records the reason |

RBDP is supported for HCD b/y. It is rejected for c/z; explicit EThcD b/y use has
not been validated. `--recon-sigma-scale` independently scales final-graph
uncertainties and defaults to `1.0`. Enabling RBDP does not change that value.
Inspect the recorded effective method and fallback reason when comparing runs.

### Combine acquisitions

A JSON manifest describes acquisitions of the same analyte. Start with
[examples/synthetic/sample.json](examples/synthetic/sample.json), then set the
neutral mass, terminus, and each dataset's path, fragmentation, charge, and top-N.
Paths are relative to the manifest. Each dataset needs a distinct ID;
`acquisition_id` identifies the underlying acquisition so that repeated views are
not treated as independent measurements. Do not put a reference sequence in a
de novo manifest.

```bash
./denovoffc combine --manifest examples/synthetic/sample.json \
    --selection explicit --datasets hcd_1 etd_1 -o result/example_pair
```

The default `--mode prefix` combines oriented prefix boundaries from candidate
reconstructions. `--selection all` uses all datasets; `best-per-series` chooses
one per ion series using resolved residue count, longest resolved run, and gap
count. These are presentation criteria, not estimates of correctness.

The default `--candidate-policy all` searches candidate tuples up to
`--max-combinations` (10,000). Incomplete source enumeration or an oversized
product produces `combination_limit`. `--candidate-policy first` explicitly uses
only the first candidate per dataset and records that the search is incomplete.

Dataset, acquisition and analyte IDs become directory names and table fields:
use letters, digits, `.`, `_`, `+` or `-`, starting with a letter or digit.
Put free text in a dataset's `label`. `--selection explicit --datasets` must
list distinct, known IDs. Two datasets of one acquisition are accepted only as
exact duplicates (same file and settings), which are combined once; choose
between different representations of one acquisition explicitly.

A dataset can name a prior reconstruction artifact instead of a map. It must
declare itself sequence-free (`kind`, `reference_assisted: false`) and record
chemistry that matches the manifest: neutral mass (within `mass_tolerance_da`,
default 0.001 Da), terminus, ion series, precursor charge and coordinate basis.
Reference-assisted, contradictory or mismatched artifacts are refused before
any output is written. A combined result cannot be combined again; combine the
original datasets instead. A legacy result that records no provenance is
refused unless you pass `--assume-sequence-free`, which is recorded with the
result. A metadata sidecar next to a map must agree with the manifest's mass,
series and charge.

### Rank unresolved gaps

Gap scoring can follow `run` or `combine` with `--gap-fill ffc`, or run separately:

```bash
./denovoffc fill-gaps \
    --reconstruction result/example_combined/reconstruction.json \
    --evidence-manifest examples/synthetic/sample.json --score-datasets all \
    -o result/example_gaps
```

Each gap is evaluated independently while other gaps remain unresolved mass
blocks. Candidate strings must fall within `--gap-tol-da` (default 0.03 Da).
If enumeration exceeds `--gap-max-candidates` (50,000), the gap is left unresolved.
The default ranking compares parental FFC matches, then shorter fill length,
then clean and total FFC matches. The default scoring depth is 500 FFCs per map,
with 20 ppm b/y and 30 ppm c/z matching tolerances.

`--fill-policy rank-only` is the default. `unique-best` adds a separate proposal
when one fill ranks first without a tie; it does not alter the original candidate.
A fully filled proposal's mass is computed from its residue symbols and terminus,
and must match the measured neutral mass within `--proposal-mass-tol-ppm`
(default 10 ppm, independent of `--gap-tol-da` and of the number of filled
gaps). A unique winner is not a confidence estimate. `fill-gaps` applies the
same provenance and chemistry checks as `combine` to its reconstruction.
Gap enumeration uses C++ when available; FFC matching currently uses Python.
`--workers` enables parallel scoring. `--gap-engine cpp` is not implemented.

### Annotate a known sequence

Supplying `--seq` selects reference-assisted annotation. For the synthetic HCD map:

```bash
./denovoffc run --data examples/synthetic/maps/hcd.tsv \
    --seq GASPVTDNEKWRYHMF --charge 3 --terminus H2O --top-list 500 \
    -o result/example_annotation
```

The precursor mass is calculated from the sequence if `--mass` is omitted.
Annotation uses the known sequence when aligning evidence and must be evaluated
separately from de novo reconstruction. Reference-assisted combination is also
available through `combine --mode annotated-master --reference-sequence`;
see `./denovoffc combine --help` for its selection options.

## Read the results

Single-map de novo runs write JSON and spreadsheet results under
`denovo/top<N>/`, plus `denovo/summary.csv` and `run.json`. Combination writes
`reconstruction.json`, per-dataset reconstructions under `runs/`, and
`combination/selection.tsv`. Gap scoring writes `gap_filling/gaps.json`,
`candidates.tsv`, `evidence.tsv`, and `proposals.json`.

| Result field | Interpretation |
| --- | --- |
| `candidates` | Structured residue/gap blocks, prefix boundaries, and provenance |
| `best_weight` / `objective_score` | Graph-search objective, not a probability |
| `returned_path_count` | Number of candidates listed |
| `exact_optimal_path_count` | Exact count as a decimal string where available; combined results may report `null` |
| `enumeration_truncated` | Whether path listing stopped before all optimal paths were returned |
| `status` | Outcome such as `ok`, `no_path`, or `combination_limit` |
| `chemistry`, `effective_parameters`, `reconstruction` | Chemistry and method settings, where supplied by the producing stage |

Numeric tokens in a displayed sequence are unresolved masses in Da. `J` denotes
an unresolved isoleucine/leucine assignment. `--max-paths` limits the number of
listed paths. `--presentation-order informative` sorts tied candidates by how
much sequence they resolve; the first candidate is not necessarily correct.

Exit code 0 means the analysis completed, including explicit no-path or
budget-limit outcomes. Check the result status before using candidates. Exit
code 1 indicates a runtime or I/O failure; code 2 indicates invalid input or an
integrity error. Full options are available through `./denovoffc <command> --help`.

## Validation and current scope

```bash
python tests/run_tests.py
python tests/run_tests.py -k rbdp
bin/denovo_tests
```

The portable suite uses synthetic fixtures and requires no private data, TopFD,
or network access. Optional measured-data checks are described in
[benchmarks/README.md](benchmarks/README.md); their data and local manifests are
not required for normal use.

Prefix combination and gap scoring are experimental. Prefix combination can use
only boundaries retained by the individual reconstructions, and its accuracy
has not been established on a broad independent benchmark. Gap evidence can
favor an incorrect fill. RBDP's reduction in the number of tied candidates does
not, by itself, establish improved sequence accuracy. The current gap dictionary
uses unmodified residues.

Implementation details and module boundaries are documented in [DESIGN.md](DESIGN.md).
