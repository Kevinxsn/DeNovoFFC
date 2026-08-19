# DeNovoMMS — Design Notes

Reference for this repository. Read this first when picking up work here; it
captures *why* things are the way they are and where the traps are, not every
line of code.

DeNovoMMS is the consolidation of the `2D_spec_dict` research tree into one
runnable pipeline. The five design docs that tree accumulated are still the
authority on their own stages and are vendored alongside the code they
describe:

| doc | covers |
|---|---|
| `src/deconvolution/deconvolution.md` | charge + monoisotopic mass assignment |
| `src/line_based_annotation/line_based.md` | line detection, the coverage table, the master TFFC-set |
| `src/annotation_pepline/annotation_pipeline.md` | the three-analysis batch runner |
| `src/de_novo/denovo.md` | de novo sequencing, the sigma model, the benchmark |
| `src/ETD/etd.md` | the c/z ladder, and why it could not be a parameter change |

This document is the layer above them: what got combined, what is new, and
what changed in the combining.

---

## 1. What this does

An **MMS map** is a table of fragment–fragment correlations (FFCs): pairs of
m/z values that co-vary across scans because both fragments came from the same
precursor. Given one map, produce the peptide.

Two routes, chosen by one question — **do we already know the sequence?**

```
                         ┌─ deconvolution (optional) ─┐
  raw or deconvolved     │                            │
  MMS map          ──────┤                            ├──► unknown ──► DE NOVO MMS
                         │  charge + monoisotopic     │
                         │  mass on every fragment    ├──► known   ──► ANNOTATION
                         └────────────────────────────┘                (de novo runs
                                                                        inside it)
```

**Why the known-peptide route does not also run de novo separately.** The
line-based annotation module's v2 path already builds a master TFFC-set and
reconstructs from it — that is what `reconstruct=True` +
`extend_master_set=True` do (`line_based.md` §8e). Running the standalone de
novo module afterwards would answer the same question with strictly *less*
information: it deliberately discards the fragment index, the b/y/internal
labels, and the snap onto exact `prefix[k]`. So on the known route the de novo
search is a **stage inside** the annotation, not an alternative to it, and the
annotation additionally reports which theoretical ion every FFC is and which
backbone bonds are covered.

The de novo module exists as a *separate* module because the unknown case is a
genuinely different algorithm — see §5.2.

---

## 2. Layout

```
DeNovoMMS/
  denovomms            the one entry point (bash shim -> pipeline/cli.py)
  DESIGN.md            this file
  README.md            how to run it

  pipeline/            NEW code -- the unifying layer
    cli.py               route selection, flags, output tree, manifest
    paths.py             where everything lives; external-tool resolution
    prep.py              load, mirror-duplicate merge, top-N cut
    deconvolve.py        wrapper around the deconvolution stage
    denovo.py            dispatches to the C++ engine or the Python reference
    annotate.py          wrapper around the three annotation analyses

  cpp/                 NEW code -- the C++ de novo engine
    include/denovo/*.hpp, src/*.cpp
    CMakeLists.txt, build.sh
  bin/denovo_mms       the built engine (gitignored)

  src/                 VENDORED from 2D_spec_dict, layout preserved (see below)
    deiso/python_code/   peptide model, merge_ffcs, line_finding, greedy_line,
                         the Stage-1 deconvolution back end, theo_patt.txt
    line_based_annotation/  line detection + the coverage table (_oa, _oa_v2)
    theoretical_mms_annotation/  per-FFC annotator + Averagine envelopes
    reconstruct/         path_finding, master_reconstruction, edges, graphs
    data_prep/           peptide_info + pep_seq.xlsx (NAME -> sequence)
    deconvolution/       the two-stage deconvolution pipeline
    annotation_pepline/  the three-analysis runner
    de_novo/             the Python de novo reference
    ETD/                 the c/z copies of the annotators

  tests/
    parity.py            C++ engine vs the Python reference -- run after ANY change
    smoke.sh             every route, end to end, on real data
```

### ⚠ `src/` mirrors the original directory layout on purpose

The vendored modules bootstrap their own `sys.path` from *relative* positions
(`deconvolution/_paths.py` computes `BASE = ..`, `denovo_core.py` appends
`../deiso/python_code`, `../line_based_annotation`, `../reconstruct`). Keeping
`src/` shaped exactly like `2D_spec_dict/` means **every vendored file imports
its siblings unchanged**, which is what makes "don't modify the existing code"
achievable rather than aspirational.

Two consequences worth knowing:

* If you move a vendored file to a "nicer" location, its imports break at load
  time and the failure looks like a missing dependency.
* `pipeline/paths.py` puts all of `src/` on `sys.path` at import. Module names
  are therefore **flat and global** — `peptide`, `edges`, `path_finding`. Two
  vendored `peptide.py` files exist (`deiso/python_code` and `reconstruct`);
  the one that wins is whichever directory comes first in `sys.path`, which
  `paths.py` fixes deterministically. Do not add another module with a name
  already in use.

### What was deliberately *not* vendored

* **TopFD** (Stage 2 of the deconvolution) — a ~38 MB binary plus TopPIC shared
  libraries. Pinning a copy here would silently fork someone else's release.
  Resolved at runtime from `$DENOVOMMS_TOPFD_BIN` → `topfd` on `PATH`, and its
  libraries from `$DENOVOMMS_TOPFD_LIB`. Absent, `--no-topfd` runs Stage 1
  alone; it places fewer rows but does not fail.
* **`reconstruct/edges.json`** — 114 MB, and reproducible in 30 s (Python) or
  80 ms (C++). Generated on demand, gitignored. See §6.1.

---

## 3. Choosing the route

| you have | flags | route |
|---|---|---|
| a map, a charge, a measured neutral mass | `--charge --mass --terminus` | **de novo** |
| the above plus the sequence | `+ --seq` | **annotation** |
| a dataset name resolvable in `pep_seq.xlsx` | `--name HAD_33_Z4_E16` | **annotation** (sequence, charge and C-terminus all resolved from the name) |

Deconvolution is a **separate axis**, not part of the route choice:

| `--deconvolve` | behaviour |
|---|---|
| `auto` (default) | run it only when the input has no `charge_A`/`charge_B` columns — i.e. when it is a raw map. The decision is printed. |
| `yes` | always, even on an already-deconvolved table (it re-deconvolves from the raw m/z) |
| `no` | never |

`auto` keys on the charge columns because that is exactly what the
deconvolution adds: a raw MMS map is m/z pairs, covariance, a ranking and
intensities, and nothing else.

---

## 4. The stages

### 4.1 Deconvolution (optional, both routes)

Assign a charge and a monoisotopic mass to every fragment. Full detail in
`deconvolution.md`; the three things to remember here:

* **⚠ It wants the CHARGED precursor mass**, `[M+zH]z+`, not the neutral one.
  The line model is `v = i·mzA + j·mzB` with no proton term, so a parental line
  sums to the charged mass. The CLI takes `--mass` as the **neutral** mass
  everywhere (that is what the de novo stage wants) and converts once, in
  `cli.charged_mass`. Handing the neutral mass straight through puts every
  line's shift off by `z · proton`.
* **`top_lines = 2000` is tuned, not arbitrary.** Bond coverage is *flat* from
  2000 to 15000 on all 7 benchmark peptides, while spurious parental pairs rise
  in 5 of 7 and detection takes 1.5–5× longer. Looking deeper into the ranking
  for lines finds more lines, not more peptide.
* **⚠ Incomplete input intensities look exactly like a pipeline failure.** The
  envelope refinement scores isotope patterns against measured intensities, so
  a peak with no intensity yields no monoisotopic mass — and TopFD *segfaults*
  on a peak list containing NaN intensities. Both symptoms point at the
  deconvolution and neither is its fault. Check `stage1_unplaced_rows`.

### 4.2 Preprocessing — and the merge-tolerance trap

Both routes share one pass: `merge_duplicate_ffcs_charge(symmetric=True,
reset_ranking=True)` collapses mirror duplicates — FFC(X, Y) and FFC(Y, X) are
one physical correlation and the A/B ordering is arbitrary — then
`prepare_ffc_data(top_n)` drops unranked rows and takes the best-ranked head.

> **⚠ The merge tolerance is NOT the same on the two routes, and it is not
> cosmetic.** `run_denovo.py` leaves the function default of **0.001**;
> `run_annotation_reconstruct.py` passes **0.002**. It changes how many rows
> survive, hence which FFCs the top-N cut admits, hence which lines are
> detected — and line detection is *non-monotonic* in top-N (a different cutoff
> yields *different* lines, not merely more). Measured on KWK 5+: 13 731 vs
> 13 684 rows, and a master set of 22 vs 23 breakpoints.
>
> Both values are kept, each on the route whose published benchmark was
> measured at it (`prep.MERGE_TOLERANCE`), so both `denovo.md` §5 and
> `line_based.md` §8e reproduce exactly. `--merge-tolerance` overrides.
> Flattening the two into one number was the first bug this consolidation
> introduced, and it is invisible without a benchmark to compare against.

**This stage stays in Python** even though everything after it is in C++
(§6.2): it runs once per peptide on ~15 k rows, so it is not a bottleneck, and
its KD-tree proximity pass is the one piece with a real third-party dependency
(scipy). The C++ engine therefore takes an **already-merged** table and its
`--top-n` does `prepare_ffc_data`'s part only.

### 4.3 De novo — the unknown-peptide route

`pipeline/denovo.py` → the C++ engine, or `src/de_novo/denovo_core.py`. §5 is
the algorithm; `denovo.md` is the full treatment.

Output per top-N: a JSON result and an `.xlsx` with sheets `lines`,
`master_set`, `reconstructions`, plus a batch `summary.csv`.

### 4.4 Annotation — the known-peptide route

`pipeline/annotate.py` runs the three analyses `annotation_pipeline.md`
describes, sharing one merged table:

1. **complementary / spurious-parental test** — console only.
2. **theoretical-MMS FFC annotation** — one `.xlsx`, run once at top 2000.
3. **line-based annotation + reconstruction** — one `.xlsx` per top-N. This is
   the one that reconstructs.

Defaults are `run_annot_rec.sh`'s working configuration (`tol_ppm 30`,
`line_delta 20 ppm`, `productivity`, `extend_master_set`, `converge`,
`loose_mock`) **minus `--accept-ambiguity`**, which is left off: measured
across 7 peptides it is net negative — it buys a bond or two but injects wrong
coordinates that can cost a correct reconstruction.

> **⚠ There is no flag for the master-set reconstruction.** It fires when
> `reconstruct` *and* `extend_master_set` are **both** true, and arrives via
> `cov_table.attrs["reconstructions"]` rather than a return value. Easy to miss
> when skimming for a parameter; there isn't one.

> **⚠ `run_line_based_reconstruct` returns a tuple**, `(coverage_table,
> all_lines)`, not a DataFrame. The second bug this consolidation introduced.

---

## 5. The de novo algorithm — the essential ideas

```
detect lines  ->  seed the master TFFC-set from the parental lines
              ->  extend with aligned internal / loss lines, in batch rounds,
                  until it converges
              ->  heaviest path through the 2D-DAG  ->  sequence
```

### 5.1 Lines, and why detection ignores the charges

A **line** is a group of FFCs sharing a reconstructed mass `v = i·mzA + j·mzB`.
Detection is **orientation-agnostic**: `(i, j)` and its mirror `(j, i)` at the
same mass are the *same physical line*, so membership is counted on the
canonical charge `(min, max)` pooling both orientations, and each member's true
orientation is resolved per-FFC afterwards (whichever of `i·mzA + j·mzB` /
`j·mzA + i·mzB` lands closer to the line mass).

Detection reads **no charge information from the table at all** — each side's
charge is *recovered* from the line the FFC lands on. Input charges can be NaN
or wrong; the line cannot be.

A line's **alignment shift** `A = ParentPair − (x + y)`, taken as the median
over its FFCs, is what classifies it: `|A| < 0.5` parental, `A` matching a
named neutral loss → loss, `A` a plausible compound residue mass → internal.
⚠ `A` comes from the **FFC masses**, never from the detection shift, which
carries an extra `(i + j − z)·proton`. A charge sweep to undo that term is a
charge *assumption*; `A` has no proton term and needs no guess.

### 5.2 ⚠ The one real design problem: tolerance

The annotation module **snapped** every master coordinate onto the exact
`prefix[k]` of the known sequence before using it (`line_based.md` §8e). It had
to: the DAG admits a gap only within a window *of the gap*, so a ~163 Da first
edge gets a 0.005 Da window, while a mock sits ~75 ppm off its bond. Un-snapped,
the true breakpoints sit in the set *structurally unreachable* while a
false-but-precise coordinate is reachable and out-scores them.

De novo we cannot snap — there is no `prefix[k]`. So:

> **Every coordinate carries its own absolute uncertainty `sigma`, derived from
> the mass that was actually measured, and every DAG edge window is
> `sigma(a) + sigma(b)`.**

A gap is a difference of two measurements, so its uncertainty is the sum of
theirs. That single change removes three separate patches upstream:

* **Per-coordinate, not one ppm number.** A breakpoint is *folded* into
  `(0, total/2]`, and **folding moves a coordinate but not its error**. A y30
  measured at 3043 Da with 8 ppm of error folds onto a 958 Da coordinate still
  carrying 0.025 Da — 27 ppm *there*. A ppm window on the folded value would
  demand 27 ppm to accept an 8 ppm measurement. The annotation module patched
  this with a `max(c, total − c)` proxy; here `coord_from_side` knows which
  side produced the coordinate and attaches that side's own error, so there is
  nothing to guess. **This is the single easiest way to lose a correct
  coordinate with no error message.**
* **The mock's tolerance falls out.** A mock is a line's *median alignment
  shift* — a difference of two large masses, not a single peak — so its error
  scales with `total`, not with `A`. `sigma_A = min(total·tol_ppm·√2/√n, cap)`.
  The annotation module needed a separate, hand-set `mock_tol_ppm = 100` for
  exactly this.
* **Corroboration replaces the snap.** Two TFFCs on one breakpoint are
  independent measurements of one quantity, so `MasterSet.add` merges them by
  inverse-variance weighting: the coordinate moves toward the better-measured
  value and sigma shrinks, floored at `sigma_floor_ppm` so it cannot claim
  better accuracy than the instrument has. A coordinate gets *closer to exact*
  the more evidence supports it — which is the de novo stand-in for teleporting
  it onto the truth.

### 5.3 The master set, and why rounds are batch

`aligned` / `active` / `unambiguous` are all defined **relative to the master
set**, so growing the set changes them. Hence rounds: judge every pending TFFC,
bank the **unambiguous** ones (their single forced image), re-derive, repeat
until the set stops growing.

Every TFFC in a round is judged against the set **as it stood at the start of
that round**, and winners are banked together at the end. That is what keeps a
round order-independent — no TFFC can be promoted or silenced by another
decided moments earlier in the same round.

> **⚠ An ambiguous TFFC's two images are atomic.** Its evidence is "one of
> these two breakpoints", never "this one". Bank both or neither; banking one
> invents a decision the data does not support. `accept_ambiguity` governs it
> in one place, and defaults **off**: across seven peptides ambiguous TFFCs
> bought **one** usable bond and supplied **four** wrong coordinates.

### 5.4 The secondary-pair mock gate

An internal line's **mock** claims: *the line is missing a fragment of mass
`A`, so the cleavage that released it is a real breakpoint at `fold(A)`*. True
when the line's FFCs are internal-ion pairs — a terminal run is gone from every
fragment on the line.

But an internal line arises a **second** way, and then the claim is false: a
**secondary** pair is two ordinary terminal ions `b_k` and `y_{n−k−r}` with an
*interior* run of `r` residues between them. `A` is that interior run and
`fold(A)` is no cleavage at all — it fabricates a breakpoint near a terminus.
Measured: HGT's three `A = 113.08 / 131.09` lines invented an N-terminal `J`,
even though HGT starts with H and ends with K.

**The two separate without ground truth.** A secondary pair's two sides are
*both* genuine terminal fragments, so **both** of the TFFC's images are already
known breakpoints; a true internal pair has at most one side that is. So a line
whose TFFCs are repeatedly "known on both images" is a secondary line and its
mock is refused (`secondary_gate`, default 0.25). Tightening that gate from 0.5
to 0.25 is what took the benchmark from 13/14 to **14/14**.

### 5.5 ETD / EThcD — three numbers, one geometry

A c/z pair is still an N-terminal fragment meeting a C-terminal fragment at one
backbone bond, so every structural test is already correct. What changes:

| | N-term ion | C-term ion | pair sum | ladder charge |
|---|---|---|---|---|
| HCD `b/y` | `prefix[k]` | `total − prefix[k]` | `M + 2p` | `i+j = z` |
| ETD `c/z·` | `b_k + NH3` | `y_j + H − NH3` | `M + H + 2p` | `i+j = z − 1` |

Folding needs "measured neutral → coordinate" to be an involution. For b/y it
is; for c/z it is **not**, until `NH3` comes off every measured side:

```
u = x − NH3 − PROTON        c ion  ->  u = prefix[k]
                            z· ion ->  u = TOTAL_STAR − prefix[k]
TOTAL_STAR = total + H − 2·NH3
```

after which the whole coordinate layer is reused untouched and simply told a
different total.

> **⚠ Both failure modes here are silent.**
> * **`--fragmentation` wrong.** Electron transfer consumes one of the
>   precursor's protons, so a c/z pair from `[M+6H]6+` carries **five**
>   charges. The ladder sits one *electron* (0.0006 Da) from where a b/y
>   parental line would — the same **mass** on a different **charge sum** — so
>   the parental pass, which selects on `i + j == parent_charge`, cannot see
>   it. Measured: an ETD map read as HCD finds **0** parental lines and 0.0%
>   coverage; read as ETD, 2 lines and 77.8%. Nothing errors.
> * **`--terminus` wrong.** An amide is 0.984 Da lighter than a free acid, so
>   every y ion is off. Paths are still found; none are right.
>
> Neither is checkable in-band. They are chemistry, and the pipeline has to be
> told.

> **⚠ Two totals are live at once.** Coordinates use `coord_total`
> (`TOTAL_STAR`); the alignment shift `A` is computed from measured masses on
> the *real* basis and must use `parent_pair = M + H + 2p`. Mixing them is a
> ~33 Da error that reads as a bad line rather than as a bug.

`TOTAL_STAR` is bookkeeping, not a mass — its implied terminus comes out
**negative** (−15.03 for a free acid). Harmless: it is only ever the y-axis
origin, and `coord_total − coord_term == prefix[n]` exactly in both modes.

---

## 6. The C++ engine

`cpp/` is a port of `src/de_novo/denovo_core.py` — **a port, not a
reimplementation**. It must agree with the Python on every reported quantity,
and `tests/parity.py` is what enforces that (§6.4).

### 6.1 What was ported, and why those parts

Profiling one peptide (`YPS6+`, top500, 5.2 s total) put the time in three
places, all of which are pure computation over small arrays — i.e. exactly what
Python is worst at and C++ is best at:

| stage | Python | why it costs |
|---|---|---|
| loading `edges.json` | **1.86 s** | 2 113 632 floats through `json.loads`, re-parsed on *every* run |
| line detection | **~1.8 s** | pooled `i·x + j·y` over all `(i,j)`, clustered and re-clustered once per line the greedy peeling finds |
| DAG builds + heaviest path | **~1 s** | the goodness test asks "would adding this breakpoint lengthen the path?" 6–39 times per peptide, each a fresh graph over ~600 nodes |

So the whole of `denovo_core` was ported:

* **`edges.cpp`** — the compound-residue-mass table is *generated* rather than
  loaded, by Minkowski-summing a bitset 20 times. 2 113 632 masses in **80 ms**,
  bit-identical to `edges.json`. The 114 MB file is never needed.
* **`lines.cpp`** — `detect_lines_oa`, `_most_informative_line_oa`,
  `_sort_and_split_oa`, `_remove_lines_oa`, `satellite_lines`.
* **`masterset.cpp`** — inverse-variance merging, the sigma floor, the view.
* **`dag.cpp`** — `build_dag`, `find_all_heaviest_paths`, the cached `Scorer`.
* **`pipeline.cpp`** — `classify_line`, `mark_isotope_satellites`,
  `_isotope_k`, `_mock_sigma`, the rounds loop, the settle pass,
  `_format_gapped`.

No dependencies beyond a C++17 compiler. `cpp/build.sh` uses CMake when
available and a single `g++` invocation otherwise.

### 6.2 What stayed in Python, and why

* **`merge_duplicate_ffcs_charge` / `prepare_ffc_data`** — §4.2. Runs once, on
  ~15 k rows, and its KD-tree pass is the one genuine scipy dependency. Porting
  it would buy ~0.3 s and risk the two implementations clustering the same data
  differently, which is precisely the class of bug this consolidation exists to
  avoid.
* **The whole annotation route.** It is not a hot loop — it is Excel writing,
  fragment-index lookups and report formatting, and it is where the science is
  still moving. A C++ copy would fork the design docs' source of truth.
* **The deconvolution.** Dominated by TopFD (an external binary) and by the
  already-vectorised numpy detector `deconvolution.md` §5 describes.
* **`evaluate.py`.** Benchmark scoring, off the hot path — and keeping it out
  of the engine is what makes "no ground truth in the pipeline" checkable.

### 6.3 Measured

Seven benchmark peptides, top500, one process per peptide:

| | Python | C++ | |
|---|---|---|---|
| per peptide, end to end | 2.2 – 3.9 s | **0.077 – 0.128 s** | **~25–30×** |
| all seven | 20.8 s | **0.70 s** | |
| edge table | 1.86 s (parse) | **0.07 s** (generate) | ~26× |
| line detection + search | ~1 – 2 s | **0.003 – 0.048 s** | ~30–300× |

The C++ figures **include** process start and a full edge-table regeneration
every run, so they are what a caller actually waits for, not an inner-loop
microbenchmark. `bin/denovo_mms --data ...` prints the split.

### 6.4 ⚠ Parity is checked, not assumed

```
PY=/path/to/python
$PY tests/parity.py              # 7 HCD peptides x {200, 500}
$PY tests/parity.py --etd        # 7 ETD/EThcD datasets x {200, 500}
```

Runs both implementations on the same prepared table and diffs **line count and
per-line category, master-set size and coordinates (to 1e-6 Da), convergence
rounds, tied-path count, and the full set of reconstructions** — not merely
"did it reconstruct the same peptide". Current state: **28 / 28 runs identical**.

Run it after *any* change to either side. If they disagree, **the Python is
right by definition** and the port has a bug.

### 6.5 Porting traps — the things that are easy to get subtly wrong

These are not hypothetical; each one is a place where a "cleaner" C++ version
would silently diverge.

* **Do not tidy the constants.** `MASS_H = 1.00784` and `PROTON = 1.00725` are
  both "a proton" and differ in the 5th decimal; `ISOTOPE_SPACING = 1.00235`
  (satellite lines) and `ISOTOPE_OFFSET = 1.00335` (alignment shifts) are both
  "the 13C step". `classify_line`'s named losses and `satellite_lines`'
  named losses are the *same three molecules with different literals*. Every
  one of these is reproduced verbatim.
* **`min()` and `max()` keep the FIRST extremum.** `_format_gapped` picks a
  residue by `min(aa.items(), key=...)`, so a tie between two residues resolves
  by the dict's insertion order — which is why `constants.hpp` stores the
  residue table as an ordered array and not a map.
* **The cluster tie-break is positional.** `_most_informative_line_oa` keeps a
  cluster only on `key > best[0]` (strictly), and clusters arrive in ascending
  mass order, so ties go to the **lowest-mass** cluster. A `>=` there changes
  which lines are found.
* **`allowed()` checks `idx` before `idx − 1`.** When both neighbours are
  inside the window the one at or *above* the gap wins, even if the lower one
  is closer. Not the nearest-match it looks like.
* **Line mass is the mean over *all* entries on the dominant charge**,
  including both orientations of a doubly-present FFC — not over the distinct
  members.
* **`by_key` / `pending` are last-wins dicts.** Two lines can share a key
  (`(i,j)` plus `A` to 3 dp), and the Python quietly keeps the later one.
* **Bank before erasing.** The rounds loop records `pending[k]` indices while
  scanning; erasing during the scan invalidates them. The Python's
  `list.remove(t)` hides this because it removes by value.
* **The topological shortcut is justified, not assumed.** The recursive DFS was
  replaced by one pass in decreasing `max(x1, x2)`. That is valid *because*
  every Down/Right move strictly increases `max(x1, x2)` and the Sink is
  terminal — so all successors are already resolved, and nodes tied on `max`
  have no edge between them.
* **Tied-path order must be preserved.** `heaviest_paths` reproduces the
  reference's neighbour order (sink, then down ascending, then right ascending)
  and its break-on-cap behaviour, so the *count* and the *listing* match.
  Persistent cons-lists replace `[node] + n_seq`: same contents, no quadratic
  copying.
* **The Scorer cache key is rounded** — coords 4 dp, sigmas 5 dp, weights 3 dp.
  A finer key is not "more correct"; it changes how many DAG builds happen and
  therefore is part of the observable behaviour.

---

## 7. Parameters worth knowing

Full lists live in the vendored docs and in `denovomms run --help`. These are
the ones that actually decide outcomes.

| knob | default | why this value |
|---|---|---|
| **de novo `--top-list`** | `200 500` | ⚠ **the dominant hyper-parameter, and lower is cleaner.** Correct reconstructions across 7 peptides: **7/7** at 200, **7/7** at 500, 4/7 at 1000, 2/7 at 2000. Coverage keeps rising past 500 but wrong coordinates rise ~10× faster, and *one wrong coordinate usually costs the whole reconstruction*. Going above 500 is a real risk, not a free coverage win. |
| de novo `--delta` | `0.015` Da | the single most valuable line-detection knob (0.02 → 0.015 took the benchmark 11/14 → 13/14). Flat over 0.010–0.015; **breaks at ≥ 0.018**, where Sort-and-Split starts merging distinct lines on the heavy peptides. Da, not ppm: at 4000 Da, 20 ppm is 0.08 Da. |
| de novo `--tol-ppm` | `25` | drives **every** sigma. Flat across 20–40. |
| `--secondary-gate` | `0.25` | §5.4. Worth 13/14 → **14/14** on its own. |
| `--accept-ambiguity` | **off** | §5.3. Net negative on both routes. |
| annotation `--top-list` | `500 2000` | the annotation has the fragment index to filter with, so it tolerates a deeper table than de novo does. |
| annotation `--tol-ppm` | `30` | 20 ppm keeps junk lines quietest but is too tight for KWK's b7 (0.026 Da off). |
| deconvolution `top_lines` | `2000` | §4.1. |
| `--merge-tolerance` | route-dependent | §4.2. |

The de novo defaults are **a plateau, not a spike** — eight neighbouring
configurations also score 14/14. That matters, because 14/14 on seven peptides
is otherwise easy to overfit.

---

## 8. Gotchas worth remembering

*Consolidated from the vendored docs plus what this repository added. The
per-stage docs have more.*

- **`--terminus` and `--fragmentation` fail silently when wrong** (§5.5). Paths
  are still found; none are right. There is no in-band warning because there
  cannot be one.
- **`--mass` is the NEUTRAL precursor mass everywhere in this CLI.** The
  deconvolution wants the charged one and the conversion happens once, inside
  `cli.charged_mass` (§4.1).
- **The merge tolerance differs between routes** (§4.2). Flattening it changes
  results without changing anything visible.
- **Sigma is referenced to the measured mass, never the folded coordinate**
  (§5.2). The easiest way to lose a correct coordinate with no error message.
- **A mock is not a measurement.** Its sigma comes from `total`, not from
  `fold(A)`.
- **Line detection is non-monotonic in top-N.** A different cutoff yields
  *different* lines, not merely more. Do not assume a top500 run contains
  everything a top200 run found.
- **One wrong coordinate usually costs the whole reconstruction.** When the
  answer is wrong, read the `#incorrect` count and the wrong coordinates
  *first* — every one carries the line that proposed it.
- **"parental" means charge-sum, not near-zero shift** in the *detector's*
  vocabulary. A neutral-loss line at −116 Da satisfies `i + j == z` and comes
  back tagged `parental`; on VEA at top 2000 that is 75 "parental" lines, one
  of which is the precursor line. The real ladder is the `|shift| < 0.5`
  subset. Reading the tag as "this is the precursor line" gives an order of
  magnitude too many.
- **Landing on a line is not the same as being deconvolved.** The line gives
  the charge; the envelope refinement still has to find a monoisotopic mass,
  and when it fails it returns NaN.
- **`denovo_core` must never import `evaluate`.** If it ever needs to, the
  thing it needs is not de novo. The engine has no way to import it at all,
  which is the stronger version of the same guarantee.
- **Two `peptide.py` modules exist** in the flat vendored namespace (§2).
- **Don't "fix" a vendored module's imports** by moving it (§2).

---

## 9. Provenance

Everything under `src/` is a **verbatim copy** from `/home/sux002/projects/2D_spec_dict`,
transferred rather than edited so the upstream tree keeps working unchanged.
The only files not copied from there are `pipeline/`, `cpp/`, `tests/`,
`denovomms`, this file and `README.md`.

If a vendored module ever *does* need editing, mark the edit `# [DeNovoMMS]`
so `grep -rn '\[DeNovoMMS\]' src/` is the complete diff against upstream — the
convention `src/ETD/` already uses with `# [CZ]`.

Large data assets carried along because the code will not run without them:
`src/deiso/python_code/theo_patt.txt` (18 MB, TopPIC's isotope-pattern table,
Stage-1 monoisotopic refinement) and
`src/theoretical_mms_annotation/james_code/Averagine_Envelopes` (17 MB, the
Averagine isotope-maximum table the per-FFC annotator needs).

---

## 10. How to run, and how to check it still works

```bash
cpp/build.sh                      # once

# unknown peptide -- de novo
./denovomms run --data map.tsv --charge 5 --mass 4001.3781 --terminus NH3

# known peptide -- annotation (de novo runs inside it)
./denovomms run --data map.tsv --seq KWKLFKK... --charge 5 --terminus NH3

# from a raw map: deconvolve first
./denovomms run --data raw_map --charge 3 --mass 1605.8474 --terminus H2O \
                --deconvolve yes

./denovomms info                  # what this install can see
```

```bash
$PY tests/parity.py               # C++ vs Python, every reported quantity
$PY tests/parity.py --etd
tests/smoke.sh                    # every route, end to end, on real data
```

README.md has the operational detail.
