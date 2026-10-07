# Datasets, filtering, and grouped comparisons

How to select and group surfaces for analysis, and how to run comparisons more complex
than a single violin plot. This is the layer between the raw per-surface CSVs the
pipeline writes and the statistics described in [statistics.md](statistics.md).

Three capabilities, each usable on its own and composable:

- **Filtering** — keep only the triangles you care about (e.g. cristae) before summarizing.
- **Selection** — choose which tomograms and surfaces enter an analysis.
- **Grouping** — attach metadata (condition, morphology, …) to tomograms so a treatment
  comparison flows from the config instead of hand-coded lists.

They are available as command-line flags (`violin`, `compare`, `compare_batch`), across
several runs at once via a `datasets:` block, and, for anything bespoke, through the
`Dataset` object in Python.

---

## TL;DR

```bash
# Isolate cristae (IMM far from the OMM junction) and compare their curvedness spread
morphometrics violin config.yml -n curvedness_VV --statistic std --filter 'IMM:OMM_dist>=20'

# Drop known-bad tomograms and tiny surfaces
morphometrics violin config.yml -n thickness --exclude-tomograms UF3 TE1 --min-triangles 500

# Treatment comparison between two conditions defined in the config `groups:` block
morphometrics compare config.yml -n curvedness_VV -g condition

# A whole study: many analyses x N conditions, possibly across several runs
morphometrics compare_batch study.yml
```

```yaml
# config.yml — all optional
statistics:
  filters: [IMM:OMM_dist>=20]      # triangle filters, ANDed
  exclude_tomograms: [UF3, TE1]    # names or globs
  min_triangles: 500
  cache: true                      # parquet parse-cache for faster re-runs
groups:
  condition:
    Tg:      [TF*, UF*]
    Vehicle: [TE*, UE*]
```

---

## Filtering: choosing triangles

A **filter** keeps only triangles whose per-triangle property satisfies a comparison,
applied to a surface *before* its feature is area-weighted-summarized. The canonical use
is isolating a subcompartment — e.g. mitochondrial cristae are the IMM triangles far from
the OMM contact site, so `IMM:OMM_dist>=20` selects them.

**Syntax** (no spaces): `[CLASS:]PROPERTY OP VALUE`, with `OP` one of `>= <= == != > <`.

| Filter | Meaning |
|---|---|
| `OMM_dist>=20` | every class that has an `OMM_dist` column, triangles ≥ 20 nm |
| `IMM:OMM_dist<12` | only the IMM class |
| `IMM:OMM_dist>=5` **and** `IMM:OMM_dist<=20` | a range — pass two filters |

Filters are **ANDed**, a **range is two filters**, and a filter scoped to a class
(`IMM:…`) leaves other classes untouched. Parsing is structured (a regex + a fixed
operator table), never `eval`, so a filter string is safe to store in a config file.
Semantics: a filter whose property is absent from a surface drops that whole surface, and
`NaN` never satisfies a comparison. A surface left with no triangles is dropped as a unit.

On the command line, `--filter` is repeatable and merged with `statistics.filters`:

```bash
morphometrics violin config.yml -n curvedness_VV --filter 'IMM:OMM_dist>=5' --filter 'IMM:OMM_dist<=20'
```

The active filter is recorded in the plot title and the per-unit CSV, so a figure is
self-documenting.

> Filtering here is **inline** — no files are written. If you instead want a persistent
> filtered surface to run the rest of the pipeline on, use `morphometrics extract_patches`,
> which materializes a new surface file.

---

## Selection: choosing tomograms and surfaces

Distinct from filtering *within* a surface, selection chooses *which units* enter the
analysis at all.

- `--include-tomograms` / `--exclude-tomograms` take names or shell globs (`TF*`, `?E*`,
  `UF3`); **exclude wins over include**. This replaces the hardcoded `if key in [...]`
  and `key[1] == "F"` lists that recur in older analysis scripts.
- `--min-triangles` / `--min-area` drop a surface or organelle that, after filtering, is
  too small to be a reliable data point.

All four are mirrored by `statistics.{include_tomograms, exclude_tomograms, min_triangles,
min_area}` config keys (command-line thresholds override the config). The number of
tomograms excluded and surfaces dropped is reported so a filter's effect is visible.

---

## Grouping: metadata for comparisons

A `groups:` config block maps each tomogram to a **bucket** per group, by name or glob:

```yaml
groups:
  condition:
    Tg:      [TF*, UF*]
    Vehicle: [TE*, UE*]
  morphology:
    fragmented: ["?F*"]
    elongated:  ["?E*"]
```

A bucket is a list of names/globs, a single glob, or one of two explicit forms:

| Bucket spec | Means |
|---|---|
| `"*"` | every tomogram (an ordinary glob — e.g. a dataset that is all one condition) |
| `rest` | every tomogram **no other bucket in this group claimed** — at most one per group |

```yaml
groups:
  drp1:
    Positive: [ts_002, ts_017]     # enumerate only the interesting set...
    Negative: rest                 # ...everything else lands here
```

An empty or missing bucket (`[]`, `null`) is an **error**, not a catch-all: read literally
it would match nothing, and treating it as "the rest" would quietly sweep every leftover
tomogram into it. (`"*"` next to another bucket is also an error — every tomogram would
match two buckets; that is what `rest` is for. A tomogram literally named `rest` can still
be listed as `[rest]`.)

Every included tomogram must land in **exactly one bucket per group** — this is validated
when the dataset loads, and a tomogram that matches zero buckets (unassigned) or more than
one (ambiguous) is a clear error listing the offenders, not a silent mistake. Exclude
tomograms you do not want assigned via `statistics.exclude_tomograms`.

Grouping (metadata) and selection (membership) are orthogonal: selection decides which
tomograms exist; grouping labels the ones that do.

---

## `morphometrics compare`: a spatially-aware treatment comparison

The command form of "is this feature different between two conditions?", done correctly.
It compares one feature between the two buckets of a group, **per membrane class**:

```bash
morphometrics compare config.yml -n curvedness_VV -g condition
```

```
Comparing 'curvedness_VV' across group 'condition': Tg vs Vehicle
  effect size: area-weighted ks; unit = tomogram(s); 2000 permutations

  IMM:  ks = 0.673, p = 0.111 (floor 0.1; 3+3 tomogram-blocks)
    mean Tg: 0.0603 [0.0563, 0.0644]
    mean Vehicle: 0.0404 [0.0371, 0.0438]
```

What it does, and why (see [statistics.md](statistics.md) for the full rationale):

- **Effect size** is an area-weighted KS (or `--statistic wasserstein`) between the two
  *pooled* condition distributions — it uses every triangle, so it is sensitive to
  differences anywhere in the distribution, not just the mean.
- **The p-value treats the tomogram as the unit of replication**, via a permutation test
  that shuffles the condition label across tomograms. This is the one thing that makes the
  comparison honest: a pooled test with *n = triangles* would report `p < 10⁻³⁰⁰` for
  almost any pair of conditions, because neighbouring triangles are not independent.
- **The `floor`** is the smallest p-value the permutation could ever return given the
  number of tomograms (`1 / C(n_a+n_b, n_a)`, doubled when n_a = n_b). With 3 vs 3 tomograms it is 0.1 — so even
  perfectly separated conditions cannot beat p ≈ 0.1. The command flags when p is at the
  floor: the fix is *more tomograms*, not more triangles.
- **Per-condition means** come with `cluster_t_interval` confidence intervals over the
  per-tomogram means, so you see the direction and magnitude, not only significance.

With `--split-components`, each organelle is a unit (more power for the effect size), but
the condition is permuted at the **tomogram** level (nested), so the p-value floor is
still set by the number of tomograms — organelles within a tomogram are not independent
replicates. `--filter` composes, so comparing *cristae* curvature between conditions is:

```bash
morphometrics compare config.yml -n curvedness_VV -g condition --filter 'IMM:OMM_dist>=20'
```

Results are written to `<feature>_<group>_compare.csv` (one row per class, with the effect
size, p-value, floor, and per-condition means + CIs).

If a group has more than two buckets, pick the pair with `--conditions A B`.

---

## Multi-dataset comparisons

Real studies often span several independently processed runs — a control dataset, a
treatment dataset, a mutant segmented months later — each with its own `work_dir`. A
**`datasets:`** block federates them into one comparison. Every analysis command
(`compare`, `compare_batch`, `variogram`) accepts such a config in place of the usual
`seg_dir`/`work_dir`:

```yaml
# study.yml
classes: [OMM, IMM, ER]          # (or segmentation_values:) -- needed to parse CSV names
statistics:
  cache: true                    # parquet cache under each work_dir
datasets:
  control:
    work_dir: /data/control/morphometrics/
    groups: {condition: {Control: "*"}}        # "*" = every tomogram of this dataset
  drug:
    work_dir: /data/drug/morphometrics/
    exclude_tomograms: [bad_ts_*]
    groups: {condition: {Drug: "*"}}
  mutant:
    work_dir: /data/mutant/morphometrics/      # used if it holds surface CSVs...
    pickle: ../mutant.pkl                      # ...otherwise this legacy Experiment pickle
    groups:
      condition:
        Mutant positive: [ts_002, ts_017]      # name-split within one dataset
        Mutant negative: rest                  # every tomogram not listed above
```

```bash
morphometrics compare study.yml -n curvedness_VV -g condition --conditions Control Drug
morphometrics compare study.yml -n IMM_dist -g dataset     # whole datasets, no groups needed
```

How it behaves:

- **Backends.** Each entry loads from its per-surface CSVs when `work_dir` has them,
  otherwise from a legacy `Experiment` `pickle`. Tomograms are discovered from the CSV names
  (or from `seg_dir/*.mrc` when the entry gives a `seg_dir`), so runs whose segmentation
  folder has moved still load. Relative paths resolve against the config file's folder.
- **A dataset with no data is an error**, naming the path that was tried — a silently
  dropped dataset would quietly change a comparison. Mark an entry `optional: true` to skip
  it instead.
- **Identity.** Units are namespaced `<dataset>/<tomogram>`, so two runs may both contain a
  `ts_001` without being confused, and the nested permutation's floor counts tomograms
  correctly.
- **Metadata.** Each entry's `groups:` is merged over any top-level `groups:` and validated
  per dataset (every tomogram in exactly one bucket). Every tomogram also carries an
  implicit **`dataset`** group whose value is its dataset name.
- Entry keys: `work_dir`, `seg_dir`, `pickle`, `groups`, `include_tomograms`,
  `exclude_tomograms`, `classes`, `radius_hit`, `optional`.

In Python, `MultiDataset` has the same collection interface as `Dataset`
(`labels`, `metadata(stratum)`, `collect_feature`), so the `spatial_stats` recipe below
works unchanged on it:

```python
from surface_morphometrics.multidataset import load_for_analysis
ds = load_for_analysis(load_config("study.yml"), "study.yml")   # Dataset or MultiDataset
records, diag = ds.collect_feature("curvedness_VV")              # strata: "control/ts_001"
```

---

## `morphometrics compare_batch`: many analyses, N conditions, one config

`compare` answers one question. A study usually asks a dozen — the same conditions
compared on distances, on the curvature of several subcompartments, on verticality, on
spread. `compare_batch` reads the whole specification from the config and runs each
analysis at **both** levels:

| Level | Unit | Output |
|---|---|---|
| per-unit summary | one area-weighted summary (peak / median / mean / std) per tomogram | violin per condition; pairwise Mann-Whitney, Welch t, summary KS |
| pooled distribution | every triangle, p-value permuted over tomograms | area-weighted KS + permutation p (with its floor) + per-condition cluster CIs |

```yaml
comparison:
  group: condition                 # which groups: block's buckets are compared
  conditions:                      # report/plot order; bare names or dicts
    - {name: Control, short: Ctrl, color: "#D55E00"}
    - {name: Drug, short: Drug, color: "#0072B2"}
    - Mutant negative
    - Mutant positive
  reference: Control               # optional: test each condition vs this one only
  # pairs: [[Veh-E, Tg-E], [Veh-F, Tg-F]]   # optional: exactly these pairs (overrides reference)
  separator_after: 2               # optional: dashed line after this violin
  reps: 5000                       # permutation replicates (default 5000)
  seed: 0
  output: compare_batch            # relative to the config file
analyses:
  - {name: imm_omm_distance, class: OMM, feature: IMM_dist, statistic: peak,
     range: [5, 25], filters: ["OMM:IMM_dist<40"], ci_statistic: median,
     title: "OMM-IMM distance", xlabel: "Distance (nm)"}
  - {name: crista_curvedness, class: IMM, feature: curvedness_VV, statistic: peak,
     range: [0, 0.1], filters: ["IMM:OMM_dist>40"]}
  - {name: crista_curvedness_spread, class: IMM, feature: curvedness_VV, statistic: std,
     filters: ["IMM:OMM_dist>40"]}            # spread: summary level only (see below)
```

```bash
morphometrics compare_batch study.yml
morphometrics compare_batch study.yml --only crista_curvedness --split-components
morphometrics compare_batch study.yml --conditions Control --conditions Drug   # a 2-way
```

Notes on the analysis fields:

- `statistic` is the per-unit summary for the violin and summary-level tests; `range` is
  the histogram range (and the binning range for `peak`).
- `ci_statistic: median` makes the confidence intervals robust to a few tomograms with a
  long tail (a mean is dragged by them).
- `distribution_test: false` skips the pooled test. It defaults to false for
  `statistic: std`: the pooled distribution does not depend on the summary statistic, so a
  spread analysis would only repeat the location analysis's pooled row under another name.
- `min_triangles` / `min_area` drop tiny units, as for `violin`.
- `statistics.filters` (top level) applies to every analysis, e.g. a data-quality cut.

For a factorial design (e.g. treatment × morphology), make one group whose buckets are the
cells of the design and list the comparisons you want in `pairs:`. The worked example in
[`examples/mitochondria/`](../examples/mitochondria/) ports the paper's 974-line
`mitochondria_statistics.py` this way: a study config for every feature comparison, plus a
short script for per-tomogram quantities (surface areas, contact-site fractions) and 2D
histograms.

Outputs, in the output folder: `summary.csv` (analysis × condition), `tests.csv` (one row
per test, with a `level` column), and `<name>_violin.svg` / `<name>_hist.svg` figures
(`--no-plots` to skip them). Runs are reproducible: the permutation seed is fixed.

---

## The `Dataset` object (Python API)

For analyses the commands do not cover, `Dataset` is the programmatic entry point. It is
the successor to `morphometrics_stats.Experiment`: same `dataset[tomogram][class]` access,
but **lazy** (assembling it only globs filenames; each CSV is read on first use and
cached), **metadata-aware** (the `groups:` block), and it feeds the `spatial_stats` tools
directly.

```python
from surface_morphometrics.config_utils import load_config
from surface_morphometrics.dataset import Dataset

config = load_config("config.yml", require=("seg_dir", "work_dir", "segmentation_values"))
ds = Dataset.from_config(config)          # applies selection, reads groups, validates them

ds.tomograms()                            # -> ['TE1', 'TF1', 'UF3', ...]
ds.metadata("TF1")                        # -> {'condition': 'Tg', 'morphology': 'fragmented'}
df = ds["TF1"]["IMM"]                      # a surface's per-triangle DataFrame (lazy, cached)

# Every IMM surface of the Tg tomograms
for surf in ds.surfaces(label="IMM", condition="Tg"):
    print(surf.tomo, surf.metadata, len(surf.load()))
```

### Feeding the statistics tools

`Dataset.collect_feature` returns per-unit `(values, areas)` arrays with all the filtering
and selection applied — exactly what the `spatial_stats` functions consume. This is how to
build a comparison the `compare` command does not express (a custom effect size, a
three-way design reduced to pairs, a non-standard unit, etc.):

```python
from surface_morphometrics import surface_filters as sf
from surface_morphometrics.spatial_stats import permutation_test, cluster_t_interval

# Cristae only, per tomogram
clauses = sf.parse_filters(["IMM:OMM_dist>=20"])
records, diag = ds.collect_feature("curvedness_VV", filters=clauses)
#   records: list of (label, unit, stratum, values, areas)

# Keep the IMM class, tag each unit with its condition, run the permutation test
imm = [(unit, stratum, v, a) for (label, unit, stratum, v, a) in records if label == "IMM"]
values     = [v for _u, _s, v, _a in imm]
areas      = [a for _u, _s, _v, a in imm]
conditions = [ds.metadata(stratum)["condition"] for _u, stratum, _v, _a in imm]

result = permutation_test(values, conditions, statistic="wasserstein",
                          unit_weights=areas, reps=5000)
print(result["observed"], result["p_value"], result["min_possible_p"])
```

Because `records` carry the `stratum` (tomogram), nested designs are a one-line change —
pass `strata=[stratum for …]` to `permutation_test` when the unit is the organelle but the
condition is assigned per tomogram. See [statistics.md](statistics.md#the-tools) for the
full menu (`estimate_neff`, `intraclass_correlation`, `cluster_bootstrap`, …).

### Caching

Reading a large run's CSVs repeatedly is the slow part of iterating on an analysis. Set
`statistics.cache: true` (or `Dataset.from_config(config, cache=True)`) to cache each
parsed surface to **parquet**, keyed by the source CSV's path/mtime/size. A later run
reads the parquet instead of re-parsing the CSV — roughly **4× faster** on a
400k-triangle surface, with parquet files about half the size. The key changes whenever a
surface is re-run, so a stale cache is never used. Parquet support comes from `pyarrow`, a
regular dependency; if a cache write ever fails (e.g. an unwritable folder) the cache turns
itself off with a note and analysis continues from the CSVs.

### Compatibility with older scripts

`Dataset.to_experiment()` / `to_pickle()` materialize a classic `Experiment` for the
pickle-based scripts in `old_scripts/`. Treat this as a convenience snapshot, not a
cache: pickles of numpy-backed objects do not load across a numpy major-version change
(the reason the cache is parquet, not pickle). Prefer rebuilding the `Dataset` from the
config.

Going the other way, `PickleDataset.from_pickle("old.pkl", groups=...)` serves an existing
`Experiment` pickle through the `Dataset` interface (filtering, grouping, collection all
work), for datasets whose CSVs no longer exist. It loads pickles written from a script
(whose classes are recorded as `__main__.Experiment`) without any `__main__` injection.
It is a migration path: regenerate the per-surface CSVs while the pickle still loads.

A different backend needs only `exists(tomo, label)` and `load(tomo, label)` overridden on
a `Dataset` subclass; everything else comes from the base class.
