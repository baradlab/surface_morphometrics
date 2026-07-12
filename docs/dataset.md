# Datasets, filtering, and grouped comparisons

How to select and group surfaces for analysis, and how to run comparisons more complex
than a single violin plot. This is the layer between the raw per-surface CSVs the
pipeline writes and the statistics described in [statistics.md](statistics.md).

Three capabilities, each usable on its own and composable:

- **Filtering** — keep only the triangles you care about (e.g. cristae) before summarizing.
- **Selection** — choose which tomograms and surfaces enter an analysis.
- **Grouping** — attach metadata (condition, morphology, …) to tomograms so a treatment
  comparison flows from the config instead of hand-coded lists.

They are available both as command-line flags (`violin`, `compare`) and, for anything
bespoke, through the `Dataset` object in Python.

---

## TL;DR

```bash
# Isolate cristae (IMM far from the OMM junction) and compare their curvedness spread
morphometrics violin config.yml -n curvedness_VV --statistic std --filter 'IMM:OMM_dist>=20'

# Drop known-bad tomograms and tiny surfaces
morphometrics violin config.yml -n thickness --exclude-tomograms UF3 TE1 --min-triangles 500

# Treatment comparison between two conditions defined in the config `groups:` block
morphometrics compare config.yml -n curvedness_VV -g condition
```

```yaml
# config.yml — all optional
statistics:
  filters: [IMM:OMM_dist>=20]      # triangle filters, ANDed
  exclude_tomograms: [UF3, TE1]    # names or globs
  min_triangles: 500
  cache: true                      # parquet parse-cache for faster re-runs (needs pyarrow)
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
  number of tomograms (`2 / C(n_a+n_b, n_a)`). With 3 vs 3 tomograms it is 0.1 — so even
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
surface is re-run, so a stale cache is never used; if `pyarrow` is unavailable the cache
degrades to a warning and analysis continues from the CSVs.

### Compatibility with older scripts

`Dataset.to_experiment()` / `to_pickle()` materialize a classic `Experiment` for the
pickle-based scripts in `old_scripts/`. Treat this as a convenience snapshot, not a
cache: pickles of numpy-backed objects do not load across a numpy major-version change
(the reason the cache is parquet, not pickle). Prefer rebuilding the `Dataset` from the
config.
