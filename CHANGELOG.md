# Changelog

All notable changes to the Surface Morphometrics toolkit are documented here.
This project loosely follows [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [2.0.0b7] — beta (unreleased)

Statistics and plotting: a cross-tomogram violin command, a unified pairwise
significance tester, and spatially-aware comparison tools that treat the tomogram
(not the triangle) as the unit of replication.

### Added
- Multi-dataset comparisons: a top-level `datasets:` block federates several separately
  processed runs (each its own `work_dir`) into one comparison, via the new
  `surface_morphometrics.multidataset.MultiDataset`. Units are namespaced
  `<dataset>/<tomogram>` (so the nested permutation floor counts tomograms correctly and
  names may repeat across runs), each dataset carries its own validated `groups:`, and every
  tomogram gets an implicit `dataset` group. A dataset with no data is an error unless marked
  `optional: true`. `compare`, `compare_batch` and `variogram` accept such configs.
- `dataset.PickleDataset` serves a legacy `Experiment` pickle through the `Dataset`
  interface (a migration path for runs whose CSVs are gone), loading script-written
  pickles (`__main__.Experiment`) without `__main__` injection; `Dataset.from_work_dir`
  discovers tomograms from surface CSV names when the segmentation folder has moved.
- `morphometrics compare_batch study.yml` — runs a config `analyses:` list across the N
  conditions of a `comparison:` block, at both the per-tomogram summary level (violins,
  Mann-Whitney / Welch / summary KS) and the pooled-distribution level (permutation test
  with its floor + cluster CIs), into tidy `summary.csv` / `tests.csv`. Per-analysis
  `filters`, `statistic`, `ci_statistic`, `distribution_test`, `min_triangles`; `--only`,
  `--conditions`, `--split-components`, fixed seed. Replaces the bespoke Drp1
  `stats_grouped.py` driver, whose 5-way output it reproduces exactly.
- `compare --conditions A B` picks two buckets of a group that has more.
- `groups:` buckets accept a single glob string, and two explicit forms: `"*"` (every
  tomogram) and the reserved word `rest` (every tomogram no other bucket in the group
  claimed; at most one per group).
- `compare_batch` `comparison.pairs` tests exactly the listed condition pairs -- e.g. a
  2x2 design comparing treatment vs vehicle within each morphology.
- `examples/mitochondria/`: the paper's `old_scripts/mitochondria_statistics.py` (974
  lines) ported to a `compare_batch` study config plus a ~150-line extras script (areas,
  ER-contact fractions, 2D histograms).
- `morphometrics variogram config.yml -n F` — opt-in spatial QC of a per-triangle feature:
  per-surface and per-class pooled semivariograms (3D distance from the CSV, or
  `--geodesic` on the graph), fitted for correlation length, nugget, sill, structured
  fraction, drift and N_eff, written to `<feature>_variogram.csv` with per-class plots.
  Backed by new `spatial_stats.euclidean_semivariogram`, poolable `VariogramSums`,
  `summarize_variogram_fit`, and a `model=` choice in `fit_correlation_length`
  (`exponential` default, `gaussian`, or `auto`): an exponential model fits a smooth
  (neighborhood-averaged) field by zeroing the nugget, the Gaussian one recovers it.
- `measure_thickness --noise-estimate` — split-half measurement-noise estimate: a sample of
  triangles (`--noise-samples`, default 5000) is re-fit from two disjoint random halves of
  its neighborhood, and var(A − B)/4 estimates the noise variance of thickness and offset
  (plus a MAD-robust version and reliability = 1 − noise/total). Written per surface to
  `work_dir/thickness_noise.csv` and reported by `variogram`, because the variogram nugget
  of a neighborhood-averaged field is suppressed and is not a noise floor.
- `morphometrics compare config.yml -n F -g GROUP` — a spatially-aware treatment
  comparison of one feature between the two conditions of a config `groups:` block, per
  membrane class. Effect size is an area-weighted KS (or `--statistic wasserstein`)
  between the pooled condition distributions; the p-value comes from a permutation test
  that treats the tomogram as the unit of replication (flat, or nested at the tomogram
  level with `--split-components`), and reports the permutation floor; per-condition means
  come with `cluster_t_interval` CIs. Composes with `--filter` (e.g. compare cristae).
  Writes `<feature>_<group>_compare.csv`. New docs: `docs/dataset.md`.
- `dataset.Dataset.collect_feature` — the shared per-unit `(values, areas)` collection now
  backing both `violin` and `compare`, so filtering, selection, and the parquet cache are
  consistent across commands (the `violin` command now reads through it and gains the
  cache). Directly consumable by `spatial_stats` for bespoke comparisons.
- `morphometrics violin config.yml --feature F` — one violin per membrane class, one
  point per tomogram, from an area-weighted mean / median / histogram-peak (mode) of
  the feature per surface. Classes lacking the feature are skipped. `--test {mwu,ttest}`
  annotates significance stars and writes a `*_tests.csv` (Mann-Whitney U, Welch t, and
  a paired summary-statistic KS, tagged with a `comparison_level` column).
- `violin --filter '[CLASS:]PROPERTY OP VALUE'` (repeatable, ANDed) keeps only triangles
  matching a per-class comparison before summarizing — e.g. `--filter 'IMM:OMM_dist>=20'`
  isolates cristae; a range is two filters. Mirrored by a `statistics.filters` config
  block, recorded in the plot title and per-unit CSV. Backed by the new
  `surface_morphometrics.surface_filters` module (structured, no `eval`, serializable).
- `violin` unit selection, choosing which surfaces enter the analysis (distinct from
  filtering triangles within them): `--include-tomograms` / `--exclude-tomograms` take
  names or glob patterns (`'TF*'`, `'?E*'`), replacing hardcoded name lists and
  filename-character hacks; `--min-triangles` / `--min-area` drop surfaces or organelles
  too small to be reliable points. Mirrored by `statistics.{include_tomograms,
  exclude_tomograms,min_triangles,min_area}` config keys; dropped-unit counts are
  reported. Backed by the new `surface_morphometrics.surface_selection` module.
- `surface_morphometrics.dataset.Dataset` — a lazy, metadata-aware index of a run's
  surfaces, successor to `morphometrics_stats.Experiment`. Keeps the `dataset[tomo][class]`
  access but reads each surface's CSV on demand (assembling the index only globs
  filenames), and carries per-tomogram metadata from a config `groups:` block that maps
  tomograms to buckets (condition, morphology, ...) by name or glob — replacing the
  hardcoded `conditions = [...]` lists and `key[1] == "F"` filename hacks. Group
  assignment is validated at load (every included tomogram in exactly one bucket per
  group, or a clear error). Metadata-aware selection via
  `dataset.surfaces(label="IMM", condition="Tg")`. Documented `statistics:` / `groups:`
  blocks in `config_template.yml`. `Experiment` stays importable for the pickle scripts.
- `Dataset` parquet parse-cache (opt-in via `statistics.cache` or `Dataset(cache=...)`):
  each surface's parsed table is cached to parquet keyed by the source CSV's path/mtime/
  size, so a later run skips the CSV parse — ~4x faster warm reads on a 400k-triangle
  surface (79 ms vs 309 ms), parquet ~2x smaller than the CSV. The key changes when a
  surface is re-run, so a stale cache is never used, and stale entries are pruned. Cache
  writes degrade gracefully to a warning if pyarrow is unavailable. `Dataset.to_pickle()`
  / `to_experiment()` export a materialized `Experiment` for older pickle-based scripts
  (documented as a convenience snapshot, not the cache, given the numpy-version caveat).
- `surface_morphometrics.spatial_stats` — spatially-aware distribution comparison.
  Triangles are strongly autocorrelated, so a pooled two-sample test with n = n_triangles
  is wildly anticonservative (Type-I ~0.9 under H0 in simulation). Provides:
  - `permutation_test` — assumption-light default; an area-weighted KS D / Wasserstein
    effect size with a p-value from permuting the condition label across tomograms.
    Validated calibrated with full power.
  - Effective sample size: `kish_neff`, `neff_from_neighbors`, a geodesic semivariogram
    with a nested-fit correlation length, and `neff_from_correlation_length` = A/(2πℓ²).
    Works for rough features (e.g. curvedness); reports non-identifiability for smooth
    ones (e.g. IMM-distance), where the permutation test should be used instead.
  - `cluster_t_interval` / `cluster_bootstrap` — confidence intervals that resample /
    summarize whole tomograms (coverage ~0.86 / ~0.77 at nominal 0.90, vs ~0.20 for the
    i.i.d. triangle bootstrap).
- Golden-file characterization tests pinning `statistics()`'s CSV output (raw stats byte-for-byte; pairwise rows cell for
  cell, numbers to 1e-12, so last-digit float noise across scipy versions passes).

### Changed
- Factored the pairwise significance testing out of the violin command into a shared
  `morphometrics_stats.pairwise_tests()`, and rewrote `statistics()` to delegate to it
  (output verified byte-identical for existing callers).
- `geodesic_semivariogram` accepts a `mask` (filtered subsets; paths still run through the
  whole surface) and can return its poolable sums.

### ⚠️ Breaking changes
- An empty `groups:` bucket (`[]` / `null`) is now an error instead of an implicit
  "everything else" bucket (unreleased behavior from earlier on this branch): write `rest`
  for the remainder or `"*"` for every tomogram.
- `statistics()`'s pairwise CSV renames its KS columns `KStest_Stars` / `KS` / `P_KS`
  to `KS_summary_Stars` / `KS_summary_stat` / `P_KS_summary`, to make explicit that this
  KS compares per-tomogram summary statistics, not pooled triangle distributions. Column
  order is unchanged, so positional readers are unaffected.

### Fixed
- `pairwise_tests` reports Mann-Whitney p = 1 ("ns") when every value in both groups is
  identical. scipy >= 1.18 returns NaN there (older versions returned 1.0), which would
  have turned those cells into "n/a" depending on the installed scipy.
- `permutation_test`'s `min_possible_p` (the permutation floor) was 2 / C(n, n_a) for
  every design; that is only right for equal group sizes. For unequal sizes it is
  1 / C(n, n_a), so the floor was overstated 2x (and `AT FLOOR` flags could trigger on
  p-values above the true floor). p-values themselves were unaffected.
- `statistics()` no longer crashes on >12 datasets (color list wrapped) or via a dead
  `except e:` clause.
- `morphometrics_stats.bootstrap` documented as deprecated for confidence intervals
  (i.i.d. triangle resampling assumes independence and under-covers badly); points at the
  `spatial_stats` cluster methods.

## [2.0.0b6] — beta

Mesh generation decoupled from pycurv (in preparation for extracting it into a
standalone importable library), density samples outside the tomogram reported as
missing instead of extrapolated, the batch confirmation prompt moved to
`refine_mesh`, and thickness recovery for poorly resolved bilayers: cubic density
sampling by default, local averaging over the full configured radius, and an opt-in
forced bilayer prior with a per-triangle flag.

### Added
- First test coverage for the meshing path beyond `mrc2xyz`. `tests/test_xyz2ply.py`
  (skipped when pymeshlab is unavailable) checks that Screened Poisson reconstructs a
  known sphere shell, that isotropic remeshing hits the requested triangle area, that a
  degenerate point cloud reports failure, and that every `xyz2ply` command-line option
  reaches `xyz_to_ply`. `tests/test_mesh_wiring.py` needs neither pymeshlab nor vtk, so
  it runs in CI: it pins that every `surface_generation` setting reaches the meshing
  subprocess carrying its configured value, and that every documented setting has a
  consumer — the two guards that would have caught the inert `ultrafine` below.
- `tests/test_version.py` pins `surface_morphometrics.__version__` to `pyproject.toml`'s
  `version`. The two had drifted apart twice (fixed in 2.0.0b3, drifted again at
  2.0.0b5) because a release bumps one and forgets the other.
- `thickness_measurements: force_bilayer_prior` (default `false`; also
  `measure_thickness --force_bilayer_prior`). The prior-recovery tier that measures
  locally merged triangles only switches on when the whole-surface average profile
  resolves two leaflets. On data where the average itself is a single flat-topped peak
  (e.g. OMM at ~1 nm/px) recovery never ran, and most triangles came back NaN. With
  the option on, that average is fitted as a bilayer anyway (accepted only if it passes
  the recovery checks: R², a separation clear of the floor, comparable leaflet
  amplitudes) and used as the prior. On four OMM test surfaces at 9.98 Å/px this took
  the measured fraction from 14–44% to 82–87%. Recovered values are lower-confidence
  (0.3–0.4 nm thinner than the few strict fits on those surfaces, and dependent on
  `average_radius`; in a synthetic blur test the strict survivors read thick and the
  recovered values were closer to the unblurred truth), so they are flagged rather
  than mixed in silently.
- Per-triangle `forced_bilayer_prior` column (1/0) in the surface graph/.vtp/.csv:
  1 where the thickness relies on a forced prior. Together with `bilayer_resolution`
  it identifies low-confidence regions.
- When the surface average does not resolve a bilayer, `measure_thickness` now says so
  and suggests sampling a less-binned tomogram (or the new option).
- `density_sampling: interpolation` (`cubic` default, or `linear`), used by both
  `sample_density` and `refine_mesh`. See Changed.

### Changed
- `vtk` is now a declared dependency in `pyproject.toml` (`vtk>=9`). It was always
  required — `ply2vtp`, `export_obj`, and `refine_mesh` all use it — but was only ever
  installed as a side effect of the conda environment, so a plain `pip install` produced
  a package that failed at runtime. It is pip-resolvable, so it belongs with the pip
  dependencies rather than with the conda-only ones.
- Mesh generation no longer depends on pycurv. `ply2vtp` imported `pycurv_io` and
  `scipy.ndimage.morphology.distance_transform_edt` without ever referencing either;
  the pycurv import was the only thing coupling the segmentation → mesh path to
  pycurv/graph-tool. The three meshing stages now need only mrcfile, numpy, pandas,
  pymeshlab, and vtk.
- `ply2vtp` imports vtk inside `ply_to_vtp` rather than at module scope, as `export_obj`
  already does, so `segmentation_to_meshes` can be imported without vtk installed.
- Dropped the pymeshlab 2022.2.post3 `PercentageValue` compatibility shim, whose only
  user was the removed `ultrafine` branch. This also removes an undeclared dependency on
  `importlib_metadata`; `environment.yml` already requires pymeshlab >= 2023.12.
- The batch "are you sure?" prompt moved from `distances_orientations` to `refine_mesh`.
  Refining every tomogram is the slowest step in the pipeline (hours per tomogram), so
  `morphometrics refine_mesh config.yml` with no `--tomogram` now asks before starting
  (`-f/--force` skips it); `distances_orientations` (~15 minutes on the tutorial data) no
  longer prompts. Its `-f/--force` is kept as a hidden no-op that prints a deprecation
  note, so existing scripts keep working; it will be removed in 2.0.
- `refine_mesh` output says "final round of refinement" rather than "accepted surface"
  (nothing is accepted until `accept_refinement`), and the "not curvature-ready" warning
  now appears only on intermediate rounds, not on the final round that pycurv runs on
  immediately afterwards.
- **Density is now sampled with a cubic B-spline by default** (previously always
  trilinear). Linear interpolation blurs most at points halfway between voxels, and at
  ~1 nm/px that extra, position-dependent blur is enough to merge a bilayer's leaflets.
  On four 9.98 Å/px OMM surfaces cubic sampling raised the strictly measured fraction
  from 14–49% to 25–58%, with medians 0.05–0.27 nm lower; on a well-resolved example
  surface thickness was unchanged (median 2.92 → 2.90 nm) while triangles with clearly
  resolved leaflets rose from 54% to 67%. Spline coefficients are computed only for
  the block of the tomogram the surface spans, so memory and run time are comparable
  to linear. Samples outside the tomogram are still NaN. **Thickness values change
  slightly: don't pool measurements made with and without it**; set
  `interpolation: linear` to reproduce earlier results.
- **Local averaging now uses the full configured radius on finely meshed surfaces.**
  `measure_thickness` and `refine_mesh` gathered neighbors with a k=500 nearest
  query, so once a ball of `average_radius` held more than 500 triangles only the
  nearest 500 were kept and the effective radius silently shrank (e.g. ~8 nm instead
  of 12 nm at ~0.4 nm² per triangle; far below 25 nm for refinement). Neighborhoods
  are now drawn from a reproducibly, uniformly thinned surface so they span the whole
  radius at the same cost (`_thickness_worker.radius_neighbors`). Surfaces whose
  neighborhoods never exceeded 500 triangles are unaffected; on denser meshes
  refinement centering and thickness now average over the radius you configured.

### Removed
- `surface_generation.ultrafine`, superseded by isotropic remeshing (`isotropic_remesh` /
  `target_area`) and by density-guided mesh refinement. It had also been inert for some
  time, by two independent paths: `run_xyz_to_ply` never passed `--ultrafine` to the
  meshing subprocess, and `xyz2ply`'s own click wrapper never forwarded it either, so
  only a direct Python call could reach the branch. Meshing output is unchanged. Configs
  that still set the key keep loading without error.

### Fixed
- **Density samples outside the tomogram were extrapolated instead of reported as
  missing.** `sample_density` called `interpn(..., fill_value=None)`, which is scipy's
  *extrapolate* setting, not its NaN default — so a linescan running off the edge of the
  volume produced invented values (linear extrapolation from the edge is unbounded) that
  the thickness fitter could not distinguish from real density. This is a routine case,
  not an edge case: the default scan range is ±10 nm along the normal, tomograms are thin
  in z, and a membrane lying flat near the top or bottom of the volume scans straight out
  of it. Out-of-bounds samples are now NaN, and `sample_density` reports how many
  triangles are affected.
  - Profiles containing NaN are excluded from neighborhood averaging
    (`_thickness_worker.usable_profile_rows`) at all three sites that average over
    neighbors. Without this the fix would have made things worse: a *single*
    out-of-bounds neighbor turned the weighted average NaN and destroyed the measurement
    for a triangle that was itself well inside the volume. Verified — a good triangle
    with one bad neighbor now measures correctly where it previously returned NaN.
  - A triangle with no usable neighbors reports NaN thickness, which is what the rest of
    the thickness code already expects, rather than a confident number from data that
    does not exist.
  - In `refine_mesh`, such a triangle is marked a failed fit so the existing
    neighbor-interpolation stage fills its offset in from triangles that did fit. A final
    guard replaces any non-finite offset with zero before vertices are displaced, since a
    NaN offset would move a vertex to NaN and corrupt the mesh irrecoverably.
  - `morphometrics_stats.histogram` no longer raises on unmeasured data. An all-NaN
    series previously reached matplotlib's range autodetection and threw
    `ValueError: autodetected range of [nan, nan] is not finite` — at the very end of
    `measure_thickness`, after every per-surface computation and most output files were
    already written. Non-finite values and their paired areas are now dropped per series,
    a series with nothing measured is omitted (with a message naming it), and a plot with
    no measured data at all is skipped rather than crashing. Verified that dropping them
    leaves the area-weighted density unchanged, since they leave both the bin counts and
    the normalization. This protects every caller, not just thickness.
  - `measure_thickness` warns, naming the file, when a surface yields no thickness at
    all, pointing at a surface outside its tomogram or a mismatched tomogram pairing —
    previously this surfaced only as a silently missing series in the summary plots.
  - **This changes results near tomogram boundaries**: thickness values that were
    previously derived from extrapolated density are now NaN. Expect fewer measured
    triangles at the top and bottom of thin tomograms, and treat prior thickness
    distributions from such regions as unreliable.
- `ply2vtp.ply_to_vtp` raised `NameError` when the vtp write failed: the error branch
  called `pexceptions.PySegInputError` but `pexceptions` was never imported, so the
  function's only error path was broken. It now raises `RuntimeError`.
- Removed a stray `print("open")` from `ply_to_vtp` that leaked into `make_meshes` output.

## [2.0.0b5] — beta

Voxel-space OBJ export.

### Added
- `export_obj --scale_to_voxels A_PER_PX` — export the mesh in voxel (pixel) space for
  overlaying on the tomogram, given the tomogram's voxel size in Å/px. The surface's
  native units are read from `surface_generation.angstroms`, so at 5 Å/px an nm-scale
  surface is doubled and an Å-scale one divided by 5. Mutually exclusive with
  `--scale_to_angstroms`.

## [2.0.0b4] — beta

Protein-patch workflow overhaul and a mesh-refinement pycurv speed fix.

### Added
- Protein-to-membrane-edge distances in the patch workflow. When per-triangle
  `thickness` is present, `generate_patches` writes a `<prefix>_headgroup_distance`
  (distance to the true membrane edge = midplane distance − thickness/2, measured
  at the patch's central triangle) per-particle into the annotated STAR, and
  `patch_statistics` reports it per-patch. The effective thickness falls back from
  the central triangle to the patch's distance-weighted mean, then to NaN.
- `patch_statistics` now reports `min_protein_distance` (closest membrane-midplane
  approach) per real patch, and keeps the measurement prefix in `region_type`
  (e.g. `ribo_patch`, `ribo_random_patch`) so multiple patch types can be compared.
- `patch_analysis.measurement_prefix` (default `ribo`): patch measurement columns
  are prefixed, so re-running with a different prefix accumulates a second patch
  type (e.g. `atp`) on the same surface.

### Changed
- **`generate_patches` edits the membrane `.gt`/`.vtp`/`.csv` in place** instead of
  writing a separate `*_patches` file set; `patch_statistics` and `extract_patches`
  default to the in-place `*.AVV_rh*` files. `--output-dir` now only affects the
  annotated STAR.
- Refinement runs pycurv (the final curvature pass) in a **fresh subprocess**. Run
  in-process at the end of a long refinement, pycurv was ~3–4× slower per NVV chunk
  (measured) due to the loaded process's memory/allocator state, not the mesh; a
  clean subprocess restores standalone speed. Refinement also frees large arrays
  and runs `gc.collect()` between iterations/surfaces.

### Fixed
- Patch headgroup distance could exceed the protein distance: it was computed per
  triangle then min-reduced, so the two minima came from different triangles (and
  NaN-thickness triangles dropped out). It is now computed at the central triangle,
  guaranteeing `headgroup_distance ≤ min_protein_distance`.

### Notes
- `generate_patches` supports RELION 3/4 STAR files (absolute tomogram-frame
  coordinates). RELION 5 centered coordinates (`rlnCenteredCoordinate*Angst`) are
  not yet supported; convert to absolute coordinates first.

## [2.0.0b3] — beta

A performance and robustness release for the mesh-refinement step, plus the move of
the repository to the Barad Lab GitHub organization.

### Changed
- **Mesh refinement now runs pycurv only once, on the final surface.** Every
  intermediate iteration (cross-correlation and dual-Gaussian) builds a fast
  lightweight graph from VTK normals and warns that the surface is not
  curvature-ready; a single full pycurv normal-vector-voting pass runs on the
  accepted surface after the loop — on every exit path, including an early
  convergence stop — so the result is always a clean, curvature-ready surface.
  Previously every dual-Gaussian iteration re-ran the slow voting (~28 min on large
  surfaces), which dominated refinement time.
- Mesh-refinement parallel fitting (dual-Gaussian offset centering and local
  thickness) now respects the configured `cores` instead of spawning one worker per
  logical CPU. Benchmarking showed throughput peaks near the configured core count
  and regresses beyond it, so this is both a correctness and a speed improvement.
- `accept_refinement` messages now refer to "intermediate" iterations (any non-final
  iteration may lack a curvature graph), not specifically cross-correlation ones.
- Repository moved to `https://github.com/baradlab/surface_morphometrics`.

### Fixed
- `__version__` in `surface_morphometrics/__init__.py` had drifted a release behind
  `pyproject.toml`; both version strings are now kept in sync.

## [2.0.0b2] — beta

A quality and robustness release on top of 2.0.0b1, focused on the density-profile
bilayer fitting (mesh refinement and thickness measurement), a forgiving centralized
config system, and packaging fixes.

### Added
- Per-triangle `bilayer_resolution` score (0–1) written to the thickness CSV — a
  reliability flag distinguishing strictly-resolved leaflets from prior-recovered
  (slightly thin) measurements.
- `morphometrics new_config --simple` — a minimal, fully-runnable starter config
  (the full annotated template is still available via the default `--verbose`).
- Centralized config defaults: omitted parameter sections fall back to documented
  values (a single source of truth, kept in sync with the template by a test), so
  partial/stripped configs run.
- Per-command required-key validation with one clean error message instead of a
  `KeyError` traceback; directory paths no longer need trailing slashes.
- Distance/orientation measurements default to all-vs-all when `intra`/`inter` are
  omitted (announced; set them explicitly empty to opt out).
- Plugins can contribute `morphometrics` subcommands via a `morphometrics.commands`
  entry point.
- GitHub Actions test workflow and unit tests for the fitting and config helpers.

### Changed
- **Dual-Gaussian bilayer fitting overhaul** (mesh refinement and thickness): leaflet
  detection by height above the shared solvent base (fixes near-universal fallback to
  a single Gaussian on clean bilayers); a shared-width, centered fit over a symmetric
  window (removes a ~0.1 nm centering bias and the collapse failure mode); and a
  two-tier gate that recovers locally-merged bilayers from the global-average prior
  while still rejecting genuine single/skewed peaks. The whole-surface average fit and
  its `_fit.svg` plot use the same model.
- Thickness measurement now applies a real quality gate (resolution + R² + physical
  range) and reports `NaN` for unmeasurable triangles instead of a spurious value.
- `accept_refinement` falls back to a surface's last available iteration (with a
  warning) when it converged before the requested step.
- `mesh_refinement` defaults now match the documented template (the code's omitted-key
  defaults had drifted): `average_radius` 25, `average_radius_min` 12,
  `max_total_offset` 8, `laplacian_iterations` 5, `laplacian_lambda` 0.5,
  `convergence_threshold` 0.05, `xcorr_iterations` [1, 2, 3].
- `export_obj`: `--angstroms` renamed to `--scale_to_angstroms`, with config-aware
  unit conversion.

### Fixed
- pycurv fork deadlock on Linux (cap `OMP_NUM_THREADS=1` before importing pycurv).
- `export_obj` reads legacy / empty VTP surfaces instead of crashing.
- NumPy 2.x compatibility (replace the removed `np.trapz`).
- Deprecated `matplotlib.cm.get_cmap` usage.

## [2.0.0b1] — beta

This release turns the collection of scripts into an installable Python package
driven by a single `morphometrics` command, and adds protein-patch /
connected-component analysis and surface export. It is a **breaking change** to
how the toolkit is invoked.

### ⚠️ Breaking changes
- The pipeline is now run through one grouped command, e.g.
  `morphometrics make_meshes config.yml` instead of `python segmentation_to_meshes.py config.yml`.
  The old `python <script>.py …` calls still work for now via **deprecation shims**
  that warn and forward; they will be removed in a future release. See the
  [migration table](README.md#upgrading-from-older-versions).
- Config keys renamed: `data_dir` → `seg_dir`, and `max_triangles` →
  `simplify_max_triangles` (only used when `simplify: true`). `make_meshes` warns
  if it detects the old names.
- Thickness measurement now writes the per-triangle `thickness`/`offset`/
  `average_width` properties **back into the surface `.gt`/`.vtp`/`.csv` in place**
  rather than creating separate `*_refined.gt`/`.vtp` files.

### Added
- Installable `surface_morphometrics` package with a `morphometrics` console entry
  point (lazy subcommands, grouped/ordered `--help`, next-step hints, shell
  completion).
- `morphometrics new_config` — write a starter `config.yml` into the working dir.
- `morphometrics fetch_example` — download a small cropped tomogram+segmentation
  test set from Zenodo (or point at the full data on EMPIAR-12534).
- Density-guided mesh refinement workflow: `refine_mesh` + `accept_refinement`
  (commit a chosen iteration in place, with `.orig.bak` backups).
- Protein-patch and connected-component analysis: `generate_patches`
  (STAR-driven, with random controls and combined-STAR basename matching),
  `label_components`, `extract_patches`, and `patch_statistics`.
- `morphometrics export_obj` — export a quantified surface to a colormapped
  OBJ + MTL for Blender / MeshLab (with `--nan-color` for unmeasured faces).
- `pyproject.toml` packaging; `pip install -e .` is wired into the conda
  environment files; basic `tests/` for the pure-logic helpers.

### Changed
- Research/paper/one-off scripts moved to `old_scripts/`.
- pycurv no longer prints the "run in parallel / this may take an hour" warnings
  (it is now about as fast as mesh generation); the slow-step note moved to
  `refine_mesh`, which is the slowest, optional step.
- README reorganized (Installation / Quick start / Pipeline / Analysis &
  visualization / Reference / Upgrading) with a table of contents.

[2.0.0b7]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b6]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b5]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b4]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b3]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b2]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b1]: https://github.com/baradlab/surface_morphometrics/releases
