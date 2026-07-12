# Statistics in Surface Morphometrics

How the distribution-comparison tools are organized, and — more importantly — *why*
they are built the way they are. Most of the design choices here are consequences of a
single fact about the data, and several were settled by simulation rather than
intuition (the numbers below are from those simulations; see
[Reproducing the validation](#reproducing-the-validation)).

---

## TL;DR — which tool for which question

| You want to… | Use | Unit of replication |
|---|---|---|
| Compare a **summary** of a feature across classes, one point per tomogram | `morphometrics violin -n <feature> --test mwu` | tomogram |
| Same, but one point per **organelle** | `morphometrics violin … --split-components` | organelle (flat; check the reported ICC) |
| Compare two **full distributions** (e.g. treatment vs control) | `spatial_stats.permutation_test(...)` | tomogram (or organelle, nested via `strata=`) |
| A **confidence interval** on a per-condition mean | `spatial_stats.cluster_t_interval(...)` | tomogram |
| A **corrected pooled KS** with an effective-N | `spatial_stats.estimate_neff(...)` then a KS with that N | tomogram-equivalent patches |

Two rules cover almost everything:

1. **The independent unit is the tomogram, not the triangle.** Triangles buy precision
   *within* a surface; they are not extra biological replicates.
2. **When in doubt, use `permutation_test`.** It is the assumption-light default and was
   the most robust option in every simulation.

---

## The one fact that drives everything: spatial autocorrelation

A membrane surface has ~10⁵ triangles, but neighbouring triangles have nearly identical
curvature / distance / thickness. They are **not independent observations**. If you
ignore this and run a pooled two-sample test with *n = number of triangles*, the test
is wildly overconfident:

> In simulation, a pooled KS test with n = n_triangles rejects the null **~89–97 % of
> the time when the two "conditions" are drawn from the *same* distribution** (it should
> reject ~5 %). This is the uncorrected `pbase` p-value.

So a naive p-value is not "a bit optimistic" — it is meaningless. Every tool here exists
to get the unit of replication right.

### Two levels at which you can compare

There are two legitimate but different questions, and the library keeps them separate on
purpose:

- **Summary-statistic level.** Reduce each tomogram to one number (an area-weighted
  mean, median, or histogram-peak/mode of the feature), then compare those ~6 numbers
  per class. `n` = number of tomograms, so the significance floor is honest (6-vs-6
  cannot go below p≈0.002). This is what `morphometrics violin --test` and the legacy
  `statistics()` do.
- **Full-distribution level.** Compare the *pooled* per-triangle distributions, but
  correct for autocorrelation so the p-value still reflects the number of tomograms.
  This is what `spatial_stats` does.

The distinction is important enough that we renamed the KS columns in `statistics()`'s
CSV to `KS_summary_*` — to make explicit that *that* KS compares per-tomogram summary
statistics, **not** the pooled triangle distributions (which need the correction below).

---

## Library map

### `morphometrics_stats.py` — summary-statistic level and plotting
- `pairwise_tests(datasets, labels, ...)` — one shared pairwise tester (Mann-Whitney U,
  Welch t, and a summary-statistic KS) used by both the violin command and `statistics()`.
- `significance_stars(p, ns=, na=)` — one place for the `****/***/**/*` thresholds.
- `statistics(...)` — the legacy paper-figure function (violins + a pairwise CSV). It now
  delegates the tests to `pairwise_tests`; its CSV output is otherwise unchanged
  (pinned byte-for-byte by golden tests).
- `ks_statistics(...)` — the legacy distribution-level KS with the effective-n correction
  (`pmito`, `prad`). Kept as-is; see the N_eff discussion below.
- `bootstrap(...)` — **deprecated for confidence intervals** (see CIs below).

### `feature_violin.py` — the `violin` command
Discovers every surface for each class, reduces each to one summary value, and plots one
violin per class with the points overlaid. `--test` annotates significance stars.
`--split-components` switches the unit from surface to organelle.

### `spatial_stats.py` — spatially-aware distribution comparison
The statistically load-bearing module.
- `permutation_test(...)` — the recommended full-distribution comparison.
- `intraclass_correlation(...)` — diagnostic for whether organelles can be treated as
  independent.
- `weighted_ks_statistic`, `weighted_wasserstein` — area-weighted effect sizes.
- `kish_neff`, `neff_from_neighbors`, `geodesic_semivariogram`,
  `fit_correlation_length`, `neff_from_correlation_length`, `estimate_neff` — effective
  sample-size estimation.
- `cluster_t_interval`, `cluster_bootstrap` — confidence intervals.

---

## Design decisions and why

### 1. The permutation test is the default full-distribution comparison

`permutation_test` keeps a real distributional **effect size** (area-weighted KS *D* or
Wasserstein distance between the pooled distributions) but obtains the **p-value by
permuting the condition label across tomograms** and recomputing the pooled statistic.

Why this is the default:
- It is **exact under exchangeability of tomograms** — no correlation model, no
  stationarity assumption, no effective-N to estimate.
- It was **calibrated in every simulation** we ran: Type-I 0.06–0.085 (nominal 0.05)
  across correlation length, covariance shape (smooth Gaussian and rough exponential),
  and unequal group sizes, with **full power** (≈1.0 to detect a mean shift of 0.3σ).
- It is **distribution-free**, so it is unaffected by the one dimension we have *not*
  fully tested (non-Gaussian marginals).

### 2. The effective-N correction is *sound*, but fragile — and not for every feature

We initially suspected the lab's `kstwo.sf(D, N_eff)` correction (`prad`/`pmito`) was a
broken heuristic. **Simulation refuted that**: given a *correct* N_eff it is calibrated
(Type-I 0.03–0.07 across the same configs above). What is broken is only the
*uncorrected* `pbase`. So the correction is not optional decoration — it is essential,
and it works.

But it has two sharp edges, both established empirically:

- **It is asymmetrically sensitive to the correlation length ℓ.** Because
  N_eff = A / (2πℓ²), underestimating ℓ overestimates N_eff and makes the test
  anticonservative: ℓ 20 % too low roughly **doubles** the Type-I error; 30 % too low
  **triples** it. Overestimating ℓ is safe (merely conservative). **Practical rule: use
  a conservative (upper) ℓ, or use the permutation test.**
- **ℓ is not always identifiable.** `estimate_neff` fits a geodesic semivariogram with a
  nested model, γ(h) = c₀ + c₁(1 − e^{−h/ℓ}) + c₂·h² (the c₂·h² term absorbs the smooth
  organelle-scale drift so it is not mistaken for correlation). On real data it recovers
  ℓ cleanly for **rough** features (curvedness: ℓ ≈ 8.6 nm) but **correctly reports
  `ok=False`** for **smooth** features (membrane-to-membrane distance: ℓ runs to the fit
  boundary at every window — the correlation spans the whole organelle, so there is no
  scale separation to fit). It fails loudly rather than returning a wrong ℓ.

**Consequence:** the N_eff path is available for rough, short-correlation features; for
smooth distance fields, `estimate_neff` refuses and the **permutation test is the answer**.
Never use the cheap `neff_from_neighbors` for a smooth feature — it uses the `radius_hit`
disk, which is far smaller than the true correlation patch, so it overstates N_eff (~3×)
and re-creates the anticonservatism.

### 3. Effect size and p-value are reported separately

Following the permutation-test philosophy, the **effect size** (KS *D*, Wasserstein) is
a descriptive quantity that uses all the triangle data, while the **p-value** carries the
(tomogram-level) uncertainty. Wasserstein distance is offered alongside KS *D* because it
is bounded, interpretable in the feature's units, and less sensitive to a single
tail than KS's sup-norm.

### 4. Confidence intervals: cluster the tomograms

The same "tomogram is the unit" principle. Coverage simulation (nominal 90 %):

| Method | Coverage | Notes |
|---|---|---|
| `cluster_t_interval` (per-tomogram summary → t-interval) | **0.86** | recommended for a mean |
| `cluster_bootstrap` (resample whole tomograms) | 0.77 | for non-mean statistics; under-covers a bit at small n |
| legacy `bootstrap` (resample individual triangles i.i.d.) | **0.20** | **broken** — assumes independence |

The legacy triangle-level bootstrap produces intervals ~5× too narrow; it is now
documented as deprecated with a pointer to the cluster methods. For a mean, the t-interval
on per-tomogram means is best-calibrated at the small tomogram counts (~6) typical here.

### 5. Weighting: everything is area-weighted, tomograms count equally

Within a surface, triangles are weighted by area (a large triangle represents more
membrane). Across tomograms, each tomogram counts **equally** (one biological replicate),
so a single large or triangle-dense tomogram cannot dominate. `cluster_t_interval` makes
this explicit; the summary statistics per surface are area-weighted.

---

## Connected components (organelles) as units

A connected component is usually one organelle within a tomogram. Splitting by component
gives more units and more power — but it is the *same* unit-of-replication problem, one
level up: organelles in one tomogram share the prep, the cell, the imaging session.

Whether they can be treated as independent samples depends on **what the treatment does**
and on the **intraclass correlation (ICC)**:

- If the biological signal genuinely varies organelle-to-organelle (e.g. a drug shifts the
  *distribution of mitochondrial functional states*, and each mitochondrion is a draw from
  that distribution), the organelle is the right unit — *provided* there is no strong
  cell-level coherence.
- If organelles within a tomogram are alike for a reason unrelated to the treatment
  (a globally stressed cell), that is clustering, and treating them as independent inflates
  false positives.

The library makes this a **data-driven choice** rather than an assumption:

- `intraclass_correlation(per_organelle_summaries, tomogram_ids)` reports the fraction of
  variance that is between-tomogram. **Low ICC → organelles are ~independent → the flat
  analysis is justified. High ICC → use the nested analysis.**
- `permutation_test(..., strata=tomogram_ids)` runs the **nested** test: the effect size
  uses every organelle, but the condition label is permuted at the **tomogram** level, so
  the p-value's floor is set by the number of tomograms. Use this when the condition is
  assigned per tomogram (the usual treatment design). `strata=None` gives the **flat**
  test (organelles fully exchangeable), valid when ICC is low.

`morphometrics violin --split-components` plots one point per organelle and prints the
ICC per class. Note that the violin compares *membrane classes*, which are **crossed**
with tomogram (every tomogram has an OMM and an ER surface), so there the split adds
per-organelle resolution plus a flat test plus the ICC check; the nested (tomogram-
stratified) test applies to *treatment* comparisons, where condition is nested in the
tomogram, via `permutation_test(strata=...)`.

---

## Visualizations

The plotting helpers live in `morphometrics_stats.py` (alongside the summary-level stats,
for backward compatibility — see [a note on file organization](#a-note-on-file-organization)).
Three are exposed as `morphometrics` commands; the rest are library functions used by the
paper-figure scripts in `old_scripts/`. **All of them are area-weighted** and write both an
`.svg` and a `.png`.

| Plot | Command / function | What it shows |
|---|---|---|
| Histogram | `morphometrics histogram file.csv -n <feature>` → `histogram()` | Area-weighted, density-normalized histogram of one feature; overlays several datasets, optional log-x and dashed lines at the weighted medians |
| 2D histogram | `morphometrics hist2d file.csv -n1 <a> -n2 <b>` → `twod_histogram()` | Area-weighted 2D histogram of two features, log-scaled color — the joint distribution / correlation |
| Violin | `morphometrics violin config.yml -n <feature>` → `violin()` | One violin per class across the dataset, one point per tomogram (or organelle); see above |
| Scatter + regression | `scatter_regression()` (library) | Per-dataset scatter of two features with a fitted regression line |
| Bar chart | `barchart()` / `double_barchart()` (library) | Bars with error bars, for one or two paired conditions |

Design notes:
- The **histogram** and **2D histogram** operate on a single CSV (one surface), for
  eyeballing a configuration or answering a reviewer; the **violin** operates across the
  whole dataset and is the one tied to the inferential machinery above (`--test`).
- Histograms use matplotlib's `density=True` normalization so differently-sized surfaces
  are comparable; the vertical lines are the *area-weighted* medians, not the unweighted
  ones.

**Not currently implemented:** kernel-density-estimate (KDE) plots and cumulative /
empirical-CDF plots. The histogram is a binned density, not a smooth KDE. Note that a
weighted ECDF is already computed internally by `spatial_stats.weighted_ks_statistic`, so
an area-weighted CDF/ECDF plot would be cheap to add on top of it if wanted — it just
isn't there yet.

### A note on file organization

Statistics and visualization currently share `morphometrics_stats.py`, and the new,
purely-statistical, simulation-validated core lives separately in `spatial_stats.py`. A
cleaner split would be plots-in-one-file / stats-in-another, but the useful seam here is
**new/validated (`spatial_stats`) vs legacy (`morphometrics_stats`)**, not plots-vs-stats:
external and `old_scripts/` analyses import the plotting functions *by name* from
`morphometrics_stats` (e.g. `from morphometrics_stats import histogram, barchart, …`), and
`statistics()` is intrinsically both (it computes tests *and* draws a violin). So a
plots/stats split would either break those imports or require re-export shims for little
practical gain, and is deliberately deferred.

---

## Caveats and open items

- **Non-Gaussian marginals** are the one untested dimension of the calibration work — the
  simulated fields are Gaussian, while real curvature/distance fields can be skewed. The
  permutation test is distribution-free and so unaffected; the N_eff-KS deserves a
  non-Gaussian check before it is trusted for a smooth, skewed feature.
- **ℓ for smooth features is genuinely not identifiable** from a single surface (no scale
  separation). This is a property of the data, not a bug; use the permutation test.
- **`morphometrics compare`** wraps `permutation_test` (with the flat/nested choice and a
  KS or Wasserstein effect size) plus `cluster_t_interval`, driven by the config `groups:`
  metadata — see [dataset.md](dataset.md#morphometrics-compare-a-spatially-aware-treatment-comparison).
  For designs it does not express, call `spatial_stats` directly on
  `Dataset.collect_feature` output.
- **`ks_statistics` still uses one `rad` for all features.** Per-feature ℓ (from
  `estimate_neff` where identifiable) would fix the anticonservatism for smooth distance
  fields; until then, do not quote `prad` for a smooth feature.

---

## Reproducing the validation

The numbers above come from Monte-Carlo simulations that generate spatially-correlated
Gaussian fields (via FFT, with a known correlation length) and measure how often each
method rejects a *true* null. The scripts live in the project scratchpad and are
summarized in `PLAN_stats_refactor.md` (Phase 4 calibration, robustness sweep, N_eff
sensitivity; Phase 3 estimator validation; Phase 5 bootstrap coverage). The library's own
correctness is covered by `tests/test_spatial_stats.py`,
`tests/test_feature_violin.py`, and the golden-file characterization tests in
`tests/test_statistics_golden.py` (which pin `statistics()`'s CSV output byte-for-byte, so
the refactor could be shown not to change any number for existing callers).
