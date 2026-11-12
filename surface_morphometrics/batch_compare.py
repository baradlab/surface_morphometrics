#! /usr/bin/env python
"""`morphometrics compare_batch` -- an N-way comparison of many analyses, from config.

`compare` answers one question (one feature, two conditions). A study usually asks a
dozen: the same set of conditions compared on distances, curvature of several
subcompartments, verticality, spread... `compare_batch` reads the whole specification
from the config and runs every analysis at both levels of comparison:

  * **per-unit summary level** -- one area-weighted summary per tomogram (or organelle):
    a violin per condition, and pairwise Mann-Whitney / Welch / summary-KS tests
    (`morphometrics_stats.pairwise_tests`);
  * **pooled-distribution level** -- an area-weighted KS effect size between the pooled
    condition distributions, with a permutation p-value that treats the tomogram as the
    unit (`spatial_stats.permutation_test`, flat or nested with --split-components), and
    per-condition `cluster_t_interval` confidence intervals. Every p-value is reported
    next to its permutation floor.

Works on a single run (`seg_dir`/`work_dir` + `groups:`) or on several runs federated by a
`datasets:` block (see `multidataset`). Config schema::

    comparison:
      group: condition              # the groups: block whose buckets are compared
      conditions:                   # report/plot order; strings or dicts
        - {name: Control, short: Ctrl, color: "#D55E00"}
        - Mutant
      reference: Control            # optional: test each condition vs this one only
      pairs: [[A, B], [C, D]]       # optional: exactly these pairs (overrides reference),
                                    # e.g. treatment vs vehicle within each morphology
      separator_after: 1            # optional: dashed line after this violin
      correction: holm              # optional: holm (default) | fdr_bh | bonferroni | none
      correction_family: analysis   # optional: analysis (default) | batch
    analyses:
      - name: imm_omm_distance      # output file stem
        class: OMM
        feature: IMM_dist
        statistic: peak             # per-unit summary: peak | median | mean | std
        range: [5, 25]              # histogram range (and peak binning)
        filters: ["OMM:IMM_dist<40"]
        ci_statistic: median        # optional: summary for the CIs (mean | median)
        distribution_test: true     # optional: false skips the pooled test
                                    # (default false for statistic: std)
        title: OMM-IMM distance     # optional plot title / axis label
        xlabel: Distance (nm)

Analyses can instead compare MEMBRANE CLASSES within one condition (e.g. OMM vs IMM vs
ER in the vehicle cells). Every tomogram carries all the classes, so this is a paired
design and uses paired tests: a Wilcoxon signed-rank / paired t-test on the per-tomogram
summaries, and a permutation test that swaps the class labels within each tomogram
(`spatial_stats.paired_permutation_test`)::

      - name: omm_vs_imm_curvedness
        compare: classes            # default `conditions`
        classes: [OMM, IMM, ER]     # at least two; classes lacking the feature are dropped
        condition: Vehicle          # optional: one bucket of comparison.group (omit = all)
        feature: curvedness_VV
        statistic: peak
        range: [0, 0.1]
        reference: OMM              # optional: test each class vs this one only
        pairs: [[OMM, IMM]]         # optional: exactly these class pairs
        center: median              # optional: per-tomogram shift before pooling
                                    # (median | mean | null; see paired_permutation_test)

Writes `summary.csv` (one row per analysis x condition, or x class) and `tests.csv` (one row per
test, with a `level` column) into the output directory, plus per-analysis violin and
histogram SVG/PNGs.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from collections import Counter
from itertools import combinations

import click
import numpy as np
import pandas as pd

from .config_utils import load_config

# Okabe-Ito, colorblind-safe; used for conditions without an explicit color.
_PALETTE = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
            "#F0E442", "#000000"]


def parse_conditions(comparison, buckets):
    """Normalize `comparison.conditions` to a list of {name, short, color} dicts.

    Entries may be bare names or dicts; when the list is absent every declared bucket of
    the group is used, in config order.
    """
    raw = comparison.get("conditions") or list(buckets)
    out = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            entry = {"name": entry}
        if "name" not in entry:
            raise ValueError(f"comparison.conditions entry {entry!r} has no `name`")
        name = str(entry["name"])
        out.append({"name": name, "short": str(entry.get("short", name)),
                    "color": entry.get("color", _PALETTE[i % len(_PALETTE)])})
    return out


def _pairs(comparison, present, reference):
    """The condition pairs to test: configured `pairs`, else vs `reference`, else all."""
    if comparison.get("pairs"):
        return [(a, b) for a, b in comparison["pairs"] if a in present and b in present]
    if reference in present:
        return [(reference, n) for n in present if n != reference]
    return list(combinations(present, 2))


def validate_analyses(analyses):
    """Check required fields and duplicate names up front, before any data is loaded."""
    from .feature_violin import STATISTICS
    from .surface_filters import parse_filters

    if not analyses:
        raise ValueError("config has no `analyses:` list")
    seen = set()
    for i, a in enumerate(analyses):
        kind = a.get("compare", "conditions")
        if kind not in ("conditions", "classes"):
            raise ValueError(f"analyses[{i}]: compare must be 'conditions' or 'classes'")
        if kind == "classes" and a.get("center", "median") not in (None, "median", "mean"):
            raise ValueError(f"analyses[{i}]: center must be median, mean or null")
        required = ("name", "class", "feature") if kind == "conditions" else \
            ("name", "classes", "feature")
        missing = [k for k in required if not a.get(k)]
        if missing:
            raise ValueError(f"analyses[{i}] is missing {', '.join(missing)}")
        if kind == "classes":
            classes = a["classes"]
            if not isinstance(classes, (list, tuple)) or len(set(classes)) < 2:
                raise ValueError(f"analysis {a['name']!r}: `classes` must list at least "
                                 "two different membrane classes")
            for pair in a.get("pairs") or []:
                if len(pair) != 2 or any(c not in classes for c in pair):
                    raise ValueError(f"analysis {a['name']!r}: pairs entry {pair} must "
                                     f"name two of its classes {list(classes)}")
            if a.get("reference") is not None and a["reference"] not in classes:
                raise ValueError(f"analysis {a['name']!r}: reference {a['reference']!r} "
                                 f"is not one of its classes {list(classes)}")
        if a["name"] in seen:
            raise ValueError(f"duplicate analysis name {a['name']!r}")
        seen.add(a["name"])
        if a.get("statistic", "peak") not in STATISTICS:
            raise ValueError(f"analysis {a['name']!r}: statistic must be one of "
                             f"{STATISTICS}")
        if a.get("ci_statistic", "mean") not in ("mean", "median"):
            raise ValueError(f"analysis {a['name']!r}: ci_statistic must be mean or median")
        parse_filters(a.get("filters"))      # raises on a malformed filter


def collect(ds, feature, label, group, filters=None, split_components=False,
            min_triangles=0, min_area=0.0):
    """Per-unit (values, areas, conditions, units, strata) for one class of one feature."""
    records, _diag = ds.collect_feature(feature, filters=filters,
                                        split_components=split_components,
                                        min_triangles=min_triangles, min_area=min_area)
    values, areas, conds, units, strata = [], [], [], [], []
    for rec_label, unit, stratum, vals, ars in records:
        if rec_label != label:
            continue
        values.append(vals)
        areas.append(ars)
        conds.append(ds.metadata(stratum).get(group))
        units.append(unit)
        strata.append(stratum)
    return values, areas, conds, units, strata


_PCOLS = ("mwu_p", "ttest_p", "ks_p", "wilcoxon_p", "ttest_rel_p", "p_permutation")


def _run_class_analysis(ds, analysis, group, output_dir, base_filters=None,
                        split_components=False, reps=5000, seed=0, plots=True,
                        correction="holm", log=print):
    """One `compare: classes` analysis: paired comparison of membrane classes.

    Returns (summary_rows, test_rows). `correction=None` leaves the p-values unadjusted
    (the caller corrects across the batch instead).
    """
    from matplotlib.colors import to_rgba

    from . import surface_filters as sf
    from .feature_violin import summary_statistic
    from .morphometrics_stats import apply_correction, histogram, paired_tests, violin
    from .spatial_stats import cluster_t_interval, paired_permutation_test

    name, feature = analysis["name"], analysis["feature"]
    classes = list(dict.fromkeys(analysis["classes"]))
    condition = analysis.get("condition")
    clauses = sf.parse_filters(list(base_filters or []) + list(analysis.get("filters") or []))
    stat = analysis.get("statistic", "peak")
    rng = analysis.get("range")
    title = analysis.get("title", name)
    ci_stat = analysis.get("ci_statistic", "mean")
    where = f"condition {condition}" if condition else "all tomograms"
    log(f"\n=== {title} [{' vs '.join(classes)}: {feature}, {where}"
        f"{', ' + sf.describe(clauses) if clauses else ''}; paired by tomogram] ===")
    if split_components:
        log("  NOTE: classes are paired by tomogram, so --split-components is ignored here.")

    # Pool each class per tomogram (organelles of different classes are not paired).
    records, _diag = ds.collect_feature(feature, filters=clauses,
                                        min_triangles=analysis.get("min_triangles", 0),
                                        min_area=analysis.get("min_area", 0.0))
    data = {c: {} for c in classes}          # class -> {tomogram: (values, areas)}
    for label, _unit, stratum, vals, ars in records:
        if label not in data:
            continue
        if condition is not None and ds.metadata(stratum).get(group) != condition:
            continue
        data[label][stratum] = (vals, ars)
    missing = [c for c in classes if not data[c]]
    if missing:
        log(f"  no surfaces with '{feature}' for {', '.join(missing)} -- dropped.")
    present = [c for c in classes if data[c]]
    if len(present) < 2:
        log("  fewer than two classes have this feature -- skipping.")
        return [], []
    log("  tomograms: " + ", ".join(f"{c}={len(data[c])}" for c in present))

    summaries = {c: {t: summary_statistic(v, a, stat, bins=100, bin_range=rng)
                     for t, (v, a) in data[c].items()} for c in present}
    palette = analysis.get("colors") or {}
    colors = [list(to_rgba(palette.get(c, _PALETTE[i % len(_PALETTE)]), alpha=0.9))
              for i, c in enumerate(present)]
    if plots:
        violin([list(summaries[c].values()) for c in present], present, title=title,
               ylabel=analysis.get("xlabel", feature),
               filename=os.path.join(output_dir, f"{name}_violin.svg"),
               custom_colors=colors)

    summary_rows = []
    for c in present:
        vals = np.asarray(list(summaries[c].values()), dtype=float)
        summary_rows.append({"analysis": name, "class": c, "feature": feature,
                             "filter": sf.describe(clauses), "statistic": stat,
                             "condition": condition or "all", "n_tomograms": len(vals),
                             "mean": vals.mean(),
                             "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan})

    if analysis.get("pairs"):
        pairs = [tuple(p) for p in analysis["pairs"] if p[0] in present and p[1] in present]
    elif analysis.get("reference") in present:
        ref = analysis["reference"]
        pairs = [(ref, c) for c in present if c != ref]
    else:
        pairs = list(combinations(present, 2))

    test_rows = []
    summary_tests = paired_tests(summaries, pairs)
    if correction:
        apply_correction(summary_tests, correction, pcols=("wilcoxon_p", "ttest_rel_p"))
    for row in summary_tests:
        row.update({"analysis": name, "level": "per_tomogram_paired_summary",
                    "feature": feature, "condition": condition or "all"})
        test_rows.append(row)
        adj = (f" ({correction}-adj {row['wilcoxon_p_adj']:.3g})"
               if "wilcoxon_p_adj" in row else "")
        log(f"  {row['class_a']} vs {row['class_b']} (per-tomogram {stat}, "
            f"{row['n_pairs']} pairs): diff {row['mean_diff']:.4g}, "
            f"Wilcoxon p={row['wilcoxon_p']:.3g}{adj}")

    if not analysis.get("distribution_test", stat != "std"):
        log("  (distribution-level permutation skipped: this analysis compares a "
            "per-tomogram summary, not the pooled distribution)")
        return summary_rows, test_rows

    if plots and rng:
        hist_data, hist_areas = [], []
        for c in present:
            v = np.concatenate([x[0] for x in data[c].values()])
            a = np.concatenate([x[1] for x in data[c].values()])
            hist_data.append(v)
            hist_areas.append(a / a.sum())
        histogram(hist_data, hist_areas, present, title=title,
                  xlabel=analysis.get("xlabel", feature),
                  filename=os.path.join(output_dir, f"{name}_hist.svg"), bins=100,
                  range=rng, custom_colors=colors,
                  custom_colors_light=[c[:3] + [0.3] for c in colors])

    perm_rows = []
    for ca, cb in pairs:
        common = sorted(set(data[ca]) & set(data[cb]))
        if not common:
            log(f"  {ca} vs {cb}: no tomogram has both -- skipping.")
            continue
        perm = paired_permutation_test(
            [data[ca][t][0] for t in common], [data[cb][t][0] for t in common],
            [data[ca][t][1] for t in common], [data[cb][t][1] for t in common],
            statistic="ks", reps=reps, seed=seed, center=analysis.get("center", "median"))
        row = {"analysis": name, "level": "pooled_distribution_paired_permutation",
               "feature": feature, "filter": sf.describe(clauses),
               "condition": condition or "all", "class_a": ca, "class_b": cb,
               "ks": perm["observed"], "ks_uncentered": perm["observed_uncentered"],
               "center": perm["center"], "p_permutation": perm["p_value"],
               "p_floor": perm["min_possible_p"],
               "at_floor": perm["p_value"] <= perm["min_possible_p"] + 1e-12,
               "exact": perm["exact"], "n_pairs": perm["n_pairs"],
               "ci_statistic": ci_stat}
        # Per-class CIs over the paired tomograms, and a CI on the paired difference.
        for tag, c in (("a", ca), ("b", cb)):
            ci = (cluster_t_interval([data[c][t][0] for t in common],
                                     [data[c][t][1] for t in common], statistic=ci_stat)
                  if len(common) >= 2 else None)
            row[f"mean_{tag}"] = ci["estimate"] if ci else np.nan
            row[f"ci_{tag}_low"] = ci["ci_low"] if ci else np.nan
            row[f"ci_{tag}_high"] = ci["ci_high"] if ci else np.nan
        if len(common) >= 2:
            from scipy.stats import t as t_dist
            fa = cluster_t_interval([data[ca][t][0] for t in common],
                                    [data[ca][t][1] for t in common], statistic=ci_stat)
            fb = cluster_t_interval([data[cb][t][0] for t in common],
                                    [data[cb][t][1] for t in common], statistic=ci_stat)
            d = np.array(fa["unit_values"]) - np.array(fb["unit_values"])
            half = t_dist.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            row.update(diff=float(d.mean()), diff_ci_low=float(d.mean() - half),
                       diff_ci_high=float(d.mean() + half))
        perm_rows.append(row)
    if correction and perm_rows:
        apply_correction(perm_rows, correction, pcols=("p_permutation",))
    for row in perm_rows:
        adj = (f", {correction}-adj {row['p_permutation_adj']:.4f}"
               if "p_permutation_adj" in row else "")
        diff = (f"; {ci_stat} diff {row['diff']:.4g} [{row['diff_ci_low']:.4g}, "
                f"{row['diff_ci_high']:.4g}]" if "diff" in row else "")
        log(f"  {row['class_a']} vs {row['class_b']} (pooled, paired): "
            f"KS={row['ks']:.3f}  p={row['p_permutation']:.4f}{adj} "
            f"(floor {row['p_floor']:.3g}{', exact' if row['exact'] else ''}){diff}"
            + ("  <-- AT FLOOR: needs more tomograms" if row["at_floor"] else ""))
        test_rows.append(row)
    if plots:
        import matplotlib.pyplot as plt
        plt.close("all")
    return summary_rows, test_rows


def run_batch(ds, comparison, analyses, output_dir, only=None, conditions=None,
              split_components=False, reps=5000, seed=0, plots=True, base_filters=None,
              correction="holm", correction_family="analysis", log=print):
    """Run every analysis; return (summary_df, tests_df) and write them to output_dir.

    p-values are adjusted for multiple comparisons with `correction` (see
    `morphometrics_stats.adjust_pvalues`) over a family set by `correction_family`:
    ``analysis`` -- the pairs tested within each analysis, separately per test and per
    level (the default); ``batch`` -- every test of that kind across the whole run.
    Raw p-values are kept; adjusted ones go in ``*_adj`` columns and drive the stars.
    """
    if correction_family not in ("analysis", "batch"):
        raise ValueError("correction_family must be 'analysis' or 'batch'")
    from matplotlib.colors import to_rgba

    from . import surface_filters as sf
    from .feature_violin import summary_statistic
    from .morphometrics_stats import apply_correction, histogram, pairwise_tests, violin
    from .spatial_stats import cluster_t_interval, permutation_test

    group = comparison.get("group", "condition")
    conds_cfg = [c for c in comparison["_conditions"]
                 if conditions is None or c["name"] in conditions]
    names = [c["name"] for c in conds_cfg]
    shorts = [c["short"] for c in conds_cfg]
    # The plotting helpers index colors as sequences (color[:3]), so hand them RGBA lists.
    colors = [list(to_rgba(c["color"], alpha=0.9)) for c in conds_cfg]
    colors_light = [list(to_rgba(c["color"], alpha=0.3)) for c in conds_cfg]
    reference = comparison.get("reference")
    if reference not in names:
        reference = None

    os.makedirs(output_dir, exist_ok=True)
    unit_word = "organelle" if split_components else "tomogram"
    log(f"Comparing {len(names)} condition(s) of '{group}': {', '.join(names)}")
    log(f"Unit of replication: {unit_word}"
        + (" (condition permuted at the tomogram level)" if split_components else ""))
    log(f"Multiple-comparison correction: {correction}, within each "
        + ("analysis" if correction_family == "analysis" else "whole batch (per test)"))

    summary_rows, test_rows = [], []
    for analysis in analyses:
        if only and analysis["name"] not in only:
            continue
        if analysis.get("compare", "conditions") == "classes":
            if conditions and analysis.get("condition") not in conditions:
                continue   # --conditions restricts class analyses to those conditions
            s_rows, t_rows = _run_class_analysis(
                ds, analysis, group, output_dir, base_filters=base_filters,
                split_components=split_components, reps=reps, seed=seed, plots=plots,
                correction=correction if correction_family == "analysis" else None,
                log=log)
            summary_rows.extend(s_rows)
            test_rows.extend(t_rows)
            continue
        name = analysis["name"]
        feature, label = analysis["feature"], analysis["class"]
        clauses = sf.parse_filters(list(base_filters or []) + list(analysis.get("filters") or []))
        stat = analysis.get("statistic", "peak")
        rng = analysis.get("range")
        title = analysis.get("title", name)

        log(f"\n=== {title} "
            f"[{label}.{feature}{', ' + sf.describe(clauses) if clauses else ''}] ===")

        values, areas, conds, units, strata = collect(
            ds, feature, label, group, filters=clauses, split_components=split_components,
            min_triangles=analysis.get("min_triangles", 0),
            min_area=analysis.get("min_area", 0.0))
        if not values:
            log("  no surfaces carry this feature -- skipping.")
            continue

        # Keep only the conditions in play, preserving the configured order.
        keep = [i for i, c in enumerate(conds) if c in names]
        values = [values[i] for i in keep]
        areas = [areas[i] for i in keep]
        conds = [conds[i] for i in keep]
        units = [units[i] for i in keep]
        strata = [strata[i] for i in keep]

        present = [n for n in names if n in set(conds)]
        counts = Counter(conds)
        log("  units: " + ", ".join(f"{n}={counts[n]}" for n in present))
        if len(present) < 2:
            log("  fewer than two conditions present -- skipping.")
            continue

        # ---- per-unit summary level (violin) ----
        by_condition = {n: [] for n in present}
        for v, a, c in zip(values, areas, conds):
            by_condition[c].append(summary_statistic(v, a, stat, bins=100, bin_range=rng))
        summaries = [by_condition[n] for n in present]
        idx = [names.index(n) for n in present]

        if plots:
            violin(summaries, [shorts[i] for i in idx], title=title,
                   ylabel=analysis.get("xlabel", feature),
                   filename=os.path.join(output_dir, f"{name}_violin.svg"),
                   custom_colors=[colors[i] for i in idx],
                   separator_after=comparison.get("separator_after"))

        for n in present:
            vals = np.asarray(by_condition[n], dtype=float)
            summary_rows.append({"analysis": name, "class": label, "feature": feature,
                                 "filter": sf.describe(clauses), "statistic": stat,
                                 "condition": n, f"n_{unit_word}s": len(vals),
                                 "mean": vals.mean(),
                                 "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan})

        wanted = _pairs(comparison, present, reference)
        wanted_set = {frozenset(p) for p in wanted}
        summary_tests = [row for row in pairwise_tests(summaries, present)
                         if not comparison.get("pairs")
                         or frozenset((row["class_a"], row["class_b"])) in wanted_set]
        if correction_family == "analysis":
            apply_correction(summary_tests, correction)
        for row in summary_tests:
            row.update({"analysis": name, "level": f"per_{unit_word}_summary"})
            test_rows.append(row)

        # ---- pooled-distribution level ----
        # The pooled distributions do not depend on `statistic`, so a spread (`std`)
        # analysis would only duplicate the location analysis on the same feature+filter
        # under a misleading name; its comparison lives at the summary level above.
        if not analysis.get("distribution_test", stat != "std"):
            log("  (distribution-level permutation skipped: this analysis compares a "
                "per-unit summary, not the pooled distribution)")
            continue

        pairs = wanted

        if plots and rng:
            hist_data, hist_areas = [], []
            for n in present:
                sel = [i for i, c in enumerate(conds) if c == n]
                hist_data.append(np.concatenate([values[i] for i in sel]))
                pooled_area = np.concatenate([areas[i] for i in sel])
                hist_areas.append(pooled_area / pooled_area.sum())
            histogram(hist_data, hist_areas, [shorts[i] for i in idx], title=title,
                      xlabel=analysis.get("xlabel", feature),
                      filename=os.path.join(output_dir, f"{name}_hist.svg"),
                      bins=100, range=rng,
                      custom_colors=[colors[i] for i in idx],
                      custom_colors_light=[colors_light[i] for i in idx])

        ci_stat = analysis.get("ci_statistic", "mean")
        perm_rows = []
        for a_name, b_name in pairs:
            sel = [i for i, c in enumerate(conds) if c in (a_name, b_name)]
            perm = permutation_test(
                [values[i] for i in sel], [conds[i] for i in sel], statistic="ks",
                unit_weights=[areas[i] for i in sel],
                strata=([strata[i] for i in sel] if split_components else None),
                reps=reps, seed=seed)

            # CIs: the unit is always the tomogram, even when organelles are test units.
            cis = {}
            for n in (a_name, b_name):
                per_tomo = {}
                for i, c in enumerate(conds):
                    if c != n:
                        continue
                    v, a = per_tomo.setdefault(strata[i], ([], []))
                    v.append(values[i])
                    a.append(areas[i])
                pooled = [(np.concatenate(v), np.concatenate(a))
                          for v, a in per_tomo.values()]
                cis[n] = (cluster_t_interval([p[0] for p in pooled],
                                             [p[1] for p in pooled], statistic=ci_stat)
                          if len(pooled) >= 2 else None)

            at_floor = perm["p_value"] <= perm["min_possible_p"] + 1e-12
            row = {"analysis": name, "level": "pooled_distribution_permutation",
                   "class": label, "feature": feature, "filter": sf.describe(clauses),
                   "class_a": a_name, "class_b": b_name,
                   "ks": perm["observed"], "p_permutation": perm["p_value"],
                   "p_floor": perm["min_possible_p"], "at_floor": at_floor,
                   "n_units_a": perm["n_units_a"], "n_units_b": perm["n_units_b"],
                   "n_tomograms_a": perm["n_blocks_a"], "n_tomograms_b": perm["n_blocks_b"],
                   "ci_statistic": ci_stat}
            for tag, n in (("a", a_name), ("b", b_name)):
                ci = cis[n]
                row[f"mean_{tag}"] = ci["estimate"] if ci else np.nan
                row[f"ci_{tag}_low"] = ci["ci_low"] if ci else np.nan
                row[f"ci_{tag}_high"] = ci["ci_high"] if ci else np.nan
            perm_rows.append((row, cis))

        if correction_family == "analysis":
            apply_correction([r for r, _ in perm_rows], correction, pcols=("p_permutation",))
        for row, cis in perm_rows:
            adj = (f", {correction}-adj {row['p_permutation_adj']:.4f}"
                   if "p_permutation_adj" in row else "")
            log(f"  {row['class_a']} vs {row['class_b']}: KS={row['ks']:.3f}  "
                f"p={row['p_permutation']:.4f}{adj} (floor {row['p_floor']:.3g})"
                + ("  <-- AT FLOOR: needs more tomograms" if row["at_floor"] else ""))
            for n in (row["class_a"], row["class_b"]):
                ci = cis[n]
                log(f"      {ci_stat} {n}: "
                    + (f"{ci['estimate']:.4g} [{ci['ci_low']:.4g}, {ci['ci_high']:.4g}]"
                       if ci else "n/a (<2 tomograms)"))
            test_rows.append(row)
        if plots:
            import matplotlib.pyplot as plt
            plt.close("all")

    if correction_family == "batch":
        # One family per level (condition vs paired-class tests are different questions).
        levels = []
        for r in test_rows:
            if r["level"] not in levels:
                levels.append(r["level"])
        for level in levels:
            rows = [r for r in test_rows if r["level"] == level]
            apply_correction(rows, correction, pcols=_PCOLS)
            log(f"\nAdjusted by {correction} across the batch: {len(rows)} {level} "
                "comparisons (see the *_adj columns in tests.csv).")

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    tests.to_csv(os.path.join(output_dir, "tests.csv"), index=False)
    log(f"\nWrote {output_dir.rstrip('/')}/summary.csv and tests.csv")
    return summary, tests


@click.command(name="compare_batch")
@click.argument("configfile", type=click.Path(exists=True))
@click.option("--only", multiple=True,
              help="Run only these analyses (by `name`; repeatable).")
@click.option("--conditions", multiple=True,
              help="Restrict to these conditions (repeatable) -- e.g. turn a 5-way "
                   "comparison into a 3-way without editing the config.")
@click.option("--split-components", is_flag=True, default=False,
              help="One unit per organelle instead of per tomogram; the condition is "
                   "then permuted at the tomogram level (nested).")
@click.option("--reps", type=int, default=None,
              help="Permutation replicates (default: comparison.reps, else 5000).")
@click.option("--seed", type=int, default=None,
              help="Permutation RNG seed (default: comparison.seed, else 0).")
@click.option("--output", default=None,
              help="Output directory (default: comparison.output, else "
                   "<config dir>/compare_batch/).")
@click.option("--correction", type=click.Choice(["holm", "fdr_bh", "bonferroni", "none"]),
              default=None,
              help="Multiple-comparison correction (default: comparison.correction, "
                   "else holm).")
@click.option("--correction-family", type=click.Choice(["analysis", "batch"]), default=None,
              help="Which tests are corrected together: the pairs within each analysis, "
                   "or every test in the run (default: comparison.correction_family, "
                   "else analysis).")
@click.option("--no-plots", is_flag=True, default=False,
              help="Write only the CSV tables, no violin/histogram figures.")
def compare_batch_cli(configfile, only, conditions, split_components, reps, seed, output,
                      correction, correction_family, no_plots):
    """Run every analysis in the config's `analyses:` list across its conditions.

    CONFIGFILE: a config with `comparison:` and `analyses:` blocks, plus either a
    `datasets:` block (several runs) or the usual seg_dir/work_dir + `groups:` (one run).
    """
    from .feature_compare import _group_buckets
    from .multidataset import load_for_analysis

    config = load_config(configfile)
    comparison = dict(config.get("comparison") or {})
    analyses = config.get("analyses") or []
    try:
        validate_analyses(analyses)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if only:
        unknown = sorted(set(only) - {a["name"] for a in analyses})
        if unknown:
            raise click.ClickException(f"--only: no analyses named {unknown}")
    if not config.get("datasets"):
        from .config_utils import require_keys
        require_keys(config, ("seg_dir", "work_dir", "segmentation_values"), configfile)

    print(f"Loading data for {configfile}")
    try:
        ds = load_for_analysis(config, configfile, verbose=True)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    group = comparison.setdefault("group", "condition")
    buckets = _group_buckets(ds, group)
    if not buckets:
        raise click.ClickException(f"no group '{group}' is defined for these tomograms.")
    try:
        comparison["_conditions"] = parse_conditions(comparison, buckets)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    unknown = [c["name"] for c in comparison["_conditions"] if c["name"] not in buckets]
    if unknown:
        raise click.ClickException(
            f"comparison.conditions {unknown} are not buckets of group '{group}' "
            f"(buckets: {buckets}).")
    names = {c["name"] for c in comparison["_conditions"]}
    for pair in comparison.get("pairs") or []:
        if len(pair) != 2 or any(p not in names for p in pair):
            raise click.ClickException(
                f"comparison.pairs entry {pair} must name two of the conditions {sorted(names)}")
    for a in analyses:
        if a.get("compare") == "classes" and a.get("condition") is not None \
                and a["condition"] not in buckets:
            raise click.ClickException(
                f"analysis {a['name']!r}: condition {a['condition']!r} is not a bucket of "
                f"group '{group}' (buckets: {buckets}).")
    if conditions:
        bad = sorted(set(conditions) - {c["name"] for c in comparison["_conditions"]})
        if bad:
            raise click.ClickException(f"--conditions: unknown condition(s) {bad}")

    from .morphometrics_stats import CORRECTIONS
    corr = correction or comparison.get("correction", "holm")
    fam = correction_family or comparison.get("correction_family", "analysis")
    if corr not in CORRECTIONS:
        raise click.ClickException(f"comparison.correction must be one of {CORRECTIONS}")
    if fam not in ("analysis", "batch"):
        raise click.ClickException("comparison.correction_family must be analysis or batch")

    stats_cfg = config.get("statistics", {}) or {}
    base_filters = stats_cfg.get("filters", []) if isinstance(stats_cfg, dict) else []
    config_dir = os.path.dirname(os.path.abspath(configfile))
    out_dir = output or os.path.join(config_dir, comparison.get("output") or "compare_batch")
    run_batch(ds, comparison, analyses, out_dir, only=set(only) or None,
              conditions=list(conditions) or None, split_components=split_components,
              reps=reps if reps is not None else comparison.get("reps", 5000),
              seed=seed if seed is not None else comparison.get("seed", 0),
              plots=not no_plots, base_filters=base_filters,
              correction=corr, correction_family=fam)


if __name__ == "__main__":
    compare_batch_cli()
