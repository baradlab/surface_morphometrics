#! /usr/bin/env python
"""`morphometrics compare` -- a spatially-aware treatment comparison between conditions.

Given a config `groups:` block that assigns each tomogram to a condition (e.g. Tg vs
Vehicle), compare one feature's distribution between the two conditions, per membrane
class. The comparison treats the tomogram (not the triangle) as the unit of replication:

  * the effect size is an area-weighted KS / Wasserstein distance between the two pooled
    condition distributions (`spatial_stats.weighted_ks_statistic` / `weighted_wasserstein`);
  * the p-value comes from `spatial_stats.permutation_test`, permuting the condition label
    across tomograms (flat) -- or, with `--split-components`, across organelles nested
    within tomograms, so the p-value floor is still set by the tomogram count;
  * per-condition area-weighted means are reported with `spatial_stats.cluster_t_interval`
    confidence intervals over per-tomogram summaries, so the reader sees the direction and
    magnitude of the difference, not only its significance.

This is the command form of the pattern that `old_scripts/mitochondria_statistics.py`
hand-coded, now driven by the config grouping and the validated `spatial_stats` machinery.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from collections import defaultdict

import click
import numpy as np
import pandas as pd

from .config_utils import load_config

STATISTICS = ("ks", "wasserstein")


def _units_for_label(records, label):
    """(values, areas, strata) lists for every unit of one class, in record order."""
    values, areas, strata = [], [], []
    for lab, _unit, stratum, v, a in records:
        if lab == label:
            values.append(v)
            areas.append(a)
            strata.append(stratum)
    return values, areas, strata


def _per_tomogram_pooled(records, label):
    """{tomogram: (values, areas)} pooling every unit (organelle) of a class per tomogram.

    Used for the cluster confidence intervals, whose unit of replication is the tomogram
    even when the permutation test splits organelles.
    """
    buckets = defaultdict(lambda: ([], []))
    for lab, _unit, stratum, v, a in records:
        if lab == label:
            vs, as_ = buckets[stratum]
            vs.append(v)
            as_.append(a)
    return {tomo: (np.concatenate(vs), np.concatenate(as_)) for tomo, (vs, as_) in buckets.items()}


def _group_buckets(ds, group):
    """Declared bucket names of `group`, in config order ([] if the group is undefined).

    For a multi-dataset this is the union over member datasets; the implicit `dataset`
    group's buckets are the member names.
    """
    members = list(ds.members.values()) if hasattr(ds, "members") else [ds]
    buckets = []
    for member in members:
        for bucket in (member.groups.get(group) or {}):
            if bucket not in buckets:
                buckets.append(bucket)
    if not buckets and hasattr(ds, "members") and group == "dataset":
        buckets = list(ds.members)
    return buckets


def _available_groups(ds):
    members = list(ds.members.values()) if hasattr(ds, "members") else [ds]
    names = ["dataset"] if hasattr(ds, "members") else []
    for member in members:
        names.extend(g for g in member.groups if g not in names)
    return names


@click.command(name="compare")
@click.argument("configfile", type=click.Path(exists=True))
@click.option("-n", "--feature", required=True,
              help="Feature column to compare between conditions (e.g. curvedness_VV).")
@click.option("-g", "--group", "group", required=True,
              help="Name of the config `groups:` block to compare across (must have "
                   "exactly two buckets, e.g. condition: {Tg, Vehicle}).")
@click.option("-s", "--statistic", type=click.Choice(STATISTICS), default="ks",
              show_default=True,
              help="Effect size between the pooled condition distributions.")
@click.option("--filter", "filters", multiple=True,
              help="Keep only triangles matching [CLASS:]PROPERTY OP VALUE before "
                   "comparing (repeatable, ANDed), e.g. 'IMM:OMM_dist>=20' to compare "
                   "cristae. Merged with config statistics.filters.")
@click.option("--split-components", is_flag=True, default=False,
              help="Use each connected component (organelle) as a unit, permuting the "
                   "condition at the tomogram level (nested). Off by default (unit = "
                   "tomogram).")
@click.option("--component-column", default="component_number", show_default=True,
              help="Per-triangle connected-component id column, for --split-components.")
@click.option("--conditions", nargs=2, default=None,
              help="The two buckets of GROUP to compare, when it has more than two "
                   "(e.g. --conditions Tg Vehicle).")
@click.option("--reps", type=int, default=2000, show_default=True,
              help="Permutation replicates for the p-value.")
@click.option("--seed", type=int, default=0, show_default=True, help="Permutation RNG seed.")
@click.option("--output", default=None,
              help="Output CSV path (default: work_dir/<feature>_<group>_compare.csv).")
def compare_cli(configfile, feature, group, statistic, filters, split_components,
                component_column, conditions, reps, seed, output):
    """Compare FEATURE between the two conditions of a config GROUP, per membrane class.

    CONFIGFILE: path to config.yml. GROUP is a `groups:` block of the config -- or, for a
    multi-dataset config (a `datasets:` block), any group defined on its datasets, or
    `dataset` to compare whole datasets against each other.
    """
    from .multidataset import load_for_analysis
    from .surface_filters import parse_filters, describe
    from .spatial_stats import permutation_test, cluster_t_interval

    config = load_config(configfile)
    multi = bool(config.get("datasets"))
    if not multi:
        from .config_utils import require_keys
        require_keys(config, ("seg_dir", "work_dir", "segmentation_values"), configfile)
    try:
        ds = load_for_analysis(config, configfile)   # validates group assignment
    except ValueError as exc:
        raise click.ClickException(str(exc))
    buckets = _group_buckets(ds, group)
    if not buckets:
        raise click.ClickException(
            f"config has no group '{group}'. Define it under `groups:` in {configfile} "
            f"(available: {_available_groups(ds) or 'none'}).")
    if conditions:
        missing = [c for c in conditions if c not in buckets]
        if missing:
            raise click.ClickException(
                f"--conditions {missing} not in group '{group}' (buckets: {buckets}).")
        buckets = list(conditions)
    if len(buckets) != 2:
        raise click.ClickException(
            f"group '{group}' has {len(buckets)} buckets {buckets}; compare needs exactly "
            "two conditions. Pick two with --conditions, or use `compare_batch` for an "
            "N-way comparison.")

    stats_cfg = config.get("statistics", {}) or {}
    config_filters = stats_cfg.get("filters", []) if isinstance(stats_cfg, dict) else []
    clauses = parse_filters(list(config_filters) + list(filters))

    records, diag = ds.collect_feature(
        feature, split_components=split_components, component_column=component_column,
        filters=clauses)
    if not records:
        raise click.ClickException(
            f"no surfaces have a '{feature}' column after selection/filtering. Check the "
            "feature name, the pipeline run, and any filters.")

    unit_name = "organelle(s)" if split_components else "tomogram(s)"
    print(f"Comparing '{feature}' across group '{group}': {buckets[0]} vs {buckets[1]}")
    print(f"  effect size: area-weighted {statistic}; unit = {unit_name}; "
          f"{reps} permutations")
    if clauses:
        print(f"  triangle filter: {describe(clauses)}")
    if diag["excluded_tomograms"]:
        print(f"  (excluded {len(diag['excluded_tomograms'])} tomogram(s) by selection)")

    labels_present = [lab for lab in ds.labels
                      if any(r[0] == lab for r in records)]
    rows = []
    for label in labels_present:
        values, areas, strata = _units_for_label(records, label)
        conds = [ds.metadata(tomo).get(group) for tomo in strata]
        keep = [i for i, c in enumerate(conds) if c in buckets]
        values = [values[i] for i in keep]
        areas = [areas[i] for i in keep]
        strata = [strata[i] for i in keep]
        conds = [conds[i] for i in keep]
        present = sorted(set(conds))
        if len(present) < 2:
            print(f"\n  {label}: only condition {present} present -- skipping.")
            continue

        perm = permutation_test(
            values, conds, statistic=statistic, unit_weights=areas,
            strata=(strata if split_components else None), reps=reps, seed=seed)

        # Cluster CIs: one area-weighted mean per tomogram, per condition.
        pooled = _per_tomogram_pooled(records, label)
        by_condition = defaultdict(list)
        for tomo, (v, a) in pooled.items():
            cond = ds.metadata(tomo).get(group)
            if cond in buckets:
                by_condition[cond].append((v, a))
        cis = {}
        for cond in (perm["condition_a"], perm["condition_b"]):
            units = by_condition.get(cond, [])
            if len(units) >= 2:
                cis[cond] = cluster_t_interval([u[0] for u in units],
                                               [u[1] for u in units], statistic="mean")
            else:
                cis[cond] = None

        row = {"class": label, "group": group, "feature": feature,
               "statistic": statistic,
               "condition_a": perm["condition_a"], "condition_b": perm["condition_b"],
               "n_units_a": perm["n_units_a"], "n_units_b": perm["n_units_b"],
               "n_blocks_a": perm["n_blocks_a"], "n_blocks_b": perm["n_blocks_b"],
               "observed": perm["observed"], "p_value": perm["p_value"],
               "min_possible_p": perm["min_possible_p"]}
        for tag, cond in (("a", perm["condition_a"]), ("b", perm["condition_b"])):
            ci = cis[cond]
            row[f"mean_{tag}"] = ci["estimate"] if ci else float("nan")
            row[f"ci_{tag}_low"] = ci["ci_low"] if ci else float("nan")
            row[f"ci_{tag}_high"] = ci["ci_high"] if ci else float("nan")
        rows.append(row)

        def _fmt(cond):
            ci = cis[cond]
            return (f"{ci['estimate']:.3g} [{ci['ci_low']:.3g}, {ci['ci_high']:.3g}]"
                    if ci else "n/a (need >=2 tomograms)")
        print(f"\n  {label}:  {statistic} = {perm['observed']:.3g}, "
              f"p = {perm['p_value']:.3g} "
              f"(floor {perm['min_possible_p']:.3g}; "
              f"{perm['n_blocks_a']}+{perm['n_blocks_b']} tomogram-blocks)")
        print(f"    mean {perm['condition_a']}: {_fmt(perm['condition_a'])}")
        print(f"    mean {perm['condition_b']}: {_fmt(perm['condition_b'])}")
        if perm["p_value"] <= perm["min_possible_p"] + 1e-12:
            print("    NOTE: p is at the permutation floor -- more tomograms would be "
                  "needed to resolve it further.")

    if not rows:
        raise click.ClickException(
            "no class had both conditions present; nothing to compare.")
    out_dir = config.get("work_dir") or (os.path.dirname(os.path.abspath(configfile)) + "/")
    out_csv = output or f"{out_dir}{feature}_{group}_compare.csv"
    if not out_csv.endswith(".csv"):
        out_csv += ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    compare_cli()
