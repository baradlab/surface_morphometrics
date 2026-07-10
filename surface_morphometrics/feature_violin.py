#! /usr/bin/env python
"""Violin plot comparing one feature across membrane classes, one point per tomogram.

For every tomogram in the dataset and every segmentation class (the keys of
`segmentation_values`), the chosen area-weighted summary statistic of the feature is
computed over that surface's triangles. Each class becomes one violin whose
distribution is the set of per-tomogram values, with one dot per tomogram overlaid.

Classes whose surfaces do not carry the feature are skipped (e.g. the IMM surface has
no `IMM_dist` column), so `--feature IMM_dist` naturally compares the classes that
measure distance to the IMM.

With --test, every pair of classes is compared with a Mann-Whitney U test and a Welch
t-test, and the plot is annotated with significance stars from the chosen one.

Usage:
  morphometrics violin config.yml --feature IMM_dist
  morphometrics violin config.yml --feature curvedness_VV --statistic peak
  morphometrics violin config.yml --feature IMM_dist --test mwu
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import click
import numpy as np
import pandas as pd

from .config_utils import load_config
from .morphometrics_stats import (pairwise_tests, violin, weighted_avg_and_std,
                                  weighted_median, weighted_histogram_peak)

STATISTICS = ("mean", "median", "peak", "std")
TESTS = ("mwu", "ttest")


def summary_statistic(values, areas, statistic, bins=100, bin_range=None):
    """Area-weighted summary of one surface's feature values.

    `mean`, `median`, and `peak` (histogram peak, i.e. the mode) summarize the surface's
    central tendency; `peak` needs `bins` and `bin_range`. `std` is the area-weighted
    standard deviation -- a measure of within-surface spread / heterogeneity rather than
    location (e.g. how variable the curvature or thickness is across one organelle).
    """
    if statistic == "mean":
        return float(weighted_avg_and_std(values, areas)[0])
    if statistic == "std":
        return float(weighted_avg_and_std(values, areas)[1])
    if statistic == "median":
        return float(weighted_median(values, areas))
    if statistic == "peak":
        if bin_range is None:
            bin_range = (float(np.min(values)), float(np.max(values)))
        return float(weighted_histogram_peak(values, areas, bins, bin_range))
    raise ValueError(f"Unknown statistic: {statistic!r} (expected one of {STATISTICS})")


def collect_feature(config, feature, split_components=False,
                    component_column="component_number"):
    """Load every surface that carries `feature`.

    Returns (labels, records) where labels is the configured class order and records
    is a list of (label, unit, stratum, values, areas) with non-finite / zero-area
    triangles removed. `unit` is the point plotted; `stratum` is the tomogram it came
    from (for intraclass-correlation and nested tests).

    With `split_components`, each surface is split by its connected-component column
    (organelles) into one record per component (id > 0), so each organelle is a unit;
    the whole surface's tomogram remains the stratum. Otherwise the whole surface is
    one unit and the stratum is the tomogram.
    """
    work_dir = config["work_dir"]
    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)
    extension = f".AVV_rh{radius_hit}.csv"
    labels = list(config["segmentation_values"].keys())
    tomograms = sorted(os.path.basename(f)[:-4] for f in glob(config["seg_dir"] + "*.mrc"))

    records = []
    for label in labels:
        for tomo in tomograms:
            path = f"{work_dir}{tomo}_{label}{extension}"
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)
            if feature not in df.columns or "area" not in df.columns:
                continue
            values = df[feature].to_numpy(dtype=float)
            areas = df["area"].to_numpy(dtype=float)
            finite = np.isfinite(values) & np.isfinite(areas) & (areas > 0)

            if split_components:
                if component_column not in df.columns:
                    continue
                comp = df[component_column].to_numpy()
                for cid in np.unique(comp[comp > 0]):
                    keep = finite & (comp == cid)
                    if keep.any():
                        records.append((label, f"{tomo}#c{int(cid)}", tomo,
                                        values[keep], areas[keep]))
            elif finite.any():
                records.append((label, tomo, tomo, values[finite], areas[finite]))
    return labels, records


@click.command(name="violin")
@click.argument("configfile", type=click.Path(exists=True))
@click.option("-n", "--feature", required=True,
              help="Feature column to compare across classes (e.g. IMM_dist).")
@click.option("-s", "--statistic", type=click.Choice(STATISTICS), default="median",
              show_default=True,
              help="Per-tomogram area-weighted summary: mean, median, peak (histogram "
                   "peak, i.e. the mode), or std (within-surface spread / heterogeneity).")
@click.option("--bins", type=int, default=100, show_default=True,
              help="Histogram bins, used only by --statistic peak.")
@click.option("--range", "bin_range", type=(float, float), default=None,
              help="Min/max of the peak histogram (default: the feature's full range "
                   "across all surfaces, so peaks are comparable between classes).")
@click.option("-t", "--test", type=click.Choice(TESTS), default=None,
              help="Run pairwise significance tests between classes and draw stars on "
                   "the plot from the chosen test: 'mwu' (Mann-Whitney U) or 'ttest' "
                   "(Welch). Both tests are always written to a *_tests.csv.")
@click.option("--split-components", is_flag=True, default=False,
              help="Treat each connected component (organelle) as a unit instead of the "
                   "whole surface. One point per organelle; reports the intraclass "
                   "correlation (ICC) so you can judge whether organelles cluster by "
                   "tomogram. Off by default (unit = tomogram).")
@click.option("--component-column", default="component_number", show_default=True,
              help="Per-triangle column holding the connected-component id (from "
                   "`morphometrics label_components`); used with --split-components.")
@click.option("--output", default=None,
              help="Output .svg path (default: work_dir/<feature>_<statistic>_violin.svg). "
                   "A matching .png and .csv of the per-unit values are written too.")
@click.option("--figuresize", nargs=2, type=float, default=(5.0, 4.0), show_default=True,
              help="Figure size in inches (x y).")
def violin_cli(configfile, feature, statistic, bins, bin_range, test, split_components,
               component_column, output, figuresize):
    """Violin plot of one FEATURE across membrane classes, one point per tomogram.

    CONFIGFILE: path to config.yml.

    Each class (from segmentation_values) gets a violin built from one value per
    tomogram: the area-weighted mean, median, histogram peak (mode), or standard
    deviation of the feature over that tomogram's surface for that class. `std`
    compares within-surface spread (heterogeneity) rather than central tendency.

    With --test, every pair of classes is compared (Mann-Whitney U, Welch t-test, and
    a Kolmogorov-Smirnov test) and the plot is annotated with significance stars from
    the chosen test: **** p<0.001, *** p<0.005, ** p<0.01, * p<0.05, otherwise "ns".
    All three are written to <feature>_<statistic>_violin_tests.csv. The KS here
    compares the per-tomogram summary values (comparison_level column), NOT the pooled
    triangle distributions with the effective-n correction in ks_statistics().

    Note that Mann-Whitney U has a p-value floor set by the number of units (with
    n=6 vs 6 it cannot go below ~0.002, so **** is unreachable no matter how separated
    the groups are). Welch's t-test has no such floor but assumes roughly normal means.

    With --split-components each connected component (organelle) is a unit instead of the
    whole surface, giving one point per organelle and reporting the intraclass
    correlation (ICC) per class. Organelles are treated as independent points (flat), so
    if the ICC is high (organelles cluster by tomogram) the between-class stars overstate
    significance -- prefer per-tomogram in that case. For a treatment comparison where the
    condition is assigned per tomogram, use spatial_stats.permutation_test(strata=...),
    which permutes at the tomogram level while still using every organelle.
    """
    config = load_config(configfile, require=("seg_dir", "work_dir", "segmentation_values"))
    labels, records = collect_feature(config, feature, split_components=split_components,
                                      component_column=component_column)
    if not records:
        extra = (f" (with connected-component column '{component_column}')"
                 if split_components else "")
        raise click.ClickException(
            f"No surfaces in {config['work_dir']} have a '{feature}' column{extra}. "
            f"Check the feature name and that the pipeline has been run.")

    # For `peak`, share one histogram range across every class so the peaks are
    # directly comparable (matching how the published scripts fixed the range).
    if statistic == "peak" and bin_range is None:
        all_values = np.concatenate([r[3] for r in records])
        bin_range = (float(all_values.min()), float(all_values.max()))

    values_by_label = {}
    units_by_label = {}
    strata_by_label = {}
    for label, unit, stratum, values, areas in records:
        stat = summary_statistic(values, areas, statistic, bins=bins, bin_range=bin_range)
        values_by_label.setdefault(label, []).append(stat)
        units_by_label.setdefault(label, []).append(unit)
        strata_by_label.setdefault(label, []).append(stratum)

    used = [label for label in labels if label in values_by_label]
    datasets = [values_by_label[label] for label in used]

    unit_name_str = "organelle(s)" if split_components else "tomogram(s)"
    stat_name = {"mean": "area-weighted mean",
                 "median": "area-weighted median",
                 "peak": "histogram peak (mode)",
                 "std": "area-weighted std (spread)"}[statistic]
    print(f"Violin plot of '{feature}' across {len(used)} class(es), statistic: {stat_name}")
    if statistic == "peak":
        print(f"  Histogram: {bins} bins over range {bin_range[0]:.3f} - {bin_range[1]:.3f}")
    for label in used:
        vals = np.asarray(values_by_label[label], dtype=float)
        line = (f"  {label}: n={len(vals)} {unit_name_str}, "
                f"{stat_name} = {vals.mean():.3f} +/- {vals.std():.3f}")
        if split_components:
            from .spatial_stats import intraclass_correlation
            icc = intraclass_correlation(values_by_label[label], strata_by_label[label])
            n_tomo = len(set(strata_by_label[label]))
            line += f"  [{n_tomo} tomogram(s), ICC={icc:.2f}]"
        print(line)
    if split_components:
        print("  NOTE: organelles are treated as independent points here (flat). If ICC is")
        print("        high, organelles cluster by tomogram and between-class stars overstate")
        print("        significance -- prefer per-tomogram (drop --split-components).")

    out_svg = output or f"{config['work_dir']}{feature}_{statistic}_violin.svg"
    if not out_svg.endswith(".svg"):
        out_svg += ".svg"

    # Tidy per-unit table alongside the figure, for downstream stats. `unit` is the
    # plotted point (tomogram, or organelle when split); `tomogram` is its stratum.
    rows = [{"unit": unit, "tomogram": stratum, "class": label, "feature": feature,
             "statistic": statistic, "value": value}
            for label in used
            for unit, stratum, value in zip(units_by_label[label], strata_by_label[label],
                                            values_by_label[label])]
    csv_path = out_svg[:-4] + ".csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Optional pairwise significance testing, annotated onto the plot.
    annotations = None
    tests_csv = None
    if test:
        if len(used) < 2:
            print(f"  WARNING: --test needs at least 2 classes with '{feature}'; "
                  f"only found {used}. Skipping tests.")
        else:
            # include_ks=True (default) -> the CSV also carries the paired
            # summary-statistic KS. comparison_level records that these tests operate
            # on one summary value per tomogram, not on pooled triangle distributions.
            test_rows = pairwise_tests(datasets, used)
            for row in test_rows:
                row["comparison_level"] = "per_tomogram_summary_statistic"
            tests_csv = out_svg[:-4] + "_tests.csv"
            cols = (["comparison_level", "class_a", "class_b", "n_a", "n_b",
                     "mean_a", "mean_b", "ci95_a", "ci95_b",
                     "mwu_U", "mwu_p", "mwu_stars", "ttest_t", "ttest_p", "ttest_stars",
                     "ks_stat", "ks_p", "ks_stars"])
            pd.DataFrame(test_rows)[cols].to_csv(tests_csv, index=False)
            star_key = "mwu_stars" if test == "mwu" else "ttest_stars"
            test_label = "Mann-Whitney U" if test == "mwu" else "Welch t-test"
            print(f"  Pairwise {test_label} (stars on plot; MWU, Welch & KS in the CSV):")
            for row in test_rows:
                print(f"    {row['class_a']} vs {row['class_b']}: "
                      f"MWU p={row['mwu_p']:.3g} ({row['mwu_stars']}), "
                      f"Welch p={row['ttest_p']:.3g} ({row['ttest_stars']}), "
                      f"KS p={row['ks_p']:.3g} ({row['ks_stars']})")
            annotations = [(used.index(row["class_a"]), used.index(row["class_b"]),
                            row[star_key]) for row in test_rows]

    violin(datasets, used,
           title=f"{feature} by class ({stat_name})",
           ylabel=f"{feature} ({statistic})",
           filename=out_svg, figsize=tuple(figuresize), annotations=annotations)
    outputs = f"{out_svg}, {out_svg[:-3]}png, and {csv_path}"
    if tests_csv:
        outputs += f", {tests_csv}"
    print(f"Wrote {outputs}")


if __name__ == "__main__":
    violin_cli()
