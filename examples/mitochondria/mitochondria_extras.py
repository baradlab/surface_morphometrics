#! /usr/bin/env python
"""The parts of the mitochondria analysis that are not per-triangle feature comparisons.

`morphometrics compare_batch mitochondria_study.yml` covers every feature comparison
(distances, curvedness, spacing, verticality, angles). This script adds the
per-tomogram quantities and pooled 2D histograms from old_scripts/mitochondria_statistics.py:

  * surface area per tomogram for each membrane (bar charts by arm, including Tg+GSK);
  * OMM-ER contact sites: per tomogram, the OMM area within 30 nm of the ER (and the ER
    area within 30 nm of the OMM), as absolute areas and as fractions;
  * 2D histograms: OMM-ER vs OMM-IMM distance, and IMM curvedness vs distance to the OMM.

It reads the same study config, so the datasets, exclusions and grouping are shared --
there is no tomogram-name parsing here.

    python mitochondria_extras.py mitochondria_study.yml
"""

import os
import sys

import numpy as np
import pandas as pd

from surface_morphometrics.batch_compare import parse_conditions
from surface_morphometrics.config_utils import load_config
from surface_morphometrics.morphometrics_stats import (barchart, pairwise_tests,
                                                       twod_histogram, violin)
from surface_morphometrics.multidataset import load_for_analysis, split_stratum

CONTACT_NM = 30          # ER contact-site distance threshold
CLASSES = ["OMM", "IMM", "ER"]


def surfaces(ds, label, **group_values):
    """(stratum, metadata, dataframe) for every tomogram that has `label` and matches."""
    for stratum in ds.tomograms():
        meta = ds.metadata(stratum)
        if any(meta.get(g) != v for g, v in group_values.items()):
            continue
        source, tomo = split_stratum(stratum)
        member = ds[source]
        if member.exists(tomo, label):
            yield stratum, meta, member.load(tomo, label)


def per_tomogram_table(ds):
    """One row per tomogram: arm/condition/morphology, areas, ER-contact areas/fractions."""
    rows = []
    for stratum in ds.tomograms():
        source, tomo = split_stratum(stratum)
        member, meta = ds[source], ds.metadata(stratum)
        row = {"tomogram": stratum, **meta}
        for label in CLASSES:
            row[f"{label}_area"] = (member.load(tomo, label)["area"].sum()
                                    if member.exists(tomo, label) else 0.0)
        if member.exists(tomo, "ER") and member.exists(tomo, "OMM"):
            omm, er = member.load(tomo, "OMM"), member.load(tomo, "ER")
            # Denominator = whole surface (triangles with no ER partner count as non-contact).
            row["OMM_contact_area"] = omm.loc[omm["ER_dist"] < CONTACT_NM, "area"].sum()
            row["ER_contact_area"] = er.loc[er["OMM_dist"] < CONTACT_NM, "area"].sum()
            row["OMM_contact_fraction"] = row["OMM_contact_area"] / row["OMM_area"]
            row["ER_contact_fraction"] = row["ER_contact_area"] / row["ER_area"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_per_tomogram(table, column, arms, pairs, colors, title, ylabel, out):
    """Violin of one per-tomogram column by arm + summary-level tests on `pairs`."""
    data = [table.loc[table["arm"] == a, column].dropna().to_numpy() for a in arms]
    present = [a for a, d in zip(arms, data) if len(d)]
    data = [d for d in data if len(d)]
    violin(data, present, title=title, ylabel=ylabel,
           filename=os.path.join(out, f"{column}_violin.svg"),
           custom_colors=[colors[a] for a in present], separator_after=2)
    wanted = {frozenset(p) for p in pairs}
    rows = [r for r in pairwise_tests(data, present)
            if frozenset((r["class_a"], r["class_b"])) in wanted]
    for r in rows:
        r["quantity"] = column
    return rows


def main(configfile):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    config = load_config(configfile)
    ds = load_for_analysis(config, configfile, verbose=True)
    comparison = config["comparison"]
    out = os.path.join(os.path.dirname(os.path.abspath(configfile)),
                       comparison.get("output", "output_final"), "extras")
    os.makedirs(out, exist_ok=True)
    conds = parse_conditions(comparison, [])
    arms = [c["name"] for c in conds]
    colors = {c["name"]: list(to_rgba(c["color"], alpha=0.9)) for c in conds}
    pairs = comparison["pairs"]

    # --- per-tomogram areas and ER contact sites ---
    table = per_tomogram_table(ds)
    table.to_csv(os.path.join(out, "per_tomogram.csv"), index=False)

    for arm, sub in table.groupby("arm", sort=False):
        bars = [sub[f"{c}_area"].mean() / 1e6 for c in CLASSES]
        errs = [sub[f"{c}_area"].std() / 1e6 for c in CLASSES]
        barchart(bars, errs, CLASSES, f"Surface area per tomogram - {arm}",
                 ylabel="Area (µm²)", filename=os.path.join(out, f"areas_{arm.replace(' ', '_')}.svg"))

    tests = []
    for column, title, ylabel in [
            ("OMM_area", "OMM area per tomogram", "Area (nm²)"),
            ("OMM_contact_area", f"OMM area within {CONTACT_NM} nm of ER", "Area (nm²)"),
            ("ER_contact_area", f"ER area within {CONTACT_NM} nm of OMM", "Area (nm²)"),
            ("OMM_contact_fraction", "Fraction of OMM contacting ER", "Fraction of OMM area"),
            ("ER_contact_fraction", "Fraction of ER contacting OMM", "Fraction of ER area")]:
        tests += compare_per_tomogram(table, column, arms, pairs, colors, title, ylabel, out)
    pd.DataFrame(tests).to_csv(os.path.join(out, "per_tomogram_tests.csv"), index=False)
    plt.close("all")

    # --- pooled 2D histograms per condition x morphology (incl. Tg+GSK) ---
    for condition in ds.group_values("condition"):
        for morphology in ("elongated", "fragmented"):
            omm = [df for _s, _m, df in surfaces(ds, "OMM", condition=condition,
                                                 morphology=morphology)
                   if "ER_dist" in df.columns]
            if omm:
                omm = pd.concat(omm)
                twod_histogram(omm["IMM_dist"], omm["ER_dist"], omm["area"],
                               "OMM-IMM Distance (nm)", "OMM-ER Distance (nm)",
                               f"{condition}: ER vs IMM distance ({morphology})",
                               bins=(30, 30), range=[[5, 30], [5, 50]], figsize=(4, 4),
                               filename=os.path.join(out, f"ER_OMM_IMM_2D_{condition}_{morphology}.svg"))
            imm = [df for _s, _m, df in surfaces(ds, "IMM", condition=condition,
                                                 morphology=morphology)]
            if imm:
                imm = pd.concat(imm)
                twod_histogram(imm["OMM_dist"], imm["curvedness_VV"], imm["area"],
                               "IMM-OMM Distance (nm)", "Curvedness (1/nm)",
                               f"{condition}: IMM curvedness vs distance ({morphology})",
                               bins=(50, 100), range=[[0, 50], [0, 0.1]], figsize=(4, 4),
                               filename=os.path.join(out, f"curvedness_vs_distance_{condition}_{morphology}.svg"))
        plt.close("all")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mitochondria_study.yml")
