#! /usr/bin/env python
"""`morphometrics variogram` -- is a feature's local variation signal, and at what scale?

A semivariogram γ(h) is half the mean squared difference of a per-triangle quantity
between triangles a distance h apart. Fitted with a nugget + exponential + drift model
(`spatial_stats.fit_correlation_length`), it summarizes a field as:

  * **correlation length ℓ** -- the spatial scale of the structured variation, and so how
    many independent patches a surface holds (N_eff = A / 2πℓ²);
  * **nugget** -- variance at zero separation (noise plus sub-resolution structure);
  * **sill** -- the structured (spatially correlated) variance; the structured fraction
    sill / (nugget + sill) is how much of the short-range variability is real pattern;
  * **drift** -- a smooth, large-scale trend (non-stationarity) absorbed separately so
    it is not mistaken for correlation.

This is opt-in QC and interpretation, not part of the pipeline. Two guardrails:

  * For neighborhood-averaged quantities -- curvature (radius_hit), thickness and offset
    (average_radius) -- neighboring triangles share their averaging windows, so the
    nugget is artificially suppressed and is NOT a measurement-noise floor. For
    thickness/offset, run `measure_thickness --noise-estimate` (split-half): when its
    `thickness_noise.csv` is present, this command reports that noise variance and the
    resulting reliability next to the variogram.
  * Distances are 3D (Euclidean) by default, which underestimates along-surface
    separation where a membrane folds back on itself (cristae). Keep --max-h modest, or
    use --geodesic (needs graph-tool and the surface .gt graphs).

Writes `<feature>_variogram.csv` (one row per surface plus a pooled row per class) and,
unless --no-plots, a `<feature>_variogram_<class>.svg` per class.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os

import click
import numpy as np
import pandas as pd

from .config_utils import load_config

# Quantities computed by averaging over a neighborhood -> suppressed nugget.
NEIGHBORHOOD_AVERAGED = ("thickness", "offset", "curvedness", "curvature", "kappa",
                         "shape_index")
NOISE_FEATURES = ("thickness", "offset")
POOLED = "(pooled)"


def _iter_surfaces(ds, labels):
    """(stratum, member Dataset, tomogram, label) for every present surface."""
    members = list(ds.members.items()) if hasattr(ds, "members") else [(None, ds)]
    for label in labels:
        for source, member in members:
            for tomo in member.tomograms():
                if member.exists(tomo, label):
                    stratum = f"{source}/{tomo}" if source else tomo
                    yield stratum, member, tomo, label


def _noise_lookup(member, tomo, label, feature):
    """Split-half noise row values for this surface from work_dir/thickness_noise.csv."""
    from .measure_thickness import NOISE_CSV

    if feature not in NOISE_FEATURES or not getattr(member, "work_dir", ""):
        return None
    path = os.path.join(member.work_dir, NOISE_CSV)
    if not os.path.isfile(path):
        return None
    cache = getattr(member, "_noise_table", None)
    if cache is None:
        cache = member._noise_table = pd.read_csv(path).set_index("surface")
    key = f"{tomo}_{label}"
    if key not in cache.index:
        return None
    row = cache.loc[key]
    col = f"{feature}_noise_var"
    if col not in row:
        return None
    return {"split_half_noise_var": float(row[col]),
            "split_half_noise_var_robust": float(row.get(f"{feature}_noise_var_robust",
                                                         np.nan))}


def _area_weighted_var(values, areas):
    mu = np.average(values, weights=areas)
    return float(np.average((values - mu) ** 2, weights=areas))


def _load_graph(member, tomo, label):
    from graph_tool import load_graph

    path = f"{member.work_dir}{tomo}_{label}.AVV_rh{member.radius_hit}.gt"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--geodesic needs the surface graph {path}")
    return load_graph(path)


def run_variogram(ds, feature, labels=None, filters=None, max_h=60.0, n_bins=30,
                  n_seeds=200, robust=True, min_pairs=30, min_triangles=200,
                  geodesic=False, seed=0, model="auto", log=print):
    """Per-surface and pooled-per-class variogram fits. Returns (rows, curves).

    `curves` maps class -> {"pooled": (h, gamma, counts, fit), "surfaces": [(h, gamma)]}
    for plotting.
    """
    from .spatial_stats import (VariogramSums, euclidean_semivariogram,
                                fit_correlation_length, geodesic_semivariogram,
                                summarize_variogram_fit)
    from .surface_filters import filter_mask

    labels = labels or ds.labels
    rows, curves = [], {}
    pooled = {}
    pooled_area = {}
    pooled_var = {}
    for stratum, member, tomo, label in _iter_surfaces(ds, labels):
        df = member.load(tomo, label)
        if feature not in df.columns or "area" not in df.columns:
            continue
        values = df[feature].to_numpy(dtype=float)
        areas = df["area"].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
        if filters:
            mask &= filter_mask(df, filters, label)
        if mask.sum() < min_triangles:
            continue
        if geodesic:
            graph = _load_graph(member, tomo, label)
            if graph.num_vertices() != len(df):
                raise ValueError(f"{stratum} {label}: graph has {graph.num_vertices()} "
                                 f"vertices but the CSV has {len(df)} rows")
            h, gamma, counts, sums = geodesic_semivariogram(
                graph, np.where(mask, values, np.nan), areas=areas, max_h=max_h,
                n_bins=n_bins, n_seeds=n_seeds, robust=robust, min_pairs=min_pairs,
                seed=seed, mask=mask, return_sums=True)
        else:
            if not all(c in df.columns for c in ("xyz_x", "xyz_y", "xyz_z")):
                raise ValueError(f"{stratum} {label}: CSV has no xyz_x/y/z columns")
            xyz = df[["xyz_x", "xyz_y", "xyz_z"]].to_numpy(dtype=float)
            h, gamma, counts, sums = euclidean_semivariogram(
                xyz, values, areas=areas, max_h=max_h, n_bins=n_bins, n_seeds=n_seeds,
                robust=robust, min_pairs=min_pairs, seed=seed, mask=mask,
                return_sums=True)
        v, a = values[mask], areas[mask]
        variance = _area_weighted_var(v, a)
        fit = fit_correlation_length(h, gamma, counts=counts, model=model)
        row = {"class": label, "surface": stratum, "feature": feature,
               "n_triangles": int(mask.sum()), "area": float(a.sum())}
        row.update(summarize_variogram_fit(fit, variance=variance, total_area=a.sum()))
        noise = _noise_lookup(member, tomo, label, feature)
        if noise:
            row.update(noise)
            row["split_half_reliability"] = (1.0 - noise["split_half_noise_var"] / variance
                                             if variance > 0 else np.nan)
        rows.append(row)
        curves.setdefault(label, {"surfaces": []})["surfaces"].append((h, gamma))
        if label not in pooled:
            pooled[label] = VariogramSums(max_h, n_bins, robust)
            pooled_area[label] = 0.0
            pooled_var[label] = []
        pooled[label] += sums
        pooled_area[label] += float(a.sum())
        pooled_var[label].append((variance, float(a.sum())))

    for label in labels:
        if label not in pooled:
            continue
        h, gamma, counts = pooled[label].result(min_pairs)
        fit = fit_correlation_length(h, gamma, counts=counts, model=model)
        var = np.average([v for v, _ in pooled_var[label]],
                         weights=[w for _, w in pooled_var[label]])
        row = {"class": label, "surface": POOLED, "feature": feature,
               "n_triangles": int(sum(r["n_triangles"] for r in rows
                                      if r["class"] == label and r["surface"] != POOLED)),
               "area": pooled_area[label]}
        # N_eff for the pooled row is per class total area; ℓ from the pooled fit.
        row.update(summarize_variogram_fit(fit, variance=var, total_area=pooled_area[label]))
        rows.append(row)
        curves[label]["pooled"] = (h, gamma, counts, fit)
        _log_class(log, label, row, rows, feature)
    return rows, curves


def _log_class(log, label, pooled_row, rows, feature):
    n_surf = sum(1 for r in rows if r["class"] == label and r["surface"] != POOLED)
    log(f"\n  {label} ({n_surf} surface(s), pooled):")
    if pooled_row["ok"]:
        log(f"    {pooled_row['model']} model: correlation length ℓ = {pooled_row['ell']:.3g} "
            f"(±{pooled_row['ell_stderr']:.2g}); nugget {pooled_row['nugget']:.3g}, "
            f"sill {pooled_row['sill']:.3g} -> structured fraction "
            f"{pooled_row['structured_fraction']:.2f}; N_eff ≈ {pooled_row['neff']:.0f}")
    else:
        log("    ℓ not identifiable (fit ran to the max lag or found no structured sill): "
            "the field is smooth on this scale. Increase --max-h, or use the permutation "
            "test rather than an effective-N correction.")
    noisy = [r for r in rows if r["class"] == label and "split_half_noise_var" in r
             and np.isfinite(r.get("split_half_noise_var", np.nan))]
    if noisy:
        rel = np.median([r["split_half_reliability"] for r in noisy])
        nsd = np.median([np.sqrt(r["split_half_noise_var"]) for r in noisy])
        log(f"    split-half noise SD (median over {len(noisy)} surface(s)): {nsd:.3g}; "
            f"reliability {rel:.2f}")


def _plot(curves, feature, label, filename):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from .spatial_stats import VARIOGRAM_MODELS

    fig, ax = plt.subplots(figsize=(5, 4))
    for h, gamma in curves[label]["surfaces"]:
        ax.plot(h, gamma, color="0.6", alpha=0.4, lw=0.8)
    h, gamma, _counts, fit = curves[label]["pooled"]
    ax.plot(h, gamma, "o", color="#0072B2", ms=4, label="pooled")
    if fit.get("params") is not None:
        hh = np.linspace(0, h.max() if len(h) else 1, 200)
        ax.plot(hh, VARIOGRAM_MODELS[fit["model"]](hh, *fit["params"]), "-",
                color="#D55E00",
                label=f"{fit['model']} fit"
                + (f" (ℓ={fit['ell']:.3g})" if fit["ok"] else " (ℓ n.i.)"))
    ax.set_xlabel("Separation h")
    ax.set_ylabel(f"γ(h) of {feature}")
    ax.set_title(f"{label}: {feature} semivariogram")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename)
    fig.savefig(filename[:-4] + ".png", dpi=150)
    plt.close(fig)


@click.command(name="variogram")
@click.argument("configfile", type=click.Path(exists=True))
@click.option("-n", "--feature", required=True,
              help="Per-triangle column to analyze (e.g. thickness, curvedness_VV).")
@click.option("-c", "--class", "classes", multiple=True,
              help="Membrane class(es) to analyze (repeatable; default: all).")
@click.option("--filter", "filters", multiple=True,
              help="Keep only triangles matching [CLASS:]PROPERTY OP VALUE (repeatable). "
                   "Merged with config statistics.filters.")
@click.option("--max-h", type=float, default=60.0, show_default=True,
              help="Largest separation (surface units, usually nm) to include.")
@click.option("--bins", type=int, default=30, show_default=True, help="Lag bins.")
@click.option("--seeds", type=int, default=200, show_default=True,
              help="Random seed triangles per surface (more = smoother, slower).")
@click.option("--min-pairs", type=int, default=30, show_default=True,
              help="Drop lag bins with fewer pairs than this.")
@click.option("--min-triangles", type=int, default=200, show_default=True,
              help="Skip surfaces with fewer (post-filter) triangles than this.")
@click.option("--geodesic", is_flag=True, default=False,
              help="Use along-surface (geodesic) distance on the .gt graph instead of 3D "
                   "distance. Slower; needs graph-tool.")
@click.option("--classical", is_flag=True, default=False,
              help="Classical (Matheron) estimator instead of the robust Cressie-Hawkins.")
@click.option("--model", type=click.Choice(["auto", "exponential", "gaussian"]),
              default="auto", show_default=True,
              help="Correlated component of the fitted model. 'auto' keeps whichever of "
                   "exponential/gaussian fits better; gaussian suits smooth "
                   "(neighborhood-averaged) fields, where an exponential fit understates "
                   "the nugget.")
@click.option("--seed", type=int, default=0, show_default=True, help="RNG seed.")
@click.option("--output", default=None,
              help="Output CSV (default: <work_dir or config dir>/<feature>_variogram.csv).")
@click.option("--no-plots", is_flag=True, default=False, help="Skip the per-class figures.")
def variogram_cli(configfile, feature, classes, filters, max_h, bins, seeds, min_pairs,
                  min_triangles, geodesic, classical, model, seed, output, no_plots):
    """Semivariogram QC of FEATURE: correlation length, nugget, sill, and noise.

    CONFIGFILE: a pipeline config (seg_dir/work_dir) or a multi-dataset config.
    """
    from .multidataset import load_for_analysis
    from .surface_filters import describe, parse_filters

    config = load_config(configfile)
    if not config.get("datasets"):
        from .config_utils import require_keys
        require_keys(config, ("seg_dir", "work_dir", "segmentation_values"), configfile)
    try:
        ds = load_for_analysis(config, configfile)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    stats_cfg = config.get("statistics", {}) or {}
    config_filters = stats_cfg.get("filters", []) if isinstance(stats_cfg, dict) else []
    try:
        clauses = parse_filters(list(config_filters) + list(filters))
    except ValueError as exc:
        raise click.ClickException(str(exc))
    labels = list(classes) or ds.labels
    unknown = [c for c in labels if c not in ds.labels]
    if unknown:
        raise click.ClickException(f"unknown class(es) {unknown}; have {ds.labels}")

    print(f"Semivariogram of '{feature}' ({'geodesic' if geodesic else '3D'} distance, "
          f"max lag {max_h:g}, {'classical' if classical else 'robust'} estimator)")
    if clauses:
        print(f"  triangle filter: {describe(clauses)}")
    if any(k in feature for k in NEIGHBORHOOD_AVERAGED):
        print(f"  NOTE: '{feature}' is neighborhood-averaged, so its nugget is suppressed "
              "and is not a noise floor."
              + (" Use measure_thickness --noise-estimate for that."
                 if feature in NOISE_FEATURES else ""))
    try:
        rows, curves = run_variogram(ds, feature, labels=labels, filters=clauses,
                                     max_h=max_h, n_bins=bins, n_seeds=seeds,
                                     robust=not classical, min_pairs=min_pairs,
                                     min_triangles=min_triangles, geodesic=geodesic,
                                     seed=seed, model=model)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise click.ClickException(str(exc))
    if not rows:
        raise click.ClickException(
            f"no surface has a '{feature}' column with >= {min_triangles} triangles after "
            "filtering.")

    out_dir = config.get("work_dir") or (os.path.dirname(os.path.abspath(configfile)) + "/")
    out_csv = output or f"{out_dir}{feature}_variogram.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    if not no_plots:
        base = os.path.dirname(os.path.abspath(out_csv))
        for label in curves:
            if "pooled" in curves[label]:
                fn = os.path.join(base, f"{feature}_variogram_{label}.svg")
                _plot(curves, feature, label, fn)
                print(f"Wrote {fn}")


if __name__ == "__main__":
    variogram_cli()
