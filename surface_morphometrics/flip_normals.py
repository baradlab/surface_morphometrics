#! /usr/bin/env python
"""Orient surface normals consistently outward, per connected component.

EXPERIMENTAL / expert users only: the orientation heuristics are not yet validated
across datasets, and are wrong for some components by construction (see below).
Always inspect the result before using oriented surfaces for analysis.

pycurv's normal-vector voting produces normals whose direction is inherited from
the meshing step, so the sign of every curvature measurement is arbitrary per
surface: two tomograms of the same organelle can disagree on whether a bulge is
positively or negatively curved, which makes pooled curvature statistics
meaningless.

This orients each connected component so that its area-weighted mean curvature is
positive -- i.e. normals point away from the estimated inside of the organelle --
and flips every sign-sensitive quantity to match.

The heuristic is right for closed-ish compartments (OMM, IMM boundary, ER). It is
wrong for *disconnected tubular cristae*: a crista still attached to the IMM is
part of that component and inherits its orientation, but an isolated crista
fragment becomes its own component and gets oriented by its own tube geometry,
which points it the opposite way.

You cannot tell which components those are until you have looked at the result, so
the workflow is orient -> inspect -> correct:

  morphometrics flip_normals config.yml                     # 1. orient everything
  # 2. inspect <name>_oriented.vtp in ParaView (docs/normals.md)
  morphometrics manual_flip <name>_oriented.gt --labels 3,7  # 3. fix what is wrong

``manual_flip`` works on the file you just inspected, using the component ids
stored in it, and flips in either direction -- unlike ``flip_normals
--exclude-labels``, which can only suppress an automatic flip and requires
re-running from the original graph.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import click
import numpy as np

from .config_utils import load_config
from .label_connected_components import label_component_numbers


# ---------------------------------------------------------------------------
# How each pycurv property behaves under a normal flip (n -> -n)
# ---------------------------------------------------------------------------
#
# The principal curvatures do not merely change sign: pycurv keeps kappa_1 as the
# maximum and kappa_2 as the minimum (verified: kappa_1 >= kappa_2 at every
# vertex), and flipping the normal exchanges which is which. So
# kappa_1' = -kappa_2 and kappa_2' = -kappa_1, which is exactly what preserves the
# ordering invariant. Everything derived from them follows:
#
#   mean_curvature_VV  = (k1 + k2) / 2            -> negated
#   gauss_curvature_VV = k1 * k2                  -> INVARIANT ((-k2)(-k1) = k1k2)
#   curvedness_VV      = sqrt((k1^2 + k2^2) / 2)  -> INVARIANT
#   shape_index_VV     = 2/pi * atan((k1+k2)/(k1-k2)) -> negated (the denominator
#                        is unchanged, since k1' - k2' = k1 - k2)
#
# All three identities were confirmed exactly against a real AVV graph.

#: Below this |mean curvature| / curvedness ratio a component has too little net
#: curvature for the sign to be meaningful, and is reported for manual checking.
AMBIGUOUS_CONFIDENCE = 0.05

#: Scalars that simply change sign.
NEGATED_SCALARS = (
    "mean_curvature_VV",
    "mean_curvature",
    "shape_index_VV",
    "shape_index_cat",
)

#: Vectors that simply change sign (the surface normals themselves).
NEGATED_VECTORS = ("n_v", "normal", "avg_normals")

#: Scalar pairs (max, min) that swap *and* change sign: max' = -min, min' = -max.
SWAPPED_NEGATED_SCALAR_PAIRS = (
    ("kappa_1", "kappa_2"),
    ("max_curvature", "min_curvature"),
)

#: Principal-direction pairs that swap, to stay paired with their curvature.
#: Their individual signs are arbitrary (they are axes, not arrows).
SWAPPED_VECTOR_PAIRS = (("t_1", "t_2"),)

#: Properties that are genuinely unchanged by a normal flip. Listed explicitly so
#: that an unrecognized property is reported rather than silently left alone.
INVARIANT_PROPERTIES = (
    "area", "xyz", "points", "orientation_class",
    "gauss_curvature", "gauss_curvature_VV", "curvedness_VV",
    "component_number", "normals_flipped",
    # distance/orientation and thickness outputs, when already computed
    "thickness", "bilayer_resolution", "t_v",
)


def classify_properties(names):
    """Split property names into (handled, unknown).

    Any property this module does not know how to transform is returned as
    unknown so the caller can warn: silently leaving a new sign-sensitive pycurv
    property untouched would corrupt it relative to the flipped normals.
    """
    handled = set(NEGATED_SCALARS) | set(NEGATED_VECTORS) | set(INVARIANT_PROPERTIES)
    for a, b in SWAPPED_NEGATED_SCALAR_PAIRS + SWAPPED_VECTOR_PAIRS:
        handled.update((a, b))
    names = list(names)
    unknown = [n for n in names if n not in handled]
    # Distance/orientation columns are per-pair and unsigned (magnitudes), so
    # anything matching those suffixes is safe to leave alone.
    unknown = [n for n in unknown
               if not n.endswith(("_dist", "_angle", "_verticality", "_number"))]
    return [n for n in names if n in handled], unknown


def component_mean_curvatures(component_numbers, mean_curvature, area):
    """Area-weighted mean of `mean_curvature` for each component id.

    Returns {component_id: area_weighted_mean}. Component 0 (the "too small to
    keep" sentinel from label_connected_components) is included so the caller can
    decide what to do with it. NaN curvatures are ignored.
    """
    component_numbers = np.asarray(component_numbers)
    mean_curvature = np.asarray(mean_curvature, dtype=float)
    area = np.asarray(area, dtype=float)
    out = {}
    for cid in np.unique(component_numbers):
        sel = component_numbers == cid
        values, weights = mean_curvature[sel], area[sel]
        good = np.isfinite(values) & np.isfinite(weights)
        total = weights[good].sum()
        out[int(cid)] = float(np.sum(values[good] * weights[good]) / total) if total > 0 else 0.0
    return out


def component_centroid_alignment(component_numbers, xyz, normals, area):
    """Area-weighted mean of n . (x - centroid) per component, normalized to [-1, 1].

    Positive means the component's normals point away from its own centre of mass.
    This is an independent second opinion on orientation, but it assumes each
    component is star-shaped (every point visible from the centroid): it is
    decisive for closed blobs like the OMM and near-meaningless for open sheets or
    folded lamellae, where the centroid can lie on the surface itself. Judge that
    by the magnitude — values near 0 mean the test does not apply.
    """
    component_numbers = np.asarray(component_numbers)
    xyz = np.asarray(xyz, dtype=float)
    normals = np.asarray(normals, dtype=float)
    area = np.asarray(area, dtype=float)
    out = {}
    for cid in np.unique(component_numbers):
        sel = component_numbers == cid
        weights = area[sel]
        if weights.sum() <= 0:
            out[int(cid)] = 0.0
            continue
        centroid = np.sum(xyz[sel] * weights[:, None], axis=0) / weights.sum()
        offsets = xyz[sel] - centroid
        lengths = np.linalg.norm(offsets, axis=1)
        good = lengths > 0
        if not good.any():
            out[int(cid)] = 0.0
            continue
        radial = offsets[good] / lengths[good, None]
        dots = np.sum(normals[sel][good] * radial, axis=1)
        out[int(cid)] = float(np.sum(dots * weights[good]) / weights[good].sum())
    return out


def flip_confidence(component_curvatures, component_curvednesses):
    """How decisive each component's flip decision is, as |mean H| / mean curvedness.

    Curvedness is the scale of curvature present regardless of sign, so this is the
    fraction of the available signal that actually points one way. A value near 0
    means the component is balanced between inward and outward curvature and the
    decision is essentially noise -- exactly the components worth checking in
    ParaView. Returns {component_id: ratio in [0, 1]}.
    """
    out = {}
    for cid, curvature in component_curvatures.items():
        scale = component_curvednesses.get(cid, 0.0)
        out[cid] = float(abs(curvature) / scale) if scale > 0 else 0.0
    return out


def decide_flips(component_scores, exclude_labels=()):
    """Which components to flip: those whose orientation score is negative.

    `component_scores` is the chosen criterion's per-component value (mean
    curvature, or centroid alignment). `exclude_labels` are left exactly as they
    are, for components where the heuristic is wrong (cristae tubes).
    Returns {component_id: bool}.
    """
    exclude = set(int(label) for label in exclude_labels)
    return {cid: (score < 0 and cid not in exclude)
            for cid, score in component_scores.items()}


def flip_properties(properties, flip_mask):
    """Apply the normal flip to `properties` for the triangles in `flip_mask`.

    `properties` maps name -> numpy array, shape (T,) for scalars and (T, 3) for
    vectors; arrays are modified in place and the dict is returned. Missing
    properties are skipped, so this works on graphs at any pipeline stage.
    """
    flip_mask = np.asarray(flip_mask, dtype=bool)
    if not flip_mask.any():
        return properties

    for name in NEGATED_SCALARS + NEGATED_VECTORS:
        if name in properties:
            properties[name][flip_mask] *= -1.0

    for high, low in SWAPPED_NEGATED_SCALAR_PAIRS:
        if high in properties and low in properties:
            new_high = -properties[low][flip_mask]
            new_low = -properties[high][flip_mask]
            properties[high][flip_mask] = new_high
            properties[low][flip_mask] = new_low

    for first, second in SWAPPED_VECTOR_PAIRS:
        if first in properties and second in properties:
            held = properties[first][flip_mask].copy()
            properties[first][flip_mask] = properties[second][flip_mask]
            properties[second][flip_mask] = held

    return properties


def parse_exclude_labels(values, surface_name):
    """Parse repeatable --exclude-labels values into ids for one surface.

    Each value is either a bare list of ids ("3,7", applying to every surface) or
    a per-surface entry ("TE1_IMM:3,7"), matching how a surface's files are named.
    A surface matches if its base name contains the given prefix, so
    "TE1_IMM" selects TE1_IMM.AVV_rh9.gt.
    """
    ids = set()
    for value in values or ():
        text = str(value).strip()
        if not text:
            continue
        if ":" in text:
            prefix, _, id_text = text.partition(":")
            if prefix.strip() not in surface_name:
                continue
        else:
            id_text = text
        for token in id_text.replace(",", " ").split():
            try:
                ids.add(int(token))
            except ValueError:
                raise click.BadParameter(
                    f"--exclude-labels expects integer component ids, got '{token}'"
                )
    return ids


# ---------------------------------------------------------------------------
# Graph-level driver
# ---------------------------------------------------------------------------


def flip_normals_single(graph_file, output_dir, exclude_values=(), min_component_size=0,
                        dry_run=False, criterion="curvature"):
    """Orient one AVV graph's normals outward per component; save .gt/.vtp/.csv."""
    from pycurv import TriangleGraph, io
    from graph_tool import load_graph
    from graph_tool.topology import label_components
    from .intradistance_verticality import export_csv

    np.bool = bool  # pycurv/graph-tool compatibility shim (deprecated numpy alias)

    base = os.path.splitext(os.path.basename(graph_file))[0]
    print(f"Processing graph: {graph_file}")
    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    graph = tg.graph

    names = list(graph.vp.keys())
    _, unknown = classify_properties(names)
    if unknown:
        print(f"  WARNING: properties with unknown flip behavior, left untouched: "
              f"{', '.join(sorted(unknown))}")
        print("           If any of these are sign-sensitive they will now disagree "
              "with the flipped normals.")

    if "mean_curvature_VV" not in names:
        raise click.ClickException(
            f"{os.path.basename(graph_file)} has no mean_curvature_VV property; "
            "run `morphometrics pycurv` on this surface first."
        )

    # Label connected components here, so this step is self-contained after
    # pycurv or refinement (matching label_connected_components' numbering).
    raw, _ = label_components(graph, directed=False)
    component_numbers = label_component_numbers(raw.get_array(),
                                                min_size=min_component_size)

    mean_curvature = np.asarray(graph.vp["mean_curvature_VV"].get_array(), dtype=float)
    area = (np.asarray(graph.vp["area"].get_array(), dtype=float)
            if "area" in names else np.ones_like(mean_curvature))

    curvatures = component_mean_curvatures(component_numbers, mean_curvature, area)
    curvedness = (np.asarray(graph.vp["curvedness_VV"].get_array(), dtype=float)
                  if "curvedness_VV" in names else np.abs(mean_curvature))
    confidence = flip_confidence(
        curvatures, component_mean_curvatures(component_numbers, curvedness, area))
    alignments = component_centroid_alignment(
        component_numbers,
        graph.vp["xyz"].get_2d_array([0, 1, 2]).T,
        graph.vp["n_v"].get_2d_array([0, 1, 2]).T,
        area)

    scores = alignments if criterion == "centroid" else curvatures
    excluded = parse_exclude_labels(exclude_values, base)
    flips = decide_flips(scores, exclude_labels=excluded)

    n_components = int(component_numbers.max())
    print(f"  {n_components} component(s); criterion '{criterion}'; "
          f"{sum(flips.values())} to flip, {len(excluded & set(scores))} excluded")
    print(f"    {'id':>4s} {'triangles':>10s} {'mean_H':>10s} {'conf':>6s} "
          f"{'centroid':>9s}   decision")
    check_these = []
    for cid in sorted(scores):
        sizes = int(np.sum(component_numbers == cid))
        mark = "FLIP" if flips[cid] else ("kept (excluded)" if cid in excluded else "kept")
        # Flag a component when the chosen criterion is weak, or when the two
        # criteria disagree; their failure modes are independent, so disagreement
        # is a better "check this one" signal than either confidence alone.
        weak = confidence[cid] < AMBIGUOUS_CONFIDENCE
        disagrees = (curvatures[cid] > 0) != (alignments[cid] > 0)
        if (weak or disagrees) and cid not in excluded:
            check_these.append(cid)
        flag = "?" if (weak or disagrees) else " "
        print(f"    {cid:>4d} {sizes:>10d} {curvatures[cid]:>+10.5f} {confidence[cid]:>6.2f} "
              f"{alignments[cid]:>+9.3f} {flag} -> {mark}")
    if check_these:
        print(f"  NOTE: components {', '.join(str(c) for c in check_these)} are marked '?': "
              "either little net\n        curvature (confidence < "
              f"{AMBIGUOUS_CONFIDENCE}) or the curvature and centroid criteria disagree.\n"
              "        Check these in ParaView first (docs/normals.md).")

    flip_mask = np.isin(component_numbers, [cid for cid, do in flips.items() if do])
    if dry_run:
        print(f"  --dry-run: {int(flip_mask.sum())} triangles would be flipped; "
              "nothing written")
        return

    _apply_flip_to_graph(graph, flip_mask)

    component_property = graph.new_vertex_property("int")
    component_property.a = component_numbers
    graph.vp["component_number"] = component_property
    flipped_property = graph.new_vertex_property("int")
    flipped_property.a = flip_mask.astype(np.int64)
    graph.vp["normals_flipped"] = flipped_property

    out_base = os.path.join(output_dir, f"{base}_oriented")
    _save_graph_outputs(tg, out_base)
    print(f"  Flipped {int(flip_mask.sum())} of {len(flip_mask)} triangles")
    print(f"  Saved: {out_base}.gt / .vtp / .csv")


def _apply_flip_to_graph(graph, flip_mask):
    """Pull every numeric vertex property into numpy, flip it, and write it back."""
    properties, is_vector = {}, {}
    for name in graph.vp.keys():
        prop = graph.vp[name]
        if prop.value_type().startswith("vector"):
            properties[name] = prop.get_2d_array([0, 1, 2]).T.astype(float)
            is_vector[name] = True
        elif prop.value_type() in ("double", "long double", "float"):
            properties[name] = np.asarray(prop.get_array(), dtype=float)
            is_vector[name] = False

    flip_properties(properties, flip_mask)

    for name, values in properties.items():
        if is_vector[name]:
            graph.vp[name].set_2d_array(values.T)
        else:
            graph.vp[name].a = values


def _save_graph_outputs(tg, out_base):
    """Save a TriangleGraph as the standard .gt / .vtp / .csv triple."""
    from pycurv import io
    from .intradistance_verticality import export_csv

    tg.graph.save(f"{out_base}.gt")
    io.save_vtp(tg.graph_to_triangle_poly(), f"{out_base}.vtp")
    export_csv(tg, f"{out_base}.csv")


def manual_flip_single(graph_file, labels, output_dir=None):
    """Flip the listed components of an already-oriented graph, in place by default.

    This is the correction step after inspecting a `flip_normals` result: it uses
    the `component_number` property stored in the file, so the ids are exactly the
    ones that were reported and that you saw in ParaView -- no re-labelling, no
    risk of renumbering. Flipping is a toggle, so running it twice on the same
    component returns it to where it started.
    """
    from pycurv import TriangleGraph
    from graph_tool import load_graph

    np.bool = bool  # pycurv/graph-tool compatibility shim (deprecated numpy alias)

    print(f"Processing graph: {graph_file}")
    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    graph = tg.graph

    if "component_number" not in graph.vp:
        raise click.ClickException(
            f"{os.path.basename(graph_file)} has no component_number property. "
            "Run `morphometrics flip_normals` first, then apply manual_flip to its "
            "*_oriented.gt output."
        )

    component_numbers = np.asarray(graph.vp["component_number"].get_array())
    requested = sorted(set(int(label) for label in labels))
    present = set(int(c) for c in np.unique(component_numbers))
    missing = [label for label in requested if label not in present]
    if missing:
        raise click.ClickException(
            f"component(s) {', '.join(str(m) for m in missing)} are not in "
            f"{os.path.basename(graph_file)} (it has {min(present)}-{max(present)})."
        )

    flip_mask = np.isin(component_numbers, requested)
    _, unknown = classify_properties(list(graph.vp.keys()))
    if unknown:
        print(f"  WARNING: properties with unknown flip behavior, left untouched: "
              f"{', '.join(sorted(unknown))}")

    for label in requested:
        print(f"  component {label:>3d}: flipping {int(np.sum(component_numbers == label))} "
              "triangles")

    _apply_flip_to_graph(graph, flip_mask)

    # normals_flipped records orientation relative to the ORIGINAL mesh, so a manual
    # flip toggles it rather than setting it.
    if "normals_flipped" in graph.vp:
        already = np.asarray(graph.vp["normals_flipped"].get_array()).astype(bool)
    else:
        already = np.zeros(len(component_numbers), dtype=bool)
    flipped_property = graph.new_vertex_property("int")
    flipped_property.a = (already ^ flip_mask).astype(np.int64)
    graph.vp["normals_flipped"] = flipped_property

    base = os.path.splitext(os.path.basename(graph_file))[0]
    destination = output_dir or os.path.dirname(os.path.abspath(graph_file))
    out_base = os.path.join(destination, base)
    _save_graph_outputs(tg, out_base)
    print(f"  Flipped {int(flip_mask.sum())} of {len(flip_mask)} triangles")
    print(f"  Saved: {out_base}.gt / .vtp / .csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(name="flip_normals")
@click.argument("configfile", type=click.Path(exists=True))
@click.option("--graph", "graph_file", type=click.Path(exists=True), default=None,
              help="Single .gt graph to process (overrides batch mode).")
@click.option("--label", default=None,
              help="Restrict batch mode to this membrane label (e.g. IMM). Default: all.")
@click.option("--exclude-labels", "--exclude_labels", "exclude_values", multiple=True,
              metavar="[SURFACE:]IDS",
              help="Component ids to leave unflipped, e.g. '3,7' for every surface or "
                   "'TE1_IMM:3,7' for one. Repeatable. Use this for cristae tubes, where "
                   "outward-facing normals give negative mean curvature.")
@click.option("--min-size", "min_size", type=int, default=None,
              help="Minimum component size in triangles (defaults to "
                   "patch_analysis.min_component_size, else 0).")
@click.option("--output-dir", "output_dir", default=None,
              help="Output directory (defaults to work_dir from config).")
@click.option("--criterion", type=click.Choice(["curvature", "centroid"]),
              default="curvature", show_default=True,
              help="How to decide outward. 'curvature' flips until area-weighted "
                   "mean_curvature_VV is positive (local; works on open sheets). "
                   "'centroid' flips until normals point away from the component's "
                   "centre of mass (assumes a star-shaped component; decisive for closed "
                   "compartments, meaningless for sheets). Both are always reported.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report each component's mean curvature and what would flip, "
                   "without writing anything.")
def flip_normals_cli(configfile, graph_file, label, exclude_values, min_size,
                     output_dir, criterion, dry_run):
    """Orient normals outward per connected component, flipping curvature signs to match.

    CONFIGFILE: path to config.yml.

    Each connected component is flipped so its area-weighted mean curvature is
    positive. Check the result in ParaView (docs/normals.md) and re-run with
    --exclude-labels for components the heuristic gets wrong.
    """
    config = load_config(configfile, require=("work_dir",))
    work_dir = config.get("work_dir", config.get("seg_dir", "./"))
    if not work_dir.endswith("/"):
        work_dir += "/"
    if output_dir is None:
        output_dir = work_dir
    elif not output_dir.endswith("/"):
        output_dir += "/"
    os.makedirs(output_dir, exist_ok=True)

    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)
    if min_size is None:
        min_size = config.get("patch_analysis", {}).get("min_component_size", 0)

    if graph_file is not None:
        flip_normals_single(graph_file, output_dir, exclude_values, min_size, dry_run,
                            criterion)
        return

    pattern = (f"{work_dir}*_{label}.AVV_rh{radius_hit}.gt" if label
               else f"{work_dir}*.AVV_rh{radius_hit}.gt")
    graphs = [g for g in sorted(glob(pattern))
              if not g.endswith(("_oriented.gt", "_components.gt"))]
    if not graphs:
        print(f"No graphs matching {pattern}")
        return
    print(f"Found {len(graphs)} graph(s) to process")
    for path in graphs:
        flip_normals_single(path, output_dir, exclude_values, min_size, dry_run, criterion)


@click.command(name="manual_flip")
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--labels", "--exclude_labels", "labels", required=True,
              metavar="IDS",
              help="Component ids to flip, e.g. '3,7'. These are read from the file's own "
                   "component_number property, so they are the ids reported by "
                   "flip_normals and shown in ParaView.")
@click.option("--output-dir", "output_dir", default=None,
              help="Write the corrected files here instead of overwriting in place.")
def manual_flip_cli(graph_file, labels, output_dir):
    """Flip specific components of an already-oriented graph.

    GRAPH_FILE: a *_oriented.gt written by `morphometrics flip_normals`.

    Use this after inspecting the result in ParaView (docs/normals.md): it corrects
    the components the automatic criterion got wrong, in either direction, without
    re-running the whole orientation pass. Overwrites the file in place unless
    --output-dir is given; flipping the same component twice returns it to where it
    started.
    """
    ids = []
    for token in str(labels).replace(",", " ").split():
        try:
            ids.append(int(token))
        except ValueError:
            raise click.BadParameter(f"--labels expects integer component ids, got '{token}'")
    if not ids:
        raise click.BadParameter("--labels needs at least one component id")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    manual_flip_single(graph_file, ids, output_dir)


if __name__ == "__main__":
    flip_normals_cli()
