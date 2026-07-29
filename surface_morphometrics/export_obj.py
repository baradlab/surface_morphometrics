#! /usr/bin/env python
"""Export a quantified surface (.vtp) to a colormapped OBJ + MTL for Blender etc.

A surface mesh from the pipeline carries per-triangle quantifications (curvature,
thickness, distances, patch ids, ...). This writes the geometry as a Wavefront
OBJ, with one chosen quantification baked into the surface color via a 1D
colormap image referenced from a paired MTL. Each triangle is flat-colored by its
value (per-triangle data is preserved exactly through per-face UVs), so the result
drops straight into Blender, MeshLab, etc.

Three files are written per surface (next to the .vtp or in --output-dir):
  <base>_<feature>.obj   geometry + per-face UVs + `mtllib`/`usemtl`
  <base>_<feature>.mtl   material referencing the colormap image via map_Kd
  <base>_<feature>.png   the colormap strip (the texture)

Usage:
  morphometrics export_obj config.yml surface.AVV_rh9.vtp --feature curvedness_VV
  morphometrics export_obj config.yml surface.vtp --list-features
  morphometrics export_obj config.yml --feature thickness          # batch over work_dir
  morphometrics export_obj config.yml surface.vtp --feature thickness --scale_to_voxels 5.0
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import click
import numpy as np
import yaml
from click.core import ParameterSource

from .config_utils import load_config


# ---------------------------------------------------------------------------
# VTP reading (vtk)
# ---------------------------------------------------------------------------


def read_surface(vtp_file):
    """Read a triangle-mesh .vtp.

    Returns (points (N,3), faces (M,3) int, cell_arrays, point_arrays) where the
    *_arrays are dicts {name: (n,) float array}. Vector arrays are split into
    per-component scalars (e.g. n_v -> n_v_x, n_v_y, n_v_z).
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # Auto-select the reader: XML PolyData (.vtp) vs legacy VTK polydata (some
    # tools / older pycurv write classic ASCII/binary polydata, sometimes even
    # with a .vtp extension, which the XML reader reads as an empty mesh).
    if vtk.vtkXMLPolyDataReader().CanReadFile(vtp_file):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetPoints() is None:
        raise click.ClickException(
            f"Could not read a non-empty mesh from {os.path.basename(vtp_file)}. "
            "The file may be empty (a surface that became empty after cleaning), "
            "corrupt, or not a polydata .vtp."
        )
    if poly.GetPolys() is None or poly.GetNumberOfPolys() == 0:
        raise click.ClickException(
            f"{os.path.basename(vtp_file)} has points but no polygons (not a "
            "triangle mesh); nothing to export."
        )

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    conn = vtk_to_numpy(poly.GetPolys().GetData())
    quads = conn.reshape(-1, 4)
    if not np.all(quads[:, 0] == 3):
        raise click.ClickException(
            f"{os.path.basename(vtp_file)} contains non-triangular faces; "
            "only triangle meshes are supported."
        )
    faces = quads[:, 1:4].astype(np.int64)

    def _scalars(field_data):
        # Single-component arrays are kept as-is; vector arrays are split into
        # per-component scalars named like the CSV export (e.g. n_v -> n_v_x/_y/_z),
        # so vector components can be used as the colored feature.
        out = {}
        for i in range(field_data.GetNumberOfArrays()):
            name = field_data.GetArrayName(i)
            arr = field_data.GetArray(i)
            if arr is None:
                continue
            data = vtk_to_numpy(arr).astype(float)
            ncomp = arr.GetNumberOfComponents()
            if ncomp == 1:
                out[name] = data
            else:
                data = data.reshape(-1, ncomp)
                suffixes = (["_x", "_y", "_z"][:ncomp] if ncomp <= 3
                            else [f"_{j}" for j in range(ncomp)])
                for j, suffix in enumerate(suffixes):
                    out[name + suffix] = data[:, j]
        return out

    cell_arrays = _scalars(poly.GetCellData())
    point_arrays = _scalars(poly.GetPointData())
    return points, faces, cell_arrays, point_arrays


def face_values(feature, faces, cell_arrays, point_arrays):
    """Per-face values for `feature` (cell data used directly; point data averaged)."""
    if feature in cell_arrays:
        return cell_arrays[feature]
    if feature in point_arrays:
        pv = point_arrays[feature]
        return np.nanmean(pv[faces], axis=1)
    available = sorted(set(cell_arrays) | set(point_arrays))
    raise click.ClickException(
        f"Feature '{feature}' not found. Available: {', '.join(available) or '(none)'}"
    )


# ---------------------------------------------------------------------------
# OBJ / MTL / colormap writing (pure; no vtk needed)
# ---------------------------------------------------------------------------


# Colormap texture layout: a `_RAMP_W`-pixel value ramp, optionally followed by a
# `_NAN_W`-pixel solid swatch that NaN faces sample.
_RAMP_W = 256
_NAN_W = 16


def write_colormap_png(path, cmap, nan_color=None, height=8):
    """Write a horizontal colormap strip image (low value -> left).

    If nan_color is given, a solid swatch of that color is appended on the right
    for NaN/unmeasured faces.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    ramp = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, _RAMP_W))[:, :3]  # (_RAMP_W, 3)
    if nan_color is not None:
        nan_rgb = np.array(to_rgb(nan_color))
        ramp = np.vstack([ramp, np.tile(nan_rgb, (_NAN_W, 1))])
    strip = np.tile(ramp[np.newaxis, :, :], (height, 1, 1))
    plt.imsave(path, strip)


def write_obj_mtl(out_base, points, faces, values, feature, cmap="viridis",
                  vmin=None, vmax=None, nan_color="lightgrey"):
    """Write <out_base>.obj/.mtl/.png, flat-coloring each face by `values`.

    NaN (unmeasured) faces are colored `nan_color` (a distinct swatch appended to
    the colormap); pass nan_color=None to instead map them to the low end.
    Returns (vmin, vmax) actually used.
    """
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if vmin is None:
        vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    span = (vmax - vmin) or 1.0
    # Map values to a texture u-coordinate. The colormap image is the ramp plus,
    # when nan_color is set, a trailing swatch that NaN faces sample.
    use_nan = nan_color is not None
    total_w = _RAMP_W + (_NAN_W if use_nan else 0)
    t = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / span, 0.0, 1.0)
    u = (t * (_RAMP_W - 1) + 0.5) / total_w
    if use_nan:
        u[~finite_mask] = (_RAMP_W + _NAN_W / 2.0) / total_w
    else:
        u[~finite_mask] = 0.5 / total_w

    obj_path = out_base + ".obj"
    mtl_path = out_base + ".mtl"
    png_path = out_base + ".png"
    mtl_name = "quant_" + feature
    write_colormap_png(png_path, cmap, nan_color=nan_color if use_nan else None)

    with open(mtl_path, "w") as m:
        m.write(f"# colormap material for '{feature}' ({cmap}, range [{vmin:.4g}, {vmax:.4g}])\n")
        m.write(f"newmtl {mtl_name}\n")
        m.write("Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n")
        m.write(f"map_Kd {os.path.basename(png_path)}\n")

    with open(obj_path, "w") as o:
        o.write("# Generated by `morphometrics export_obj`\n")
        o.write(f"# feature: {feature}  range: [{vmin:.6g}, {vmax:.6g}]  cmap: {cmap}\n")
        o.write(f"mtllib {os.path.basename(mtl_path)}\n")
        o.write(f"o {os.path.basename(out_base)}\n")
        o.write(f"usemtl {mtl_name}\n")
        # Vertices
        for p in points:
            o.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        # One texture coord per face (sampling the colormap at the face value)
        for uf in u:
            o.write(f"vt {uf:.6f} 0.5\n")
        # Faces: all three corners of face k share that face's vt index (k+1)
        for k, f in enumerate(faces):
            vt = k + 1
            a, b, c = (f + 1)
            o.write(f"f {a}/{vt} {b}/{vt} {c}/{vt}\n")
    return vmin, vmax


def coordinate_scale(source_angstroms, scale_to_angstroms=True, scale_to_voxels=None):
    """Factor converting the surface's native coordinates to the requested output units.

    `source_angstroms` is the surface's native unit (config surface_generation.angstroms:
    True for Angstroms, False for nm). With `scale_to_voxels` (a voxel size in A/px) the
    output is in voxel/pixel space; otherwise output is Angstroms or nm per
    `scale_to_angstroms`. Nothing is ever double-converted.
    """
    if scale_to_voxels is not None:
        if scale_to_voxels <= 0:
            raise click.UsageError("--scale_to_voxels must be a positive voxel size in Angstroms/px.")
        # native -> Angstroms -> voxels
        return (1.0 if source_angstroms else 10.0) / scale_to_voxels
    if scale_to_angstroms and not source_angstroms:
        return 10.0    # surface in nm -> Angstrom output
    if not scale_to_angstroms and source_angstroms:
        return 0.1     # surface in Angstrom -> nm output
    return 1.0         # already in the requested units


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(name="export_obj")
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("vtp", required=False, default=None)
@click.option("--feature", default=None,
              help="Per-triangle quantification to colormap (e.g. curvedness_VV, thickness).")
@click.option("--cmap", default="viridis", show_default=True, help="Matplotlib colormap name.")
@click.option("--vmin", type=float, default=None, help="Lower color limit (default: 2nd percentile).")
@click.option("--vmax", type=float, default=None, help="Upper color limit (default: 98th percentile).")
@click.option("--nan-color", "nan_color", default="lightgrey", show_default=True,
              help="Color for NaN/unmeasured faces (any matplotlib color); 'none' maps them to the low end.")
@click.option("--pattern", default=None,
              help="Batch glob within work_dir (default: *.AVV_rh{radius_hit}.vtp).")
@click.option("--output-dir", "output_dir", default=None,
              help="Output directory (defaults to the .vtp's directory / work_dir).")
@click.option("--list-features", is_flag=True, default=False,
              help="List the colorable per-triangle/-vertex arrays in the VTP and exit.")
@click.option("--scale_to_angstroms", "scale_to_angstroms", type=bool, default=True,
              show_default=True,
              help="Output coordinates in Angstroms (default, since downstream tools like "
                   "ChimeraX and Blender molecular nodes work in Angstrom space). The "
                   "surface's native units are read from surface_generation.angstroms and "
                   "coordinates are converted only as needed. Pass false to output in nm.")
@click.option("--scale_to_voxels", "scale_to_voxels", type=float, default=None,
              metavar="ANGSTROMS_PER_PIXEL",
              help="Output coordinates in voxel (pixel) space instead, given the tomogram's "
                   "voxel size in Angstroms/px (e.g. 5.0 doubles an nm-scale surface and "
                   "divides an Angstrom-scale one by 5). Mutually exclusive with "
                   "--scale_to_angstroms.")
@click.pass_context
def export_obj_cli(ctx, configfile, vtp, feature, cmap, vmin, vmax, nan_color, pattern, output_dir,
                   list_features, scale_to_angstroms, scale_to_voxels):
    """Export quantified surface(s) to colormapped OBJ + MTL for visualization.

    CONFIGFILE: path to config.yml.
    VTP: a surface .vtp to export; if omitted, all matching surfaces in work_dir.
    """
    if scale_to_voxels is not None and ctx.get_parameter_source("scale_to_angstroms") is not ParameterSource.DEFAULT:
        raise click.UsageError("--scale_to_voxels and --scale_to_angstroms are mutually exclusive; "
                               "--scale_to_voxels already puts the output in voxel space.")

    config = load_config(configfile, require=("work_dir",))
    work_dir = config.get("work_dir", config.get("seg_dir", "./"))
    if not work_dir.endswith("/"):
        work_dir += "/"
    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)

    # Decide the coordinate scale factor from the surface's native units (config)
    # and the requested output units, so we never double-convert.
    source_angstroms = config.get("surface_generation", {}).get("angstroms", False)
    coord_scale = coordinate_scale(source_angstroms, scale_to_angstroms, scale_to_voxels)

    # --list-features inspects one VTP and exits.
    if list_features:
        if vtp is None:
            raise click.UsageError("Provide a VTP file to --list-features.")
        _, _, cell_arrays, point_arrays = read_surface(vtp)
        click.echo(f"Per-triangle (cell) arrays: {', '.join(sorted(cell_arrays)) or '(none)'}")
        click.echo(f"Per-vertex (point) arrays:  {', '.join(sorted(point_arrays)) or '(none)'}")
        return

    if not feature:
        raise click.UsageError("--feature is required (or use --list-features to see options).")

    nan_color_arg = None if str(nan_color).lower() == "none" else nan_color

    if vtp is not None:
        vtps = [vtp]
    else:
        pattern = pattern or f"*.AVV_rh{radius_hit}.vtp"
        vtps = sorted(glob(work_dir + pattern))
        if not vtps:
            print(f"No surfaces matching {work_dir}{pattern}")
            return
    print(f"Exporting {len(vtps)} surface(s), feature '{feature}', cmap '{cmap}'")

    for vtp_file in vtps:
        points, faces, cell_arrays, point_arrays = read_surface(vtp_file)
        if coord_scale != 1.0:
            points = points * coord_scale
        values = face_values(feature, faces, cell_arrays, point_arrays)
        dest = output_dir or os.path.dirname(os.path.abspath(vtp_file))
        os.makedirs(dest, exist_ok=True)
        base = os.path.splitext(os.path.basename(vtp_file))[0]
        out_base = os.path.join(dest, f"{base}_{feature}")
        lo, hi = write_obj_mtl(out_base, points, faces, values, feature, cmap, vmin, vmax,
                               nan_color=nan_color_arg)
        print(f"  {os.path.basename(vtp_file)} -> {os.path.basename(out_base)}.obj "
              f"(range [{lo:.4g}, {hi:.4g}])")


if __name__ == "__main__":
    export_obj_cli()
