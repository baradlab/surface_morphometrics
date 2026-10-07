"""
Density sampling for surface morphometrics pipeline.

This script samples tomogram density values along normal vectors for each triangle
in a surface mesh. The resulting density profiles can be used for thickness measurements,
membrane analysis, and other downstream analyses.

Pipeline Citation: Barad BA*, Medina M*, Fuentes D, Wiseman RL, Grotjahn DA.
Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline.
J Cell Biol 2023.

Thickness Citation: Medina M, Chang Y-T, Rahmani H, Frank M, Khan Z, Fuentes D, Heberle FA, Waxham MN,
Barad BA, Grotjahn DA. Surface Morphometrics reveals local membrane thickness variation in organellar
subcompartments. J Cell Biol 2025.
"""

import numpy as np
import mrcfile
import scipy.interpolate as interp
from glob import glob
from pathlib import Path
from sys import argv
import yaml
import click

from .config_utils import load_config


def load_graph_data(filename, voxsize):
    """
    Load xyz coordinates and normal vectors from a graph-tool .gt file.

    Parameters
    ----------
    filename : str
        Path to the .gt file
    voxsize : float
        The voxel size of the mrc data (for converting surface coords to voxel coords)

    Returns
    -------
    tuple
        xyz coordinates (3, n_vertices), n_v normal vectors (3, n_vertices), and graph object
    """
    from graph_tool import load_graph  # lazy: keeps the module importable without graph-tool

    graph = load_graph(filename)

    # Get xyz coordinates and convert from surface units (nm) to voxel units
    xyz = graph.vp.xyz.get_2d_array([0, 1, 2]) / voxsize

    # Get normal vectors and scale to convert nm steps to voxel steps
    # n_v is a unit vector; dividing by voxsize converts nm distance to voxel distance
    n_v = graph.vp.n_v.get_2d_array([0, 1, 2]) / voxsize

    return xyz, n_v, graph

def load_mrc(filename, angstroms=False):
    """
    Load mrc data from an mrc file using mrcfile.

    Parameters
    ----------
    filename : str
        Name of the mrc file
    angstroms : bool
        If True, keep angstrom units; if False, convert to nm

    Returns
    -------
    tuple
        data array, data_matrix for interpolation, voxel size, and origin
    """
    with mrcfile.open(filename, permissive=True) as mrc:
        print(mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z)
        if angstroms:
            origin = (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z)
            voxsize = mrc.voxel_size.x

        else:
            origin = (mrc.header.origin.x/10, mrc.header.origin.y/10, mrc.header.origin.z/10)
            voxsize = mrc.voxel_size.x/10 # Convert from Angstroms to nm

        print(voxsize)
        data = mrc.data
        data = np.swapaxes(data,0,2)
        # data = np.flip(data, axis=2)
        print(data.shape)
        data_matrix = (np.arange(data.shape[0]),np.arange(data.shape[1]),np.arange(data.shape[2]))
    return data,data_matrix, voxsize, origin

INTERPOLATION_ORDERS = {"linear": 1, "cubic": 3}


def _interpolation_order(interpolation):
    """Spline order for a ``density_sampling.interpolation`` setting."""
    try:
        return INTERPOLATION_ORDERS[str(interpolation).lower()]
    except KeyError:
        raise ValueError(f"density_sampling interpolation must be one of "
                         f"{sorted(INTERPOLATION_ORDERS)}, got {interpolation!r}") from None


def _spline_sample(data, points, order=3, margin=8):
    """Sample ``data`` at fractional voxel ``points`` (n, 3) with a cubic B-spline.

    Linear interpolation blurs most at points halfway between voxels, which at
    ~1 nm/px is enough to merge a bilayer's two leaflets; a cubic spline does not add
    that extra, position-dependent blur. Points outside the volume are NaN, exactly
    as for linear sampling. Spline coefficients are computed only for the block of
    the tomogram the points touch (plus ``margin`` voxels, past which a cubic
    prefilter's influence has decayed below 1e-4), so memory scales with the surface's
    extent rather than the whole tomogram.
    """
    from scipy import ndimage

    shape = np.array(data.shape)
    values = np.full(len(points), np.nan)
    inside = np.all((points >= 0) & (points <= shape - 1), axis=1)
    if not inside.any():
        return values
    pts = points[inside]
    lo = np.maximum(np.floor(pts.min(axis=0)).astype(int) - margin, 0)
    hi = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + margin + 1, shape)
    block = np.asarray(data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    coeffs = ndimage.spline_filter(block, order=order, output=np.float32, mode="mirror")
    values[inside] = ndimage.map_coordinates(coeffs, (pts - lo).T, order=order,
                                             prefilter=False, mode="mirror")
    return values


def interpolate(data, data_matrix, xyz, n_v, sample_spacing=0.25, angstroms=False, scan_range=10,
                interpolation="cubic"):
    """
    Interpolate the values of the mrc data along each normal vector.

    Stepping from -scan_range to +scan_range in parameterized steps, interpolate
    the values of the mrc data along each normal vector using scipy.interpn.

    Parameters
    ----------
    data : ndarray
        The mrc data
    data_matrix : tuple
        Coordinate arrays for interpolation
    xyz : ndarray
        The xyz coordinates of the faces
    n_v : ndarray
        The normal vectors
    sample_spacing : float
        Distance in nm between samples (default: 0.25)
    angstroms : bool
        If True, scale samples to angstroms
    scan_range : float
        Half-range in nm to scan along normal vectors (default: 10)
    interpolation : str
        "cubic" (default; cubic B-spline) or "linear" (trilinear, the only
        behavior in earlier versions).

    Returns
    -------
    ndarray
        Interpolated values array (n_triangles x nsamples)
    """
    nsamples = int(2 * scan_range / sample_spacing) + 1
    samples = np.linspace(-scan_range, scan_range, nsamples)
    if angstroms:
        samples = samples * 10.

    # Build all query points in one vectorized step.
    # xyz: (3, n_tri), n_v: (3, n_tri), samples: (nsamples,)
    # all_points[s, i, :] = xyz[:, i] + samples[s] * n_v[:, i]
    n_tri = xyz.shape[1]
    all_points = xyz.T[None, :, :] + samples[:, None, None] * n_v.T[None, :, :]
    # shape: (nsamples, n_tri, 3) → flatten to (nsamples*n_tri, 3)
    all_points_flat = all_points.reshape(-1, 3)

    # fill_value=np.nan (NOT None, which would extrapolate): a sample outside the
    # tomogram has no measured density, and linear extrapolation from the edge is
    # unbounded, so it would silently invent values. Membranes lying flat near the
    # top or bottom of a thin tomogram scan straight out of the volume, so this is a
    # routine case, not an edge case. Downstream, a profile containing NaN is
    # excluded from neighborhood averaging (see _thickness_worker.usable_profile_rows).
    order = _interpolation_order(interpolation)
    if order == 1:
        values_flat = interp.interpn(data_matrix, data, all_points_flat,
                                     method="linear", bounds_error=False, fill_value=np.nan)
    else:
        values_flat = _spline_sample(data, all_points_flat, order=order)
    # reshape (nsamples, n_tri) then transpose to (n_tri, nsamples)
    value_array = values_flat.reshape(nsamples, n_tri).T

    outside = ~np.isfinite(value_array)
    if outside.any():
        partly = np.count_nonzero(outside.any(axis=1))
        wholly = np.count_nonzero(outside.all(axis=1))
        print(f"  {partly} of {n_tri} triangles ({100.0 * partly / n_tri:.1f}%) have "
              f"linescans reaching outside the tomogram; {wholly} fall outside entirely.")
        print("  Those samples are NaN and are excluded from thickness measurement "
              "rather than extrapolated.")
    return value_array


def sample_density_single(mrc_file, graph_file, sample_spacing=0.25, scan_range=10, angstroms=False,
                          lowpass_sigma=0, interpolation="cubic"):
    """
    Sample density values for a single graph file.

    This is a reusable function that can be called by other scripts.

    Parameters
    ----------
    mrc_file : str
        Path to the tomogram MRC file
    graph_file : str
        Path to the graph-tool .gt file
    sample_spacing : float
        Distance in nm between samples (default: 0.25)
    scan_range : float
        Half-range in nm to scan along normal vectors (default: 10)
    angstroms : bool
        If True, use angstrom units (default: False)
    lowpass_sigma : float
        Sigma in nm for 3D Gaussian low-pass filter applied to tomogram before
        sampling. Set to 0 to disable filtering (default: 0).
    interpolation : str
        "cubic" (default) or "linear"; see :func:`interpolate`.

    Returns
    -------
    tuple
        (value_array, x_positions, voxsize) where value_array is (n_triangles x nsamples),
        x_positions is the array of sample positions, and voxsize is the voxel size
    """
    # Load MRC data
    data, data_matrix, voxsize, origin = load_mrc(mrc_file, angstroms=angstroms)

    # Apply 3D low-pass filter to tomogram if requested
    if lowpass_sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Convert sigma from nm to voxels
        sigma_voxels = lowpass_sigma / voxsize
        print(f"  Applying 3D low-pass filter (sigma={lowpass_sigma} nm = {sigma_voxels:.2f} voxels)...")
        data = gaussian_filter(data, sigma=sigma_voxels)

    # Load graph data
    xyz, n_v, graph = load_graph_data(graph_file, voxsize)

    # Sample density along normals
    value_array = interpolate(data, data_matrix, xyz, n_v,
                              sample_spacing=sample_spacing, angstroms=angstroms,
                              scan_range=scan_range, interpolation=interpolation)

    # Generate x positions
    nsamples = int(2 * scan_range / sample_spacing) + 1
    x_positions = np.linspace(-scan_range, scan_range, nsamples)
    if angstroms:
        x_positions = x_positions * 10.

    return value_array, x_positions, voxsize


def sample_density_for_tomogram(filename, work_dir, angstroms=False, sample_spacing=0.25, scan_range=10, radius_hit=None,
                                interpolation="cubic"):
    """
    Sample density values from a tomogram along surface normal vectors.

    Parameters
    ----------
    filename : str
        Path to the tomogram MRC file
    work_dir : str
        Working directory containing the graph-tool .gt files
    angstroms : bool
        If True, use angstrom units
    sample_spacing : float
        Distance in nm between samples (default: 0.25)
    scan_range : float
        Half-range in nm to scan along normal vectors
    radius_hit : int or None
        If specified, only process files with this radius_hit value
    interpolation : str
        "cubic" (default) or "linear"; see :func:`interpolate`.
    """
    mrcbase = filename.split(".mrc")[0].split("/")[-1]
    print(f"Processing {mrcbase}")

    # Build glob pattern based on radius_hit - look for .gt files
    if radius_hit is not None:
        files = glob(work_dir + mrcbase + f"*.AVV_rh{radius_hit}.gt")
    else:
        files = glob(work_dir + mrcbase + f"*.AVV_rh*.gt")

    if not files:
        print(f"No graph files (.gt) found for {mrcbase}")
        return

    print(f"Found {len(files)} files to process")

    # Process each graph file
    for file in files:
        print(f"Processing {file}")
        value_array, positions, voxsize = sample_density_single(
            filename, file,
            sample_spacing=sample_spacing, scan_range=scan_range, angstroms=angstroms,
            interpolation=interpolation
        )
        # Save the interpolated values to a csv file (same basename as .gt file)
        header = ",".join([f"{p:.4f}" for p in positions])
        output_file = file[:-3] + "_sampling.csv"
        print(f"Saving to {output_file}")
        np.savetxt(output_file, value_array, delimiter=",", header=header, comments="")

@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.argument('mrcfile', required=False, default=None)
@click.option('--sample_spacing', type=float, default=None, help='Distance in nm between samples (overrides config)')
@click.option('--scan_range', type=float, default=None, help='Half-range in nm to scan (overrides config)')
def sample_density_cli(configfile, mrcfile, sample_spacing, scan_range):
    """
    Sample tomogram density along surface normal vectors.

    For each triangle in the surface meshes, samples density values from the
    tomogram along the normal vector. Output can be used for thickness
    measurements and other downstream analyses.

    CONFIGFILE: Path to the config.yml file

    MRCFILE: Optional path to a specific tomogram MRC file to process.
             If not provided, processes all MRC files in tomo_dir.
    """
    run_sample_density(configfile, mrcfile, sample_spacing_override=sample_spacing,
                       scan_range_override=scan_range)


def run_sample_density(configfile, mrcfile=None, sample_spacing_override=None, scan_range_override=None):
    """
    Main function to run density sampling from config file.

    Parameters
    ----------
    configfile : str
        Path to config.yml
    mrcfile : str or None
        Optional specific tomogram MRC file to process
    sample_spacing_override : float or None
        Override sample_spacing from config
    scan_range_override : float or None
        Override scan_range from config
    """
    # Load config. tomo_dir (tomogram MRCs) and work_dir (curvature CSVs + sampling
    # output) are both required for density sampling.
    config = load_config(configfile, require=("work_dir", "tomo_dir"))
    tomo_dir = config["tomo_dir"]
    work_dir = config["work_dir"]

    # Get density sampling settings from config (falls back to thickness_measurements for compatibility)
    density_config = config.get("density_sampling", config.get("thickness_measurements", {}))
    angstroms = config.get("surface_generation", {}).get("angstroms", False)
    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", None)

    # Set sampling parameters (CLI overrides take precedence)
    sample_spacing = sample_spacing_override if sample_spacing_override is not None else density_config.get("sample_spacing", 0.25)
    scan_range = scan_range_override if scan_range_override is not None else density_config.get("scan_range", 10)
    interpolation = density_config.get("interpolation", "cubic")
    _interpolation_order(interpolation)  # fail fast on a typo, before any tomogram loads
    nsamples = int(2 * scan_range / sample_spacing) + 1

    # Warn if sample_spacing doesn't divide evenly into scan_range
    if (scan_range % sample_spacing) != 0:
        print(f"WARNING: sample_spacing ({sample_spacing}) does not divide evenly into scan_range ({scan_range}).")
        print(f"         Actual spacing will be {2 * scan_range / (nsamples - 1):.4f} nm.")

    print(f"Density sampling settings:")
    print(f"  Tomogram directory: {tomo_dir}")
    print(f"  Work directory: {work_dir}")
    print(f"  Angstroms: {angstroms}")
    print(f"  Sample spacing: {sample_spacing} nm")
    print(f"  Scan range: {scan_range} nm")
    print(f"  N samples: {nsamples} (computed from spacing and range)")
    print(f"  Interpolation: {interpolation}")
    if radius_hit:
        print(f"  Radius hit: {radius_hit}")

    # Determine which MRC files to process
    if mrcfile:
        mrcs = [mrcfile]
    else:
        mrcs = glob(tomo_dir + "*.mrc")

    if not mrcs:
        print(f"No MRC files found in {tomo_dir}")
        return

    print(f"Found {len(mrcs)} MRC file(s) to process")

    # Process each MRC file
    for mrc in mrcs:
        sample_density_for_tomogram(mrc, work_dir, angstroms=angstroms, sample_spacing=sample_spacing,
                                    scan_range=scan_range, radius_hit=radius_hit,
                                    interpolation=interpolation)


if __name__ == "__main__":
    sample_density_cli()
