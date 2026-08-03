"""
Iterative mesh refinement for surface morphometrics pipeline.

This script implements density-guided mesh refinement that moves surface vertices
toward the true membrane center by:
1. Sampling density profiles along normal vectors
2. Fitting dual Gaussians to identify membrane leaflet peaks
3. Computing displacement vectors to center the surface on the membrane
4. Iteratively refining vertex positions with damped movement

Pipeline Citation: Barad BA*, Medina M*, Fuentes D, Wiseman RL, Grotjahn DA.
Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline.
J Cell Biol 2023.

Thickness Citation: Medina M, Chang Y-T, Rahmani H, Frank M, Khan Z, Fuentes D, Heberle FA, Waxham MN,
Barad BA, Grotjahn DA. Surface Morphometrics reveals local membrane thickness variation in organellar
subcompartments. J Cell Biol 2025.
"""

# Cap OpenMP to a single thread BEFORE importing graph-tool / pycurv. graph-tool
# loads OpenMP, and pycurv runs fork-based multiprocessing passes (fork is the
# default on Linux); forking a process that already has OpenMP threads running
# deadlocks -- the "First pass: running pycurv" hang. Mirrors run_pycurv.py and
# must run before the pycurv/graph_tool imports below.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import gc

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import spatial
from glob import glob
from pathlib import Path
import yaml
import click

from .config_utils import load_config
import vtk
import multiprocessing as mp

from pycurv import TriangleGraph
from pycurv import pycurv_io as io
from graph_tool import load_graph
from tqdm import tqdm
from matplotlib import pyplot as plt

from .sample_density import load_mrc, sample_density_single
from .measure_thickness import find_mins, dual_gaussian, monogaussian
from ._thickness_worker import (init_worker, fit_triangle_chunk_offsets,
                               init_local_thickness_worker, compute_thickness_chunk,
                               _dual_gaussian_centered, _dual_gaussian_shared_width,
                               _seed_bilayer_center, _symmetric_fit_window,
                               MIN_THICKNESS, MAX_THICKNESS)
# NOTE: curvature.run_pycurv is invoked in a subprocess (see run_pycurv_refinement),
# not imported here, so pycurv runs in a clean process rather than inheriting this
# one's memory state.


def compute_local_thicknesses_parallel(value_array, xyz, x_positions, average_radius, n_workers=None):
    """
    Compute local thickness for all triangles using parallel processing.

    Parameters
    ----------
    value_array : np.ndarray
        Density sampling data (n_triangles x n_samples)
    xyz : np.ndarray
        Triangle center coordinates (n_triangles, 3)
    x_positions : np.ndarray
        Sample positions along normal vectors
    average_radius : float
        Radius for local averaging
    n_workers : int or None
        Number of worker processes (None = use all CPUs)

    Returns
    -------
    np.ndarray
        Array of thickness values (np.nan for failed fits)
    """
    from scipy import spatial

    num_triangles = len(value_array)
    tree = spatial.cKDTree(xyz)

    # Batch query all neighbors
    distances, neighbors = tree.query(xyz, k=500, distance_upper_bound=average_radius, workers=-1)

    # Set up multiprocessing
    ctx = mp.get_context('spawn')
    if n_workers is None:
        n_workers = ctx.cpu_count()

    # Create chunks
    chunk_size = max(100, num_triangles // (n_workers * 4))
    chunks = [list(range(i, min(i + chunk_size, num_triangles)))
              for i in range(0, num_triangles, chunk_size)]

    # Run parallel computation
    with ctx.Pool(n_workers, initializer=init_local_thickness_worker,
                  initargs=(value_array, distances, neighbors, x_positions)) as pool:
        chunk_results = list(tqdm(
            pool.imap(compute_thickness_chunk, chunks),
            total=len(chunks),
            desc="  Computing local thicknesses"
        ))
        pool.close()
        pool.join()

    # Flatten results
    thicknesses = np.array([t for chunk in chunk_results for t in chunk])
    return thicknesses


def build_triangle_to_vertex_mapping(surf):
    """
    Build a mapping from triangle indices to their vertex IDs.

    Parameters
    ----------
    surf : vtkPolyData
        The VTK surface mesh

    Returns
    -------
    dict
        Dictionary mapping triangle index to list of 3 vertex IDs
    """
    triangle_vertices = {}
    for i in range(surf.GetNumberOfCells()):
        cell = surf.GetCell(i)
        pt_ids = [cell.GetPointId(j) for j in range(3)]
        triangle_vertices[i] = pt_ids
    return triangle_vertices


def compute_triangle_offsets(graph, sampling_data, x_positions, average_radius, monolayer=False,
                              use_xcorr=False, fallback_offset=None, global_fit_params=None,
                              n_workers=None):
    """
    Compute the offset from membrane center for each triangle.

    Uses locally averaged density profiles and either Gaussian fitting or
    cross-correlation to find the membrane center position. Parallelized
    using multiprocessing.

    Parameters
    ----------
    graph : graph_tool.Graph
        The triangle graph with xyz and normal vector properties
    sampling_data : pd.DataFrame
        Density sampling data with one row per triangle
    x_positions : np.ndarray
        The x positions (distances) for the sampling data columns
    average_radius : float
        Radius for local averaging of density profiles
    monolayer : bool
        If True, fit a single Gaussian (for high defocus data where bilayer
        is not resolved). If False (default), fit dual Gaussian for bilayer.
        Ignored when use_xcorr=True.
    use_xcorr : bool
        If True, use cross-correlation with global average profile to find
        offsets. More robust to noise but less informative (no sigma values).
    fallback_offset : float or None
        If provided, triangles where the Gaussian fit fell back to centroid
        (method=3) or failed entirely (method=0) will use this value instead
        of the per-triangle median. Intended for passing the global average
        midpoint so that poor-fit triangles get a reliable centering correction.
    n_workers : int or None
        Number of worker processes for the parallel Gaussian fitting. If None,
        falls back to the machine's logical CPU count (multiprocessing.cpu_count()).
        Pass the pipeline `cores` setting to respect the configured core budget.

    Returns
    -------
    tuple
        (offsets, sigmas1, sigmas2) - Arrays of center offsets and Gaussian widths.
        For monolayer or xcorr mode, sigmas2 will be NaN.
    """
    # Get coordinates and build KD-tree for local averaging
    xyz = graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()
    xyztree = spatial.cKDTree(xyz)

    num_triangles = len(sampling_data)
    thickness_arr = sampling_data.to_numpy()

    if use_xcorr:
        mode_str = "cross-correlation"
    elif monolayer:
        mode_str = "monolayer (single Gaussian)"
    else:
        mode_str = "bilayer (dual Gaussian)"
    print(f"Fitting mode: {mode_str}")

    # Compute reference profile for cross-correlation mode.
    # Use the global average of all normalized profiles as the xcorr reference.
    # This makes xcorr return LOCAL deviations from the current average position —
    # i.e., pure local sharpening.  The mean xcorr offset is zero-meaned before
    # applying to prevent systematic drift.  Global centering is handled separately
    # by the Gaussian iterations via an explicit dual Gaussian fit.
    reference = None
    if use_xcorr:
        print("  Computing global average profile for cross-correlation...")
        all_profiles = thickness_arr.copy() * -1
        all_profiles = all_profiles - all_profiles.min(axis=1, keepdims=True)
        row_sums = all_profiles.sum(axis=1, keepdims=True)
        valid_rows = row_sums.flatten() > 0
        all_profiles[valid_rows] = all_profiles[valid_rows] / (80/81 * row_sums[valid_rows])
        reference = np.mean(all_profiles[valid_rows], axis=0)

    # Batch KDTree query - query all points at once
    print(f"  Running batch KDTree query for {num_triangles} triangles...")
    distances, neighbor_indices = xyztree.query(xyz, k=500, distance_upper_bound=average_radius, workers=-1)

    if use_xcorr:
        # Serial processing for xcorr - faster than multiprocessing for simple computation
        from ._thickness_worker import _xcorr_offset
        print(f"  Computing offsets using cross-correlation (serial)...")

        results = []
        for i in tqdm(range(num_triangles), desc="  Computing offsets"):
            # Get valid neighbors
            valid_mask = distances[i] != np.inf
            l = distances[i][valid_mask]
            neighbors = neighbor_indices[i][valid_mask]

            if len(neighbors) == 0:
                results.append((0, np.nan, np.nan, 0))
                continue

            # Compute weighted average profile
            weights = 1.0 / (1.0 + l)
            dat = np.average(thickness_arr[neighbors], weights=weights, axis=0) * -1
            dat = dat - dat.min()
            dat_sum = dat.sum()

            if dat_sum == 0:
                results.append((0, np.nan, np.nan, 0))
                continue

            dat = dat / (80/81 * dat_sum)

            # Cross-correlation with reference
            offset = _xcorr_offset(dat, reference, x_positions)
            results.append((offset, np.nan, np.nan, 4))
    else:
        # Parallel fitting using multiprocessing for Gaussian fitting
        # Use spawn context explicitly to avoid conflicts with pycurv's multiprocessing
        print(f"  Fitting gaussians using multiprocessing...")
        ctx = mp.get_context('spawn')
        if n_workers is None:
            n_workers = ctx.cpu_count()
        print(f"  Using {n_workers} worker processes")

        # Create chunks of indices for each worker
        chunk_size = max(100, num_triangles // (n_workers * 4))
        chunks = [list(range(i, min(i + chunk_size, num_triangles)))
                  for i in range(0, num_triangles, chunk_size)]

        # Use initializer to share data once per worker
        with ctx.Pool(n_workers, initializer=init_worker,
                      initargs=(thickness_arr, distances, neighbor_indices, x_positions, monolayer,
                                reference, use_xcorr, global_fit_params)) as pool:
            chunk_results = list(tqdm(
                pool.imap(fit_triangle_chunk_offsets, chunks),
                total=len(chunks),
                desc="  Computing offsets"
            ))
            pool.close()
            pool.join()
            pool.terminate()  # Ensure complete cleanup before pycurv runs

        # Flatten results from chunks
        results = [r for chunk in chunk_results for r in chunk]

    offsets = np.array([r[0] for r in results])
    sigmas1 = np.array([r[1] for r in results])
    sigmas2 = np.array([r[2] for r in results])
    fit_methods = np.array([r[3] for r in results])

    # Report fit method statistics
    n_dual = np.sum(fit_methods == 1)
    n_single = np.sum(fit_methods == 2)
    n_centroid = np.sum(fit_methods == 3)
    n_xcorr = np.sum(fit_methods == 4)
    n_failed = np.sum(fit_methods == 0)
    if use_xcorr:
        print(f"  Fit methods: xcorr={n_xcorr} ({100*n_xcorr/num_triangles:.1f}%), "
              f"failed={n_failed}")
    else:
        print(f"  Fit methods: dual={n_dual} ({100*n_dual/num_triangles:.1f}%), "
              f"single={n_single} ({100*n_single/num_triangles:.1f}%), "
              f"centroid={n_centroid} ({100*n_centroid/num_triangles:.1f}%), "
              f"failed={n_failed}")

    # Count successful fits (Gaussian fits or xcorr, not centroid fallback)
    fit_success = (fit_methods == 1) | (fit_methods == 2) | (fit_methods == 4)

    # Robust offset processing to prevent divergence
    # 1. Clamp extreme offsets (more than 2x the scan range is clearly wrong)
    max_reasonable_offset = x_positions[-1]
    offsets = np.clip(offsets, -max_reasonable_offset, max_reasonable_offset)

    # 2. Replace outlier offsets with local median (among good Gaussian fits)
    valid_offsets = offsets[fit_success]
    if len(valid_offsets) > 0:
        median_offset = np.median(valid_offsets)
        mad = np.median(np.abs(valid_offsets - median_offset))
        outlier_threshold = 5 * max(mad, 0.5)
        outliers = np.abs(offsets - median_offset) > outlier_threshold
        n_outliers = np.sum(outliers & fit_success)
        if n_outliers > 0:
            print(f"  Clamped {n_outliers} outlier offsets (>{outlier_threshold:.2f} nm from median)")
            offsets[outliers] = np.clip(offsets[outliers],
                                        median_offset - outlier_threshold,
                                        median_offset + outlier_threshold)

    # 3. Triangles with failed or centroid-fallback fits get neighbor interpolation.
    # Interpolate offsets from nearby triangles that DID fit successfully, weighted
    # by distance.  This preserves local spatial structure (the bilayer may genuinely
    # be at different positions in different regions) better than a single global value.
    # Fall back to global midpoint (or per-triangle median) only when there are no
    # good-fit neighbors within 2× average_radius.
    needs_replacement = (fit_methods == 0) | (fit_methods == 3)
    n_replaced = np.sum(needs_replacement)
    if n_replaced > 0 and np.any(fit_success):
        final_fallback = fallback_offset if fallback_offset is not None else (median_offset if len(valid_offsets) > 0 else 0.0)
        good_indices = np.where(fit_success)[0]
        bad_indices = np.where(needs_replacement)[0]
        good_xyz = xyz[good_indices]
        good_offsets = offsets[good_indices]
        bad_xyz = xyz[bad_indices]

        # Build tree from good-fit triangles and query up to 2× average_radius
        good_tree = spatial.cKDTree(good_xyz)
        k = min(20, len(good_indices))
        interp_dists, interp_local_idx = good_tree.query(
            bad_xyz, k=k, distance_upper_bound=2 * average_radius
        )
        if k == 1:
            interp_dists = interp_dists[:, np.newaxis]
            interp_local_idx = interp_local_idx[:, np.newaxis]

        n_no_neighbor = 0
        for j, tri_idx in enumerate(bad_indices):
            valid = interp_dists[j] != np.inf
            if np.any(valid):
                w = 1.0 / (1.0 + interp_dists[j][valid])
                offsets[tri_idx] = np.average(good_offsets[interp_local_idx[j][valid]], weights=w)
            else:
                offsets[tri_idx] = final_fallback
                n_no_neighbor += 1

        fallback_src = f"global midpoint {final_fallback:+.3f} nm" if fallback_offset is not None else f"median {final_fallback:+.3f} nm"
        print(f"  Interpolated {n_replaced - n_no_neighbor} centroid/failed fits from good neighbors; "
              f"{n_no_neighbor} used {fallback_src}")
    elif n_replaced > 0:
        # No good fits at all — use final fallback for everything
        final_fallback = fallback_offset if fallback_offset is not None else 0.0
        offsets[needs_replacement] = final_fallback
        print(f"  No good fits available; replaced {n_replaced} with global midpoint {final_fallback:+.3f} nm")

    # 4a. For Gaussian iterations: center the offset field on the global midpoint.
    # Per-triangle dual Gaussian fits underestimate the global centering correction
    # because locally-averaged profiles are biased by the current surface offset — the
    # local fits converge to small values relative to the true global shift needed.
    # Fix: decompose offset_i = global_midpoint + (local_offset_i - mean(local_offsets))
    # This forces the mean correction to match the global midpoint (explicit centering)
    # while preserving local variation structure above the mean.
    if not use_xcorr and fallback_offset is not None:
        local_mean = np.mean(offsets)
        offsets = fallback_offset + (offsets - local_mean)
        print(f"  Global centering: mean {local_mean:+.3f} → {fallback_offset:+.3f} nm "
              f"(Δ={fallback_offset - local_mean:+.3f} nm, preserving local variation)")

    # 4b. Zero-mean xcorr offsets to remove systematic bias.
    # The xcorr reference is the global average of raw per-triangle profiles, while
    # each local comparison uses a spatially-averaged profile.  That shape mismatch
    # (and any global offset of the surface) introduces a constant bias in all xcorr
    # offsets.  Zero-meaning makes xcorr purely local sharpening — global centering
    # is handled by the Gaussian iterations via explicit dual Gaussian fit.
    if use_xcorr:
        xcorr_mask = fit_methods == 4
        if np.any(xcorr_mask):
            mean_xcorr = np.mean(offsets[xcorr_mask])
            offsets[xcorr_mask] -= mean_xcorr
            print(f"  Removed mean xcorr bias ({mean_xcorr:+.3f} nm) — local sharpening only")

    return offsets, sigmas1, sigmas2


def smooth_offset_field(offsets, xyz, smoothing_radius=10.0, iterations=1):
    """
    Spatially smooth the offset field to reduce roughness.

    Parameters
    ----------
    offsets : np.ndarray
        Raw offsets for each triangle
    xyz : np.ndarray
        Triangle center coordinates (n_triangles, 3)
    smoothing_radius : float
        Radius for neighbor averaging (nm)
    iterations : int
        Number of smoothing iterations

    Returns
    -------
    np.ndarray
        Smoothed offsets
    """
    from scipy import spatial

    tree = spatial.cKDTree(xyz)
    smoothed = offsets.copy()

    for _ in range(iterations):
        new_smoothed = np.zeros_like(smoothed)
        # Query all neighbors within radius
        neighbors_list = tree.query_ball_tree(tree, smoothing_radius)

        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) > 0:
                # Weighted average: closer neighbors have more influence
                neighbor_xyz = xyz[neighbors]
                distances = np.linalg.norm(neighbor_xyz - xyz[i], axis=1)
                weights = 1.0 / (1.0 + distances)
                new_smoothed[i] = np.average(smoothed[neighbors], weights=weights)
            else:
                new_smoothed[i] = smoothed[i]

        smoothed = new_smoothed

    return smoothed


def laplacian_smooth_mesh(surf, iterations=1, lambda_factor=0.5):
    """
    Apply Laplacian smoothing to mesh vertices.

    Moves each vertex toward the centroid of its neighbors, reducing
    high-frequency roughness while preserving overall shape.

    Parameters
    ----------
    surf : vtkPolyData
        The VTK surface mesh (modified in place)
    iterations : int
        Number of smoothing iterations
    lambda_factor : float
        Smoothing strength (0-1). Higher = more smoothing.

    Returns
    -------
    vtkPolyData
        Smoothed surface
    """
    from scipy.sparse import csr_matrix, diags
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    n_points = surf.GetNumberOfPoints()
    points = surf.GetPoints()

    positions = vtk_to_numpy(points.GetData()).astype(np.float64)

    # Build sparse normalized adjacency matrix from cell connectivity.
    # Each triangle contributes 6 directed edges (3 pairs of corner vertices).
    rows, cols = [], []
    for cell_idx in range(surf.GetNumberOfCells()):
        cell = surf.GetCell(cell_idx)
        pt_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        for j in range(len(pt_ids)):
            for k in range(len(pt_ids)):
                if j != k:
                    rows.append(pt_ids[j])
                    cols.append(pt_ids[k])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, n_points))
    row_sums = np.asarray(adj.sum(axis=1)).flatten()
    L = diags(1.0 / np.maximum(row_sums, 1e-10)) @ adj  # row-normalized: L[i,j] = 1/degree

    # Iterative Laplacian smoothing: one sparse matmul per iteration
    for _ in range(iterations):
        positions = positions + lambda_factor * (L @ positions - positions)

    # Write back using bulk VTK update
    new_pts = numpy_to_vtk(positions.astype(np.float32), deep=True)
    points.SetData(new_pts)

    return surf


def apply_vertex_displacements(surf, triangle_vertices, offsets, normals, damping_factor=0.6,
                                max_displacement=3.0, original_positions=None, max_total_offset=None):
    """
    Apply damped displacements to mesh vertices based on triangle offsets.

    For each vertex, averages the impulses from all attached triangles and
    applies a damped movement toward the membrane center.

    Parameters
    ----------
    surf : vtkPolyData
        The VTK surface mesh (modified in place)
    triangle_vertices : dict
        Mapping from triangle index to vertex IDs
    offsets : np.ndarray
        Center offset for each triangle (in nm along normal direction)
    normals : np.ndarray
        Normal vectors for each triangle (3, n_triangles)
    damping_factor : float
        Fraction of calculated impulse to apply (0-1)
    max_displacement : float
        Maximum displacement allowed per vertex per iteration (nm).
        Prevents divergence from bad fits.
    original_positions : np.ndarray or None
        Original vertex positions (n_vertices, 3). If provided along with
        max_total_offset, cumulative displacement from original is limited.
    max_total_offset : float or None
        Maximum total displacement from original surface (nm).
        Only used if original_positions is provided.

    Returns
    -------
    np.ndarray
        Array of displacement magnitudes for each vertex
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    points = surf.GetPoints()
    n_vertices = points.GetNumberOfPoints()

    # Build (n_tri, 3) vertex-index array from the triangle→vertex dict.
    # triangle_vertices covers all indices 0..n_tri-1 with exactly 3 ids each.
    n_tri = len(triangle_vertices)
    tri_vert_arr = np.array([triangle_vertices[i] for i in range(n_tri)], dtype=np.int32)
    # tri_vert_arr: (n_tri, 3)

    # Compute per-triangle impulse vectors: offsets * normals (n_tri, 3)
    impulses = offsets[:, None] * normals.T  # (n_tri, 3)

    # Accumulate impulse sums and counts per vertex via advanced indexing.
    # Each triangle contributes its impulse to 3 vertices.
    impulse_sums = np.zeros((n_vertices, 3))
    counts = np.zeros(n_vertices)
    flat_verts = tri_vert_arr.ravel()                  # (n_tri*3,)
    flat_impulses = np.repeat(impulses, 3, axis=0)     # (n_tri*3, 3)
    np.add.at(impulse_sums, flat_verts, flat_impulses)
    np.add.at(counts, flat_verts, 1)

    avg_impulses = impulse_sums / np.maximum(counts, 1)[:, None] * damping_factor

    # Vectorized per-iteration clamping
    mags = np.linalg.norm(avg_impulses, axis=1)
    clamp_iter = mags > max_displacement
    n_clamped_iter = int(clamp_iter.sum())
    if n_clamped_iter:
        scale = max_displacement / mags[clamp_iter]
        avg_impulses[clamp_iter] *= scale[:, None]

    # Bulk-read current positions via shared numpy view
    old_positions = vtk_to_numpy(points.GetData()).astype(np.float64)
    new_positions = old_positions + avg_impulses

    # Vectorized total-offset clamping
    n_clamped_total = 0
    if original_positions is not None and max_total_offset is not None:
        total_disp = new_positions - original_positions
        total_mags = np.linalg.norm(total_disp, axis=1)
        clamp_total = total_mags > max_total_offset
        n_clamped_total = int(clamp_total.sum())
        if n_clamped_total:
            scale = max_total_offset / total_mags[clamp_total]
            new_positions[clamp_total] = (original_positions[clamp_total]
                                          + total_disp[clamp_total] * scale[:, None])

    displacements = np.linalg.norm(new_positions - old_positions, axis=1)

    # Bulk-write updated positions back to VTK
    new_pts = numpy_to_vtk(new_positions.astype(np.float32), deep=True)
    points.SetData(new_pts)

    if n_clamped_iter > 0:
        print(f"  Clamped {n_clamped_iter} vertices to max per-iteration displacement of {max_displacement} nm")
    if n_clamped_total > 0:
        print(f"  Clamped {n_clamped_total} vertices to max total offset of {max_total_offset} nm from original")

    return displacements


def recompute_normals(surf):
    """
    Recompute surface normals after vertex displacement, stripping stale cell/point data.

    Removes all non-geometry arrays (curvature values, areas, etc. from prior pycurv
    output) so the saved VTP contains only updated geometry and fresh normals.

    Parameters
    ----------
    surf : vtkPolyData
        The VTK surface mesh

    Returns
    -------
    vtkPolyData
        Surface with recomputed normals and no stale auxiliary data
    """
    # Strip all stale cell and point data arrays before recomputing normals.
    # When surf was loaded from a pycurv AVV output, it carries curvature arrays
    # (kappa1, kappa2, area, etc.) that become stale after vertex displacement.
    for data in (surf.GetCellData(), surf.GetPointData()):
        names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
        for name in names:
            data.RemoveArray(name)

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(surf)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.SplittingOff()  # prevent vertex duplication at feature edges
    normals_filter.Update()
    return normals_filter.GetOutput()


def run_pycurv_refinement(vtp_file, output_base, pixel_size, radius_hit, cores=6,
                          use_gpu=False, gpu_device=None, gpu_batch_size=1024):
    """
    Run pycurv normal vector voting on a refined surface.

    Invokes curvature.run_pycurv in a fresh subprocess (see the body) so it runs
    with standalone-pycurv performance instead of inheriting the long-lived
    refinement process's memory/allocator state, which was measured to slow NVV
    3-4x per chunk.

    Parameters
    ----------
    vtp_file : str
        Path to the VTP surface file (will be copied to .surface.vtp if needed)
    output_base : str
        Base path for output files (without extension)
    pixel_size : float
        Pixel size in nm (not used - surface is already scaled)
    radius_hit : float
        Radius for normal vector voting
    cores : int
        Number of cores for parallel processing
    use_gpu : bool
        If True, dispatch to pycurv-gpu in the subprocess
    gpu_device : str or None
        Optional torch device for pycurv-gpu
    gpu_batch_size : int
        SSSP/voting batch size for pycurv-gpu

    Returns
    -------
    tuple
        (graph_file, surface_file) paths to the output files
    """
    import shutil

    # curvature.run_pycurv expects filename.surface.vtp format
    output_dir = str(Path(output_base).parent.resolve())
    if not output_dir.endswith("/"):
        output_dir += "/"
    basename = Path(output_base).name

    # Copy/rename the vtp file to match expected naming convention
    surface_vtp = f"{output_dir}{basename}.surface.vtp"
    vtp_file_resolved = str(Path(vtp_file).resolve())
    surface_vtp_resolved = str(Path(surface_vtp).resolve())
    if vtp_file_resolved != surface_vtp_resolved:
        shutil.copy(vtp_file, surface_vtp)

    # Delete all stale pycurv cached files so pycurv rebuilds from the newly
    # displaced surface rather than reusing any old cached graphs.
    # Must include NVV_rh*.gt and AVV_rh*.gt: pycurv skips NVV if NVV_rh*.gt
    # exists, and uses that stale graph for curvature estimation, which then
    # propagates wrong normals into the next iteration's profile sampling.
    # Also delete pycurv-gpu's .gpu_normals.npz Pass-1 cache for the same reason.
    stale_exts = [
        ".scaled_cleaned.gt", ".scaled_cleaned.vtp",
        f".NVV_rh{radius_hit}.gt",
        f".NVV_rh{radius_hit}.gpu_normals.npz",
        f".AVV_rh{radius_hit}.gt", f".AVV_rh{radius_hit}.vtp",
        f".AVV_rh{radius_hit}.csv", f".AVV_rh{radius_hit}_sampling.csv",
    ]
    for ext in stale_exts:
        stale = f"{output_dir}{basename}{ext}"
        if os.path.exists(stale):
            os.remove(stale)

    # Run pycurv in a FRESH SUBPROCESS rather than in-process.
    #
    # When run inline, pycurv forks its worker pool from the long-lived refinement
    # process (large resident memory, warm machine, accumulated allocator state).
    # That was measured ~3-4x slower per NVV chunk (~5.9 vs ~1.6 s/chunk) than the
    # IDENTICAL surface processed by a standalone `morphometrics pycurv` run, and
    # freeing memory / gc.collect() before the fork did not recover the speed. A
    # clean child process gets a fresh address space and its own memory arenas,
    # reproducing standalone-pycurv performance. We replicate the exact in-process
    # call (scale=1 since the surface is already in nm; min_component=0 keeps all
    # components during refinement) and stream the child's output to this process.
    import subprocess
    import sys
    runner = (
        "import os, sys\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"  # before importing graph-tool/pycurv
        "from surface_morphometrics import curvature\n"
        "surf, outdir, rh_s, cores = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])\n"
        "use_gpu = sys.argv[5] == '1'\n"
        "gpu_device = sys.argv[6] or None\n"
        "gpu_batch_size = int(sys.argv[7])\n"
        "try:\n"
        "    rh = int(rh_s)\n"          # preserve int vs float so AVV_rh<N> filenames match
        "except ValueError:\n"
        "    rh = float(rh_s)\n"
        "curvature.run_pycurv(surf, outdir, scale=1.0, radius_hit=rh,\n"
        "                     min_component=0, exclude_borders=0, cores=cores,\n"
        "                     use_gpu=use_gpu, gpu_device=gpu_device,\n"
        "                     gpu_batch_size=gpu_batch_size)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner,
         f"{basename}.surface.vtp", output_dir, str(radius_hit), str(cores),
         "1" if use_gpu else "0", gpu_device or "", str(gpu_batch_size)],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pycurv subprocess failed (exit {proc.returncode}) for "
            f"{basename}.surface.vtp"
        )

    graph_file = f"{output_base}.AVV_rh{radius_hit}.gt"
    surface_file = f"{output_base}.AVV_rh{radius_hit}.vtp"
    if not os.path.exists(graph_file):
        raise RuntimeError(
            f"pycurv subprocess finished but did not produce {graph_file}"
        )

    return graph_file, surface_file


def build_lightweight_graph(surf, output_gt_path):
    """
    Build a minimal graph_tool Graph from a displaced VTK surface.

    Extracts triangle centroids (xyz) and VTK cell normals (n_v) directly from
    the surface without running pycurv NVV.  Sufficient for the next iteration's
    density sampling and KDTree queries.  Runs in seconds vs ~28 min for NVV.

    Parameters
    ----------
    surf : vtkPolyData
        Surface with VTK-computed cell normals (output of recompute_normals).
    output_gt_path : str
        Path to write the .gt file.

    Returns
    -------
    str
        Path to the saved .gt file.
    """
    from vtk.util import numpy_support
    import graph_tool as gt

    # Extract vertex positions
    pts = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())  # (n_points, 3)

    # Extract triangle connectivity as a flat array [3, v0, v1, v2, 3, v3, ...]
    polys_data = numpy_support.vtk_to_numpy(surf.GetPolys().GetData())
    connectivity = polys_data.reshape(-1, 4)[:, 1:]  # (n_triangles, 3)
    centroids = pts[connectivity].mean(axis=1)        # (n_triangles, 3), nm units

    # Extract VTK cell normals
    cell_normals_data = surf.GetCellData().GetNormals()
    if cell_normals_data is not None:
        normals = numpy_support.vtk_to_numpy(cell_normals_data)  # (n_triangles, 3)
    else:
        normals = np.zeros((len(centroids), 3))
        normals[:, 2] = 1.0

    n_triangles = len(centroids)

    # Build minimal graph_tool Graph with only the properties sampling needs
    g = gt.Graph(directed=False)
    g.add_vertex(n_triangles)
    xyz_prop = g.new_vertex_property("vector<double>")
    n_v_prop = g.new_vertex_property("vector<double>")
    for i in range(n_triangles):
        xyz_prop[i] = centroids[i]
        n_v_prop[i] = normals[i]
    g.vp['xyz'] = xyz_prop
    g.vp['n_v'] = n_v_prop
    g.save(output_gt_path)
    return output_gt_path


def finalize_surface_with_pycurv(surface_vtp, mrc_file, output_base, pixel_size,
                                 radius_hit, sample_spacing, scan_range, angstroms,
                                 cores, average_radius, compute_thickness=True,
                                 use_gpu=False, gpu_device=None, gpu_batch_size=1024):
    """Run full pycurv on an already-refined surface to produce the final output.

    Intermediate refinement iterations skip pycurv (they build fast lightweight
    VTK-normal graphs that are not curvature-ready).  This runs the single, full
    pycurv normal-vector-voting pass on the final accepted surface so the user
    gets a clean, curvature-ready graph/surface to move forward with.  It does
    NOT move any vertices — the geometry is already final; it only computes
    curvature (and, optionally, the local-thickness distribution).

    Intended to be called exactly once, after the refinement loop, on every exit
    path (all iterations completed OR an early convergence stop).

    Parameters
    ----------
    surface_vtp : str
        Path to the final refined surface (.surface.vtp) to run pycurv on.
    mrc_file : str
        Tomogram, used only if compute_thickness is True.
    output_base : str
        Base path for the pycurv outputs (matches the final iteration).
    compute_thickness : bool
        If True, also measure the local thickness distribution on the finalized
        surface (for the convergence histogram).
    use_gpu : bool
        If True, dispatch final pycurv to pycurv-gpu.
    gpu_device : str or None
        Optional torch device for pycurv-gpu.
    gpu_batch_size : int
        SSSP/voting batch size for pycurv-gpu.

    Returns
    -------
    dict
        'graph_file', 'surface_file', and thickness stats
        ('local_thicknesses', 'mean_thickness', 'std_thickness').
    """
    print("\n=== Finalizing accepted surface with pycurv ===")
    # pycurv runs in its own subprocess (see run_pycurv_refinement), so it no longer
    # inherits this process's memory. Still collect here to shrink this process's
    # resident footprint while the pycurv subprocess and its workers run, reducing
    # overall system memory pressure on a machine loaded by the full refinement run.
    gc.collect()
    print("Running full pycurv normal vector voting for a curvature-ready final surface...")
    graph_file, surface_file = run_pycurv_refinement(
        surface_vtp, output_base, pixel_size, radius_hit, cores,
        use_gpu=use_gpu, gpu_device=gpu_device, gpu_batch_size=gpu_batch_size)

    result = {
        'graph_file': graph_file,
        'surface_file': surface_file,
        'local_thicknesses': np.array([]),
        'mean_thickness': np.nan,
        'std_thickness': np.nan,
    }

    if compute_thickness:
        print("Computing local thickness distribution on the finalized surface...")
        refined_value_array, refined_x_positions, _ = sample_density_single(
            mrc_file, graph_file,
            sample_spacing=sample_spacing, scan_range=scan_range, angstroms=angstroms)
        rtg = TriangleGraph()
        rtg.graph = load_graph(graph_file)
        refined_xyz = rtg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()
        local_thicknesses = compute_local_thicknesses_parallel(
            refined_value_array, refined_xyz, refined_x_positions, average_radius,
            n_workers=cores)
        valid = local_thicknesses[~np.isnan(local_thicknesses)]
        print(f"  Valid thickness measurements: {len(valid)}/{len(local_thicknesses)}")
        result['local_thicknesses'] = valid
        result['mean_thickness'] = np.mean(valid) if len(valid) > 0 else np.nan
        result['std_thickness'] = np.std(valid) if len(valid) > 0 else np.nan
        # Free the large density-sampling arrays / graph before returning.
        del refined_value_array, refined_xyz, local_thicknesses, rtg
        gc.collect()

    return result


def refine_mesh_iteration(graph_file, vtp_file, mrc_file, output_base, pixel_size,
                          radius_hit, average_radius, damping_factor, sample_spacing,
                          scan_range, angstroms, cores, monolayer=False,
                          original_positions=None, max_total_offset=None, use_xcorr=False,
                          smooth_offsets=True, offset_smoothing_radius=None,
                          laplacian_iterations=0, laplacian_lambda=0.5, lowpass_sigma=0,
                          run_full_pycurv=True, compute_thickness=False,
                          use_gpu=False, gpu_device=None, gpu_batch_size=1024):
    """
    Perform a single iteration of mesh refinement.

    Parameters
    ----------
    graph_file : str
        Path to input graph file
    vtp_file : str
        Path to input VTP surface file
    mrc_file : str
        Path to tomogram MRC file
    output_base : str
        Base path for output files
    pixel_size : float
        Pixel size in nm
    radius_hit : float
        Radius for normal vector voting
    average_radius : float
        Radius for local density averaging
    damping_factor : float
        Fraction of impulse to apply
    sample_spacing : float
        Distance between density samples
    scan_range : float
        Half-range for density sampling
    angstroms : bool
        Whether using angstrom units
    cores : int
        Number of CPU cores
    monolayer : bool
        If True, fit single Gaussian for high-defocus data
    original_positions : np.ndarray or None
        Original vertex positions from first iteration, for limiting total offset
    max_total_offset : float or None
        Maximum total displacement from original surface (nm)
    use_xcorr : bool
        If True, use cross-correlation instead of Gaussian fitting
    smooth_offsets : bool
        If True, spatially smooth offset field before applying
    offset_smoothing_radius : float or None
        Radius for offset smoothing. If None, uses average_radius.
    laplacian_iterations : int
        Number of Laplacian smoothing iterations after displacement (0 = disabled)
    laplacian_lambda : float
        Laplacian smoothing strength (0-1)
    lowpass_sigma : float
        Sigma in nm for 3D Gaussian low-pass filter on tomogram before sampling (0 = disabled)
    run_full_pycurv : bool
        If True, run full pycurv normal-vector voting to produce a curvature-ready
        graph/surface. If False (used for every intermediate iteration), build a
        fast lightweight VTK-normal graph instead and warn that the surface is not
        curvature-ready. The single full pycurv pass is run once on the final
        accepted surface by finalize_surface_with_pycurv() after the loop.
    compute_thickness : bool
        If True, compute local thickness distribution for all triangles (slow ~2 min).
        Only needed on the final iteration for the convergence histogram.

    Returns
    -------
    dict
        Statistics for this iteration
    """
    # Load graph and surface
    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    surf = io.load_poly(vtp_file)

    # Sample density along normals.
    # For xcorr iterations, apply the configured lowpass filter.
    # For Gaussian iterations, always sample unfiltered so the bilayer's two peaks
    # remain sharp enough for the global average dual-Gaussian fit.
    iter_lowpass_sigma = lowpass_sigma if use_xcorr else 0
    print("Sampling density along normal vectors...")
    value_array, x_positions, voxsize = sample_density_single(
        mrc_file, graph_file,
        sample_spacing=sample_spacing, scan_range=scan_range, angstroms=angstroms,
        lowpass_sigma=iter_lowpass_sigma
    )
    # Convert to DataFrame with position headers
    header = [f"{p:.4f}" for p in x_positions]
    sampling_data = pd.DataFrame(value_array, columns=header)

    # For Gaussian iterations, fit the global average profile with a dual Gaussian
    # and use its midpoint as a uniform offset for all triangles.  The global
    # average has far better SNR than any individual triangle profile, so the
    # midpoint estimate is much more reliable than the mean of per-triangle fits.
    global_center_offset = np.nan
    global_sigma1 = np.nan
    global_sigma2 = np.nan
    global_fit_params = None
    if not use_xcorr:
        print("Fitting dual Gaussian to global average profile for centering...")
        all_profiles_raw = value_array * -1
        all_profiles_raw = all_profiles_raw - all_profiles_raw.min(axis=1, keepdims=True)
        row_sums_raw = all_profiles_raw.sum(axis=1, keepdims=True)
        valid_raw = row_sums_raw.flatten() > 0
        all_profiles_raw[valid_raw] /= (80 / 81 * row_sums_raw[valid_raw])
        pre_disp_avg = np.mean(all_profiles_raw[valid_raw], axis=0)

        try:
            mid_g = len(pre_disp_avg) // 2
            lm_g = np.argmin(pre_disp_avg[:mid_g])
            rm_g = np.argmin(pre_disp_avg[mid_g:]) + mid_g
            ag = x_positions[lm_g + 2:rm_g - 2]
            bg = pre_disp_avg[lm_g + 2:rm_g - 2]
            center_seed, half_seed, n_resolved = _seed_bilayer_center(ag, bg)
            if len(ag) >= 7 and n_resolved >= 2:
                # Center + half-separation parameterization keeps the two leaflet
                # peaks apart so the global fit cannot collapse into one leaflet.
                # Restrict the fit to a window around the seeded bilayer so distant
                # density (adjacent membranes, CTF ringing) cannot pull the fit out.
                agf, bgf = _symmetric_fit_window(ag, bg, center_seed, half_seed)
                # Shared leaflet width: keeps asymmetric outer density from tilting
                # the fit and shifting the bilayer center off the true midpoint.
                p0g = [0.02, 0.02, 1.5, center_seed, half_seed, 0]
                bounds_g = ([0.005, 0.005, 0.8, center_seed - 3.0, MIN_THICKNESS / 2.0, -1],
                            [0.04,  0.04,  2.2, center_seed + 3.0, MAX_THICKNESS / 2.0,  1])
                popt_g, _ = opt.curve_fit(_dual_gaussian_shared_width, agf, bgf, p0g, bounds=bounds_g)
                center_g, half_g = popt_g[3], popt_g[4]
                c1_g, c2_g = center_g - half_g, center_g + half_g
                global_center_offset = center_g
                global_sigma1 = popt_g[2]
                global_sigma2 = popt_g[2]
                global_fit_params = (c1_g, popt_g[2], c2_g, popt_g[2])
                print(f"  Global avg profile: c1={c1_g:.3f} nm, c2={c2_g:.3f} nm, "
                      f"center={global_center_offset:+.3f} nm, "
                      f"thickness={2.0 * half_g:.3f} nm")
            else:
                print(f"  Global average bilayer not resolved (n_peaks={n_resolved}) "
                      "or too few points; per-triangle fits will self-seed.")
        except Exception as e:
            print(f"  Global avg profile dual Gaussian fit failed: {e}")

    # Compute triangle offsets from density profiles.
    # For Gaussian iterations: run per-triangle dual Gaussian fitting with the
    # global average midpoint as a reliable fallback for poor-fit triangles.
    # Xcorr sharpening in earlier iterations should have made the bilayer peaks
    # visible enough to fit, but any triangle that falls back to centroid or
    # fails entirely gets the global midpoint instead of a noisy per-triangle guess.
    print("Computing triangle offsets from density profiles...")
    gaussian_fallback = global_center_offset if (not use_xcorr and not np.isnan(global_center_offset)) else None
    if gaussian_fallback is not None:
        print(f"  Global avg midpoint {gaussian_fallback:+.3f} nm available as fallback for poor fits")
    offsets, sigmas1, sigmas2 = compute_triangle_offsets(
        tg.graph, sampling_data, x_positions, average_radius, monolayer=monolayer,
        use_xcorr=use_xcorr, fallback_offset=gaussian_fallback,
        global_fit_params=global_fit_params, n_workers=cores
    )

    # Get normal vectors and coordinates from graph
    normals = tg.graph.vp.n_v.get_2d_array([0, 1, 2])
    xyz = tg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()

    # Smooth offset field to reduce roughness
    if smooth_offsets:
        smoothing_r = offset_smoothing_radius if offset_smoothing_radius else average_radius
        print(f"Smoothing offset field (radius={smoothing_r:.1f} nm)...")
        offsets = smooth_offset_field(offsets, xyz, smoothing_radius=smoothing_r, iterations=1)

    # Build triangle to vertex mapping
    triangle_vertices = build_triangle_to_vertex_mapping(surf)

    # Apply vertex displacements
    print(f"Applying vertex displacements (damping={damping_factor})...")
    displacements = apply_vertex_displacements(
        surf, triangle_vertices, offsets, normals, damping_factor,
        original_positions=original_positions, max_total_offset=max_total_offset
    )

    # Laplacian smoothing to reduce high-frequency roughness
    if laplacian_iterations > 0:
        print(f"Applying Laplacian smoothing ({laplacian_iterations} iterations, lambda={laplacian_lambda})...")
        surf = laplacian_smooth_mesh(surf, iterations=laplacian_iterations, lambda_factor=laplacian_lambda)

    # Recompute normals
    print("Recomputing surface normals...")
    refined_surf = recompute_normals(surf)

    # Save refined surface
    refined_vtp = f"{output_base}.surface.vtp"
    io.save_vtp(refined_surf, refined_vtp)
    print(f"Saved refined surface: {refined_vtp}")

    # Full pycurv normal-vector voting is the slow step (~28 min on large surfaces),
    # so it is skipped on every intermediate iteration: we build a fast lightweight
    # graph from VTK normals instead (good enough for the next iteration's density
    # sampling and KDTree).  Full pycurv runs ONCE, on the final accepted surface,
    # via finalize_surface_with_pycurv() after the refinement loop — including when
    # the loop stops early on convergence.  Intermediate surfaces are therefore NOT
    # curvature-ready.
    if run_full_pycurv:
        print("Running pycurv normal vector voting...")
        new_graph_file, new_surface_file = run_pycurv_refinement(
            refined_vtp, output_base, pixel_size, radius_hit, cores,
            use_gpu=use_gpu, gpu_device=gpu_device, gpu_batch_size=gpu_batch_size,
        )
        sampling_csv = f"{output_base}.AVV_rh{radius_hit}_sampling.csv"
    else:
        print("Building lightweight graph from VTK normals (skipping pycurv NVV)...")
        print("  WARNING: intermediate surface — pycurv was NOT run, so this surface has")
        print("           no curvature data and is not ready for downstream analysis.")
        print("           A full pycurv pass runs automatically on the final surface;")
        print("           to use THIS mid-stage surface, run `morphometrics pycurv` on it.")
        new_graph_file = build_lightweight_graph(refined_surf, f"{output_base}.lightweight.gt")
        new_surface_file = refined_vtp
        print(f"  Lightweight graph: {new_graph_file}")
        sampling_csv = f"{output_base}.lightweight_sampling.csv"
    header = ",".join([f"{p:.4f}" for p in x_positions])
    np.savetxt(sampling_csv, sampling_data.values, delimiter=",", header=header, comments="")
    print(f"Saved sampling data: {sampling_csv}")

    # Sample density from the REFINED surface for the profile plot
    # This shows the actual state after this iteration's refinement
    print("Sampling density from refined surface for profile plot...")
    refined_value_array, refined_x_positions, _ = sample_density_single(
        mrc_file, new_graph_file,
        sample_spacing=sample_spacing, scan_range=scan_range, angstroms=angstroms
    )

    # Compute global average profile from the refined surface
    all_profiles = refined_value_array * -1  # Invert like in worker
    all_profiles = all_profiles - all_profiles.min(axis=1, keepdims=True)
    row_sums = all_profiles.sum(axis=1, keepdims=True)
    valid_rows = row_sums.flatten() > 0
    all_profiles[valid_rows] = all_profiles[valid_rows] / (80/81 * row_sums[valid_rows])
    avg_profile = np.mean(all_profiles[valid_rows], axis=0)

    # Fit dual Gaussian to average profile to get sigmas
    avg_profile_sigma1 = np.nan
    avg_profile_sigma2 = np.nan
    avg_profile_thickness = np.nan
    try:
        # Find peak and minima for fitting bounds
        ipk = refined_x_positions[np.argmax(avg_profile)]
        mid = len(avg_profile) // 2
        left_min = np.argmin(avg_profile[:mid])
        right_min = np.argmin(avg_profile[mid:]) + mid

        a = refined_x_positions[left_min+2:right_min-2]
        b = avg_profile[left_min+2:right_min-2]

        if len(a) >= 7:
            p0 = [0.02, ipk-0.5, 1.5, 0.02, ipk+0.5, 1.5, 0]
            bounds = ([0.005, ipk-6, 0.8, 0.005, ipk-1.5, 0.8, -1],
                      [0.04, ipk+1.5, 2.2, 0.04, ipk+6, 2.2, 1])
            popt, _ = opt.curve_fit(dual_gaussian, a, b, p0, bounds=bounds)
            avg_profile_sigma1 = popt[2]
            avg_profile_sigma2 = popt[5]
            avg_profile_thickness = np.abs(popt[4] - popt[1])
    except Exception:
        pass

    # Compute local averaged profiles for 50 random triangles (for visualization)
    # Load refined graph for neighbor computation
    refined_tg = TriangleGraph()
    refined_tg.graph = load_graph(new_graph_file)
    refined_xyz = refined_tg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()

    n_sample = min(50, len(refined_value_array))
    sample_indices = np.random.choice(len(refined_value_array), n_sample, replace=False)
    sample_tree = spatial.cKDTree(refined_xyz)
    sample_distances, sample_neighbors = sample_tree.query(
        refined_xyz[sample_indices], k=500, distance_upper_bound=average_radius, workers=-1
    )

    sample_profiles = []
    for i in range(n_sample):
        valid_mask = sample_distances[i] != np.inf
        l = sample_distances[i][valid_mask]
        neighbors = sample_neighbors[i][valid_mask]

        if len(neighbors) == 0:
            sample_profiles.append(None)
            continue

        weights = 1.0 / (1.0 + l)
        dat = np.average(refined_value_array[neighbors], weights=weights, axis=0) * -1
        dat = dat - dat.min()
        dat_sum = dat.sum()
        if dat_sum > 0:
            dat = dat / (80/81 * dat_sum)
        sample_profiles.append(dat)

    # Compute local thicknesses for all triangles using parallel dual Gaussian fitting.
    # Only run on the final iteration (compute_thickness=True) — it adds ~2 min per surface.
    if compute_thickness:
        local_thicknesses = compute_local_thicknesses_parallel(
            refined_value_array, refined_xyz, refined_x_positions, average_radius,
            n_workers=cores
        )
        valid_thicknesses = local_thicknesses[~np.isnan(local_thicknesses)]
        print(f"  Valid thickness measurements: {len(valid_thicknesses)}/{len(local_thicknesses)}")
    else:
        valid_thicknesses = np.array([])

    # Compute statistics
    valid_sigmas1 = sigmas1[~np.isnan(sigmas1)]
    valid_sigmas2 = sigmas2[~np.isnan(sigmas2)]
    stats = {
        'mean_sigma1': np.mean(valid_sigmas1) if len(valid_sigmas1) > 0 else np.nan,
        'mean_sigma2': np.mean(valid_sigmas2) if len(valid_sigmas2) > 0 else np.nan,
        'std_sigma1': np.std(valid_sigmas1) if len(valid_sigmas1) > 0 else np.nan,
        'std_sigma2': np.std(valid_sigmas2) if len(valid_sigmas2) > 0 else np.nan,
        'mean_offset': np.mean(np.abs(offsets)),
        'signed_mean_offset': np.mean(offsets),
        'std_offset': np.std(offsets),
        'mean_displacement': np.mean(displacements),
        'max_displacement': np.max(displacements),
        'graph_file': new_graph_file,
        'surface_file': new_surface_file,
        'avg_profile': avg_profile,
        'x_positions': refined_x_positions,
        'sample_profiles': sample_profiles,
        'local_thicknesses': valid_thicknesses,
        'mean_thickness': np.mean(valid_thicknesses) if len(valid_thicknesses) > 0 else np.nan,
        'std_thickness': np.std(valid_thicknesses) if len(valid_thicknesses) > 0 else np.nan,
        'avg_profile_sigma1': avg_profile_sigma1,
        'avg_profile_sigma2': avg_profile_sigma2,
        'avg_profile_thickness': avg_profile_thickness
    }

    # Free the large per-iteration arrays, surfaces, and graphs before returning.
    # None of these are kept in `stats`; releasing them each iteration keeps the
    # process from accumulating memory across iterations and surfaces, which is what
    # drives the swapping/slowdown when the final pycurv forks its worker pool.
    del value_array, sampling_data, refined_value_array, refined_xyz
    del surf, refined_surf, tg, refined_tg, all_profiles
    gc.collect()

    return stats


def refine_mesh(config_file, iterations=5, damping_factor=0.6, output_dir=None,
                component=None, tomogram=None, mrc_file_override=None, monolayer=None,
                max_total_offset=None, use_xcorr=None, xcorr_iterations=None,
                smooth_offsets=None, laplacian_iterations=None, laplacian_lambda=None,
                lowpass_sigma=None):
    """
    Iteratively refine mesh positions using density-guided vertex movement.

    Parameters
    ----------
    config_file : str
        Path to the config.yml file
    iterations : int
        Number of refinement iterations
    damping_factor : float
        Fraction of calculated impulse to apply (0-1)
    output_dir : str or None
        Output directory (defaults to work_dir)
    component : str or None
        Specific component to process (e.g., "OMM")
    tomogram : str or None
        Specific tomogram basename to process
    mrc_file_override : str or None
        If specified, use this MRC file instead of auto-detecting
    monolayer : bool or None
        If True, fit single Gaussian (for high-defocus data where bilayer
        is not resolved). If None, reads from config.
    max_total_offset : float or None
        Maximum total displacement from original surface (nm).
        Prevents divergence. If None, reads from config (default: 5.0).
    use_xcorr : bool or None
        If True, use cross-correlation for all iterations. Superseded by
        xcorr_iterations if specified. If None, reads from config.
    xcorr_iterations : int or None
        Number of initial iterations using cross-correlation before switching
        to dual Gaussian fitting. If None, reads from config (falls back to
        use_xcorr flag).
    smooth_offsets : bool or None
        If True, spatially smooth offset field before applying. If None, reads from config.
    laplacian_iterations : int or None
        Number of Laplacian smoothing iterations (0=disabled). If None, reads from config.
    laplacian_lambda : float or None
        Laplacian smoothing strength (0-1). If None, reads from config.
    lowpass_sigma : float or None
        Sigma in nm for 3D Gaussian low-pass filter on tomogram before sampling (0=disabled). If None, reads from config.

    Returns
    -------
    dict
        Dictionary with iteration statistics
    """
    # Load config. Refinement reads tomograms (tomo_dir) and curvature graphs
    # (work_dir), so both are required.
    config = load_config(config_file, require=("work_dir", "tomo_dir"))
    work_dir = config["work_dir"]
    tomo_dir = config["tomo_dir"]

    if output_dir is None:
        output_dir = work_dir
    elif not output_dir.endswith("/"):
        output_dir += "/"

    # Get settings from config
    curvature_config = config.get("curvature_measurements", {})
    density_config = config.get("density_sampling", config.get("thickness_measurements", {}))
    thickness_config = config.get("thickness_measurements", {})
    refinement_config = config.get("mesh_refinement", {})

    pixel_size = config.get("surface_generation", {}).get("pixel_size", 1.0)
    # Try to get pixel size from a known location or default
    # Note: pixel_size might need to be determined from the data
    radius_hit = curvature_config.get("radius_hit", 9)
    use_gpu = curvature_config.get("use_gpu", False)
    gpu_device = curvature_config.get("gpu_device")
    gpu_batch_size = curvature_config.get("gpu_batch_size", 1024)
    # Average radius: prefer mesh_refinement, fall back to thickness_measurements
    average_radius = refinement_config.get("average_radius", thickness_config.get("average_radius", 12))
    # Progressive radius reduction settings
    average_radius_decay = refinement_config.get("average_radius_decay", 1.0)  # 1.0 = no decay
    average_radius_min = refinement_config.get("average_radius_min", 3.0)  # Minimum radius
    sample_spacing = density_config.get("sample_spacing", 0.25)
    scan_range = density_config.get("scan_range", 10)
    angstroms = config.get("surface_generation", {}).get("angstroms", False)
    cores = config.get("cores", 6)

    # Override from refinement config if present
    iterations = refinement_config.get("iterations", iterations)
    damping_factor = refinement_config.get("damping_factor", damping_factor)
    # Monolayer mode: CLI override > config > default (False)
    if monolayer is None:
        monolayer = refinement_config.get("monolayer", False)
    # Max total offset: CLI override > config > default (5.0 nm)
    if max_total_offset is None:
        max_total_offset = refinement_config.get("max_total_offset", 5.0)
    # xcorr_iterations: which iterations use cross-correlation (set of 1-based iteration numbers)
    # Accepts int (first N iterations), list of iteration numbers, or None (falls back to use_xcorr)
    # Priority: xcorr_iterations arg > config xcorr_iterations > use_xcorr flag
    if xcorr_iterations is None:
        xcorr_iterations = refinement_config.get("xcorr_iterations", None)
    if xcorr_iterations is None:
        if use_xcorr is None:
            use_xcorr = refinement_config.get("use_xcorr", False)
        xcorr_iterations = set(range(1, iterations + 1)) if use_xcorr else set()
    elif isinstance(xcorr_iterations, int):
        xcorr_iterations = set(range(1, xcorr_iterations + 1))
    else:
        xcorr_iterations = set(xcorr_iterations)
    # Smoothing parameters: CLI override > config > defaults
    if smooth_offsets is None:
        smooth_offsets = refinement_config.get("smooth_offsets", True)
    offset_smoothing_radius = refinement_config.get("offset_smoothing_radius", None)  # None = use average_radius
    if laplacian_iterations is None:
        laplacian_iterations = refinement_config.get("laplacian_iterations", 1)
    if laplacian_lambda is None:
        laplacian_lambda = refinement_config.get("laplacian_lambda", 0.3)
    if lowpass_sigma is None:
        lowpass_sigma = refinement_config.get("lowpass_sigma", 0)
    convergence_threshold = refinement_config.get("convergence_threshold", None)

    # Find files to process
    if component:
        components = [component]
    else:
        components = thickness_config.get("components", list(config.get("segmentation_values", {}).keys()))

    print(f"Mesh refinement settings:")
    print(f"  Work directory: {work_dir}")
    print(f"  Tomogram directory: {tomo_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Iterations: {iterations}")
    print(f"  Damping factor: {damping_factor}")
    print(f"  Average radius: {average_radius} nm")
    if average_radius_decay < 1.0:
        print(f"  Average radius decay: {average_radius_decay} (min: {average_radius_min} nm)")
    print(f"  Radius hit: {radius_hit}")
    print(f"  Max total offset: {max_total_offset} nm")
    all_iters = set(range(1, iterations + 1))
    if xcorr_iterations == all_iters:
        print(f"  Fitting method: cross-correlation (all {iterations} iterations)")
    elif len(xcorr_iterations) == 0:
        if monolayer:
            print(f"  Fitting method: monolayer (single Gaussian)")
        else:
            print(f"  Fitting method: bilayer (dual Gaussian)")
    else:
        xcorr_list = sorted(xcorr_iterations)
        gaussian_list = sorted(all_iters - xcorr_iterations)
        print(f"  Fitting method: xcorr on iter(s) {xcorr_list}, dual Gaussian on iter(s) {gaussian_list}")
    # Smoothing settings
    if smooth_offsets:
        smooth_r = offset_smoothing_radius if offset_smoothing_radius else "same as average_radius"
        print(f"  Offset smoothing: enabled (radius={smooth_r})")
    else:
        print(f"  Offset smoothing: disabled")
    if laplacian_iterations > 0:
        print(f"  Laplacian smoothing: {laplacian_iterations} iterations, lambda={laplacian_lambda}")
    if lowpass_sigma > 0:
        print(f"  Low-pass filter: sigma={lowpass_sigma} nm (applied to tomogram)")
    if convergence_threshold is not None:
        print(f"  Convergence threshold: {convergence_threshold} nm (early stop per phase)")
    print(f"  Components: {components}")

    os.makedirs(output_dir, exist_ok=True)

    # Find tomograms first, then match graph files to each (like sample_density.py)
    all_stats = {}

    # Get list of tomograms to process
    if mrc_file_override:
        mrc_files = [mrc_file_override]
    elif tomogram:
        mrc_files = glob(f"{tomo_dir}{tomogram}*.mrc")
    else:
        mrc_files = glob(f"{tomo_dir}*.mrc")

    if not mrc_files:
        print(f"No MRC files found in {tomo_dir}")
        return all_stats

    print(f"Found {len(mrc_files)} tomogram(s) to process")

    # Mesh refinement improves every surface and is the slowest pipeline step, but
    # it now runs pycurv's normal-vector voting only ONCE per surface — on the final
    # accepted surface, after all iterations complete (or after an early convergence
    # stop).  Intermediate iterations build fast lightweight VTK-normal graphs and
    # are NOT curvature-ready.
    print("")
    print("=" * 70)
    print("NOTE: Mesh refinement is the slowest pipeline step, but it runs pycurv")
    print("      only ONCE per surface — on the final accepted surface (after all")
    print("      iterations, or after an early convergence stop). Intermediate")
    print("      per-iteration surfaces skip pycurv and are NOT curvature-ready;")
    print("      run `morphometrics pycurv` on one only if you want curvature for")
    print(f"      that specific mid-stage surface. x{len(mrc_files)} tomogram(s).")
    print("=" * 70)

    for mrc_file in mrc_files:
        mrc_basename = Path(mrc_file).stem
        print(f"\n{'='*60}")
        print(f"Processing tomogram: {mrc_basename}")
        print(f"{'='*60}")

        for comp in components:
            print(f"\n--- Component: {comp} ---")

            # Find graph files matching this tomogram and component
            pattern = f"{work_dir}{mrc_basename}*{comp}.AVV_rh{radius_hit}.gt"
            graph_files = glob(pattern)

            if not graph_files:
                print(f"  No graph files found matching: {pattern}")
                continue

            print(f"  Found {len(graph_files)} graph file(s)")

            for graph_file in graph_files:
                basename = Path(graph_file).stem.rsplit(f".AVV_rh{radius_hit}", 1)[0]
                print(f"\n  Processing: {basename}")
                print(f"  Using tomogram: {mrc_file}")

                # Get actual pixel size from MRC
                _, _, voxsize, _ = load_mrc(mrc_file, angstroms=angstroms)
                pixel_size = voxsize

                # Initialize for iteration tracking
                current_graph = graph_file
                current_vtp = graph_file.replace(".gt", ".vtp")

                # Store original vertex positions to limit total displacement
                original_surf = io.load_poly(current_vtp)
                original_points = original_surf.GetPoints()
                n_vertices = original_points.GetNumberOfPoints()
                original_positions = np.array([original_points.GetPoint(i) for i in range(n_vertices)])
                print(f"  Stored {n_vertices} original vertex positions for offset limiting")

                iteration_stats = []
                xcorr_phase_converged = False  # True when xcorr iters can be skipped

                # Sample initial profile (before any refinement) for comparison
                print("\n=== Initial profile (before refinement) ===")
                print("Sampling initial density profile...")
                init_value_array, init_x_positions, _ = sample_density_single(
                    mrc_file, current_graph,
                    sample_spacing=sample_spacing, scan_range=scan_range, angstroms=angstroms
                )
                # Compute normalized average profile
                init_profiles = init_value_array * -1
                init_profiles = init_profiles - init_profiles.min(axis=1, keepdims=True)
                init_row_sums = init_profiles.sum(axis=1, keepdims=True)
                init_valid = init_row_sums.flatten() > 0
                init_profiles[init_valid] = init_profiles[init_valid] / (80/81 * init_row_sums[init_valid])
                init_avg_profile = np.mean(init_profiles[init_valid], axis=0)

                # Compute sample profiles for initial state
                init_tg = TriangleGraph()
                init_tg.graph = load_graph(current_graph)
                init_xyz = init_tg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()

                n_sample = min(50, len(init_value_array))
                init_sample_indices = np.random.choice(len(init_value_array), n_sample, replace=False)
                init_tree = spatial.cKDTree(init_xyz)
                init_sample_dist, init_sample_neighbors = init_tree.query(
                    init_xyz[init_sample_indices], k=500, distance_upper_bound=average_radius, workers=-1
                )

                init_sample_profiles = []
                for i in range(n_sample):
                    valid_mask = init_sample_dist[i] != np.inf
                    l = init_sample_dist[i][valid_mask]
                    neighbors = init_sample_neighbors[i][valid_mask]

                    if len(neighbors) == 0:
                        init_sample_profiles.append(None)
                        continue

                    weights = 1.0 / (1.0 + l)
                    dat = np.average(init_value_array[neighbors], weights=weights, axis=0) * -1
                    dat = dat - dat.min()
                    dat_sum = dat.sum()
                    if dat_sum > 0:
                        dat = dat / (80/81 * dat_sum)
                    init_sample_profiles.append(dat)

                # Fit dual Gaussian to initial average profile to get sigmas
                init_avg_profile_sigma1 = np.nan
                init_avg_profile_sigma2 = np.nan
                init_avg_profile_thickness = np.nan
                try:
                    ipk = init_x_positions[np.argmax(init_avg_profile)]
                    mid = len(init_avg_profile) // 2
                    left_min = np.argmin(init_avg_profile[:mid])
                    right_min = np.argmin(init_avg_profile[mid:]) + mid

                    a = init_x_positions[left_min+2:right_min-2]
                    b = init_avg_profile[left_min+2:right_min-2]

                    if len(a) >= 7:
                        p0 = [0.02, ipk-0.5, 1.5, 0.02, ipk+0.5, 1.5, 0]
                        bounds = ([0.005, ipk-6, 0.8, 0.005, ipk-1.5, 0.8, -1],
                                  [0.04, ipk+1.5, 2.2, 0.04, ipk+6, 2.2, 1])
                        popt, _ = opt.curve_fit(dual_gaussian, a, b, p0, bounds=bounds)
                        init_avg_profile_sigma1 = popt[2]
                        init_avg_profile_sigma2 = popt[5]
                        init_avg_profile_thickness = np.abs(popt[4] - popt[1])
                except Exception:
                    pass

                # Skip local thickness computation for the initial surface — it's
                # only needed on the final iteration for the convergence histogram.
                init_valid_thicknesses = np.array([])

                # Store as iteration 0
                iteration_stats.append({
                    'iteration': 0,
                    'average_radius': average_radius,
                    'mean_sigma1': np.nan,
                    'mean_sigma2': np.nan,
                    'std_sigma1': np.nan,
                    'std_sigma2': np.nan,
                    'mean_offset': np.nan,
                    'signed_mean_offset': np.nan,
                    'std_offset': np.nan,
                    'mean_displacement': 0.0,
                    'max_displacement': 0.0,
                    'graph_file': current_graph,
                    'surface_file': current_vtp,
                    'avg_profile': init_avg_profile,
                    'x_positions': init_x_positions,
                    'sample_profiles': init_sample_profiles,
                    'local_thicknesses': init_valid_thicknesses,
                    'mean_thickness': np.mean(init_valid_thicknesses) if len(init_valid_thicknesses) > 0 else np.nan,
                    'std_thickness': np.std(init_valid_thicknesses) if len(init_valid_thicknesses) > 0 else np.nan,
                    'avg_profile_sigma1': init_avg_profile_sigma1,
                    'avg_profile_sigma2': init_avg_profile_sigma2,
                    'avg_profile_thickness': init_avg_profile_thickness
                })
                print("  Initial profile captured.")
                if len(init_valid_thicknesses) > 0:
                    print(f"  Initial mean thickness: {np.mean(init_valid_thicknesses):.3f} +/- {np.std(init_valid_thicknesses):.3f} nm")
                if not np.isnan(init_avg_profile_sigma1):
                    print(f"  Initial avg profile sigmas: {init_avg_profile_sigma1:.3f}, {init_avg_profile_sigma2:.3f} nm")

                # Plot initial sample profiles
                fig_init, ax_init = plt.subplots(figsize=(10, 6))
                for profile in init_sample_profiles:
                    if profile is not None:
                        ax_init.plot(init_x_positions, profile, color='gray', alpha=0.3, linewidth=0.8)
                ax_init.plot(init_x_positions, init_avg_profile, color='red', linewidth=2.5,
                            label='Global average')
                ax_init.set_xlabel('Distance from surface (nm)')
                ax_init.set_ylabel('Normalized density')
                ax_init.set_title('Local Averaged Profiles (50 random triangles) - Initial')
                ax_init.legend(loc='upper right')
                ax_init.grid(True, alpha=0.3)
                ax_init.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
                plt.tight_layout()
                init_samples_file = f"{output_dir}{basename}_samples_iter0.png"
                plt.savefig(init_samples_file, dpi=150)
                plt.close()

                for iter_num in range(1, iterations + 1):
                    # Calculate current averaging radius with progressive decay
                    # Radius decays each iteration: radius * decay^(iter-1), but not below minimum
                    current_avg_radius = max(
                        average_radius * (average_radius_decay ** (iter_num - 1)),
                        average_radius_min
                    )

                    iter_use_xcorr = iter_num in xcorr_iterations

                    # Skip remaining xcorr iterations if the xcorr phase already
                    # converged. Gaussian iterations are never skipped this way —
                    # they still perform the actual density-guided refinement.
                    if xcorr_phase_converged and iter_use_xcorr:
                        continue

                    print(f"\n=== Iteration {iter_num}/{iterations} ===")
                    if average_radius_decay < 1.0:
                        print(f"  Averaging radius: {current_avg_radius:.2f} nm")

                    # Create output base name for this iteration
                    iter_output_base = f"{output_dir}{basename}_refined_iter{iter_num}"

                    # Run refinement iteration
                    if 0 < len(xcorr_iterations) < iterations:
                        method_label = "xcorr" if iter_use_xcorr else "dual Gaussian"
                        print(f"  Fitting method this iteration: {method_label}")
                    stats = refine_mesh_iteration(
                        current_graph, current_vtp, mrc_file, iter_output_base,
                        pixel_size, radius_hit, current_avg_radius, damping_factor,
                        sample_spacing, scan_range, angstroms, cores,
                        monolayer=monolayer,
                        original_positions=original_positions,
                        max_total_offset=max_total_offset,
                        use_xcorr=iter_use_xcorr,
                        smooth_offsets=smooth_offsets,
                        offset_smoothing_radius=offset_smoothing_radius,
                        laplacian_iterations=laplacian_iterations,
                        laplacian_lambda=laplacian_lambda,
                        lowpass_sigma=lowpass_sigma,
                        run_full_pycurv=False,
                        compute_thickness=False
                    )

                    stats['iteration'] = iter_num
                    stats['average_radius'] = current_avg_radius
                    iteration_stats.append(stats)

                    # Generate cumulative profile plot showing all iterations so far
                    fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
                    cmap = plt.cm.viridis

                    for j, prev_stat in enumerate(iteration_stats):
                        if 'avg_profile' in prev_stat and 'x_positions' in prev_stat:
                            x_pos = prev_stat['x_positions']
                            profile = prev_stat['avg_profile']
                            prev_iter = prev_stat['iteration']
                            color = cmap(j / max(len(iteration_stats) - 1, 1))
                            label = "Initial" if prev_iter == 0 else f"Iter {prev_iter}"
                            linewidth = 2.0 if prev_iter == 0 or prev_iter == iter_num else 1.0
                            linestyle = '--' if prev_iter == 0 else '-'
                            alpha = 1.0 if prev_iter == iter_num else 0.6
                            ax_iter.plot(x_pos, profile, color=color, label=label,
                                        linewidth=linewidth, linestyle=linestyle, alpha=alpha)

                    ax_iter.set_xlabel('Distance from surface (nm)')
                    ax_iter.set_ylabel('Normalized density')
                    ax_iter.set_title(f'Average Density Profile - After Iteration {iter_num}')
                    ax_iter.legend(loc='upper right')
                    ax_iter.grid(True, alpha=0.3)
                    ax_iter.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

                    plt.tight_layout()
                    iter_profile_file = f"{output_dir}{basename}_profile_iter{iter_num}.png"
                    plt.savefig(iter_profile_file, dpi=150)
                    plt.close()

                    # Plot individual triangle profiles for this iteration
                    if 'sample_profiles' in stats and 'x_positions' in stats:
                        fig_samples, ax_samples = plt.subplots(figsize=(10, 6))
                        x_pos = stats['x_positions']

                        # Plot individual profiles in light gray
                        for profile in stats['sample_profiles']:
                            if profile is not None:
                                ax_samples.plot(x_pos, profile, color='gray', alpha=0.3, linewidth=0.8)

                        # Overlay the average profile in bold
                        ax_samples.plot(x_pos, stats['avg_profile'], color='red', linewidth=2.5,
                                       label='Global average')

                        ax_samples.set_xlabel('Distance from surface (nm)')
                        ax_samples.set_ylabel('Normalized density')
                        ax_samples.set_title(f'Local Averaged Profiles (50 random triangles) - Iteration {iter_num}')
                        ax_samples.legend(loc='upper right')
                        ax_samples.grid(True, alpha=0.3)
                        ax_samples.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Surface')

                        plt.tight_layout()
                        samples_file = f"{output_dir}{basename}_samples_iter{iter_num}.png"
                        plt.savefig(samples_file, dpi=150)
                        plt.close()

                    if not np.isnan(stats['mean_thickness']):
                        print(f"  Mean thickness: {stats['mean_thickness']:.3f} +/- {stats['std_thickness']:.3f} nm")
                    if not np.isnan(stats['avg_profile_sigma1']):
                        print(f"  Avg profile sigmas: {stats['avg_profile_sigma1']:.3f}, {stats['avg_profile_sigma2']:.3f} nm")
                    print(f"  Mean |offset|: {stats['mean_offset']:.3f} nm  (signed mean: {stats['signed_mean_offset']:+.3f} nm)")
                    print(f"  Mean displacement: {stats['mean_displacement']:.3f} nm")

                    # Update for next iteration
                    current_graph = stats['graph_file']
                    current_vtp = stats['surface_file']

                    # Pycurv's cleaning can change vertex count.  If it did, the
                    # original_positions baseline is stale and will cause an IndexError
                    # in apply_vertex_displacements.  Reset it to the current surface so
                    # the max_total_offset guard remains valid going forward.
                    check_surf = io.load_poly(current_vtp)
                    new_n = check_surf.GetPoints().GetNumberOfPoints()
                    if new_n != len(original_positions):
                        print(f"  Note: vertex count changed {len(original_positions)} → {new_n} "
                              f"(pycurv cleaning); resetting displacement baseline")
                        original_positions = np.array(
                            [check_surf.GetPoints().GetPoint(i) for i in range(new_n)]
                        )

                    # Early stopping check.
                    # XCorr phase: if converged, skip remaining xcorr iters but continue
                    # to the Gaussian phase (which performs the actual refinement).
                    # Gaussian phase: if converged, stop the loop entirely. The final
                    # full-pycurv pass runs after the loop regardless of how it exits,
                    # so an early stop still yields a clean, curvature-ready surface.
                    mean_offset = stats.get('mean_offset', np.nan)
                    if (convergence_threshold is not None
                            and not np.isnan(mean_offset)
                            and mean_offset < convergence_threshold):
                        remaining_xcorr = [i for i in sorted(xcorr_iterations) if i > iter_num]
                        if iter_use_xcorr and remaining_xcorr:
                            print(f"  XCorr converged (|offset|={mean_offset:.3f} nm < {convergence_threshold:.3f} nm). "
                                  f"Skipping {len(remaining_xcorr)} remaining xcorr iteration(s).")
                            xcorr_phase_converged = True
                        elif not iter_use_xcorr:
                            print(f"  Converged (|offset|={mean_offset:.3f} nm < {convergence_threshold:.3f} nm). "
                                  f"Stopping early after iteration {iter_num}; "
                                  f"running final pycurv on this surface.")
                            break

                # Finalize the accepted surface: run full pycurv ONCE so the user
                # gets a clean, curvature-ready final surface. Intermediate iterations
                # skip pycurv (lightweight VTK-normal graphs only), so this single
                # pass is what makes the output usable — and it runs on every exit
                # path here, including the early convergence stop above.
                if len(iteration_stats) > 1:
                    fin = finalize_surface_with_pycurv(
                        current_vtp, mrc_file, iter_output_base, pixel_size,
                        radius_hit, sample_spacing, scan_range, angstroms, cores,
                        current_avg_radius, compute_thickness=True,
                        use_gpu=use_gpu, gpu_device=gpu_device,
                        gpu_batch_size=gpu_batch_size)
                    final_entry = iteration_stats[-1]
                    final_entry['graph_file'] = fin['graph_file']
                    final_entry['surface_file'] = fin['surface_file']
                    final_entry['local_thicknesses'] = fin['local_thicknesses']
                    final_entry['mean_thickness'] = fin['mean_thickness']
                    final_entry['std_thickness'] = fin['std_thickness']
                    current_graph = fin['graph_file']
                    current_vtp = fin['surface_file']
                    print(f"Final curvature-ready surface: {fin['surface_file']}")
                    if not np.isnan(fin['mean_thickness']):
                        print(f"  Final mean thickness: {fin['mean_thickness']:.3f} "
                              f"+/- {fin['std_thickness']:.3f} nm")

                # Save iteration statistics (full, including iteration 0)
                stats_df_full = pd.DataFrame(iteration_stats)
                stats_file = f"{output_dir}{basename}_refinement_stats.csv"
                stats_df_full.to_csv(stats_file, index=False)
                print(f"\nSaved refinement statistics: {stats_file}")

                # Filter out iteration 0 for convergence plots (it has no fit statistics)
                stats_df = stats_df_full[stats_df_full['iteration'] > 0].copy()

                # Generate convergence plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Check fitting mode from sigma values
                is_xcorr = stats_df['mean_sigma1'].isna().all()
                is_monolayer = stats_df['mean_sigma2'].isna().all() and not is_xcorr

                # Thickness distribution histogram (final iteration)
                ax1 = axes[0, 0]
                iters = stats_df['iteration']
                # Get local thicknesses from the final iteration
                final_stats = iteration_stats[-1]
                if 'local_thicknesses' in final_stats and len(final_stats['local_thicknesses']) > 0:
                    thicknesses = final_stats['local_thicknesses']
                    ax1.hist(thicknesses, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    mean_t = np.mean(thicknesses)
                    std_t = np.std(thicknesses)
                    ax1.axvline(mean_t, color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {mean_t:.2f} nm')
                    ax1.axvline(mean_t - std_t, color='red', linestyle=':', alpha=0.7)
                    ax1.axvline(mean_t + std_t, color='red', linestyle=':', alpha=0.7)
                    ax1.set_title(f'Local Thickness Distribution (Final)\nN={len(thicknesses)}, σ={std_t:.2f} nm')
                    ax1.legend()
                else:
                    ax1.text(0.5, 0.5, 'No valid thickness\nmeasurements',
                            ha='center', va='center', transform=ax1.transAxes,
                            fontsize=12, color='gray')
                    ax1.set_title('Local Thickness Distribution')
                ax1.set_xlabel('Thickness (nm)')
                ax1.set_ylabel('Count')
                ax1.grid(True, alpha=0.3)

                # Offset plot
                ax2 = axes[0, 1]
                ax2.errorbar(iters, stats_df['mean_offset'], yerr=stats_df['std_offset'],
                            marker='o', capsize=3, color='green', label='Mean |offset|')
                if 'signed_mean_offset' in stats_df.columns:
                    ax2.plot(iters, stats_df['signed_mean_offset'], marker='s', color='orange',
                            linestyle='--', label='Signed mean offset')
                    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
                    ax2.legend(fontsize=8)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Offset (nm)')
                ax2.set_title('Center Offset Evolution')
                ax2.grid(True, alpha=0.3)

                # Displacement plot
                ax3 = axes[1, 0]
                ax3.plot(iters, stats_df['mean_displacement'], marker='o', label='Mean')
                ax3.plot(iters, stats_df['max_displacement'], marker='s', label='Max')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Displacement (nm)')
                ax3.set_title('Vertex Displacement')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Average profile sigmas evolution (from fitting the global average profile)
                ax4 = axes[1, 1]
                # Extract avg_profile_sigma values from iteration_stats (including iteration 0)
                iter_nums = []
                sigma1_vals = []
                sigma2_vals = []
                for stat in iteration_stats:
                    iter_nums.append(stat['iteration'])
                    sigma1_vals.append(stat.get('avg_profile_sigma1', np.nan))
                    sigma2_vals.append(stat.get('avg_profile_sigma2', np.nan))

                iter_nums = np.array(iter_nums)
                sigma1_vals = np.array(sigma1_vals)
                sigma2_vals = np.array(sigma2_vals)

                has_sigma1 = not np.all(np.isnan(sigma1_vals))
                has_sigma2 = not np.all(np.isnan(sigma2_vals))

                if has_sigma1 or has_sigma2:
                    if has_sigma1:
                        ax4.plot(iter_nums, sigma1_vals, marker='o', color='blue',
                                label='Sigma 1 (avg profile)')
                    if has_sigma2:
                        ax4.plot(iter_nums, sigma2_vals, marker='s', color='green',
                                label='Sigma 2 (avg profile)')
                    ax4.set_title('Average Profile Gaussian Widths')
                    ax4.legend()
                else:
                    ax4.text(0.5, 0.5, 'No valid sigma fits\nfor average profile',
                            ha='center', va='center', transform=ax4.transAxes,
                            fontsize=12, color='gray')
                    ax4.set_title('Average Profile Gaussian Widths')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Sigma (nm)')
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = f"{output_dir}{basename}_refinement_convergence.png"
                plt.savefig(plot_file, dpi=150)
                plt.close()
                print(f"Saved convergence plot: {plot_file}")

                # Generate profile evolution plot
                fig_profile, ax_profile = plt.subplots(figsize=(10, 6))

                # Use a colormap to show iteration progression
                cmap = plt.cm.viridis
                n_iters = len(iteration_stats)

                for i, iter_stat in enumerate(iteration_stats):
                    if 'avg_profile' in iter_stat and 'x_positions' in iter_stat:
                        x_pos = iter_stat['x_positions']
                        profile = iter_stat['avg_profile']
                        color = cmap(i / max(n_iters - 1, 1))
                        iter_num = iter_stat['iteration']
                        label = "Initial" if iter_num == 0 else f"Iter {iter_num}"
                        linewidth = 2.0 if iter_num == 0 or iter_num == iterations else 1.5
                        linestyle = '--' if iter_num == 0 else '-'
                        ax_profile.plot(x_pos, profile, color=color,
                                       label=label, linewidth=linewidth,
                                       linestyle=linestyle, alpha=0.8)

                ax_profile.set_xlabel('Distance from surface (nm)')
                ax_profile.set_ylabel('Normalized density')
                ax_profile.set_title('Average Density Profile Evolution')
                ax_profile.legend(loc='upper right')
                ax_profile.grid(True, alpha=0.3)
                ax_profile.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Surface')

                # Add colorbar to show iteration progression
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_iters - 1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_profile, label='Iteration')

                plt.tight_layout()
                profile_plot_file = f"{output_dir}{basename}_profile_evolution.png"
                plt.savefig(profile_plot_file, dpi=150)
                plt.close()
                print(f"Saved profile evolution plot: {profile_plot_file}")

                all_stats[basename] = iteration_stats

                # Release per-surface state and reclaim memory before the next
                # surface, so resident memory doesn't grow across surfaces
                # (IMM, OMM, ...). This keeps the next surface's final pycurv fork
                # from running under accumulated memory pressure.
                plt.close('all')
                del original_positions, original_surf, original_points
                gc.collect()

    return all_stats


@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.option('--iterations', '-n', type=int, default=5,
              help='Number of refinement iterations (default: 5)')
@click.option('--damping', '-d', type=float, default=0.6,
              help='Damping factor for vertex movement (default: 0.6)')
@click.option('--output', '-o', type=str, default=None,
              help='Output directory (defaults to work_dir from config)')
@click.option('--component', '-c', type=str, default=None,
              help='Specific component to process (e.g., OMM)')
@click.option('--tomogram', '-t', type=str, default=None,
              help='Specific tomogram basename to process')
@click.option('--mrc', '-m', type=click.Path(exists=True), default=None,
              help='Specific tomogram MRC file to use (overrides auto-detection)')
@click.option('--monolayer', is_flag=True, default=False,
              help='Fit single Gaussian for high-defocus data where bilayer is not resolved')
@click.option('--max-offset', type=float, default=None,
              help='Maximum total displacement from original surface in nm (default: 5.0)')
@click.option('--xcorr', is_flag=True, default=False,
              help='Use cross-correlation with global average instead of Gaussian fitting for all iterations')
@click.option('--xcorr-iterations', type=int, default=None,
              help='Number of initial iterations using cross-correlation, then switch to dual Gaussian fitting')
@click.option('--no-smooth', is_flag=True, default=False,
              help='Disable offset field smoothing')
@click.option('--laplacian', '-l', type=int, default=None,
              help='Number of Laplacian smoothing iterations (default: from config, typically 1)')
@click.option('--laplacian-lambda', type=float, default=None,
              help='Laplacian smoothing strength 0-1 (default: from config, typically 0.3)')
@click.option('--lowpass', type=float, default=None,
              help='3D Gaussian low-pass filter sigma in nm applied to tomogram (default: from config, 0=disabled)')
def refine_mesh_cli(configfile, iterations, damping, output, component, tomogram, mrc, monolayer, max_offset, xcorr,
                    xcorr_iterations, no_smooth, laplacian, laplacian_lambda, lowpass):
    """
    Iteratively refine mesh positions using density-guided vertex movement.

    This script moves surface vertices toward the true membrane center by:
    1. Sampling density profiles along normal vectors
    2. Fitting Gaussians or cross-correlating to find membrane center
    3. Smoothing the offset field to reduce noise
    4. Computing displacement vectors to center the surface
    5. Applying Laplacian smoothing to reduce roughness
    6. Running pycurv to refine normal vectors
    7. Iterating until convergence

    CONFIGFILE: Path to the config.yml file

    Use --monolayer for high-defocus tomograms where the bilayer cannot be
    resolved. This fits a single Gaussian instead of dual Gaussians, which is
    less accurate but still helps center the surface on the membrane.

    Use --xcorr to use cross-correlation with the global average profile instead
    of Gaussian fitting for all iterations. More robust to noise and unusual
    profile shapes, but doesn't provide sigma (peak width) values.

    Use --xcorr-iterations N to use cross-correlation for the first N iterations
    then switch to dual Gaussian fitting. Useful when the mesh starts far from
    the membrane center (xcorr aligns quickly) and you want the precision of
    Gaussian fitting for the final iterations.

    Smoothing options: By default, offset smoothing and Laplacian smoothing are
    enabled to reduce surface roughness. Use --no-smooth to disable offset
    smoothing, or --laplacian 0 to disable Laplacian smoothing.
    """
    # Convert flags to None if not set, so config can provide default
    monolayer_arg = True if monolayer else None
    xcorr_arg = True if xcorr else None
    smooth_arg = False if no_smooth else None  # False disables, None uses config
    stats = refine_mesh(
        configfile,
        iterations=iterations,
        damping_factor=damping,
        output_dir=output,
        component=component,
        tomogram=tomogram,
        mrc_file_override=mrc,
        monolayer=monolayer_arg,
        max_total_offset=max_offset,
        use_xcorr=xcorr_arg,
        xcorr_iterations=xcorr_iterations,
        smooth_offsets=smooth_arg,
        laplacian_iterations=laplacian,
        laplacian_lambda=laplacian_lambda,
        lowpass_sigma=lowpass,
    )

    print("\n" + "="*60)
    print("Refinement complete!")
    print("="*60)

    def _sigma_str(stat):
        s1, s2 = stat['mean_sigma1'], stat['mean_sigma2']
        if np.isnan(s1):
            return "N/A (xcorr)"
        elif np.isnan(s2):
            return f"{s1:.3f} nm (monolayer)"
        else:
            return f"{(s1+s2)/2:.3f} nm"

    # Print summary
    for basename, iter_stats in stats.items():
        if iter_stats:
            first = iter_stats[0]
            last = iter_stats[-1]
            print(f"\n{basename}:")
            print(f"  Initial sigma: {_sigma_str(first)}")
            print(f"  Final sigma:   {_sigma_str(last)}")
            print(f"  Initial offset: {first['mean_offset']:.3f} nm  (signed: {first.get('signed_mean_offset', float('nan')):+.3f} nm)")
            print(f"  Final offset:   {last['mean_offset']:.3f} nm  (signed: {last.get('signed_mean_offset', float('nan')):+.3f} nm)")


if __name__ == "__main__":
    refine_mesh_cli()
