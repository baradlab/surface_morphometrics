"""
Membrane thickness measurement for surface morphometrics pipeline.

Analyze density sampling CSV files to measure membrane thickness by fitting
dual gaussians to density profiles. Writes per-triangle thickness back into the
surface graph/.vtp/.csv in place, and generates summary statistics and plots.

Pipeline Citation: Barad BA*, Medina M*, Fuentes D, Wiseman RL, Grotjahn DA.
Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline.
J Cell Biol 2023.

Thickness Citation: Medina M, Chang Y-T, Rahmani H, Frank M, Khan Z, Fuentes D, Heberle FA, Waxham MN,
Barad BA, Grotjahn DA. Surface Morphometrics reveals local membrane thickness variation in organellar
subcompartments. J Cell Biol 2025.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
import scipy.signal as signal
import scipy.stats as stats
from scipy import spatial
from glob import glob
from pathlib import Path
import os
import yaml
import click

from .config_utils import load_config
import multiprocessing as mp

from tqdm import tqdm
from .morphometrics_stats import histogram

def find_mins(y):
    """Find the indices of minimum values on left and right halves of the curve."""
    mid = int(np.round(len(y)/2))
    left_side = np.argmin(y[:mid])
    right_side = np.argmin(y[mid:]) + mid
    return left_side, right_side, y

def sinc(x, A, mu, sigma):
    """Squared sinc peak of amplitude A centered at mu with width parameter sigma."""
    return A * (np.sin(np.pi*(x-mu)*sigma) / (np.pi*(x-mu)*sigma))**2

def dual_sinc(x, p):
    """Sum of two squared-sinc peaks plus an offset; p = (a1, mu1, s1, a2, mu2, s2, offset)."""
    return sinc(x,*p[0:3])+sinc(x,*p[3:6])+p[6]

def gauss(x, p):
    """Unit-area Gaussian; p = (mean, stdev)."""
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

def monogaussian(x, h, c, w):
    """Single Gaussian of height h, center c, and width (sigma) w."""
    return h*np.exp(-(x-c)**2/(2*w**2))

def dual_gaussian(x, h1, c1, w1, h2, c2, w2, o):
    """Sum of two Gaussians plus offset o; params (h, c, w) per peak."""
    return monogaussian(x,h1,c1,w1)+monogaussian(x,h2,c2,w2)+o

# Fit a gaussian to a series of 21 points and return the thickness
def fit_gaussian(x, thickness_set, skipedge=3):
    x = x[skipedge:-1*skipedge]
    thickness_set = thickness_set[skipedge:-1*skipedge]
    p0 = [0, 4]
    errfunc = lambda p,a,b: gauss(a,p)-b
    p1, success = opt.leastsq(errfunc, p0[:], args=(x,thickness_set))
    fwhm = 2*np.sqrt(2*np.log(2))*p1[1]
    return p1, fwhm


def func(x, *args):
    """Sum of exponentials: sum_i a_i * exp(-b_i * x); args alternate (a1, b1, a2, b2, ...)."""
    x = x.reshape(-1, 1)
    a = np.array(args[0::2]).reshape(1, -1)
    b = np.array(args[1::2]).reshape(1, -1)
    return np.sum(a * np.exp(-b * x), axis=1)



def peak_fit(x, y):
    """Use scipy peak width to return a FWHM"""
    # peak = np.argmax(y) 
    # peaks = [peak]
    # peaks, _ = signal.find_peaks(y, 0.13)
    peaks = [np.argmax(y)]
    results_half = signal.peak_widths(y, peaks, rel_height=0.5)
    width = results_half[0][0]
    if width == 0:
        return -1,0,0,0
    height = results_half[1][0]
    h0 = results_half[2][0]+x[0]
    h1=  results_half[3][0]+x[0]
    return width, height, h0, h1
    

def find_two_peaks(x,y):
    peaks, _ = signal.find_peaks(y)
    if len(peaks)<2:
        return 0, 0, 0
    peaks = peaks[::-1]
    pos = np.argsort(y[peaks])
    peaks = np.take_along_axis(peaks, pos, axis=0)
    peak1 = x[peaks[-1]]
    peak2 = x[peaks[-2]]
    width = np.abs(peak1-peak2)

    return width, peak1, peak2


from ._thickness_worker import (init_worker, fit_triangle_chunk,
                                _seed_bilayer_center, _compute_r_squared,
                                _dual_gaussian_shared_width, _symmetric_fit_window,
                                R2_THRESHOLD, MIN_THICKNESS, MAX_THICKNESS)


def _global_bilayer_prior(thickness_set, x):
    """Fit the whole-surface average density profile to a bilayer.

    Returns ``(c1, w, c2, w)`` to seed the per-triangle recovery tier (so locally
    merged triangles on a clearly-bilayer surface can still be measured), or ``None``
    if the surface average does not resolve a bilayer.
    """
    avg = thickness_set.mean(axis=0).to_numpy() * -1
    avg = avg - avg.min()
    total = avg.sum()
    if total <= 0:
        return None
    avg = avg / (80 / 81 * total)
    mid = len(avg) // 2
    lm = np.argmin(avg[:mid]); rm = np.argmin(avg[mid:]) + mid
    a, b = x[lm + 2:rm - 2], avg[lm + 2:rm - 2]
    if len(a) < 7:
        return None
    cs, hs, n_resolved = _seed_bilayer_center(a, b)
    if n_resolved < 2:
        return None
    try:
        af, bf = _symmetric_fit_window(a, b, cs, hs)
        p, _ = opt.curve_fit(_dual_gaussian_shared_width, af, bf,
                             [0.02, 0.02, 1.5, cs, hs, 0],
                             bounds=([0.005, 0.005, 0.8, cs - 3, MIN_THICKNESS / 2.0, -1],
                                     [0.04, 0.04, 2.2, cs + 3, MAX_THICKNESS / 2.0, 1]))
    except Exception:
        return None
    c, half, w = p[3], p[4], p[2]
    return (c - half, w, c + half, w)


def split_half_neighbors(distances, neighbor_indices, rows, half_labels):
    """Neighbor tables for two disjoint halves of each sampled triangle's neighborhood.

    `distances` / `neighbor_indices` are cKDTree.query outputs (missing neighbors have
    distance inf and index n). `rows` selects the sampled triangles; `half_labels` (n,)
    assigns every triangle of the surface to half 0 or 1 once, so the two halves of
    every neighborhood are disjoint sets of density profiles. Returns
    (dist_a, idx_a, dist_b, idx_b) restricted to `rows`, with the other half's
    neighbors masked to inf -- directly usable by the thickness fit workers.
    """
    d = distances[rows]
    idx = neighbor_indices[rows]
    n = len(half_labels)
    valid = np.isfinite(d) & (idx < n)
    half = np.where(valid, half_labels[np.minimum(idx, n - 1)], -1)
    dist_a = np.where(half == 0, d, np.inf)
    dist_b = np.where(half == 1, d, np.inf)
    return dist_a, idx, dist_b, idx


def split_half_noise(est_a, est_b, full_values, full_weights=None):
    """Measurement-noise variance of a neighborhood fit from two half-neighborhood fits.

    Each half averages ~half the profiles, so its noise variance is ~2x the full fit's;
    the two halves are independent, so var(A - B) = 4 * noise_var(full). Returns a dict
    with the classical and a MAD-based robust estimate (fits throw occasional outliers),
    the field's total (area-weighted) variance, and the reliability
    1 - noise_var / total_var -- the fraction of the field's variance that is signal.

    Caveats: density profiles of adjacent triangles sample overlapping voxels, so the
    halves are not perfectly independent and the noise is somewhat underestimated; only
    triangles where both halves (and the full fit) succeeded contribute.
    """
    a, b = np.asarray(est_a, dtype=float), np.asarray(est_b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    diff = a[both] - b[both]
    full = np.asarray(full_values, dtype=float)
    w = np.ones_like(full) if full_weights is None else np.asarray(full_weights, dtype=float)
    fm = np.isfinite(full) & np.isfinite(w) & (w > 0)
    out = {"n_sampled": int(len(a)), "n_both": int(both.sum()),
           "noise_var": np.nan, "noise_var_robust": np.nan, "total_var": np.nan,
           "reliability": np.nan, "reliability_robust": np.nan}
    if fm.sum() > 1:
        mu = np.average(full[fm], weights=w[fm])
        out["total_var"] = float(np.average((full[fm] - mu) ** 2, weights=w[fm]))
    if len(diff) >= 10:
        out["noise_var"] = float(np.var(diff, ddof=1) / 4.0)
        mad = np.median(np.abs(diff - np.median(diff)))
        out["noise_var_robust"] = float((1.4826 * mad) ** 2 / 4.0)
        if out["total_var"] > 0:
            out["reliability"] = float(1.0 - out["noise_var"] / out["total_var"])
            out["reliability_robust"] = float(1.0 - out["noise_var_robust"] / out["total_var"])
    return out


def _fit_rows(thickness_arr, distances, neighbor_indices, x, global_fit_params, n_workers):
    """Run the per-triangle dual-Gaussian fit over every row of the given tables."""
    n = len(distances)
    chunk_size = max(50, n // (n_workers * 4))
    chunks = [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]
    with mp.Pool(n_workers, initializer=init_worker,
                 initargs=(thickness_arr, distances, neighbor_indices, x,
                           False, None, False, global_fit_params)) as pool:
        results = [r for chunk in pool.imap(fit_triangle_chunk, chunks) for r in chunk]
    return (np.array([r[0] for r in results], dtype=float),
            np.array([r[1] for r in results], dtype=float))


def process_single_surface(filename, average_radius, output_dir, noise_samples=0,
                           noise_seed=0):
    """
    Process a single thickness sampling file: compute per-triangle thickness,
    generate plots, and write the thickness back into the surface graph/.vtp/.csv
    in place.

    Parameters
    ----------
    filename : str
        Path to the thickness sampling CSV file
    average_radius : float
        Radius for local averaging in thickness calculations
    output_dir : str
        Directory for output files
    noise_samples : int
        If > 0, also run a split-half noise estimate on this many randomly chosen
        triangles (see :func:`split_half_noise`); the result is returned under
        ``component_info['noise']``. Costs ~2 extra fits per sampled triangle.
    noise_seed : int
        RNG seed for the half assignment and triangle sample.

    Returns
    -------
    tuple
        (component_info_dict, per_surface_thickness, areas, width, x, fig2, ax2)
    """
    # Lazy imports to avoid loading in worker processes
    from pycurv import TriangleGraph, io as pycurv_io
    from graph_tool import load_graph
    from .intradistance_verticality import export_csv

    tsname = Path(filename).stem.split(".")[0]
    comp_num = Path(filename).stem.split("_")[-5].split(".")[0]
    print(f"Processing {tsname}")

    # Load thickness csv file (header contains relative positions)
    thickness_set = pd.read_csv(filename, header=0)

    # Extract x positions from column headers
    x = np.array([float(col) for col in thickness_set.columns])
    graph_file = filename[:-13] + ".gt"
    csv_outfile = filename[:-13] + ".csv"
    # Thickness is written back into the original graph/surface in place
    # (no separate "_refined" copies).
    graph_file_final = graph_file

    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    print(tg.graph.vp.points[0], tg.graph.vp.xyz[0])

    areas = tg.graph.vp.area.get_array()
    xyz_2d = tg.graph.vp.xyz.get_2d_array([0, 1, 2])
    xx, yy, zz = xyz_2d
    xyz = xyz_2d.T
    xyztree = spatial.cKDTree(xyz)

    avg_x = np.average(xx, weights=areas)
    avg_y = np.average(yy, weights=areas)
    avg_z = np.average(zz, weights=areas)
    total_area = np.sum(areas)

    curvedness = tg.graph.vp.curvedness_VV.get_array()
    rad_curv = 1.0 / curvedness  # Vectorized
    rad_avg = np.average(rad_curv, weights=areas)
    rad_std = np.sqrt(np.cov(rad_curv, aweights=areas))

    surface_file = filename[:-13] + ".vtp"

    fig2, ax2 = plt.subplots()

    # Pre-convert DataFrame to NumPy for faster access
    thickness_arr = thickness_set.to_numpy()
    n_triangles = len(rad_curv)

    # Batch KDTree query - query all points at once (much faster than per-point)
    print(f"  Running batch KDTree query for {n_triangles} triangles...")
    distances, neighbor_indices = xyztree.query(xyz, k=500, distance_upper_bound=average_radius, workers=-1)

    # Per-triangle thickness calculation using multiprocessing
    print(f"  Fitting dual gaussians using multiprocessing...")
    n_workers = mp.cpu_count()

    # Create chunks of indices for each worker
    chunk_size = max(100, n_triangles // (n_workers * 4))
    chunks = [list(range(i, min(i + chunk_size, n_triangles)))
              for i in range(0, n_triangles, chunk_size)]

    # Whole-surface average bilayer (prior). When the surface clearly resolves a
    # bilayer, this lets the per-triangle recovery tier measure locally-merged
    # triangles too; each measurement carries a resolution score so the recovered
    # (lower-confidence, slightly thin) values stay distinguishable.
    global_fit_params = _global_bilayer_prior(thickness_set, x)
    if global_fit_params is not None:
        print(f"  Global average resolved a bilayer; recovery tier enabled "
              f"(thickness {abs(global_fit_params[2] - global_fit_params[0]):.2f} nm).")
    else:
        print("  Global average did not resolve a bilayer; per-triangle fits stay strict.")

    # Use initializer to share data once per worker (avoids repeated pickling)
    with mp.Pool(n_workers, initializer=init_worker,
                 initargs=(thickness_arr, distances, neighbor_indices, x,
                           False, None, False, global_fit_params)) as pool:
        chunk_results = list(tqdm(
            pool.imap(fit_triangle_chunk, chunks),
            total=len(chunks),
            desc="  Fitting triangles"
        ))

    # Flatten results from chunks
    results = [r for chunk in chunk_results for r in chunk]
    per_surface_thickness = [r[0] for r in results]
    per_triangle_offset = [r[1] for r in results]
    per_triangle_resolution = [r[2] for r in results]

    noise = None
    if noise_samples and noise_samples > 0:
        rng = np.random.default_rng(noise_seed)
        rows = np.sort(rng.choice(n_triangles, size=min(int(noise_samples), n_triangles),
                                  replace=False))
        half_labels = rng.integers(0, 2, n_triangles)
        dist_a, idx_a, dist_b, idx_b = split_half_neighbors(
            distances, neighbor_indices, rows, half_labels)
        print(f"  Split-half noise estimate on {len(rows)} triangles...")
        thick_a, off_a = _fit_rows(thickness_arr, dist_a, idx_a, x, global_fit_params,
                                   n_workers)
        thick_b, off_b = _fit_rows(thickness_arr, dist_b, idx_b, x, global_fit_params,
                                   n_workers)
        full_t = np.asarray(per_surface_thickness, dtype=float)
        # Offsets are reported as 0 where the fit failed; mark those missing.
        off_a = np.where(np.isfinite(thick_a), off_a, np.nan)
        off_b = np.where(np.isfinite(thick_b), off_b, np.nan)
        full_o = np.where(np.isfinite(full_t), np.asarray(per_triangle_offset, float), np.nan)
        noise = {"thickness": split_half_noise(thick_a, thick_b, full_t, areas),
                 "offset": split_half_noise(off_a, off_b, full_o, areas)}
        for name, res in noise.items():
            print(f"    {name}: noise SD {np.sqrt(res['noise_var']):.3f} nm "
                  f"(robust {np.sqrt(res['noise_var_robust']):.3f}), field SD "
                  f"{np.sqrt(res['total_var']):.3f} nm, reliability "
                  f"{res['reliability']:.2f} ({res['n_both']}/{res['n_sampled']} paired)")

    # Plot a sample of profiles for visualization
    for i in range(0, n_triangles, 5000):
        valid_mask = distances[i] != np.inf
        l = distances[i][valid_mask]
        neighbors = neighbor_indices[i][valid_mask]
        if len(neighbors) > 0:
            weights = 1.0 / (1.0 + l)
            dat = np.average(thickness_arr[neighbors], weights=weights, axis=0) * -1
            dat = dat - dat.min()
            dat_sum = dat.sum()
            if dat_sum > 0:
                dat = dat / (80/81 * dat_sum)
                ax2.plot(x, dat)

    # Average thickness calculation
    avg = thickness_set.mean(axis=0) * -1
    avg = avg - min(avg)
    avg = avg / (80/81 * sum(avg))
    mins = find_mins(avg)

    ipk = x[np.argmax(avg)]
    a = x[mins[0]+2:mins[1]-2]
    b = avg[mins[0]+2:mins[1]-2]

    # Fit the whole-surface average with the same shared-width, symmetric-window model
    # that drives the per-triangle measurements: two equal-width leaflets symmetric
    # about the center, so an asymmetric baseline cannot tilt the components and bias
    # the center. Repackaged into the legacy (h1, c1, w1, h2, c2, w2, o) layout so the
    # plot/CSV below are unchanged -- with w1 == w2 this equals dual_gaussian exactly.
    # Quality gate as for the per-triangle fit: width is NaN unless the average
    # resolves two leaflets and the fit is good and physical.
    center_seed, half_seed, avg_n_resolved = _seed_bilayer_center(a, b)
    try:
        af, bf = _symmetric_fit_window(a, b, center_seed, half_seed)
        pN, _ = opt.curve_fit(
            _dual_gaussian_shared_width, af, bf,
            [0.02, 0.02, 1.5, center_seed, half_seed, 0.0],
            bounds=([0.005, 0.005, 0.8, center_seed - 3.0, MIN_THICKNESS / 2.0, -1],
                    [0.04, 0.04, 2.2, center_seed + 3.0, MAX_THICKNESS / 2.0, 1]))
        h1, h2, w, center, half, o = pN
        p3 = np.array([h1, center - half, w, h2, center + half, w, o])
        width = 2.0 * half
        r2_avg = _compute_r_squared(bf, _dual_gaussian_shared_width(af, *pN))
        if (avg_n_resolved < 2 or r2_avg <= R2_THRESHOLD
                or not (MIN_THICKNESS <= width <= MAX_THICKNESS)):
            width = np.nan
    except Exception:
        p3 = np.array([0.0, ipk, 1.0, 0.0, ipk, 1.0, 0.0])
        width = np.nan

    # Update graph with thickness properties
    average_width_prop = tg.graph.new_vertex_property("float")
    average_width_prop.a = [width] * len(thickness_set)
    thick = tg.graph.new_vertex_property("float")
    thick.a = per_surface_thickness
    offset = tg.graph.new_vertex_property("float")
    offset.a = per_triangle_offset
    # Per-triangle reliability flag for `thickness`: 1 = two leaflets cleanly
    # resolved (high confidence); < MIN_LEAFLET_HEIGHT_RATIO (0.5) = thickness only
    # obtained by prior recovery (lower confidence, reads slightly thin).
    bilayer_resolution = tg.graph.new_vertex_property("float")
    bilayer_resolution.a = per_triangle_resolution

    tg.graph.vp.average_width = average_width_prop
    tg.graph.vp.thickness = thick
    tg.graph.vp.offset = offset
    tg.graph.vp.bilayer_resolution = bilayer_resolution
    tg.graph.save(graph_file_final)

    surf = tg.graph_to_triangle_poly()
    pycurv_io.save_vtp(surf, surface_file)
    export_csv(tg, csv_outfile)

    del tg

    # Component info for CSV output
    component_info = {
        'tsname': tsname,
        'comp_num': comp_num,
        'avg_x': avg_x,
        'avg_y': avg_y,
        'avg_z': avg_z,
        'total_area': total_area,
        'rad_avg': rad_avg,
        'rad_std': rad_std,
        'width': width,
        'p3': p3,
        'avg': avg,
        'noise': noise,
    }

    return component_info, per_surface_thickness, areas/np.sum(areas), width, x, fig2, ax2


NOISE_CSV = "thickness_noise.csv"


def write_noise_table(rows, path):
    """Upsert per-surface split-half noise rows into `path` (keyed by `surface`)."""
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        old = old[~old["surface"].isin(new["surface"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def run_measure_thickness(config, output_dir=None, noise_samples=0, noise_seed=0):
    """
    Main function to run thickness plotting and analysis.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config.yml
    output_dir : str or None
        Output directory for plots (defaults to work_dir)
    noise_samples, noise_seed :
        Split-half noise estimate settings (off when noise_samples == 0); the per-surface
        results are written to ``thickness_noise.csv`` in work_dir.
    """
    # Get settings from config
    work_dir = config.get("work_dir", config.get("seg_dir", "./"))
    if not work_dir.endswith("/"):
        work_dir += "/"

    if output_dir is None:
        output_dir = work_dir
    elif not output_dir.endswith("/"):
        output_dir += "/"

    # Get thickness settings
    thickness_config = config.get("thickness_measurements", {})
    components = thickness_config.get("components", [])
    average_radius = thickness_config.get("average_radius", 12)
    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)
    # Note: nsamples and scan_range are now read from CSV headers (set by sample_density.py)

    if not components:
        print("No components specified in config.yml thickness_measurements section")
        return

    print(f"Thickness plots settings:")
    print(f"  Work directory: {work_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Components: {components}")
    print(f"  Average radius: {average_radius}")
    print(f"  Radius hit: {radius_hit}")

    # Find files for each component
    filenames = {component: [] for component in components}
    for component in components:
        basename = work_dir + f"*{component}.AVV_rh{radius_hit}_sampling.csv"
        fileset = glob(basename)
        filenames[component].extend(fileset)
        print(f"  Found {len(fileset)} files for {component}")

    # x positions are now read from CSV headers in process_single_surface

    thickness_measurements = {component: [] for component in components}
    area_measurements = {component: [] for component in components}
    widths = {component: [] for component in components}
    noise_rows = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    component_list_file = os.path.join(output_dir, "component_list.csv")
    with open(component_list_file, "w") as compfile:
        compfile.write("TS,Component Type,Component Number,Centroid X,Centroid Y,Centroid Z,"
                       "Total Area,Radius of Curvature,Rad_Curv STD,Thickness,"
                       "Peak1 Position,Peak1 Sigma,Peak2 Position,Peak2 Sigma\n")

        for component in components:
            fig, ax = plt.subplots()
            print(f"\nProcessing component: {component}")

            for index, filename in enumerate(filenames[component]):
                info, per_surface_thickness, norm_areas, width, x, fig2, ax2 = \
                    process_single_surface(filename, average_radius, output_dir,
                                           noise_samples=noise_samples,
                                           noise_seed=noise_seed)
                if info.get('noise'):
                    row = {"surface": info['tsname'], "component": component,
                           "average_radius": average_radius}
                    for qty, res in info['noise'].items():
                        row.update({f"{qty}_{k}": v for k, v in res.items()})
                    noise_rows.append(row)

                thickness_measurements[component].extend(per_surface_thickness)
                area_measurements[component].extend(norm_areas)
                widths[component].append(width)

                # Write component info
                p3 = info['p3']
                compfile.write(f"{info['tsname']},{component},{info['comp_num']},"
                               f"{info['avg_x']:.1f},{info['avg_y']:.1f},{info['avg_z']:.1f},"
                               f"{info['total_area']:.1f},{info['rad_avg']:.2f},{info['rad_std']:.2f},"
                               f"{width:.2f},{p3[1]:.2f},{p3[2]:.2f},{p3[4]:.2f},{p3[5]:.2f}\n")

                # Generate individual surface plot
                fig3, ax3 = plt.subplots()
                scan_range = (x[-1] - x[0]) / 2  # Derive scan_range from x positions
                ax.plot(x, info['avg'], label=index)
                ax3.plot(x, info['avg'], label="data")
                ax3.plot(x, dual_gaussian(x, *p3), "-.", label="Dual Gaussian Fit")
                ax3.plot(x, monogaussian(x, *p3[0:3]) + p3[6], "--", label="Gaussian 1")
                ax3.plot(x, monogaussian(x, *p3[3:6]) + p3[6], "--", label="Gaussian 2")
                ax3.axvspan(p3[1], p3[4], facecolor='g', alpha=0.1, label="Dual Gauss Span")
                ax3.legend()
                ax3.set_xlabel("Distance (nm)")
                ax3.set_xlim(x[0], x[-1])
                ax3.set_ylabel("Density")
                ax3.set_title(f"{info['tsname']} - {component} {info['comp_num']}")

                fig2.savefig(os.path.join(output_dir, f"thickness_{component}.png"))
                fig3.savefig(os.path.join(output_dir,
                             f"thickness_average_{info['tsname']}_{component}_{info['comp_num']}_fit.svg"))
                plt.close(fig=fig2)
                plt.close(fig=fig3)

            # Save component summary plot
            ax.set_xlabel("Distance (nm)")
            if 'x' in dir():  # Use x from last processed file
                ax.set_xlim(x[0], x[-1])
            ax.set_ylabel("Density")
            ax.legend()
            ax.set_title(f"{component} - All Curves")
            fig.savefig(os.path.join(output_dir, f"{component}_Averages.png"))
            fig.clear()
            plt.close()

    # Print summary statistics
    thicknesses = []
    for component in components:
        if widths[component]:
            thicknesses.append(widths[component])
            print(f"{component} - mean: {np.mean(widths[component]):.3f}, "
                  f"stdev: {np.std(widths[component]):.3f}")

    # Statistical tests (if we have at least 2 components with data)
    if len(thicknesses) >= 2:
        res = stats.ttest_ind(thicknesses[0], thicknesses[1])
        print(f"Student's T Test pval: {res.pvalue}, df: {res.df}")

    # Generate violin plot
    if thicknesses:
        fig4, ax4 = plt.subplots()
        ax4.violinplot(thicknesses, showmedians=True)
        ax4.set_xticks(range(1, len(components)+1))
        ax4.set_xticklabels(components)
        fig4.savefig(os.path.join(output_dir, "violin.svg"))

        with open(os.path.join(output_dir, "violin.csv"), "w") as violin:
            for component in components:
                if widths[component]:
                    vstring = ",".join([str(i) for i in widths[component]])
                    violin.write(f"{component},{vstring}\n")

    # Generate histogram
    thickness_data = [thickness_measurements[c] for c in components if thickness_measurements[c]]
    area_data = [area_measurements[c] for c in components if area_measurements[c]]
    labels_with_data = [c for c in components if thickness_measurements[c]]
    if thickness_data and area_data:
        histogram(data=thickness_data, areas=area_data,
                  labels=labels_with_data, title="Thickness Comparison", xlabel="Thickness (nm)")

    if noise_rows:
        path = write_noise_table(noise_rows, os.path.join(work_dir, NOISE_CSV))
        print(f"Split-half noise estimates written to {path}")

    print(f"\nOutput files written to {output_dir}")


@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default=None,
              help='Output directory for plots (defaults to work_dir from config)')
@click.option('--average_radius', type=float, default=None,
              help='Radius for local averaging (overrides config)')
@click.option('--noise-estimate', 'noise_estimate', is_flag=True, default=False,
              help='Also estimate per-surface measurement noise of thickness/offset by '
                   'fitting each sampled triangle from two disjoint random halves of its '
                   'neighborhood (var(A-B)/4). Writes thickness_noise.csv to work_dir.')
@click.option('--noise-samples', type=int, default=5000, show_default=True,
              help='Triangles per surface used for --noise-estimate.')
@click.option('--noise-seed', type=int, default=0, show_default=True,
              help='RNG seed for --noise-estimate.')
def measure_thickness_cli(configfile, output, average_radius, noise_estimate,
                          noise_samples, noise_seed):
    """
    Measure membrane thickness from density sampling data.

    Fits dual gaussians to density profiles to estimate membrane thickness
    for each triangle in the surface mesh. Writes the thickness properties back
    into the surface graph/.vtp/.csv in place, with summary statistics and plots.

    CONFIGFILE: Path to the config.yml file
    """
    config = load_config(configfile, require=("work_dir",))

    # Override config settings if specified on command line
    if average_radius is not None:
        if "thickness_measurements" not in config:
            config["thickness_measurements"] = {}
        config["thickness_measurements"]["average_radius"] = average_radius

    run_measure_thickness(config, output_dir=output,
                          noise_samples=noise_samples if noise_estimate else 0,
                          noise_seed=noise_seed)


if __name__ == "__main__":
    measure_thickness_cli()
