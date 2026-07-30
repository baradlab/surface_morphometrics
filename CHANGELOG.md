# Changelog

All notable changes to the Surface Morphometrics toolkit are documented here.
This project loosely follows [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [2.0.0b6] — beta (unreleased)

Mesh generation decoupled from pycurv, in preparation for extracting it into a
standalone importable library.

### Added
- First test coverage for the meshing path beyond `mrc2xyz`. `tests/test_xyz2ply.py`
  (skipped when pymeshlab is unavailable) checks that Screened Poisson reconstructs a
  known sphere shell, that isotropic remeshing hits the requested triangle area, that a
  degenerate point cloud reports failure, and that every `xyz2ply` command-line option
  reaches `xyz_to_ply`. `tests/test_mesh_wiring.py` needs neither pymeshlab nor vtk, so
  it runs in CI: it pins that every `surface_generation` setting reaches the meshing
  subprocess carrying its configured value, and that every documented setting has a
  consumer — the two guards that would have caught the inert `ultrafine` below.
- `tests/test_version.py` pins `surface_morphometrics.__version__` to `pyproject.toml`'s
  `version`. The two had drifted apart twice (fixed in 2.0.0b3, drifted again at
  2.0.0b5) because a release bumps one and forgets the other.

### Changed
- `vtk` is now a declared dependency in `pyproject.toml` (`vtk>=9`). It was always
  required — `ply2vtp`, `export_obj`, and `refine_mesh` all use it — but was only ever
  installed as a side effect of the conda environment, so a plain `pip install` produced
  a package that failed at runtime. It is pip-resolvable, so it belongs with the pip
  dependencies rather than with the conda-only ones.
- Mesh generation no longer depends on pycurv. `ply2vtp` imported `pycurv_io` and
  `scipy.ndimage.morphology.distance_transform_edt` without ever referencing either;
  the pycurv import was the only thing coupling the segmentation → mesh path to
  pycurv/graph-tool. The three meshing stages now need only mrcfile, numpy, pandas,
  pymeshlab, and vtk.
- `ply2vtp` imports vtk inside `ply_to_vtp` rather than at module scope, as `export_obj`
  already does, so `segmentation_to_meshes` can be imported without vtk installed.
- Dropped the pymeshlab 2022.2.post3 `PercentageValue` compatibility shim, whose only
  user was the removed `ultrafine` branch. This also removes an undeclared dependency on
  `importlib_metadata`; `environment.yml` already requires pymeshlab >= 2023.12.

### Removed
- `surface_generation.ultrafine`, superseded by isotropic remeshing (`isotropic_remesh` /
  `target_area`) and by density-guided mesh refinement. It had also been inert for some
  time, by two independent paths: `run_xyz_to_ply` never passed `--ultrafine` to the
  meshing subprocess, and `xyz2ply`'s own click wrapper never forwarded it either, so
  only a direct Python call could reach the branch. Meshing output is unchanged. Configs
  that still set the key keep loading without error.

### Fixed
- `ply2vtp.ply_to_vtp` raised `NameError` when the vtp write failed: the error branch
  called `pexceptions.PySegInputError` but `pexceptions` was never imported, so the
  function's only error path was broken. It now raises `RuntimeError`.
- Removed a stray `print("open")` from `ply_to_vtp` that leaked into `make_meshes` output.

## [2.0.0b5] — beta

Voxel-space OBJ export.

### Added
- `export_obj --scale_to_voxels A_PER_PX` — export the mesh in voxel (pixel) space for
  overlaying on the tomogram, given the tomogram's voxel size in Å/px. The surface's
  native units are read from `surface_generation.angstroms`, so at 5 Å/px an nm-scale
  surface is doubled and an Å-scale one divided by 5. Mutually exclusive with
  `--scale_to_angstroms`.

## [2.0.0b4] — beta

Protein-patch workflow overhaul and a mesh-refinement pycurv speed fix.

### Added
- Protein-to-membrane-edge distances in the patch workflow. When per-triangle
  `thickness` is present, `generate_patches` writes a `<prefix>_headgroup_distance`
  (distance to the true membrane edge = midplane distance − thickness/2, measured
  at the patch's central triangle) per-particle into the annotated STAR, and
  `patch_statistics` reports it per-patch. The effective thickness falls back from
  the central triangle to the patch's distance-weighted mean, then to NaN.
- `patch_statistics` now reports `min_protein_distance` (closest membrane-midplane
  approach) per real patch, and keeps the measurement prefix in `region_type`
  (e.g. `ribo_patch`, `ribo_random_patch`) so multiple patch types can be compared.
- `patch_analysis.measurement_prefix` (default `ribo`): patch measurement columns
  are prefixed, so re-running with a different prefix accumulates a second patch
  type (e.g. `atp`) on the same surface.

### Changed
- **`generate_patches` edits the membrane `.gt`/`.vtp`/`.csv` in place** instead of
  writing a separate `*_patches` file set; `patch_statistics` and `extract_patches`
  default to the in-place `*.AVV_rh*` files. `--output-dir` now only affects the
  annotated STAR.
- Refinement runs pycurv (the final curvature pass) in a **fresh subprocess**. Run
  in-process at the end of a long refinement, pycurv was ~3–4× slower per NVV chunk
  (measured) due to the loaded process's memory/allocator state, not the mesh; a
  clean subprocess restores standalone speed. Refinement also frees large arrays
  and runs `gc.collect()` between iterations/surfaces.

### Fixed
- Patch headgroup distance could exceed the protein distance: it was computed per
  triangle then min-reduced, so the two minima came from different triangles (and
  NaN-thickness triangles dropped out). It is now computed at the central triangle,
  guaranteeing `headgroup_distance ≤ min_protein_distance`.

### Notes
- `generate_patches` supports RELION 3/4 STAR files (absolute tomogram-frame
  coordinates). RELION 5 centered coordinates (`rlnCenteredCoordinate*Angst`) are
  not yet supported; convert to absolute coordinates first.

## [2.0.0b3] — beta

A performance and robustness release for the mesh-refinement step, plus the move of
the repository to the Barad Lab GitHub organization.

### Changed
- **Mesh refinement now runs pycurv only once, on the final surface.** Every
  intermediate iteration (cross-correlation and dual-Gaussian) builds a fast
  lightweight graph from VTK normals and warns that the surface is not
  curvature-ready; a single full pycurv normal-vector-voting pass runs on the
  accepted surface after the loop — on every exit path, including an early
  convergence stop — so the result is always a clean, curvature-ready surface.
  Previously every dual-Gaussian iteration re-ran the slow voting (~28 min on large
  surfaces), which dominated refinement time.
- Mesh-refinement parallel fitting (dual-Gaussian offset centering and local
  thickness) now respects the configured `cores` instead of spawning one worker per
  logical CPU. Benchmarking showed throughput peaks near the configured core count
  and regresses beyond it, so this is both a correctness and a speed improvement.
- `accept_refinement` messages now refer to "intermediate" iterations (any non-final
  iteration may lack a curvature graph), not specifically cross-correlation ones.
- Repository moved to `https://github.com/baradlab/surface_morphometrics`.

### Fixed
- `__version__` in `surface_morphometrics/__init__.py` had drifted a release behind
  `pyproject.toml`; both version strings are now kept in sync.

## [2.0.0b2] — beta

A quality and robustness release on top of 2.0.0b1, focused on the density-profile
bilayer fitting (mesh refinement and thickness measurement), a forgiving centralized
config system, and packaging fixes.

### Added
- Per-triangle `bilayer_resolution` score (0–1) written to the thickness CSV — a
  reliability flag distinguishing strictly-resolved leaflets from prior-recovered
  (slightly thin) measurements.
- `morphometrics new_config --simple` — a minimal, fully-runnable starter config
  (the full annotated template is still available via the default `--verbose`).
- Centralized config defaults: omitted parameter sections fall back to documented
  values (a single source of truth, kept in sync with the template by a test), so
  partial/stripped configs run.
- Per-command required-key validation with one clean error message instead of a
  `KeyError` traceback; directory paths no longer need trailing slashes.
- Distance/orientation measurements default to all-vs-all when `intra`/`inter` are
  omitted (announced; set them explicitly empty to opt out).
- Plugins can contribute `morphometrics` subcommands via a `morphometrics.commands`
  entry point.
- GitHub Actions test workflow and unit tests for the fitting and config helpers.

### Changed
- **Dual-Gaussian bilayer fitting overhaul** (mesh refinement and thickness): leaflet
  detection by height above the shared solvent base (fixes near-universal fallback to
  a single Gaussian on clean bilayers); a shared-width, centered fit over a symmetric
  window (removes a ~0.1 nm centering bias and the collapse failure mode); and a
  two-tier gate that recovers locally-merged bilayers from the global-average prior
  while still rejecting genuine single/skewed peaks. The whole-surface average fit and
  its `_fit.svg` plot use the same model.
- Thickness measurement now applies a real quality gate (resolution + R² + physical
  range) and reports `NaN` for unmeasurable triangles instead of a spurious value.
- `accept_refinement` falls back to a surface's last available iteration (with a
  warning) when it converged before the requested step.
- `mesh_refinement` defaults now match the documented template (the code's omitted-key
  defaults had drifted): `average_radius` 25, `average_radius_min` 12,
  `max_total_offset` 8, `laplacian_iterations` 5, `laplacian_lambda` 0.5,
  `convergence_threshold` 0.05, `xcorr_iterations` [1, 2, 3].
- `export_obj`: `--angstroms` renamed to `--scale_to_angstroms`, with config-aware
  unit conversion.

### Fixed
- pycurv fork deadlock on Linux (cap `OMP_NUM_THREADS=1` before importing pycurv).
- `export_obj` reads legacy / empty VTP surfaces instead of crashing.
- NumPy 2.x compatibility (replace the removed `np.trapz`).
- Deprecated `matplotlib.cm.get_cmap` usage.

## [2.0.0b1] — beta

This release turns the collection of scripts into an installable Python package
driven by a single `morphometrics` command, and adds protein-patch /
connected-component analysis and surface export. It is a **breaking change** to
how the toolkit is invoked.

### ⚠️ Breaking changes
- The pipeline is now run through one grouped command, e.g.
  `morphometrics make_meshes config.yml` instead of `python segmentation_to_meshes.py config.yml`.
  The old `python <script>.py …` calls still work for now via **deprecation shims**
  that warn and forward; they will be removed in a future release. See the
  [migration table](README.md#upgrading-from-older-versions).
- Config keys renamed: `data_dir` → `seg_dir`, and `max_triangles` →
  `simplify_max_triangles` (only used when `simplify: true`). `make_meshes` warns
  if it detects the old names.
- Thickness measurement now writes the per-triangle `thickness`/`offset`/
  `average_width` properties **back into the surface `.gt`/`.vtp`/`.csv` in place**
  rather than creating separate `*_refined.gt`/`.vtp` files.

### Added
- Installable `surface_morphometrics` package with a `morphometrics` console entry
  point (lazy subcommands, grouped/ordered `--help`, next-step hints, shell
  completion).
- `morphometrics new_config` — write a starter `config.yml` into the working dir.
- `morphometrics fetch_example` — download a small cropped tomogram+segmentation
  test set from Zenodo (or point at the full data on EMPIAR-12534).
- Density-guided mesh refinement workflow: `refine_mesh` + `accept_refinement`
  (commit a chosen iteration in place, with `.orig.bak` backups).
- Protein-patch and connected-component analysis: `generate_patches`
  (STAR-driven, with random controls and combined-STAR basename matching),
  `label_components`, `extract_patches`, and `patch_statistics`.
- `morphometrics export_obj` — export a quantified surface to a colormapped
  OBJ + MTL for Blender / MeshLab (with `--nan-color` for unmeasured faces).
- `pyproject.toml` packaging; `pip install -e .` is wired into the conda
  environment files; basic `tests/` for the pure-logic helpers.

### Changed
- Research/paper/one-off scripts moved to `old_scripts/`.
- pycurv no longer prints the "run in parallel / this may take an hour" warnings
  (it is now about as fast as mesh generation); the slow-step note moved to
  `refine_mesh`, which is the slowest, optional step.
- README reorganized (Installation / Quick start / Pipeline / Analysis &
  visualization / Reference / Upgrading) with a table of contents.

[2.0.0b6]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b5]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b4]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b3]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b2]: https://github.com/baradlab/surface_morphometrics/releases
[2.0.0b1]: https://github.com/baradlab/surface_morphometrics/releases
