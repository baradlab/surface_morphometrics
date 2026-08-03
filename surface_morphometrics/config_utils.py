#! /usr/bin/env python
"""Shared helpers for loading the pipeline config.yml."""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import copy

import click
import yaml

# Directory settings that the pipeline joins with filenames by plain string
# concatenation (e.g. work_dir + "*.gt"), so they must end in a path separator.
DIR_KEYS = ("seg_dir", "tomo_dir", "work_dir")

# Sensible defaults for every parameter section, so a stripped-down config that
# omits them still runs as documented. These mirror config_template.yml (kept in
# sync by tests/test_config_utils.py) and are the single source of truth for default
# values. User-data keys with no sensible default are intentionally excluded and left
# for the user / each command: top-level segmentation_values and exp_name, and within
# sections the membership/selection lists (distance intra/inter, thickness components,
# patch_analysis membrane_label).
DEFAULTS = {
    "cores": 6,
    "surface_generation": {
        "angstroms": False,
        "ultrafine": False,
        "extrapolation_distance": 1.5,
        "isotropic_remesh": True,
        "target_area": 1.0,
        "simplify": False,
        "simplify_max_triangles": 300000,
        "octree_depth": 9,
        "point_weight": 0.7,
        "neighbor_count": 400,
        "smoothing_iterations": 1,
    },
    "curvature_measurements": {
        "radius_hit": 9,
        "min_component": 30,
        "exclude_borders": 1,
        "use_gpu": False,
        "gpu_device": None,
        "gpu_batch_size": 1024,
    },
    "distance_and_orientation_measurements": {
        "mindist": 3,
        "maxdist": 400,
        "tolerance": 0.1,
        "verticality": True,
        "relative_orientation": True,
        # intra (which surfaces) and inter (which pairs) are left out on purpose so the
        # command can tell "omitted" (-> default to all surfaces / all pairs) from
        # "explicitly empty" (-> measure nothing). See measure_distances_orientations.
    },
    "density_sampling": {
        "sample_spacing": 0.25,
        "scan_range": 10,
    },
    "thickness_measurements": {
        "average_radius": 12,
        "fit_curve": True,
    },
    "patch_analysis": {
        "patch_radius": 12,
        "particle_max_distance": 24,
        "generate_random": True,
        "random_min_distance": 12,
        "random_seed": 0,
        "star_dir": "./star/",
        "star_pattern": "{tomo}.star",
        "star_tomo_column": None,
        "star_coord_columns": ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"],
        "star_pixelsize_column": "rlnPixelSize",
        "star_coords_in_pixels": True,
        "annotate_star": True,
        "min_component_size": 0,
        "statistics_properties": ["curvedness_VV", "thickness"],
    },
    "mesh_refinement": {
        "iterations": 6,
        "damping_factor": 0.9,
        "monolayer": False,
        "xcorr_iterations": [1, 2, 3],
        "average_radius": 25,
        "average_radius_decay": 1.0,
        "average_radius_min": 12.0,
        "max_total_offset": 8,
        "smooth_offsets": True,
        "offset_smoothing_radius": 25,
        "laplacian_iterations": 5,
        "laplacian_lambda": 0.5,
        "lowpass_sigma": 0,
        "convergence_threshold": 0.05,
    },
}

# Human-readable purpose of each setting, used in the missing-required-key error.
_KEY_HELP = {
    "seg_dir": "directory of segmentation MRC files",
    "work_dir": "directory for pipeline output files",
    "tomo_dir": "directory of tomogram MRC files (needed for density sampling, "
                "thickness, and mesh refinement)",
    "segmentation_values": "mapping of feature name to its segmentation label value "
                           "(e.g. OMM: 1)",
}


def normalize_dirs(config):
    """Append a trailing "/" to any set directory path that lacks one.

    Mutates and returns ``config``. Only ``seg_dir``/``tomo_dir``/``work_dir`` that
    are non-empty strings are touched; empty or unset values are left alone so each
    command can apply its own defaulting.
    """
    if isinstance(config, dict):
        for key in DIR_KEYS:
            value = config.get(key)
            if isinstance(value, str) and value and not value.endswith("/"):
                config[key] = value + "/"
    return config


def merge_defaults(config):
    """Fill in default parameter sections so partial configs run as documented.

    Each section in :data:`DEFAULTS` is merged shallowly into ``config``: any
    parameter the user did not set takes its default, while user-set values (and any
    extra keys, like a section's membership lists) are kept. Mutates and returns
    ``config``. Defaults are deep-copied so the module-level :data:`DEFAULTS` is never
    mutated by downstream code.
    """
    if not isinstance(config, dict):
        return config
    # Back-compat: density-sampling settings used to live under thickness_measurements.
    # Carry them over before defaults are applied so old configs keep their values.
    if "density_sampling" not in config and isinstance(config.get("thickness_measurements"), dict):
        legacy = {k: config["thickness_measurements"][k]
                  for k in ("sample_spacing", "scan_range")
                  if k in config["thickness_measurements"]}
        if legacy:
            config["density_sampling"] = legacy
    for section, default in DEFAULTS.items():
        if isinstance(default, dict):
            merged = copy.deepcopy(default)
            user = config.get(section)
            if isinstance(user, dict):
                merged.update(user)
            config[section] = merged
        elif config.get(section) is None:
            config[section] = copy.deepcopy(default)
    return config


def resolve_distance_targets(dist_settings, labels):
    """Resolve which surfaces/pairs to measure distances for, defaulting to all-vs-all.

    Omitting ``intra``/``inter`` (key absent) opts in to the all-vs-all default: every
    surface measured against itself, and every unordered pair against each other.
    Setting either explicitly -- including an empty ``[]`` / ``{}`` -- is honored as
    given, so an explicit empty means "measure nothing". Returns
    ``(intra, inter, intra_defaulted, inter_defaulted)`` where the booleans say whether
    the all-vs-all default was applied (so the caller can announce it).
    """
    labels = list(labels)
    intra = dist_settings.get("intra")
    intra_defaulted = intra is None
    if intra_defaulted:
        intra = labels
    inter = dist_settings.get("inter")
    inter_defaulted = inter is None
    if inter_defaulted:
        # Upper triangle only; callers write each pair bidirectionally.
        inter = {labels[i]: labels[i + 1:] for i in range(len(labels) - 1) if labels[i + 1:]}
    return intra, inter, intra_defaulted, inter_defaulted


def require_keys(config, required, configfile=""):
    """Raise a clean CLI error if any ``required`` top-level key is unset.

    A key counts as unset if it is missing or maps to an empty/None value. All
    missing keys are reported together so the user fixes the config in one pass.
    ``required`` is per-command: each command demands only the settings it uses
    (e.g. tomogram steps require ``tomo_dir``; analysis/export steps need only
    ``work_dir``).
    """
    missing = [key for key in required if not (isinstance(config, dict) and config.get(key))]
    if missing:
        target = configfile or "config.yml"
        details = "\n".join(
            f"  - {key}" + (f"  ({_KEY_HELP[key]})" if key in _KEY_HELP else "")
            for key in missing
        )
        raise click.ClickException(
            f"{target} is missing required setting(s):\n{details}\n"
            "Add them to your config.yml (start one with `morphometrics new_config`)."
        )


def load_config(configfile, require=()):
    """Load a pipeline config YAML, normalizing directory paths and validating.

    A config written without trailing slashes (e.g. ``work_dir: ./morphometrics``)
    behaves identically to one with them, since the pipeline builds paths by string
    concatenation. Missing parameter sections are filled from :data:`DEFAULTS`
    (:func:`merge_defaults`), so partial/stripped configs run as documented.
    ``require`` is a tuple of top-level keys this command needs; if any is missing the
    user gets one clean error (see :func:`require_keys`) instead of a ``KeyError``
    traceback partway through the run.
    """
    with open(configfile) as handle:
        config = yaml.safe_load(handle) or {}
    normalize_dirs(config)
    merge_defaults(config)
    require_keys(config, require, configfile)
    return config
