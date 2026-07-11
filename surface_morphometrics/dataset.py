#! /usr/bin/env python
"""A lazy, metadata-aware index of a morphometrics run's per-surface tables.

`Dataset` is the successor to `morphometrics_stats.Experiment`. It shares the useful
access pattern -- `dataset[tomogram][class]` gives a surface's per-triangle dataframe --
but sheds the parts that made `Experiment` painful:

- **Lazy.** Building the index only globs filenames; a surface's CSV is read (and cached)
  the first time it is accessed, so assembling a dataset over a large run is cheap.
- **Config-driven grouping instead of hardcoded lists.** A `groups:` config block maps
  tomograms to metadata (condition, morphology, ...) by name or glob pattern, replacing
  the `conditions = [...]` lists and `key[1] == "F"` filename hacks in the research
  scripts. Assignment is validated at load: every included tomogram must land in exactly
  one bucket per group, or a clear error is raised.
- **Metadata-aware selection.** `dataset.surfaces(label="IMM", condition="Tg")` iterates
  the surfaces matching a class and/or any group values, so a treatment comparison flows
  from the grouping rather than from re-deriving membership per script.

Grouping (metadata) is orthogonal to selection (which tomograms exist at all): the latter
reuses the phase-2 `statistics.include_tomograms` / `exclude_tomograms` config keys via
`surface_selection`.

Example config::

    groups:
      condition:
        Tg:      ["TF*", "UF*"]
        Vehicle: ["TE*", "UE*"]
      morphology:
        fragmented: ["?F*"]
        elongated:  ["?E*"]
    statistics:
      exclude_tomograms: ["UF3", "TE1"]
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import pandas as pd

from .surface_selection import _matches_any, select_tomograms


def assign_groups(tomograms, groups):
    """Map each tomogram to its bucket in every group, validating the assignment.

    `groups` is `{group_name: {bucket_label: [name_or_glob, ...]}}`. Returns
    `{tomogram: {group_name: bucket_label}}`. Raises ValueError listing every problem if
    any tomogram matches zero buckets (unassigned) or more than one bucket (ambiguous) in
    a group -- so a mislabeled tomogram is a load-time error, not a silent mistake.
    """
    assignment = {}
    problems = []
    for tomo in tomograms:
        meta = {}
        for group_name, buckets in groups.items():
            matched = [label for label, patterns in buckets.items()
                       if _matches_any(tomo, patterns or [])]
            if len(matched) == 1:
                meta[group_name] = matched[0]
            elif not matched:
                problems.append(
                    f"  {tomo}: no bucket in group '{group_name}' "
                    f"(buckets: {', '.join(buckets)})")
            else:
                problems.append(
                    f"  {tomo}: matches {len(matched)} buckets in group '{group_name}': "
                    f"{', '.join(matched)}")
        assignment[tomo] = meta
    if problems:
        raise ValueError(
            "config `groups:` does not assign every tomogram to exactly one bucket per "
            "group. Fix the patterns, or exclude these tomograms via "
            "statistics.exclude_tomograms:\n" + "\n".join(problems))
    return assignment


class SurfaceView:
    """One `(tomogram, class)` surface: its path, existence, and (lazy) dataframe."""

    def __init__(self, dataset, tomo, label):
        self.dataset = dataset
        self.tomo = tomo
        self.label = label

    @property
    def path(self):
        return self.dataset.surface_path(self.tomo, self.label)

    def exists(self):
        return os.path.isfile(self.path)

    def load(self):
        """The per-triangle dataframe (read once, then cached on the dataset)."""
        return self.dataset.load(self.tomo, self.label)

    @property
    def metadata(self):
        return self.dataset.metadata(self.tomo)

    def __repr__(self):
        return f"SurfaceView({self.tomo!r}, {self.label!r})"


class TomogramView:
    """Metadata plus per-class surface access for one tomogram.

    `dataset[tomo][label]` returns the class's dataframe (lazy), mirroring the old
    `Experiment[tomo][label]` access. Use `.surface(label)` for a non-loading handle.
    """

    def __init__(self, dataset, tomo):
        self.dataset = dataset
        self.tomo = tomo

    @property
    def metadata(self):
        return self.dataset.metadata(self.tomo)

    def classes(self):
        """The configured classes whose surface file exists for this tomogram."""
        return [label for label in self.dataset.labels
                if self.dataset.surface(self.tomo, label).exists()]

    def surface(self, label):
        return self.dataset.surface(self.tomo, label)

    def __getitem__(self, label):
        return self.dataset.load(self.tomo, label)

    def __contains__(self, label):
        return self.dataset.surface(self.tomo, label).exists()

    def __repr__(self):
        return f"TomogramView({self.tomo!r}, metadata={self.metadata})"


class Dataset:
    """A lazy index of a morphometrics run's surfaces, with per-tomogram metadata."""

    def __init__(self, work_dir, labels, tomograms, radius_hit=9, groups=None, name=None):
        """Build an index. Prefer :meth:`from_config`.

        Args:
            work_dir: directory of per-surface ``*.AVV_rh{rh}.csv`` tables (trailing /).
            labels: ordered class names (from ``segmentation_values``).
            tomograms: tomogram base names that make up this dataset (already selected).
            radius_hit: pycurv radius-hit, part of the surface-file extension.
            groups: ``{group: {bucket: [patterns]}}`` metadata mapping (validated here).
            name: optional label for the dataset.
        """
        self.work_dir = work_dir
        self.labels = list(labels)
        self.tomogram_names = list(tomograms)
        self.radius_hit = radius_hit
        self.extension = f".AVV_rh{radius_hit}.csv"
        self.name = name
        self.groups = groups or {}
        self._metadata = assign_groups(self.tomogram_names, self.groups)
        self._cache = {}

    @classmethod
    def from_config(cls, config, name=None):
        """Build a Dataset from a loaded pipeline config.

        Discovers tomograms from ``seg_dir/*.mrc``, applies the phase-2
        ``statistics.include_tomograms`` / ``exclude_tomograms`` selection, and reads the
        top-level ``groups:`` block for metadata.
        """
        work_dir = config["work_dir"]
        radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)
        labels = list(config["segmentation_values"].keys())
        all_tomograms = sorted(os.path.basename(f)[:-4]
                               for f in glob(config["seg_dir"] + "*.mrc"))
        stats_cfg = config.get("statistics", {}) or {}
        if not isinstance(stats_cfg, dict):
            stats_cfg = {}
        used, _excluded = select_tomograms(
            all_tomograms, stats_cfg.get("include_tomograms"),
            stats_cfg.get("exclude_tomograms"))
        return cls(work_dir, labels, used, radius_hit=radius_hit,
                   groups=config.get("groups") or {}, name=name)

    # --- paths and lazy loading ---

    def surface_path(self, tomo, label):
        return f"{self.work_dir}{tomo}_{label}{self.extension}"

    def surface(self, tomo, label):
        return SurfaceView(self, tomo, label)

    def load(self, tomo, label):
        """Read (and cache) a surface's per-triangle dataframe."""
        key = (tomo, label)
        if key not in self._cache:
            path = self.surface_path(tomo, label)
            if not os.path.isfile(path):
                raise KeyError(f"no surface file for tomogram {tomo!r}, class {label!r} "
                               f"(expected {path})")
            self._cache[key] = pd.read_csv(path)
        return self._cache[key]

    # --- metadata ---

    def metadata(self, tomo):
        """The group metadata dict for a tomogram (e.g. {'condition': 'Tg'})."""
        return dict(self._metadata.get(tomo, {}))

    # --- access / iteration / selection ---

    def tomograms(self):
        return list(self.tomogram_names)

    def __getitem__(self, tomo):
        if tomo not in self._metadata:
            raise KeyError(f"tomogram {tomo!r} is not in this dataset")
        return TomogramView(self, tomo)

    def __iter__(self):
        for tomo in self.tomogram_names:
            yield TomogramView(self, tomo)

    def __len__(self):
        return len(self.tomogram_names)

    def __contains__(self, tomo):
        return tomo in self._metadata

    def surfaces(self, label=None, exists_only=True, **group_values):
        """Iterate SurfaceViews matching a class and/or group-metadata values.

        `label` limits to one class (default: every configured class). Keyword args select
        by group metadata, e.g. `surfaces(label="IMM", condition="Tg")` yields the IMM
        surfaces of every Tg tomogram. With `exists_only` (default), only surfaces whose
        file is present are yielded.
        """
        wanted_labels = [label] if label is not None else self.labels
        for tomo in self.tomogram_names:
            meta = self._metadata.get(tomo, {})
            if any(meta.get(g) != v for g, v in group_values.items()):
                continue
            for lab in wanted_labels:
                view = SurfaceView(self, tomo, lab)
                if exists_only and not view.exists():
                    continue
                yield view

    def __repr__(self):
        return (f"Dataset(name={self.name!r}, {len(self.tomogram_names)} tomogram(s), "
                f"labels={self.labels}, groups={list(self.groups)})")
