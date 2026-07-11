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

import hashlib
import os
from glob import glob

import pandas as pd

from .surface_selection import _matches_any, select_tomograms


def _surface_cache_key(path):
    """A short content-independent key for a source CSV: its path, mtime, and size.

    Cheap (a single stat, no file read) and sufficient to detect that a surface has been
    re-run: any edit changes the mtime and usually the size, so the key changes and the
    stale parquet is bypassed.
    """
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


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

    def __init__(self, work_dir, labels, tomograms, radius_hit=9, groups=None, name=None,
                 cache_dir=None):
        """Build an index. Prefer :meth:`from_config`.

        Args:
            work_dir: directory of per-surface ``*.AVV_rh{rh}.csv`` tables (trailing /).
            labels: ordered class names (from ``segmentation_values``).
            tomograms: tomogram base names that make up this dataset (already selected).
            radius_hit: pycurv radius-hit, part of the surface-file extension.
            groups: ``{group: {bucket: [patterns]}}`` metadata mapping (validated here).
            name: optional label for the dataset.
            cache_dir: directory for the persistent parquet parse-cache (created on first
                write). ``None`` disables it (in-memory caching only).
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
        self.cache_dir = cache_dir
        self._parquet_disabled = False   # set if pyarrow is missing at runtime

    @classmethod
    def from_config(cls, config, name=None, cache=None):
        """Build a Dataset from a loaded pipeline config.

        Discovers tomograms from ``seg_dir/*.mrc``, applies the phase-2
        ``statistics.include_tomograms`` / ``exclude_tomograms`` selection, and reads the
        top-level ``groups:`` block for metadata.

        The parquet parse-cache is enabled by the ``cache`` argument, or by
        ``statistics.cache`` in the config if ``cache`` is None: ``True`` uses a default
        ``.morphometrics_cache/`` under ``work_dir``; a string is used as the cache
        directory; ``False`` disables it.
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
        if cache is None:
            cache = stats_cfg.get("cache", False)
        cache_dir = cls._resolve_cache_dir(cache, work_dir)
        return cls(work_dir, labels, used, radius_hit=radius_hit,
                   groups=config.get("groups") or {}, name=name, cache_dir=cache_dir)

    @staticmethod
    def _resolve_cache_dir(cache, work_dir):
        """Turn a ``cache`` config value (bool / path / None) into a cache directory."""
        if not cache:
            return None
        if cache is True:
            return work_dir + ".morphometrics_cache/"
        cache = str(cache)
        return cache if cache.endswith("/") else cache + "/"

    # --- paths and lazy loading ---

    def surface_path(self, tomo, label):
        return f"{self.work_dir}{tomo}_{label}{self.extension}"

    def surface(self, tomo, label):
        return SurfaceView(self, tomo, label)

    def load(self, tomo, label):
        """Read (and cache) a surface's per-triangle dataframe.

        In-memory-cached per process. When a `cache_dir` is set, the parsed table is also
        cached to parquet keyed by the source CSV's path/mtime/size, so a later run (or a
        later Dataset) skips the CSV parse -- the slow step -- and reads parquet instead.
        Editing/re-running the surface changes the key, so a stale cache is never used.
        """
        key = (tomo, label)
        if key in self._cache:
            return self._cache[key]
        path = self.surface_path(tomo, label)
        if not os.path.isfile(path):
            raise KeyError(f"no surface file for tomogram {tomo!r}, class {label!r} "
                           f"(expected {path})")
        df = self._read_with_cache(tomo, label, path)
        self._cache[key] = df
        return df

    def _read_with_cache(self, tomo, label, path):
        """Read `path`, going through the parquet parse-cache when one is configured."""
        if not self.cache_dir or self._parquet_disabled:
            return pd.read_csv(path)
        cpath = f"{self.cache_dir}{tomo}_{label}-{_surface_cache_key(path)}.parquet"
        if os.path.isfile(cpath):
            try:
                return pd.read_parquet(cpath)
            except Exception:
                pass   # unreadable/corrupt cache -> fall through and re-parse
        df = pd.read_csv(path)
        self._write_cache(tomo, label, cpath, df)
        return df

    def _write_cache(self, tomo, label, cpath, df):
        """Best-effort parquet write; disable the cache (with a note) if it can't work."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # A fresh source (new mtime/size) yields a new key; drop the old entry so the
            # cache directory does not accumulate one parquet per edit.
            for stale in glob(f"{self.cache_dir}{tomo}_{label}-*.parquet"):
                if stale != cpath:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            df.to_parquet(cpath, index=False)
        except Exception as exc:
            self._parquet_disabled = True
            print(f"  NOTE: parquet caching disabled ({type(exc).__name__}: {exc}). "
                  "Install pyarrow to enable it; analysis continues from the CSVs.")

    def to_experiment(self, name=None):
        """Materialize an `morphometrics_stats.Experiment` (loads every present surface).

        A convenience bridge for the pickle-based `old_scripts/`; loses the lazy loading
        and the group metadata.
        """
        from .morphometrics_stats import Experiment, Tomogram
        exp = Experiment(name or self.name or "dataset")
        for tomo in self.tomogram_names:
            t = Tomogram(tomo, [], [])
            for label in self[tomo].classes():
                t[label] = self.load(tomo, label)
            exp[tomo] = t
        return exp

    def to_pickle(self, path, name=None):
        """Export a materialized `Experiment` pickle for continuity with older scripts.

        This is a convenience snapshot, NOT the cache: pickles of numpy-backed objects do
        not load across a numpy major-version change (the reason the cache is parquet, not
        pickle). Prefer re-building the Dataset from the config where possible.
        """
        import pickle
        with open(path, "wb") as handle:
            pickle.dump(self.to_experiment(name=name), handle)
        return path

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
