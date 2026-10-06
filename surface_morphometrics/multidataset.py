#! /usr/bin/env python
"""Federate several morphometrics runs into one comparison.

A :class:`~surface_morphometrics.dataset.Dataset` indexes one run (one `work_dir`). Real
comparisons often span several independent runs -- e.g. a control dataset, a treatment
dataset, and a mutant dataset segmented and processed separately. `MultiDataset` presents
a set of named member datasets through the same collection interface as a single
`Dataset`, so `compare`, `compare_batch` and bespoke `spatial_stats` scripts never loop
over sources themselves.

Identity: units are namespaced ``<source>/<tomogram>`` (``<source>/<tomogram>#cN`` when
splitting components), and the stratum for nested permutation is ``<source>/<tomogram>``.
Two sources may therefore contain a tomogram with the same name without the units being
confused, and the permutation floor counts tomograms correctly.

Metadata: every member carries its own `groups:` (merged over any top-level `groups:`),
validated per member. Each tomogram additionally gets an implicit ``dataset`` group whose
value is its source name, so ``compare -g dataset`` compares whole runs without any
grouping config.

Config schema::

    classes: [OMM, IMM, ER]          # or segmentation_values: {OMM: 1, ...}
    statistics: {cache: true}         # applies to every CSV member
    datasets:
      control:
        work_dir: /data/control/morphometrics/
        groups: {condition: {Control: "*"}}
      mutant:
        work_dir: /data/mutant/morphometrics/   # used if it has surface CSVs...
        pickle: mutant.pkl                      # ...else this legacy Experiment pickle
        exclude_tomograms: [bad_tomo_*]
        groups:
          condition:
            Mutant positive: [tomoA, tomoB]
            Mutant negative: rest
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

from .dataset import Dataset, PickleDataset

SEP = "/"
SOURCE_GROUP = "dataset"


def split_stratum(stratum):
    """``'source/tomo'`` -> ``('source', 'tomo')``."""
    source, _, tomo = stratum.partition(SEP)
    return source, tomo


class MultiDataset:
    """Several named `Dataset`s presented as one, with namespaced units."""

    def __init__(self, members, name=None):
        """`members` is an ordered ``{source_name: Dataset}`` mapping."""
        if not members:
            raise ValueError("MultiDataset needs at least one member dataset")
        for source in members:
            if SEP in source:
                raise ValueError(f"dataset name {source!r} may not contain {SEP!r}")
        self.members = dict(members)
        self.name = name
        labels = []
        for ds in self.members.values():
            labels.extend(lab for lab in ds.labels if lab not in labels)
        self.labels = labels

    # --- construction ---

    @classmethod
    def from_config(cls, config, base_dir=None, cache=None, verbose=False):
        """Build from a config's top-level ``datasets:`` block (see module docstring).

        Each entry picks its backend by what is present: surface CSVs under `work_dir`
        (preferred), otherwise a legacy `pickle`. Relative paths resolve against
        `base_dir` (the config file's directory). An entry with no data raises, unless
        it sets ``optional: true``, in which case it is skipped with a note.
        """
        entries = config.get("datasets") or {}
        if not isinstance(entries, dict) or not entries:
            raise ValueError("config has no `datasets:` block")
        base_dir = base_dir or os.getcwd()
        top_groups = config.get("groups") or {}
        stats_cfg = config.get("statistics", {}) or {}
        if cache is None:
            cache = stats_cfg.get("cache", False) if isinstance(stats_cfg, dict) else False
        default_labels = _config_labels(config)
        radius_hit = (config.get("curvature_measurements") or {}).get("radius_hit", 9)

        members = {}
        for source, entry in entries.items():
            entry = entry or {}
            ds = _build_member(source, entry, base_dir, top_groups, default_labels,
                               radius_hit, cache)
            if ds is None:
                if entry.get("optional"):
                    print(f"  NOTE: dataset '{source}' has neither surface CSVs nor a "
                          "pickle -- skipped (optional).")
                    continue
                raise ValueError(
                    f"dataset '{source}' has no data: no *.AVV_rh{radius_hit}.csv under "
                    f"work_dir={entry.get('work_dir')!r} and no readable "
                    f"pickle={entry.get('pickle')!r}. Fix the path or mark it "
                    "`optional: true`.")
            members[source] = ds
            if verbose:
                backend = "pickle" if isinstance(ds, PickleDataset) else "csv"
                print(f"  {source:16s} [{backend:6s}] {len(ds):3d} tomogram(s)")
        if not members:
            raise ValueError("no dataset in `datasets:` could be loaded")
        return cls(members)

    # --- metadata ---

    def metadata(self, stratum):
        """Group metadata for ``'source/tomo'``, plus the implicit ``dataset`` group."""
        source, tomo = split_stratum(stratum)
        meta = {SOURCE_GROUP: source}
        meta.update(self.members[source].metadata(tomo))
        return meta

    def tomograms(self):
        """Every namespaced ``source/tomo`` across members."""
        return [f"{s}{SEP}{t}" for s, ds in self.members.items() for t in ds.tomograms()]

    @property
    def excluded_tomograms(self):
        return [f"{s}{SEP}{t}" for s, ds in self.members.items()
                for t in ds.excluded_tomograms]

    def group_values(self, group):
        """Ordered distinct values of `group` across all tomograms (None if missing)."""
        seen = []
        for stratum in self.tomograms():
            value = self.metadata(stratum).get(group)
            if value not in seen:
                seen.append(value)
        return seen

    def __len__(self):
        return sum(len(ds) for ds in self.members.values())

    def __getitem__(self, source):
        return self.members[source]

    def __iter__(self):
        return iter(self.members.items())

    # --- collection ---

    def collect_feature(self, feature, split_components=False,
                        component_column="component_number", filters=None,
                        min_triangles=0, min_area=0.0):
        """Same contract as :meth:`Dataset.collect_feature`, over every member.

        Records are ``(label, unit, stratum, values, areas)`` with `unit` and `stratum`
        namespaced by source; diagnostics are summed/concatenated across members.
        """
        records = []
        diagnostics = {"used_tomograms": [], "excluded_tomograms": [], "dropped_units": 0}
        for source, ds in self.members.items():
            recs, diag = ds.collect_feature(
                feature, split_components=split_components,
                component_column=component_column, filters=filters,
                min_triangles=min_triangles, min_area=min_area)
            for label, unit, stratum, values, areas in recs:
                records.append((label, f"{source}{SEP}{unit}", f"{source}{SEP}{stratum}",
                                values, areas))
            diagnostics["used_tomograms"] += [f"{source}{SEP}{t}"
                                              for t in diag["used_tomograms"]]
            diagnostics["excluded_tomograms"] += [f"{source}{SEP}{t}"
                                                  for t in diag["excluded_tomograms"]]
            diagnostics["dropped_units"] += diag["dropped_units"]
        # Keep the class order of `self.labels` (members already iterate label-major).
        order = {lab: i for i, lab in enumerate(self.labels)}
        records.sort(key=lambda r: order.get(r[0], len(order)))
        return records, diagnostics

    def __repr__(self):
        parts = ", ".join(f"{s}={len(ds)}" for s, ds in self.members.items())
        return f"MultiDataset({parts}; labels={self.labels})"


def _config_labels(config):
    """Class names from ``classes:`` or ``segmentation_values:`` (None if neither)."""
    if config.get("classes"):
        return list(config["classes"])
    if config.get("segmentation_values"):
        return list(config["segmentation_values"].keys())
    return None


def _resolve(path, base_dir):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _build_member(source, entry, base_dir, top_groups, default_labels, radius_hit, cache):
    """One member Dataset from a `datasets:` entry, or None if it has no data."""
    groups = dict(top_groups)
    groups.update(entry.get("groups") or {})
    labels = list(entry["classes"]) if entry.get("classes") else default_labels
    rh = entry.get("radius_hit", radius_hit)
    include, exclude = entry.get("include_tomograms"), entry.get("exclude_tomograms")

    work_dir = _resolve(entry.get("work_dir"), base_dir)
    if work_dir and not work_dir.endswith("/"):
        work_dir += "/"
    has_csvs = bool(work_dir and os.path.isdir(work_dir)
                    and glob(f"{work_dir}*.AVV_rh{rh}.csv"))
    if has_csvs:
        if labels is None:
            raise ValueError(
                f"dataset '{source}': cannot tell classes from CSV names alone; set a "
                "top-level `classes:` list (or `segmentation_values:`), or `classes:` on "
                "the entry.")
        seg_dir = _resolve(entry.get("seg_dir"), base_dir)
        if seg_dir and os.path.isdir(seg_dir):
            sub = {"work_dir": work_dir, "seg_dir": seg_dir.rstrip("/") + "/",
                   "segmentation_values": {lab: i for i, lab in enumerate(labels)},
                   "curvature_measurements": {"radius_hit": rh}, "groups": groups,
                   "statistics": {"include_tomograms": include,
                                  "exclude_tomograms": exclude, "cache": cache}}
            return Dataset.from_config(sub, name=source)
        return Dataset.from_work_dir(work_dir, labels, radius_hit=rh, groups=groups,
                                     name=source, cache=cache, include_tomograms=include,
                                     exclude_tomograms=exclude)

    pickle_path = _resolve(entry.get("pickle"), base_dir)
    if pickle_path and os.path.isfile(pickle_path):
        return PickleDataset.from_pickle(pickle_path, labels=labels, groups=groups,
                                         name=source, include_tomograms=include,
                                         exclude_tomograms=exclude)
    return None


def load_for_analysis(config, configfile=None, verbose=False):
    """The dataset an analysis command should use for `config`.

    A `MultiDataset` when the config has a ``datasets:`` block, otherwise the single
    `Dataset` of its ``seg_dir``/``work_dir`` -- the two share the collection interface
    (`labels`, `metadata(stratum)`, `collect_feature`), so callers need not care.
    """
    if config.get("datasets"):
        base = os.path.dirname(os.path.abspath(configfile)) if configfile else None
        return MultiDataset.from_config(config, base_dir=base, verbose=verbose)
    return Dataset.from_config(config)
