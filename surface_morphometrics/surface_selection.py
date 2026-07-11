#! /usr/bin/env python
"""Unit-level selection for the statistics tools.

Where `surface_filters` masks *triangles within* a surface, this module chooses *which
units* (tomograms, and the surfaces/organelles they contain) enter an analysis at all:

- **By tomogram identity** -- an include and/or exclude list of names or glob patterns,
  replacing the hardcoded `if key in [...]` / `key[1] == "F"` hacks that recur in the
  research scripts. A tomogram is used when it matches an include pattern (or none are
  given) and matches no exclude pattern.
- **By size** -- a surface or organelle is dropped when, after triangle filtering, it has
  fewer than `min_triangles` triangles or less than `min_area` total area, so tiny noisy
  fragments do not become spurious data points.

Patterns use shell-glob semantics (`fnmatch`): `TF*`, `?E*`, `UF3`. Selection is
structured and serializable, so an analysis can be pinned in a config `statistics:` block.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import fnmatch


def _matches_any(name, patterns):
    """True if `name` matches any of the glob `patterns`."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def select_tomogram(name, include=None, exclude=None):
    """Whether tomogram `name` is used, given include/exclude name-or-glob lists.

    An empty/None `include` means "all tomograms"; a non-empty `include` keeps only
    matches. `exclude` always wins over `include` (a name matching both is dropped).
    """
    include = include or []
    exclude = exclude or []
    if include and not _matches_any(name, include):
        return False
    if exclude and _matches_any(name, exclude):
        return False
    return True


def select_tomograms(names, include=None, exclude=None):
    """Split `names` into (used, excluded) lists, preserving order."""
    used, excluded = [], []
    for name in names:
        (used if select_tomogram(name, include, exclude) else excluded).append(name)
    return used, excluded


def passes_size(n_triangles, total_area, min_triangles=0, min_area=0.0):
    """Whether a unit meets the minimum triangle-count and total-area thresholds."""
    return n_triangles >= (min_triangles or 0) and total_area >= (min_area or 0.0)


def describe_selection(include=None, exclude=None, min_triangles=0, min_area=0.0):
    """A short human-readable summary of an active selection (empty string if none)."""
    parts = []
    if include:
        parts.append("include " + " ".join(include))
    if exclude:
        parts.append("exclude " + " ".join(exclude))
    if min_triangles:
        parts.append(f"min_triangles={min_triangles}")
    if min_area:
        parts.append(f"min_area={min_area:g}")
    return ", ".join(parts)
