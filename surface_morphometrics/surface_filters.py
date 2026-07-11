#! /usr/bin/env python
"""Structured triangle-level filters for the statistics tools.

A filter keeps triangles whose per-triangle property satisfies a comparison, applied to a
surface BEFORE its feature is summarized -- e.g. isolate mitochondrial cristae by keeping
only IMM triangles far from the OMM junction (`IMM:OMM_dist>=20`) and then quantify their
curvature. Filters are ANDed. Each is a single comparison; use two filters for a range.

Syntax (no spaces):  [CLASS:]PROPERTY OP VALUE
  OMM_dist>=20        # every class that has an OMM_dist column
  IMM:OMM_dist<12     # only the IMM class
  IMM:OMM_dist>=5     # }
  IMM:OMM_dist<=20    # } together: IMM triangles with 5 <= OMM_dist <= 20 (cristae)
OP is one of  >=  <=  ==  !=  >  < .

Parsing is structured (a regex + a fixed operator table), never `eval`, so a filter string
can go safely into a config file.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import operator
import re

import numpy as np

_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq,
        "!=": operator.ne, ">": operator.gt, "<": operator.lt}

# [CLASS:]PROPERTY OP VALUE  -- OP alternation lists two-char ops before one-char ones.
_CLAUSE = re.compile(
    r"^(?:(?P<cls>[A-Za-z0-9_]+):)?(?P<prop>[A-Za-z0-9_]+)"
    r"(?P<op>>=|<=|==|!=|>|<)(?P<val>[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)$")


def parse_filter(spec):
    """Parse one '[CLASS:]PROPERTY OP VALUE' string into a clause dict.

    Returns {cls, prop, op, value, spec}; `cls` is None for an unscoped filter.
    Raises ValueError on a malformed spec.
    """
    m = _CLAUSE.match(str(spec).strip())
    if not m:
        raise ValueError(
            f"bad filter {spec!r}; expected [CLASS:]PROPERTY OP VALUE with OP one of "
            f">= <= == != > <  (e.g. 'IMM:OMM_dist>=20')")
    return {"cls": m.group("cls"), "prop": m.group("prop"), "op": m.group("op"),
            "value": float(m.group("val")), "spec": m.group(0)}


def parse_filters(specs):
    """Parse a list of filter strings into clause dicts (empty list for None)."""
    return [parse_filter(s) for s in (specs or [])]


def clauses_for_class(clauses, label):
    """The clauses that apply to class `label`; unscoped clauses apply to every class."""
    return [c for c in (clauses or []) if c["cls"] is None or c["cls"] == label]


def filter_mask(df, clauses, label):
    """Boolean row mask for the clauses that apply to class `label`.

    A clause whose property is absent from this surface excludes every row (the surface
    cannot satisfy the filter). NaN values never satisfy a comparison, so they are
    excluded by any active clause. Returns all-True when no clause applies.
    """
    n = len(df)
    applicable = clauses_for_class(clauses, label)
    if not applicable:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for c in applicable:
        if c["prop"] not in df.columns:
            return np.zeros(n, dtype=bool)
        col = df[c["prop"]].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            mask &= _OPS[c["op"]](col, c["value"])
    return mask


def describe(clauses):
    """A short human-readable summary of a clause list, e.g. 'IMM:OMM_dist>=20, ...'."""
    return ", ".join(c["spec"] for c in (clauses or []))
