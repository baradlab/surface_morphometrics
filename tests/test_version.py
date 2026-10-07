"""The package's two version strings must agree.

`surface_morphometrics.__version__` and `pyproject.toml`'s `version` have drifted
apart twice (fixed in 2.0.0b3, drifted again at 2.0.0b5) because a release bumps one
and forgets the other. This pins them together.
"""
import os
import tomllib

import pytest

import surface_morphometrics

PYPROJECT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml"
)


@pytest.mark.skipif(not os.path.exists(PYPROJECT),
                    reason="not running from a source checkout")
def test_dunder_version_matches_pyproject():
    with open(PYPROJECT, "rb") as handle:
        declared = tomllib.load(handle)["project"]["version"]
    assert surface_morphometrics.__version__ == declared, (
        "surface_morphometrics/__init__.py and pyproject.toml disagree on the "
        "version; a release must bump both."
    )
