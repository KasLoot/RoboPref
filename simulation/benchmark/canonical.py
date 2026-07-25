"""Cross-platform canonicalisation for benchmark catalog values."""

from __future__ import annotations

import math
from typing import Any


CATALOG_FLOAT_DECIMAL_PLACES = 12
CATALOG_FLOAT_ABS_TOLERANCE = 1e-12
CATALOG_FLOAT_REL_TOLERANCE = 1e-12


def stable_float(value: float) -> float:
    """Return a finite float with platform-independent catalog precision."""

    if not math.isfinite(value):
        raise ValueError("Benchmark catalog floats must be finite.")
    result = round(value, CATALOG_FLOAT_DECIMAL_PLACES)
    return 0.0 if result == 0.0 else result


def catalog_values_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values while tolerating only negligible float drift."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            catalog_values_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            catalog_values_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(actual) is float and type(expected) is float:
        return math.isclose(
            actual,
            expected,
            rel_tol=CATALOG_FLOAT_REL_TOLERANCE,
            abs_tol=CATALOG_FLOAT_ABS_TOLERANCE,
        )
    return type(actual) is type(expected) and actual == expected
