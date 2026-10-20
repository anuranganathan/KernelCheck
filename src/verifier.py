"""Parse program output and compare it with the expected values.

Output protocol: a test program prints its results to stdout as
whitespace-separated numbers and nothing else. Diagnostics go to stderr.
"""

from dataclasses import dataclass


@dataclass
class Comparison:
    passed: bool
    detail: str
    mismatch_count: int = 0
    first_mismatch_index: int | None = None
    max_abs_error: float | None = None


def parse_output(stdout: str, output_type: str) -> list[float]:
    """Turn program stdout into a list of numbers. Raises ValueError on junk."""
    convert = int if output_type == "int" else float
    values = []
    for position, token in enumerate(stdout.split()):
        try:
            values.append(convert(token))
        except ValueError:
            raise ValueError(f"token {position} is not a valid {output_type}: {token!r}") from None
    return values


def values_close(expected: float, actual: float, rel_tol: float, abs_tol: float) -> bool:
    """True if |actual - expected| <= max(rel_tol * |expected|, abs_tol).

    The relative part scales with the magnitude of the expected value; the
    absolute part handles expected values at or near zero, where any relative
    tolerance would shrink to nothing. NaN never compares close to anything.
    """
    return abs(actual - expected) <= max(rel_tol * abs(expected), abs_tol)


def compare(expected: list[float], actual: list[float], output_type: str,
            rel_tol: float = 0.0, abs_tol: float = 0.0) -> Comparison:
    """Integers must match exactly; floats must be within the given tolerance."""
    if len(expected) != len(actual):
        return Comparison(False, f"expected {len(expected)} values but got {len(actual)}",
                          mismatch_count=abs(len(expected) - len(actual)))

    mismatches = 0
    first_index = None
    max_error = 0.0
    for i, (e, a) in enumerate(zip(expected, actual)):
        if output_type == "int":
            ok = e == a
        else:
            ok = values_close(e, a, rel_tol, abs_tol)
            # abs() of a NaN difference is NaN; keep max_error meaningful.
            error = abs(a - e)
            if error == error:
                max_error = max(max_error, error)
        if not ok:
            mismatches += 1
            if first_index is None:
                first_index = i

    max_abs_error = max_error if output_type == "float" else None
    if mismatches == 0:
        if output_type == "float":
            detail = (f"all {len(expected)} values within tolerance "
                      f"(rel={rel_tol:g}, abs={abs_tol:g}); max abs error {max_error:.3g}")
        else:
            detail = f"all {len(expected)} values match exactly"
        return Comparison(True, detail, max_abs_error=max_abs_error)

    e, a = expected[first_index], actual[first_index]
    detail = (f"{mismatches} of {len(expected)} values differ; first at index "
              f"{first_index}: expected {e!r}, got {a!r}")
    return Comparison(False, detail, mismatches, first_index, max_abs_error)
