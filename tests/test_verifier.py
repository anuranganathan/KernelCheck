import math

import pytest

from src.verifier import compare, parse_output, values_close


def test_parse_int_output():
    assert parse_output("1 2\n-3\n", "int") == [1, 2, -3]


def test_parse_float_output():
    assert parse_output("0.5 -1.25e+02\n", "float") == [0.5, -125.0]


def test_parse_empty_output():
    assert parse_output("", "int") == []


def test_parse_rejects_garbage():
    with pytest.raises(ValueError, match="token 1"):
        parse_output("1 hello 3", "int")


def test_parse_rejects_float_for_int_test():
    with pytest.raises(ValueError):
        parse_output("1.5", "int")


def test_exact_match_passes():
    result = compare([1, 2, 3], [1, 2, 3], "int")
    assert result.passed
    assert result.mismatch_count == 0


def test_exact_mismatch_reports_first_index():
    result = compare([1, 2, 3, 4], [1, 9, 3, 7], "int")
    assert not result.passed
    assert result.mismatch_count == 2
    assert result.first_mismatch_index == 1


def test_length_mismatch_fails():
    result = compare([1, 2, 3], [1, 2], "int")
    assert not result.passed
    assert "expected 3 values but got 2" in result.detail


def test_float_within_relative_tolerance():
    # 1000.0 vs 1000.005 differs by 5e-6 relative
    assert compare([1000.0], [1000.005], "float", rel_tol=1e-5, abs_tol=0).passed


def test_float_outside_relative_tolerance():
    result = compare([1000.0], [1000.1], "float", rel_tol=1e-5, abs_tol=0)
    assert not result.passed
    assert result.max_abs_error == pytest.approx(0.1)


def test_float_near_zero_uses_absolute_tolerance():
    # A relative tolerance alone would reject any non-zero answer here.
    assert values_close(0.0, 5e-7, rel_tol=1e-5, abs_tol=1e-6)
    assert not values_close(0.0, 5e-6, rel_tol=1e-5, abs_tol=1e-6)


def test_float_nan_never_passes():
    assert not compare([1.0], [math.nan], "float", rel_tol=1, abs_tol=1).passed


def test_exact_comparison_for_int_ignores_tolerance():
    assert not compare([100], [101], "int", rel_tol=0.5, abs_tol=5).passed
