import json

import pytest

import config
from src.manifest import ManifestError, expected_values, load_manifest


def write_manifest(tmp_path, tests, sources=("prog.cu",)):
    for name in sources:
        (tmp_path / name).write_text("// placeholder\n")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tests": tests}))
    return path


def entry(**overrides):
    base = {
        "id": "t1",
        "description": "a test",
        "source": "prog.cu",
        "args": ["4"],
        "expected_behavior": "adds numbers",
        "verification": "reference",
        "reference": "vec_add_int",
        "output_type": "int",
    }
    base.update(overrides)
    return base


def test_valid_manifest_loads(tmp_path):
    tests = load_manifest(write_manifest(tmp_path, [entry()]))
    assert len(tests) == 1
    assert tests[0].id == "t1"
    assert tests[0].source == tmp_path / "prog.cu"
    assert tests[0].args == ["4"]


def test_missing_manifest_file(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(path)


def test_missing_tests_list(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"something": []}))
    with pytest.raises(ManifestError, match="'tests' list"):
        load_manifest(path)


def test_missing_field_is_reported(tmp_path):
    bad = entry()
    del bad["expected_behavior"]
    with pytest.raises(ManifestError, match="missing field 'expected_behavior'"):
        load_manifest(write_manifest(tmp_path, [bad]))


def test_missing_source_file(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(write_manifest(tmp_path, [entry(source="other.cu")]))


def test_duplicate_ids(tmp_path):
    with pytest.raises(ManifestError, match="duplicate id 't1'"):
        load_manifest(write_manifest(tmp_path, [entry(), entry()]))


def test_unknown_reference(tmp_path):
    with pytest.raises(ManifestError, match="unknown reference"):
        load_manifest(write_manifest(tmp_path, [entry(reference="does_not_exist")]))


def test_all_problems_reported_together(tmp_path):
    with pytest.raises(ManifestError) as info:
        load_manifest(write_manifest(tmp_path, [entry(output_type="complex"),
                                                entry(id="t2", verification="magic")]))
    assert "output_type" in str(info.value)
    assert "verification" in str(info.value)


def test_expected_output_strategy(tmp_path):
    tests = load_manifest(write_manifest(tmp_path, [
        entry(verification="expected_output", expected_output=[0, 1, 4], reference=None)]))
    assert expected_values(tests[0]) == [0, 1, 4]


def test_reference_strategy(tmp_path):
    tests = load_manifest(write_manifest(tmp_path, [entry()]))
    assert expected_values(tests[0]) == [-68, -48, -28, -45]


def test_project_manifest_is_valid():
    """The real test_cases/manifest.json must load, and every reference must run."""
    tests = load_manifest(config.MANIFEST_PATH)
    assert len(tests) >= 8
    for test in tests:
        assert len(expected_values(test)) > 0
