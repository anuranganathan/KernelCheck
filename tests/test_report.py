import json
from unittest import mock

import pytest

import main
from src import pipeline, report
from src.environment import Environment

NO_CUDA = Environment(nvcc_path=None, nvcc_version=None, gpu_names=[])


def make_record(test_id, config_name, status, compile_s=None, kernel_ms=None):
    return {
        "test_id": test_id, "config": config_name, "status": status,
        "compile": {"status": "SUCCESS" if compile_s else "SKIPPED_NO_CUDA",
                    "duration_s": compile_s, "output": ""},
        "run": {"status": "OK" if kernel_ms else "NOT_RUN", "kernel_ms": kernel_ms, "message": ""},
        "verification": {"detail": "details here"},
    }


def test_summary_of_empty_results():
    summary = report.summarize([])
    assert summary["total"] == summary["passed"] == summary["failed"] == summary["skipped"] == 0
    assert summary["by_config"] == {}
    text = report.format_summary(report.build_report(NO_CUDA, {}, None, [], []))
    assert "No tests were run" in text


def test_summary_counts():
    records = [make_record("a", "O0", pipeline.PASS), make_record("a", "O2", pipeline.OUTPUT_MISMATCH),
               make_record("b", "O0", pipeline.COMPILE_ERROR),
               make_record("b", "O2", pipeline.SKIPPED_NO_CUDA)]
    summary = report.summarize(records)
    assert summary["total"] == 4
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["skipped"] == 1
    assert summary["by_config"]["O2"][pipeline.OUTPUT_MISMATCH] == 1
    assert summary["by_config"]["O0"][pipeline.COMPILE_ERROR] == 1


def test_json_report_round_trip(tmp_path):
    records = [make_record("a", "O0", pipeline.PASS)]
    data = report.build_report(NO_CUDA, {"O0": ["-O0"]}, "sm_75", records, [])
    path = tmp_path / "out" / "report.json"
    report.write_json(data, path)
    loaded = json.loads(path.read_text())
    assert loaded["summary"]["passed"] == 1
    assert loaded["environment"]["can_compile"] is False
    assert loaded["arch"] == "sm_75"
    assert loaded["results"][0]["test_id"] == "a"


def test_no_charts_without_measurements(tmp_path):
    records = [make_record("a", "O0", pipeline.SKIPPED_NO_CUDA)]
    assert report.make_charts(records, tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_charts_drawn_from_measurements(tmp_path):
    pytest.importorskip("matplotlib")
    # Synthetic numbers, only to exercise the plotting code in a temp folder.
    records = [make_record("a", "O0", pipeline.PASS, compile_s=1.5, kernel_ms=0.02),
               make_record("a", "O2", pipeline.PASS, compile_s=1.7, kernel_ms=0.01)]
    written = report.make_charts(records, tmp_path)
    assert [p.name for p in written] == ["compile_times.png", "kernel_times.png"]
    assert all(p.stat().st_size > 0 for p in written)


def test_cli_run_without_cuda_skips_everything(tmp_path, capsys):
    """End-to-end run of `main.py run` with CUDA reported as missing."""
    output = tmp_path / "report.json"
    with mock.patch("src.environment.detect_environment", return_value=NO_CUDA):
        exit_code = main.main(["run", "--output", str(output), "--tests", "int_add,squares_literal"])
    assert exit_code == 0
    data = json.loads(output.read_text())
    assert data["summary"]["total"] == 4                 # 2 tests x 2 default configs
    assert data["summary"]["skipped"] == 4
    assert data["summary"]["passed"] == 0
    assert (tmp_path / "report_summary.txt").is_file()
    assert not (tmp_path / "compile_times.png").exists()
    assert "NOT FOUND" in capsys.readouterr().out


def test_cli_rejects_unknown_config_and_test(capsys):
    assert main.main(["run", "--configs", "O9"]) == 2
    assert main.main(["run", "--tests", "nope"]) == 2


def test_cli_analyze_missing_ptx(capsys):
    assert main.main(["analyze-ptx", "does_not_exist.ptx"]) == 2
    assert "File not found" in capsys.readouterr().err
