from pathlib import Path
from unittest import mock

import pytest

from src import compiler, pipeline, runner
from src.environment import Environment
from tests.helpers import fake_program, squares_test

NO_CUDA = Environment(nvcc_path=None, nvcc_version=None, gpu_names=[])
NVCC_ONLY = Environment(nvcc_path="/fake/nvcc", nvcc_version="12.4", gpu_names=[])
FULL_CUDA = Environment(nvcc_path="/fake/nvcc", nvcc_version="12.4", gpu_names=["Fake GPU"])


@pytest.mark.parametrize("compile_status, run_status, output_ok, expected", [
    (compiler.SKIPPED_NO_CUDA, None, None, pipeline.SKIPPED_NO_CUDA),
    (compiler.COMPILE_ERROR, None, None, pipeline.COMPILE_ERROR),
    (compiler.TIMEOUT, None, None, pipeline.TIMEOUT),
    (compiler.SUCCESS, None, None, pipeline.SKIPPED_NO_CUDA),
    (compiler.SUCCESS, runner.SKIPPED_NO_CUDA, None, pipeline.SKIPPED_NO_CUDA),
    (compiler.SUCCESS, runner.RUNTIME_ERROR, None, pipeline.RUNTIME_ERROR),
    (compiler.SUCCESS, runner.TIMEOUT, None, pipeline.TIMEOUT),
    (compiler.SUCCESS, runner.OK, False, pipeline.OUTPUT_MISMATCH),
    (compiler.SUCCESS, runner.OK, True, pipeline.PASS),
])
def test_classify(compile_status, run_status, output_ok, expected):
    assert pipeline.classify(compile_status, run_status, output_ok) == expected


def test_missing_cuda_is_skipped_not_passed(tmp_path):
    record, actual = pipeline.run_test(squares_test(tmp_path), "O2", NO_CUDA,
                                       build_dir=tmp_path / "build")
    assert record["status"] == pipeline.SKIPPED_NO_CUDA
    assert record["verification"]["passed"] is None
    assert record["run"]["status"] == "NOT_RUN"
    assert actual is None


def run_with_fake_binary(tmp_path, script_body, env=FULL_CUDA):
    """Pretend nvcc succeeded and produced `script_body` as the executable."""
    program = fake_program(tmp_path, script_body)
    built = compiler.CompileResult(compiler.SUCCESS, executable=program, duration_s=0.1,
                                   exit_code=0, ptx_error="no PTX in this test")
    with mock.patch("src.compiler.compile_test", return_value=built):
        return pipeline.run_test(squares_test(tmp_path), "O2", env, build_dir=tmp_path / "build")


def test_compiled_but_no_gpu_is_skipped(tmp_path):
    record, _ = run_with_fake_binary(tmp_path, "echo 0 1 4 9", env=NVCC_ONLY)
    assert record["status"] == pipeline.SKIPPED_NO_CUDA
    assert record["compile"]["status"] == compiler.SUCCESS
    assert record["run"]["status"] == runner.SKIPPED_NO_CUDA


def test_correct_output_passes(tmp_path):
    record, actual = run_with_fake_binary(tmp_path, "echo 0 1 4 9\necho kernel_ms=0.01 >&2")
    assert record["status"] == pipeline.PASS
    assert actual == [0, 1, 4, 9]
    assert record["run"]["kernel_ms"] == 0.01
    assert record["verification"]["actual"] == {"count": 4, "first_values": [0, 1, 4, 9]}


def test_wrong_output_is_mismatch(tmp_path):
    record, _ = run_with_fake_binary(tmp_path, "echo 0 1 5 9")
    assert record["status"] == pipeline.OUTPUT_MISMATCH
    assert record["verification"]["first_mismatch_index"] == 2


def test_garbage_output_is_mismatch(tmp_path):
    record, actual = run_with_fake_binary(tmp_path, "echo hello world")
    assert record["status"] == pipeline.OUTPUT_MISMATCH
    assert "unparseable" in record["verification"]["detail"]
    assert actual is None


def test_failing_program_is_runtime_error(tmp_path):
    record, _ = run_with_fake_binary(tmp_path, "exit 3")
    assert record["status"] == pipeline.RUNTIME_ERROR
    assert record["run"]["exit_code"] == 3


def make_record(test_id, config_name, status, output_type="int", compile_status=compiler.SUCCESS):
    return {"test_id": test_id, "config": config_name, "status": status,
            "compile": {"status": compile_status},
            "verification": {"output_type": output_type}}


def test_configs_that_agree_are_not_flagged():
    records = [make_record("t", "O0", "PASS"), make_record("t", "O2", "PASS")]
    outputs = {("t", "O0"): [1, 2], ("t", "O2"): [1, 2]}
    [result] = pipeline.compare_configurations(records, outputs)
    assert result["outputs_identical"] is True
    assert not result["candidate_issue"]


def test_mismatch_in_one_config_is_a_candidate_issue():
    records = [make_record("t", "O0", "PASS"), make_record("t", "O2", "OUTPUT_MISMATCH")]
    outputs = {("t", "O0"): [1, 2], ("t", "O2"): [1, 3]}
    [result] = pipeline.compare_configurations(records, outputs)
    assert result["candidate_issue"]
    assert any("O2" in reason for reason in result["reasons"])
    assert any("requiring investigation" in note for note in result["notes"])


def test_compile_error_in_only_one_config_is_flagged():
    records = [make_record("t", "O0", "PASS"),
               make_record("t", "G", "COMPILE_ERROR", compile_status=compiler.COMPILE_ERROR)]
    [result] = pipeline.compare_configurations(records, {("t", "O0"): [1]})
    assert result["candidate_issue"]
    assert result["outputs_identical"] is None   # only one config produced output


def test_float_differences_within_tolerance_are_only_a_note():
    records = [make_record("t", "O0", "PASS", "float"), make_record("t", "O2", "PASS", "float")]
    outputs = {("t", "O0"): [1.0], ("t", "O2"): [1.0000001]}
    [result] = pipeline.compare_configurations(records, outputs)
    assert result["outputs_identical"] is False
    assert not result["candidate_issue"]
    assert result["notes"]


def test_skipped_tests_are_not_flagged():
    records = [make_record("t", "O0", "SKIPPED_NO_CUDA", compile_status=compiler.SKIPPED_NO_CUDA)]
    [result] = pipeline.compare_configurations(records, {})
    assert not result["candidate_issue"]


SAMPLE_PTX = Path(__file__).parent / "data" / "sample_add.ptx"


def test_ptx_is_analysed_when_available(tmp_path):
    built = compiler.CompileResult(compiler.SUCCESS, ptx_path=SAMPLE_PTX)
    ptx = pipeline.analyze_ptx(built)
    assert ptx["available"]
    assert ptx["stats"]["total_instructions"] == 22
    assert len(ptx["sha256"]) == 64


def test_missing_ptx_is_reported_as_skipped(tmp_path):
    record, _ = run_with_fake_binary(tmp_path, "echo 0 1 4 9")
    assert record["ptx"] == {"available": False, "reason": "no PTX in this test"}
    built = compiler.CompileResult(compiler.SUCCESS, ptx_path=tmp_path / "gone.ptx")
    assert "could not read PTX" in pipeline.analyze_ptx(built)["reason"]


def test_ptx_compared_across_configurations():
    ptx = pipeline.analyze_ptx(compiler.CompileResult(compiler.SUCCESS, ptx_path=SAMPLE_PTX))
    records = [make_record("t", "O0", "PASS"), make_record("t", "O2", "PASS")]
    for record in records:
        record["ptx"] = ptx
    [result] = pipeline.compare_configurations(records, {})
    comparison = result["ptx_comparisons"]["O0 vs O2"]
    assert comparison["identical_statistics"]
    assert comparison["text_identical"]


def test_non_finite_values_are_json_safe():
    import json
    preview = pipeline._preview([1.0, float("nan"), float("inf")])
    assert preview["first_values"] == [1.0, "nan", "inf"]
    json.dumps(preview, allow_nan=False)    # raises if NaN slipped through
