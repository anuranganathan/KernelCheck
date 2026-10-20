"""Integration tests that use the real nvcc and a real NVIDIA GPU.

Run with:  python -m pytest -m cuda -v
Without CUDA these tests are reported as SKIPPED (never as passed).
"""

import pytest

import config
from src import compiler, environment, pipeline
from src.manifest import TestCase, load_manifest

ENV = environment.detect_environment()
needs_nvcc = pytest.mark.skipif(not ENV.can_compile, reason="nvcc not found")
needs_gpu = pytest.mark.skipif(not ENV.can_run, reason="nvcc or NVIDIA GPU not found")
pytestmark = pytest.mark.cuda

TESTS = {t.id: t for t in load_manifest(config.MANIFEST_PATH)}


@needs_nvcc
def test_nvcc_version_detected():
    assert ENV.nvcc_version is not None


@needs_nvcc
def test_real_compile_produces_executable_and_ptx(tmp_path):
    result = compiler.compile_test(TESTS["int_add"], "O2", config.COMPILER_CONFIGS["O2"],
                                   ENV.nvcc_path, tmp_path, config.COMPILE_TIMEOUT_S)
    assert result.status == compiler.SUCCESS, result.output
    assert result.executable.is_file()
    assert result.ptx_path is not None and result.ptx_path.is_file()
    stats = pipeline.analyze_ptx(result)["stats"]
    assert any("add_kernel" in name for name in stats["entries"])
    assert stats["memory_by_space"].get("st.global", 0) >= 1


@needs_nvcc
def test_real_compile_error_is_classified(tmp_path):
    source = tmp_path / "broken.cu"
    source.write_text("__global__ void k() { undeclared_variable = 1; }\nint main() { return 0; }\n")
    broken = TestCase(id="broken", description="", source=source, expected_behavior="",
                      verification="expected_output", output_type="int", expected_output=[])
    result = compiler.compile_test(broken, "O2", config.COMPILER_CONFIGS["O2"], ENV.nvcc_path,
                                   tmp_path / "build", config.COMPILE_TIMEOUT_S)
    assert result.status == compiler.COMPILE_ERROR
    assert "undeclared_variable" in result.output


@needs_gpu
@pytest.mark.parametrize("test_id", ["int_add", "int_divmod", "float_arith", "loop_collatz",
                                     "boundary_guard", "size_1", "reduce_int", "squares_literal"])
def test_program_passes_on_gpu(test_id, tmp_path):
    record, actual = pipeline.run_test(TESTS[test_id], "O2", ENV, build_dir=tmp_path)
    assert record["status"] == pipeline.PASS, record["verification"]["detail"] + record["run"].get("stderr", "")
    assert record["run"]["kernel_ms"] is not None
    assert actual is not None


@needs_gpu
def test_two_configurations_agree_on_integer_test(tmp_path):
    records, outputs = [], {}
    for config_name in ("O0", "O2"):
        record, actual = pipeline.run_test(TESTS["size_257"],
                                           config_name, ENV, build_dir=tmp_path)
        records.append(record)
        outputs[(record["test_id"], config_name)] = actual
    [comparison] = pipeline.compare_configurations(records, outputs)
    assert comparison["outputs_identical"] is True
    assert not comparison["candidate_issue"]
    assert "O0 vs O2" in comparison["ptx_comparisons"]
