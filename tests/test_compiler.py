"""Compiler tests with subprocess.run mocked, so no CUDA toolkit is needed."""

import subprocess
from pathlib import Path
from unittest import mock

from src import compiler
from src.manifest import TestCase


def make_test(tmp_path) -> TestCase:
    source = tmp_path / "prog.cu"
    source.write_text("// placeholder\n")
    return TestCase(id="prog", description="", source=source, expected_behavior="",
                    verification="reference", output_type="int", reference="vec_add_int")


def fake_nvcc(returncode=0, stderr="", create_output=True, ptx_returncode=0):
    """Return a stand-in for subprocess.run that behaves like nvcc."""
    def run(command, **kwargs):
        assert isinstance(command, list)          # never a shell string
        assert kwargs.get("shell") is not True
        is_ptx = "-ptx" in command
        code = ptx_returncode if is_ptx else returncode
        if code == 0 and create_output:
            Path(command[command.index("-o") + 1]).write_text("output")
        return subprocess.CompletedProcess(command, code, stdout="", stderr=stderr)
    return run


def compile_with(tmp_path, run, flags=("-O2",), arch=None):
    with mock.patch("subprocess.run", side_effect=run) as mocked:
        result = compiler.compile_test(make_test(tmp_path), "O2", list(flags), "/fake/nvcc",
                                       tmp_path / "build", timeout=5, arch=arch)
    return result, mocked


def test_missing_nvcc_skips_compilation(tmp_path):
    result = compiler.compile_test(make_test(tmp_path), "O2", ["-O2"], None,
                                   tmp_path / "build", timeout=5)
    assert result.status == compiler.SKIPPED_NO_CUDA
    assert result.executable is None


def test_successful_compile_produces_executable_and_ptx(tmp_path):
    result, mocked = compile_with(tmp_path, fake_nvcc())
    assert result.status == compiler.SUCCESS
    assert result.executable == tmp_path / "build" / "O2" / "prog"
    assert result.ptx_path == tmp_path / "build" / "O2" / "prog.ptx"
    assert result.duration_s is not None and result.duration_s >= 0
    assert mocked.call_count == 2            # executable + PTX


def test_command_is_an_argument_list_with_flags(tmp_path):
    result, _ = compile_with(tmp_path, fake_nvcc(), flags=("-O0", "-Xptxas", "-O0"), arch="sm_75")
    assert result.command[0] == "/fake/nvcc"
    assert result.command[1:4] == ["-O0", "-Xptxas", "-O0"]
    assert "-arch=sm_75" in result.command


def test_compile_error_is_classified(tmp_path):
    message = "prog.cu(3): error: identifier \"x\" is undefined"
    result, mocked = compile_with(tmp_path, fake_nvcc(returncode=1, stderr=message))
    assert result.status == compiler.COMPILE_ERROR
    assert result.exit_code == 1
    assert "undefined" in result.output
    assert result.executable is None
    assert mocked.call_count == 1            # no PTX attempt after a failure


def test_zero_exit_without_output_file_is_an_error(tmp_path):
    result, _ = compile_with(tmp_path, fake_nvcc(create_output=False))
    assert result.status == compiler.COMPILE_ERROR


def test_compile_timeout_is_classified(tmp_path):
    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    result, _ = compile_with(tmp_path, hang)
    assert result.status == compiler.TIMEOUT
    assert "did not finish" in result.output


def test_nvcc_that_cannot_start_is_a_compile_error(tmp_path):
    def broken(command, **kwargs):
        raise PermissionError("not executable")
    result, _ = compile_with(tmp_path, broken)
    assert result.status == compiler.COMPILE_ERROR
    assert "could not start nvcc" in result.output


def test_ptx_failure_keeps_executable(tmp_path):
    result, _ = compile_with(tmp_path, fake_nvcc(ptx_returncode=1, stderr="ptx problem"))
    assert result.status == compiler.SUCCESS
    assert result.ptx_path is None
    assert "ptx problem" in result.ptx_error


def test_stale_executable_is_removed_before_compiling(tmp_path):
    stale = tmp_path / "build" / "O2" / "prog"
    stale.parent.mkdir(parents=True)
    stale.write_text("old binary")
    result, _ = compile_with(tmp_path, fake_nvcc(returncode=1))
    assert result.status == compiler.COMPILE_ERROR
    assert not stale.exists()
