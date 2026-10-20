import subprocess
from unittest import mock

from src import runner
from tests.helpers import fake_program


def test_parse_kernel_time():
    assert runner.parse_kernel_time("note\nkernel_ms=0.0123\n") == 0.0123
    assert runner.parse_kernel_time("no timing here") is None


# --- real subprocesses (shell scripts stand in for CUDA programs) ---

def test_successful_program(tmp_path):
    program = fake_program(tmp_path, "echo 1 2 3\necho kernel_ms=0.5 >&2")
    result = runner.run_executable(program, [], timeout=10)
    assert result.status == runner.OK
    assert result.exit_code == 0
    assert result.stdout.split() == ["1", "2", "3"]
    assert result.kernel_ms == 0.5
    assert result.wall_time_s > 0


def test_arguments_are_passed_as_a_list(tmp_path):
    program = fake_program(tmp_path, 'echo "$1|$2"')
    # The second argument would be dangerous if it were given to a shell.
    result = runner.run_executable(program, ["42", "; rm -rf x"], timeout=10)
    assert result.stdout.strip() == "42|; rm -rf x"


def test_nonzero_exit_is_runtime_error(tmp_path):
    program = fake_program(tmp_path, "echo 'CUDA error cudaErrorNoDevice' >&2\nexit 1")
    result = runner.run_executable(program, [], timeout=10)
    assert result.status == runner.RUNTIME_ERROR
    assert result.exit_code == 1
    assert "cudaErrorNoDevice" in result.stderr


def test_crash_by_signal_is_runtime_error(tmp_path):
    program = fake_program(tmp_path, "kill -SEGV $$")
    result = runner.run_executable(program, [], timeout=10)
    assert result.status == runner.RUNTIME_ERROR
    assert "signal 11" in result.message


def test_slow_program_times_out(tmp_path):
    program = fake_program(tmp_path, "sleep 5")
    result = runner.run_executable(program, [], timeout=0.5)
    assert result.status == runner.TIMEOUT


def test_missing_executable_is_runtime_error(tmp_path):
    result = runner.run_executable(tmp_path / "does_not_exist", [], timeout=10)
    assert result.status == runner.RUNTIME_ERROR
    assert "could not start" in result.message


# --- mocked ---

def test_timeout_with_partial_bytes_output(tmp_path):
    error = subprocess.TimeoutExpired(["prog"], 1, output=b"1 2", stderr=b"partial")
    with mock.patch("subprocess.run", side_effect=error):
        result = runner.run_executable(tmp_path / "prog", [], timeout=1)
    assert result.status == runner.TIMEOUT
    assert result.stderr == "partial"
