"""Execute compiled test programs and capture what they did."""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Run statuses
OK = "OK"
RUNTIME_ERROR = "RUNTIME_ERROR"
TIMEOUT = "TIMEOUT"
SKIPPED_NO_CUDA = "SKIPPED_NO_CUDA"


@dataclass
class RunResult:
    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    wall_time_s: float | None = None   # whole process, includes CUDA start-up
    kernel_ms: float | None = None     # GPU time reported by the program itself
    message: str = ""


def parse_kernel_time(stderr: str) -> float | None:
    """Read the 'kernel_ms=<value>' line printed by KernelTimer in kc_common.h."""
    times = [float(m) for m in re.findall(r"kernel_ms=([0-9]+(?:\.[0-9]+)?)", stderr)]
    return sum(times) if times else None


def run_executable(executable: Path, args: list[str], timeout: float) -> RunResult:
    start = time.perf_counter()
    try:
        process = subprocess.run([str(executable), *args], capture_output=True,
                                 text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return RunResult(TIMEOUT, wall_time_s=time.perf_counter() - start,
                         stderr=_as_text(exc.stderr),
                         message=f"program did not finish within {timeout} s")
    except OSError as exc:
        return RunResult(RUNTIME_ERROR, message=f"could not start program: {exc}")
    wall_time = time.perf_counter() - start

    result = RunResult(OK, exit_code=process.returncode, stdout=process.stdout,
                       stderr=process.stderr, wall_time_s=wall_time,
                       kernel_ms=parse_kernel_time(process.stderr))
    if process.returncode < 0:
        # On Linux/macOS a negative code means the process was killed by a signal
        # (e.g. -11 = SIGSEGV).
        result.status = RUNTIME_ERROR
        result.message = f"terminated by signal {-process.returncode}"
    elif process.returncode != 0:
        result.status = RUNTIME_ERROR
        result.message = f"exit code {process.returncode}"
    return result


def _as_text(data) -> str:
    """TimeoutExpired gives bytes even when text=True was requested."""
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data
