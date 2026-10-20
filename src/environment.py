"""Detect whether the CUDA toolchain and an NVIDIA GPU are available.

These checks never assume CUDA exists: on a MacBook (or any machine without
the CUDA toolkit) they simply report "unavailable" so that the rest of the
framework can mark CUDA-dependent work as SKIPPED_NO_CUDA.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Environment:
    nvcc_path: str | None
    nvcc_version: str | None
    gpu_names: list[str]

    @property
    def can_compile(self) -> bool:
        return self.nvcc_path is not None

    @property
    def can_run(self) -> bool:
        return self.can_compile and len(self.gpu_names) > 0


def find_nvcc() -> str | None:
    """Locate nvcc without hard-coding a path.

    Order: the NVCC environment variable, then $CUDA_HOME/bin/nvcc,
    then whatever 'nvcc' is on PATH.
    """
    explicit = os.environ.get("NVCC")
    if explicit and Path(explicit).is_file():
        return explicit

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)

    return shutil.which("nvcc")


def parse_nvcc_version(text: str) -> str | None:
    """Extract e.g. '12.4' from the output of 'nvcc --version'."""
    match = re.search(r"release (\d+\.\d+)", text)
    return match.group(1) if match else None


def get_nvcc_version(nvcc_path: str) -> str | None:
    try:
        result = subprocess.run([nvcc_path, "--version"], capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_nvcc_version(result.stdout)


def detect_gpus() -> list[str]:
    """Return GPU names reported by 'nvidia-smi -L', or [] if none/unavailable."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        result = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    # Lines look like: "GPU 0: Tesla T4 (UUID: GPU-...)"
    names = []
    for line in result.stdout.splitlines():
        match = re.match(r"GPU \d+: (.+?) \(UUID", line.strip())
        if match:
            names.append(match.group(1))
    return names


def detect_environment() -> Environment:
    nvcc = find_nvcc()
    version = get_nvcc_version(nvcc) if nvcc else None
    return Environment(nvcc_path=nvcc, nvcc_version=version, gpu_names=detect_gpus())


def describe(env: Environment) -> str:
    """Human-readable environment summary, including what will be skipped."""
    lines = []
    if env.nvcc_path:
        lines.append(f"nvcc        : {env.nvcc_path} (CUDA {env.nvcc_version or 'unknown version'})")
    else:
        lines.append("nvcc        : NOT FOUND - CUDA compilation unavailable")
    if env.gpu_names:
        lines.append(f"NVIDIA GPU  : {', '.join(env.gpu_names)}")
    else:
        lines.append("NVIDIA GPU  : NOT FOUND - CUDA execution unavailable")

    if not env.can_compile:
        lines.append("\nCUDA tests will be reported as SKIPPED_NO_CUDA. Host-only features "
                     "(unit tests, manifest validation, references, PTX analysis of existing "
                     "files) still work. See README section 'NVIDIA GPU setup'.")
    elif not env.can_run:
        lines.append("\nPrograms will be compiled (and PTX analysed) but not executed; "
                     "execution results will be SKIPPED_NO_CUDA.")
    return "\n".join(lines)
