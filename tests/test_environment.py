import subprocess
from unittest import mock

from src import environment


def test_missing_nvcc_is_reported(monkeypatch):
    monkeypatch.delenv("NVCC", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    with mock.patch("shutil.which", return_value=None):
        env = environment.detect_environment()
    assert env.nvcc_path is None
    assert not env.can_compile
    assert not env.can_run
    assert "NOT FOUND" in environment.describe(env)


def test_nvcc_env_variable_takes_priority(monkeypatch, tmp_path):
    fake = tmp_path / "nvcc"
    fake.write_text("")
    monkeypatch.setenv("NVCC", str(fake))
    assert environment.find_nvcc() == str(fake)


def test_parse_nvcc_version():
    text = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 12.4, V12.4.131\n")
    assert environment.parse_nvcc_version(text) == "12.4"
    assert environment.parse_nvcc_version("garbage") is None


def test_detect_gpus_parses_nvidia_smi():
    output = "GPU 0: Tesla T4 (UUID: GPU-1234)\n"
    completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
    with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         mock.patch("subprocess.run", return_value=completed):
        assert environment.detect_gpus() == ["Tesla T4"]


def test_detect_gpus_without_nvidia_smi():
    with mock.patch("shutil.which", return_value=None):
        assert environment.detect_gpus() == []


def test_nvcc_without_gpu_can_compile_but_not_run():
    env = environment.Environment("/usr/local/cuda/bin/nvcc", "12.4", [])
    assert env.can_compile and not env.can_run
