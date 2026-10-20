"""Small helpers shared by the unit tests."""

import stat
from pathlib import Path

from src.manifest import TestCase


def fake_program(directory: Path, body: str, name: str = "fake_prog") -> Path:
    """Create an executable shell script that stands in for a compiled CUDA program."""
    path = directory / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def squares_test(tmp_path: Path) -> TestCase:
    source = tmp_path / "squares.cu"
    source.write_text("// placeholder\n")
    return TestCase(id="squares", description="squares", source=source,
                    expected_behavior="0 1 4 9", verification="expected_output",
                    output_type="int", expected_output=[0, 1, 4, 9])
