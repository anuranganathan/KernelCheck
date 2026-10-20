"""Central configuration for KernelCheck.

Everything a user might reasonably want to change lives here, as plain
Python values. There is deliberately no config-file format to learn.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_CASES_DIR = PROJECT_ROOT / "test_cases"
MANIFEST_PATH = TEST_CASES_DIR / "manifest.json"
BUILD_DIR = PROJECT_ROOT / "build"
RESULTS_DIR = PROJECT_ROOT / "results"

# Seconds. Compilation of these small programs normally takes a few seconds;
# execution should take well under one second (most of it is CUDA context
# creation), so these limits only catch genuinely stuck processes.
COMPILE_TIMEOUT_S = 120
RUN_TIMEOUT_S = 30

# Compiler configurations that can be compared against each other.
#
# Note on what these flags actually affect:
#   * nvcc's own -O<n> option controls *host* code optimisation.
#   * -Xptxas -O<n> forwards the level to ptxas, the PTX -> SASS assembler.
#     It does not change the PTX text itself.
#   * -G generates debuggable device code and disables most device-side
#     optimisation, which normally does change the PTX noticeably.
# So identical PTX for "O0" and "O2" is an expected, legitimate outcome.
COMPILER_CONFIGS: dict[str, list[str]] = {
    "O0": ["-O0", "-Xptxas", "-O0"],
    "O2": ["-O2", "-Xptxas", "-O2"],
    "G": ["-G", "-O0"],
}
DEFAULT_CONFIGS = ["O0", "O2"]

# Default floating-point tolerance, used when a manifest entry does not give
# its own. A value passes if |actual - expected| <= max(rel * |expected|, abs).
# float32 has ~7 significant decimal digits, so 1e-5 relative leaves room for
# a few units of rounding error (e.g. from FMA contraction) without hiding
# real errors.
DEFAULT_REL_TOL = 1e-5
DEFAULT_ABS_TOL = 1e-6

# How many output values to store in the JSON report per test. Full outputs
# can contain tens of thousands of numbers.
OUTPUT_PREVIEW_LEN = 16
