# KernelCheck: An Automated CUDA Compiler Testing and Verification Framework

A final-year B.E. (CSE) project. KernelCheck takes a small suite of CUDA
programs and compiles each one with NVIDIA's `nvcc` under several compiler
configurations. It runs the programs on an NVIDIA GPU, checks every output
against an independent reference, and inspects the PTX the compiler generated.
The results go into a JSON report and a readable text summary.

> **About the results in this README.** The project was developed on an Apple
> Silicon MacBook, where CUDA is not available. Every output shown here is
> labelled either as **real output** (captured on that Mac) or as an
> **illustrative example** (it shows the format only, nothing was measured).
> This README contains no GPU measurements and claims no compiler bugs.

---

## 1. Abstract

Compilers are trusted to translate programs without changing what they mean.
For GPU compilers that trust is hard to check by hand: the code is
massively parallel, it goes through several translation stages
(CUDA C++ → PTX → SASS), and floating-point results are allowed to differ in
the last bits. KernelCheck automates a basic verification workflow. A manifest
describes the test programs. Each program is compiled under different
optimisation settings and executed. Its output is checked against a trusted CPU
reference, using exact comparison for integers and a documented tolerance for
floats. Each outcome is classified into one of six statuses, and any
disagreement between configurations is flagged as a *candidate issue requiring
investigation*. A small PTX analyser reports static instruction statistics and
how they change between configurations.

## 2. Problem statement

When a CUDA program gives a wrong answer, the cause could be the program, the
compiler, the runtime, or the test itself. Checking compiler output manually
for every optimisation level is slow and easy to get wrong. We need a
repeatable, automated way to:

- compile the same programs under different compiler configurations,
- decide objectively whether each output is correct,
- tell compile failures, crashes, hangs and wrong answers apart, and
- record everything in a form that can be compared over time.

## 3. Motivation

Compiler-verification teams (for example in GPU compiler groups) rely on large
automated test suites, differential testing across optimisation levels, and
inspection of intermediate representations. This project reproduces the core
ideas at a scale one student can build, run and fully explain.

## 4. Objectives

1. Maintain a manifest of CUDA test programs with explicit expected behaviour.
2. Compile them with `nvcc` (safely, with timeouts) and record the results.
3. Execute them on an NVIDIA GPU and verify outputs against references.
4. Compare compiler configurations (`-O0`, `-O2`, optionally `-G`).
5. Analyse the generated PTX.
6. Produce JSON and human-readable reports.
7. Keep everything that does not need CUDA runnable on a Mac.

## 5. Technologies used

| Technology | Used for |
|---|---|
| Python 3.10+ (standard library) | the framework itself (`subprocess`, `json`, `re`, `argparse`, `dataclasses`) |
| CUDA C++ / `nvcc` | test programs and compilation |
| PTX | the compiler output being analysed |
| pytest | unit tests and integration tests |
| Matplotlib (optional) | charts, only from real measurements |
| Git | version control |

## 6. System architecture

```
                 test_cases/manifest.json          reference/references.py
                           |                                 |
                           v                                 v
 main.py  --->  src/manifest.py  (load + validate)  --->  expected values
    |                      |
    |                      v
    |            src/pipeline.py  (for each test x configuration)
    |              |-- src/compiler.py     nvcc -> build/<cfg>/<id>, <id>.ptx
    |              |-- src/runner.py       run executable, timeout, kernel time
    |              |-- src/verifier.py     parse output, exact / tolerance compare
    |              |-- src/ptx_analyzer.py static PTX statistics
    |              '-- classify -> PASS / OUTPUT_MISMATCH / COMPILE_ERROR /
    |                              RUNTIME_ERROR / TIMEOUT / SKIPPED_NO_CUDA
    |                      |
    |                      v
    |            compare_configurations()  -> candidate issues, PTX diffs
    v                      |
 src/environment.py        v
 (nvcc? GPU?)       src/report.py -> results/report.json, report_summary.txt, *.png
```

More detail is in [docs/architecture.md](docs/architecture.md).

### Project structure

```
KernelCheck/
  main.py                 command-line interface
  config.py               paths, timeouts, compiler configurations, tolerances
  src/
    environment.py        finds nvcc (NVCC, CUDA_HOME, PATH) and GPUs (nvidia-smi)
    manifest.py           loads and validates the manifest
    compiler.py           runs nvcc, classifies compile results
    runner.py             runs programs, captures output, timeouts, kernel time
    verifier.py           output parsing and comparison
    ptx_analyzer.py       line-based PTX statistics and comparison
    pipeline.py           compile -> run -> verify -> classify, config comparison
    report.py             JSON report, text summary, charts
  reference/references.py Python reference implementations
  test_cases/             manifest.json, kc_common.h, 11 CUDA programs
  tests/                  unit tests, CUDA integration tests, sample PTX
  docs/                   architecture.md, test_plan.md
  build/, results/        generated output (git-ignored)
```

## 7. Workflow

1. `main.py run` loads the manifest. Invalid entries stop the run, and every problem is listed.
2. `environment.py` detects `nvcc` and GPUs. Whatever is missing becomes `SKIPPED_NO_CUDA`.
3. For every test and every selected configuration:
   1. **Compile** with `nvcc <flags> -o build/<cfg>/<id>`, then again with `-ptx`.
   2. **Run** the executable with the manifest's arguments and a timeout.
   3. **Verify**: parse stdout into numbers and compare them with the reference.
   4. **Classify** the result into one of the six statuses.
   5. **Analyse** the PTX, if it was produced.
4. **Compare configurations** for each test: statuses, outputs and PTX statistics.
5. **Report**: write `results/report.json` and `results/report_summary.txt`,
   plus charts if real timings exist.

### Test program protocol

Every CUDA program follows the same simple contract (see `test_cases/kc_common.h`):

- **stdout** contains only the results, as whitespace-separated numbers.
- **stderr** contains diagnostics, plus one `kernel_ms=<t>` line measured with CUDA events.
- Every CUDA API call goes through `CUDA_CHECK`. Kernel launches are checked with
  `cudaGetLastError()`, and asynchronous errors are caught at the next synchronisation.
  Any error exits with a non-zero code, which the framework reports as `RUNTIME_ERROR`.

### Outcome classification

| Status | Meaning |
|---|---|
| `PASS` | compiled, ran, and the output matched the expected values |
| `OUTPUT_MISMATCH` | ran, but the output was wrong, the wrong length, or not parseable |
| `COMPILE_ERROR` | nvcc returned an error (or produced no executable) |
| `RUNTIME_ERROR` | non-zero exit code, or killed by a signal (e.g. segfault) |
| `TIMEOUT` | compilation or execution exceeded its time limit |
| `SKIPPED_NO_CUDA` | nvcc or an NVIDIA GPU was not available. **Never counted as a pass.** |

## 8. What is compiler verification?

Compiler verification asks whether a compiler preserves the meaning of the
programs it translates. Formal approaches try to *prove* this, and only a few
research compilers manage it. Most industrial compiler testing is *empirical*.
KernelCheck uses three such techniques:

- **Reference (oracle) testing.** Compare the program's output with an
  independently computed correct answer (`reference/references.py`).
- **Differential testing.** Compile the same program in different ways (here,
  different optimisation settings) and check that the results agree. A
  disagreement points to a component worth investigating.
- **Intermediate-representation inspection.** Look at what the compiler
  actually produced (PTX) to see how a setting changed the code.

**A failing test is not automatically a compiler bug.** It can also come from
undefined behaviour in the test (e.g. signed overflow, out-of-bounds access),
a data race, legitimate floating-point differences (such as FMA contraction or
summation order), a wrong reference, or a framework bug. That is why
KernelCheck says *"candidate issue requiring investigation"* and never
"compiler bug". The test programs are written to avoid undefined behaviour:
inputs are chosen so that nothing overflows, there is no division by zero, and
every write stays in bounds.

### Floating-point tolerance

Floats pass if `|actual - expected| <= max(rel_tol * |expected|, abs_tol)`.
The default is `rel_tol = 1e-5`, `abs_tol = 1e-6` (`config.py`). The reason:
float32 carries about 7 significant digits, so a few units of rounding error
appear at about 1e-6 to 1e-7 relative. The tolerances are backed by
evidence, not guessed. Unit tests in `tests/test_references.py` emulate
float32 arithmetic in Python with and without FMA:

- `float_arith` showed a worst-case relative error of about 9e-8 against the
  double-precision reference, well inside 1e-5.
- `reduce_float` uses `rel_tol = 2e-5`. Its atomic-add order on the GPU is not
  fixed. Across 200 emulated orders the worst error was 4.0e-6, while dropping
  any single block changes the sum by at least 3.8e-5. So 2e-5 tolerates
  rounding but still catches a lost block. (An earlier, looser 1e-4 would
  have missed more than half of all possible lost blocks.)

## 9. What is PTX?

PTX (Parallel Thread Execution) is NVIDIA's *virtual* instruction set.
`nvcc` compiles CUDA C++ in stages:

```
 .cu  --(front end + NVVM optimiser)-->  PTX  --(ptxas)-->  SASS (real GPU machine code)
       host code -----------------------------(g++/clang)--> host object
                     all linked into one executable ("fat binary")
```

PTX looks like assembly (`ld.global.u32`, `add.s32`, `bar.sync`, `@%p1 bra`),
but it has unlimited virtual registers and is not executed directly. `ptxas`
(or the driver, at run time) optimises it further into SASS. This matters when
reading KernelCheck's output:

- `nvcc -O<n>` mainly affects **host** code. `-Xptxas -O<n>` affects the
  **PTX → SASS** step, *after* PTX is produced. So the PTX for `O0` and `O2`
  may well be **identical**, and that is a correct result, not a failure.
- `-G` (device debug) disables device optimisation and normally changes the PTX
  substantially. Use `--configs O0,O2,G` to see that.
- **PTX instruction counts do not measure performance.** The counts are static
  (a loop body counts once) and ptxas rewrites the code afterwards.

The analyser is deliberately simple and line-based. It depends on nvcc's
one-statement-per-line layout and does not attempt to parse PTX fully.

## 10. Installation

```bash
git clone <your-repo-url> KernelCheck
cd KernelCheck
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # pytest; matplotlib is optional
```

## 11. MacBook (host-only) setup

macOS on Apple Silicon has no CUDA support. Do not try to install NVIDIA
drivers, and note that Apple's GPU cannot run CUDA. On a Mac you can still run
everything that does not need CUDA:

```bash
python3 main.py check-env                    # reports nvcc / GPU as NOT FOUND
python3 main.py list                         # show the 13 test definitions
python3 main.py validate                     # validate manifest, compute all references
python3 -m pytest -m "not cuda" -v           # 98 host-only unit tests
python3 main.py analyze-ptx tests/data/sample_add.ptx tests/data/sample_reduce.ptx
python3 main.py run                          # every result: SKIPPED_NO_CUDA, report still written
```

## 12. NVIDIA GPU setup

You need a Linux machine with an NVIDIA GPU, the NVIDIA driver, and the CUDA
Toolkit (which provides `nvcc`). Confirm the setup with:

```bash
nvidia-smi          # shows the GPU and driver
nvcc --version      # shows the CUDA toolkit version
```

If `nvcc` is not on `PATH`, set `CUDA_HOME=/usr/local/cuda` or `NVCC=/path/to/nvcc`.

**Option A: Google Colab** (free GPU when available, often a T4 = `sm_75`).
Choose *Runtime → Change runtime type → GPU*, then in cells:

```bash
!nvidia-smi
!nvcc --version
!git clone <your-repo-url> KernelCheck
%cd KernelCheck
!pip install -q pytest matplotlib
!python main.py check-env
!python -m pytest -m cuda -v
!python main.py run --arch sm_75 --configs O0,O2,G
```

(If you upload a zip instead of cloning, unzip it and `%cd` into the folder.)

**Option B: any Linux GPU machine or cloud VM** with the CUDA Toolkit installed
(university lab, AWS/GCP/Azure GPU instance). Run the commands in section 13.

`--arch` is optional. Without it, nvcc uses its default target and the driver
JIT-compiles the embedded PTX for your GPU. Passing your GPU's architecture
(see `nvidia-smi --query-gpu=compute_cap --format=csv`, e.g. 7.5 → `sm_75`)
makes the PTX and SASS match the hardware you test on.

## 13. Commands

| Purpose | Command |
|---|---|
| Environment check | `python3 main.py check-env` |
| Show manifest | `python3 main.py list` |
| Validate manifest and references | `python3 main.py validate` |
| Host-only unit tests | `python3 -m pytest -m "not cuda" -v` |
| CUDA integration tests | `python3 -m pytest -m cuda -v` |
| Full run (O0 vs O2) | `python3 main.py run` |
| Include device-debug config | `python3 main.py run --configs O0,O2,G` |
| Selected tests, given arch | `python3 main.py run --tests int_add,reduce_float --arch sm_75` |
| Analyse one PTX file | `python3 main.py analyze-ptx build/O2/int_add.ptx` |
| Compare two PTX files | `python3 main.py analyze-ptx build/O2/loop_collatz.ptx build/G/loop_collatz.ptx` |

`main.py run` exits with code 1 if any test failed, 0 otherwise. Skipped
tests do not count as failures, but they are reported separately.

## 14. The test cases

| ID | Program | What it exercises | Verification |
|---|---|---|---|
| `int_add` | vec_add_int.cu | element-wise integer add | reference, exact |
| `int_mul` | vec_mul_int.cu | integer multiply, mixed signs | reference, exact |
| `int_divmod` | int_divmod.cu | `/` and `%` with negatives (truncation toward zero) | reference, exact |
| `float_arith` | float_arith.cu | float mul/div/add/sub, possible FMA contraction | reference, rel 1e-5 |
| `saxpy` | saxpy.cu | classic SAXPY | reference, rel 1e-5 |
| `loop_collatz` | collatz.cu | data-dependent loops and branches (warp divergence) | reference, exact |
| `boundary_guard` | boundary_guard.cu | partial last block; guard region must stay untouched | reference, exact |
| `size_1`, `size_257`, `size_65537` | vec_size.cu | grid-stride loop at different array sizes | reference, exact |
| `reduce_int` | reduce_int.cu | shared-memory tree reduction, `__syncthreads`, `atomicAdd` | reference, exact |
| `reduce_float` | reduce_float.cu | same in float; order-dependent rounding | reference, rel 2e-5 |
| `squares_literal` | squares.cu | literal expected output in the manifest | expected_output |

Full rationale: [docs/test_plan.md](docs/test_plan.md).

## 15. Example test report

**Real output**, captured on the development MacBook (no CUDA) with
`python3 main.py run --tests int_add` (abbreviated):

```
nvcc        : NOT FOUND - CUDA compilation unavailable
NVIDIA GPU  : NOT FOUND - CUDA execution unavailable
...
Total results : 2  (tests x configurations)
Passed        : 0
Failed        : 0
...
Skipped (no CUDA) : 2

test             config status           kernel ms  detail
int_add          O0     SKIPPED_NO_CUDA          -  nvcc not found; compilation not attempted
int_add          O2     SKIPPED_NO_CUDA          -  nvcc not found; compilation not attempted
```

**Illustrative example only.** This shows the *structure* of one record in
`results/report.json` from a GPU machine. Every value is a placeholder, not a
measurement:

```json
{
  "test_id": "int_add",
  "config": "O2",
  "compiler_flags": ["-O2", "-Xptxas", "-O2"],
  "compiler_version": "<nvcc version>",
  "status": "PASS",
  "compile": {"status": "SUCCESS", "duration_s": "<seconds>", "exit_code": 0, "command": ["nvcc", "..."], "output": ""},
  "run": {"status": "OK", "exit_code": 0, "wall_time_s": "<seconds>", "kernel_ms": "<ms>", "message": "", "stderr": "kernel_ms=<ms>"},
  "verification": {"strategy": "reference", "passed": true, "detail": "all 1024 values match exactly",
                   "expected": {"count": 1024, "first_values": [-68, -48, -28, -45]},
                   "actual":   {"count": 1024, "first_values": ["..."]}},
  "ptx": {"available": true, "file": "int_add.ptx", "sha256": "<hash>",
          "stats": {"total_instructions": "<n>", "category_counts": {"...": "..."}, "memory_by_space": {"...": "..."}}}
}
```

The `configuration_comparison` section holds one entry per test, containing the
status for each configuration, whether the outputs are identical, any PTX
differences, and `candidate_issue` with its reasons.

Once you run KernelCheck on a GPU, replace this section with your own real
`report_summary.txt` and note the GPU, driver and CUDA version.

## 16. Testing strategy

- **Unit tests (98, host-only).** `subprocess.run` is mocked for nvcc, so
  compile success, failure, timeout and missing-nvcc can all be tested without
  CUDA. The runner is tested against *real* subprocesses: small shell scripts
  stand in for CUDA programs and exercise success, non-zero exit,
  crash-by-signal, timeout and argument passing. Other tests cover the
  verifier, the manifest validator, the references (against hand-checked
  values, plus float32 emulation that justifies the tolerances), the PTX
  analyser (hand-written sample PTX), the report, and the CLI.
- **Integration tests (12, `-m cuda`).** Real nvcc and a real GPU: a real
  compile with PTX, a real compile error, 8 programs end to end, and an O0 vs O2
  agreement check. Without CUDA they are **skipped**, never passed.
- **CUDA syntax check (manual, development only).** The `.cu` files were checked
  with Apple clang's CUDA front end (`-fsyntax-only -Wall -Wextra`) against a
  stub header, with no errors or warnings. This catches typos and API misuse,
  but it is **not** a compile with nvcc and says nothing about runtime behaviour.

## 17. Limitations

- **Not yet executed on a GPU.** Real CUDA results only exist once you run
  KernelCheck on an NVIDIA machine.
- Small suite (13 tests). It demonstrates the method; it is not a coverage claim.
- The PTX analyser is line-based and static. It does not parse PTX fully,
  does not look at SASS, and does not measure performance.
- Kernel times come from a single run, with no warm-up and no statistics. They
  show trends, not benchmarks.
- `-O0` / `-O2` mostly affect host code and ptxas. Identical PTX between them
  is expected.
- Detecting a GPU relies on `nvidia-smi`.
- Expected outputs come from Python references. A bug in a reference would
  show up as `OUTPUT_MISMATCH`, which is why the references have their own tests.

## 18. Possible future improvements

- Disassemble SASS with `cuobjdump -sass` and compare it across configurations.
- Repeat runs and report the median and spread of kernel times.
- Add more configurations (`--use_fast_math`, `-fmad=false`, different `-arch`).
- Generate random test programs (in the spirit of Csmith) for broader differential testing.
- Add sanitizer runs (`compute-sanitizer`) to detect races and out-of-bounds accesses.
- Compare results across CUDA toolkit versions.
