# KernelCheck test plan

There are two kinds of tests here, and they should not be confused:

- **CUDA test cases**: the programs in `test_cases/` that KernelCheck runs to
  test the *compiler*.
- **Framework tests**: the pytest tests in `tests/` that check *KernelCheck
  itself*.

## 1. CUDA test cases

All inputs are generated deterministically inside each program, using the same
formulas as `reference/references.py`. They are chosen to avoid undefined
behaviour: no signed overflow, no division by zero, no out-of-bounds access.

| ID | Purpose / edge case | Input design | Verification |
|---|---|---|---|
| `int_add` | baseline integer arithmetic | a ∈ [-50, 50], b ∈ [-18, 18], n = 1024 | exact |
| `int_mul` | multiplication with mixed signs | same inputs; products fit in int32 | exact |
| `int_divmod` | C++ truncating division and the sign of `%` | divisors in [-4, 4] with 0 replaced by 5 | exact. The reference uses `c_div`, because Python's `//` floors |
| `float_arith` | float32 `* / + -`; FMA contraction is allowed | x, y exactly representable; y ≥ 1 | rel 1e-5, abs 1e-6 |
| `saxpy` | the classic kernel | results exactly representable | rel 1e-5 (would also pass exactly) |
| `loop_collatz` | data-dependent `while`/`if`, warp divergence | starts 1..1000; steps capped at 10000 | exact |
| `boundary_guard` | partial last block: 1000 elements, 4 × 256 threads | 32 guard ints set to -1 must remain -1 | exact |
| `size_1` / `size_257` / `size_65537` | grid-stride loop at very small, odd, and larger-than-grid sizes | fixed grid of 4096 threads | exact |
| `reduce_int` | shared memory, `__syncthreads`, `atomicAdd` | n = 1,000,003 (not a multiple of 256) | exact |
| `reduce_float` | non-associative float addition, unordered atomics | n = 2^20 | rel 2e-5 (see below) |
| `squares_literal` | the literal-expected-output path | 8 threads | exact against the manifest list |

**Tolerance justification.** `tests/test_references.py` emulates float32
arithmetic with Python's `struct`/`array` modules:

- `float_arith`: maximum relative error about 9e-8, both with and without FMA.
- `reduce_float`: worst 4.0e-6 across 200 random block orders. A single
  lost block would change the sum by between 3.8e-5 and 7.8e-4 relative.
  So `rel_tol = 2e-5` is 5x the worst rounding error seen, yet below the
  smallest lost-block effect. A first choice of 1e-4 was rejected after
  this analysis, because it would miss 2195 of the 4096 possible lost blocks.

## 2. Framework unit tests (host-only: `pytest -m "not cuda"`)

| File | Covers |
|---|---|
| `test_manifest.py` | valid load; missing file; invalid JSON; missing field; missing source; duplicate ids; unknown reference; all errors reported together; both verification strategies; the real manifest |
| `test_environment.py` | missing nvcc; `$NVCC` priority; version parsing; `nvidia-smi` parsing; no GPU |
| `test_compiler.py` (mocked nvcc) | missing nvcc → skip; success with PTX; argument list and flags; compile error; zero exit without output; timeout; nvcc not startable; PTX failure keeps executable; stale binary removed |
| `test_runner.py` (real subprocesses) | success and `kernel_ms`; safe argument passing; non-zero exit; crash by signal; timeout; missing executable |
| `test_verifier.py` | parsing; garbage output; exact match/mismatch; length mismatch; relative and absolute tolerance; NaN |
| `test_references.py` | C division semantics; hand-checked values; input ranges; float32 emulation within tolerance |
| `test_pipeline.py` | `classify` truth table; no CUDA → skip; nvcc without GPU → skip; PASS / mismatch / garbage / runtime error with fake binaries; PTX available/missing; candidate-issue rules |
| `test_ptx_analyzer.py` | statistics of two sample files; comments, labels, directives; `-G` debug sections; multi-line `call`; generic memory; empty input; missing file; comparison |
| `test_report.py` | empty results; counts; JSON round trip; no charts without data; CLI end to end without CUDA; CLI argument errors |

## 3. Integration tests (`pytest -m cuda`, needs nvcc + GPU)

- The nvcc version is detected.
- A real compile produces an executable and PTX containing the expected kernel and a `st.global`.
- A real syntax error is classified as `COMPILE_ERROR`.
- Eight manifest programs `PASS` end to end, with `kernel_ms` reported.
- O0 and O2 give identical integer output, with no candidate issue.

Without CUDA all 12 are reported as **skipped**.

## 4. Not covered

- Real GPU execution on the development Mac (impossible there).
- SASS-level inspection, performance benchmarking, and multi-GPU behaviour.
- Undefined-behaviour detection (`compute-sanitizer` would be the right tool).
