# KernelCheck architecture

## Design goals

1. **Honest results.** Missing CUDA becomes `SKIPPED_NO_CUDA`, never a pass.
   Discrepancies become *candidate issues*, never "compiler bugs".
2. **Host-first.** Everything that can run without CUDA does, including most
   of the testing (by mocking `subprocess`).
3. **Small and explainable.** Plain functions and dataclasses, no class
   hierarchies, no plugins, and only the standard library in the core.

## Modules

| Module | Responsibility | Key functions |
|---|---|---|
| `config.py` | paths, timeouts, compiler configurations, default tolerances | constants only |
| `src/environment.py` | locate nvcc (`$NVCC`, `$CUDA_HOME/bin`, `PATH`), read its version, detect GPUs via `nvidia-smi -L` | `detect_environment`, `describe` |
| `src/manifest.py` | load `manifest.json`, report *all* validation problems at once, compute expected values | `load_manifest`, `expected_values` |
| `reference/references.py` | independent Python implementations producing expected outputs | `REFERENCES` dict |
| `src/compiler.py` | build nvcc argument lists, compile executable + PTX, classify | `compile_test` |
| `src/runner.py` | run an executable with a timeout, capture output, read `kernel_ms` | `run_executable` |
| `src/verifier.py` | parse stdout into numbers, exact/tolerance comparison | `parse_output`, `compare` |
| `src/ptx_analyzer.py` | line-based PTX statistics and comparison | `analyze_ptx_text`, `compare_ptx` |
| `src/pipeline.py` | glue: compile → run → verify → classify, then cross-config comparison | `run_test`, `classify`, `compare_configurations` |
| `src/report.py` | JSON report, text summary, optional charts | `build_report`, `format_summary`, `make_charts` |
| `main.py` | argparse CLI | `check-env`, `list`, `validate`, `run`, `analyze-ptx` |

## Data flow for one (test, configuration)

```
TestCase ──► compile_test ──► CompileResult(status, executable, ptx_path, duration, output)
                                   │ SUCCESS and GPU present
                                   ▼
                             run_executable ──► RunResult(status, stdout, stderr, kernel_ms)
                                   │ OK
                                   ▼
            expected_values ──► compare(parse_output(stdout)) ──► Comparison(passed, detail)
                                   │
                                   ▼
               classify(compile status, run status, passed) ──► final status
                                   │
                                   ▼
                         record (dict, JSON-ready) + full output values
```

`classify` works because the stage statuses reuse the final status names:
a compile `TIMEOUT` and a run `TIMEOUT` both become the final `TIMEOUT`, and
the same holds for `COMPILE_ERROR`, `RUNTIME_ERROR` and `SKIPPED_NO_CUDA`.

## Cross-configuration comparison

After every configuration has run, `compare_configurations` groups the records
by test and flags a **candidate issue** when any of these is true:

- some configuration produced `OUTPUT_MISMATCH`, `RUNTIME_ERROR`, or a run `TIMEOUT`;
- the test compiles under some configurations but not others;
- *integer* outputs differ between configurations.

Float outputs that differ between configurations while each stays within
tolerance are only recorded as a note. That is legitimate (FMA, summation
order). PTX statistics and a SHA-256 of the PTX text are compared against the
first configuration, so the report can distinguish "identical PTX" from
"different text, same counts" and from "different counts".

## Safety decisions

- **No shell.** Every `subprocess.run` call takes an argument list and never
  sets `shell=True`. Manifest arguments are passed to the program as-is and
  never interpreted. A unit test checks this with the argument `"; rm -rf x"`.
- **Timeouts** on every nvcc call and every program run (`config.py`).
- **Stale artefacts are deleted before compiling**, so a failed compile can
  never cause an old executable to be run and reported.
- **No unsafe test programs.** The edge cases (`boundary_guard`) write only
  inside memory they allocated themselves.

## Where things are stored

- Test *definitions*: `test_cases/` (committed).
- Build artefacts: `build/<config>/<test_id>` and `.ptx` (git-ignored).
- Results: `results/report.json`, `report_summary.txt`, charts (git-ignored).
