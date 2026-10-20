"""Run one test through compile -> run -> verify, and classify the outcome.

Final statuses:
    PASS             compiled, ran, output matched the expected values
    OUTPUT_MISMATCH  compiled and ran, but the output was wrong or unparseable
    COMPILE_ERROR    nvcc failed
    RUNTIME_ERROR    the program exited with an error or crashed
    TIMEOUT          compilation or execution took too long
    SKIPPED_NO_CUDA  nvcc or an NVIDIA GPU was not available (never a pass)
"""

import hashlib
import math

import config
from src import compiler, ptx_analyzer, runner
from src.environment import Environment
from src.manifest import TestCase, expected_values
from src.verifier import Comparison, compare, parse_output

PASS = "PASS"
OUTPUT_MISMATCH = "OUTPUT_MISMATCH"
COMPILE_ERROR = "COMPILE_ERROR"
RUNTIME_ERROR = "RUNTIME_ERROR"
TIMEOUT = "TIMEOUT"
SKIPPED_NO_CUDA = "SKIPPED_NO_CUDA"
ALL_STATUSES = [PASS, OUTPUT_MISMATCH, COMPILE_ERROR, RUNTIME_ERROR, TIMEOUT, SKIPPED_NO_CUDA]

INVESTIGATE_NOTE = ("Candidate issue requiring investigation. Possible causes include a "
                    "compiler bug, undefined behaviour in the test, floating-point rounding, "
                    "a data race, a wrong reference implementation, or a framework error.")


def classify(compile_status: str, run_status: str | None, output_ok: bool | None) -> str:
    """Combine the stage results into one final status.

    The compile and run statuses deliberately use the same names as the final
    statuses (COMPILE_ERROR, TIMEOUT, ...), so most cases pass straight through.
    """
    if compile_status != compiler.SUCCESS:
        return compile_status
    if run_status is None or run_status == runner.SKIPPED_NO_CUDA:
        return SKIPPED_NO_CUDA
    if run_status != runner.OK:
        return run_status
    return PASS if output_ok else OUTPUT_MISMATCH


def tolerance_for(test: TestCase) -> tuple[float, float]:
    rel = config.DEFAULT_REL_TOL if test.rel_tol is None else test.rel_tol
    abs_ = config.DEFAULT_ABS_TOL if test.abs_tol is None else test.abs_tol
    return rel, abs_


def _preview(values: list | None) -> dict | None:
    if values is None:
        return None
    # NaN/inf are not valid JSON numbers, so store them as text.
    shown = [v if math.isfinite(v) else str(v) for v in values[:config.OUTPUT_PREVIEW_LEN]]
    return {"count": len(values), "first_values": shown}


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def analyze_ptx(built: compiler.CompileResult) -> dict:
    if built.ptx_path is None:
        return {"available": False,
                "reason": built.ptx_error or "PTX not generated (compilation did not succeed)"}
    try:
        text = built.ptx_path.read_text()
    except OSError as exc:
        return {"available": False, "reason": f"could not read PTX: {exc}"}
    return {"available": True,
            "file": built.ptx_path.name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "stats": ptx_analyzer.analyze_ptx_text(text)}


def run_test(test: TestCase, config_name: str, env: Environment, arch: str | None = None,
             build_dir=config.BUILD_DIR) -> tuple[dict, list | None]:
    """Compile, run and verify one test under one compiler configuration.

    Returns (JSON-ready record, full list of actual output values or None).
    """
    flags = config.COMPILER_CONFIGS[config_name]
    expected = expected_values(test)
    rel_tol, abs_tol = tolerance_for(test)

    built = compiler.compile_test(test, config_name, flags, env.nvcc_path, build_dir,
                                  config.COMPILE_TIMEOUT_S, arch)

    ran = None
    if built.status == compiler.SUCCESS and env.can_run:
        ran = runner.run_executable(built.executable, test.args, config.RUN_TIMEOUT_S)

    actual = None
    comparison = None
    if ran is not None and ran.status == runner.OK:
        try:
            actual = parse_output(ran.stdout, test.output_type)
            comparison = compare(expected, actual, test.output_type, rel_tol, abs_tol)
        except ValueError as exc:
            comparison = Comparison(False, f"unparseable program output: {exc}")

    status = classify(built.status, ran.status if ran else None,
                      comparison.passed if comparison else None)

    if ran is not None:
        run_info = {"status": ran.status, "exit_code": ran.exit_code,
                    "wall_time_s": ran.wall_time_s, "kernel_ms": ran.kernel_ms,
                    "message": ran.message, "stderr": _tail(ran.stderr)}
    elif built.status == compiler.SUCCESS:
        run_info = {"status": runner.SKIPPED_NO_CUDA, "message": "no NVIDIA GPU detected"}
    else:
        reason = ("nvcc not found" if built.status == compiler.SKIPPED_NO_CUDA
                  else "compilation did not succeed")
        run_info = {"status": "NOT_RUN", "message": reason}

    record = {
        "test_id": test.id,
        "description": test.description,
        "source": test.source.name,
        "args": test.args,
        "config": config_name,
        "compiler_flags": flags,
        "compiler_version": env.nvcc_version,
        "status": status,
        "compile": {
            "status": built.status,
            "duration_s": built.duration_s,
            "exit_code": built.exit_code,
            "command": built.command,
            "output": _tail(built.output),
        },
        "run": run_info,
        "verification": {
            "strategy": test.verification,
            "output_type": test.output_type,
            "tolerance": {"rel": rel_tol, "abs": abs_tol} if test.output_type == "float" else None,
            "passed": comparison.passed if comparison else None,
            "detail": comparison.detail if comparison else "not verified (program did not run)",
            "mismatch_count": comparison.mismatch_count if comparison else None,
            "first_mismatch_index": comparison.first_mismatch_index if comparison else None,
            "max_abs_error": comparison.max_abs_error if comparison else None,
            "expected": _preview(expected),
            "actual": _preview(actual),
        },
        "ptx": analyze_ptx(built),
    }
    return record, actual


def compare_configurations(records: list[dict], outputs: dict[tuple[str, str], list]) -> list[dict]:
    """Look across compiler configurations for each test and flag anything suspicious.

    `outputs` maps (test_id, config) -> full list of output values.
    """
    by_test: dict[str, list[dict]] = {}
    for record in records:
        by_test.setdefault(record["test_id"], []).append(record)

    comparisons = []
    for test_id, test_records in by_test.items():
        statuses = {r["config"]: r["status"] for r in test_records}
        reasons, notes = [], []

        for r in test_records:
            if r["status"] in (OUTPUT_MISMATCH, RUNTIME_ERROR) or \
               (r["status"] == TIMEOUT and r["compile"]["status"] == compiler.SUCCESS):
                reasons.append(f"{r['status']} under configuration {r['config']}")

        compiled = {r["config"] for r in test_records
                    if r["compile"]["status"] == compiler.SUCCESS}
        failed_compile = {r["config"] for r in test_records if r["status"] == COMPILE_ERROR}
        if compiled and failed_compile:
            reasons.append(f"compiles under {sorted(compiled)} but not under {sorted(failed_compile)}")

        produced = [(r["config"], outputs[(test_id, r["config"])]) for r in test_records
                    if (test_id, r["config"]) in outputs]
        outputs_identical = None
        if len(produced) >= 2:
            outputs_identical = all(values == produced[0][1] for _, values in produced[1:])
            if not outputs_identical:
                if test_records[0]["verification"]["output_type"] == "int":
                    reasons.append("integer outputs differ between configurations")
                else:
                    notes.append("float outputs differ between configurations; acceptable only "
                                 "if every configuration is within tolerance of the reference")

        # PTX: compare every other configuration against the first one that has PTX.
        with_ptx = [r for r in test_records if r.get("ptx", {}).get("available")]
        ptx_comparisons = {}
        for other in with_ptx[1:]:
            base = with_ptx[0]
            result = ptx_analyzer.compare_ptx(base["ptx"]["stats"], other["ptx"]["stats"],
                                              base["config"], other["config"])
            result["text_identical"] = base["ptx"]["sha256"] == other["ptx"]["sha256"]
            ptx_comparisons[f"{base['config']} vs {other['config']}"] = result

        comparisons.append({
            "test_id": test_id,
            "statuses": statuses,
            "outputs_identical": outputs_identical,
            "ptx_comparisons": ptx_comparisons,
            "candidate_issue": bool(reasons),
            "reasons": reasons,
            "notes": notes + ([INVESTIGATE_NOTE] if reasons else []),
        })
    return comparisons
