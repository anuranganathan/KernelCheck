"""Build the JSON report, the text summary and (optionally) charts."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from src import pipeline
from src.environment import Environment

FAILED_STATUSES = [pipeline.OUTPUT_MISMATCH, pipeline.COMPILE_ERROR,
                   pipeline.RUNTIME_ERROR, pipeline.TIMEOUT]

# Fixed categorical colours, one per configuration, in this order.
# Checked with a colour-vision-deficiency validator; always paired with a
# legend and value labels, so colour is never the only cue.
CONFIG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def _count(records: list[dict], status: str) -> int:
    return sum(1 for r in records if r["status"] == status)


def summarize(records: list[dict]) -> dict:
    """Counts per status, overall and per compiler configuration."""
    by_config: dict[str, dict] = {}
    for config_name in dict.fromkeys(r["config"] for r in records):
        subset = [r for r in records if r["config"] == config_name]
        by_config[config_name] = {s: _count(subset, s) for s in pipeline.ALL_STATUSES}
    return {
        "total": len(records),
        "passed": _count(records, pipeline.PASS),
        "failed": sum(_count(records, s) for s in FAILED_STATUSES),
        "skipped": _count(records, pipeline.SKIPPED_NO_CUDA),
        "by_status": {s: _count(records, s) for s in pipeline.ALL_STATUSES},
        "by_config": by_config,
    }


def build_report(env: Environment, configs: dict[str, list[str]], arch: str | None,
                 records: list[dict], comparisons: list[dict]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nvcc_path": env.nvcc_path,
            "nvcc_version": env.nvcc_version,
            "gpus": env.gpu_names,
            "can_compile": env.can_compile,
            "can_run": env.can_run,
        },
        "configurations": configs,
        "arch": arch,
        "summary": summarize(records),
        "results": records,
        "configuration_comparison": comparisons,
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def format_summary(report: dict) -> str:
    summary = report["summary"]
    env = report["environment"]
    lines = [
        "KernelCheck summary",
        "===================",
        f"Generated : {report['generated_at']}",
        f"Compiler  : {'nvcc ' + (env['nvcc_version'] or '?') if env['nvcc_path'] else 'nvcc not found'}",
        f"GPU       : {', '.join(env['gpus']) or 'none detected'}",
        "",
        f"Total results : {summary['total']}  (tests x configurations)",
        f"Passed        : {summary['passed']}",
        f"Failed        : {summary['failed']}",
        f"  Compile errors  : {summary['by_status'][pipeline.COMPILE_ERROR]}",
        f"  Runtime errors  : {summary['by_status'][pipeline.RUNTIME_ERROR]}",
        f"  Output mismatch : {summary['by_status'][pipeline.OUTPUT_MISMATCH]}",
        f"  Timeouts        : {summary['by_status'][pipeline.TIMEOUT]}",
        f"Skipped (no CUDA) : {summary['skipped']}",
    ]
    if summary["total"] == 0:
        lines.append("\nNo tests were run.")
        return "\n".join(lines)

    lines += ["", "By configuration:"]
    for config_name, counts in summary["by_config"].items():
        nonzero = ", ".join(f"{s}={n}" for s, n in counts.items() if n)
        lines.append(f"  {config_name:<4} {nonzero}")

    lines += ["", f"{'test':<16} {'config':<6} {'status':<16} {'kernel ms':>9}  detail"]
    for r in report["results"]:
        kernel = r["run"].get("kernel_ms")
        kernel_text = f"{kernel:.4f}" if kernel is not None else "-"
        lines.append(f"{r['test_id']:<16} {r['config']:<6} {r['status']:<16} {kernel_text:>9}  "
                     f"{_one_line_detail(r)}")

    flagged = [c for c in report["configuration_comparison"] if c["candidate_issue"]]
    lines += ["", f"Candidate issues requiring investigation: {len(flagged)}"]
    for c in flagged:
        lines.append(f"  {c['test_id']}: " + "; ".join(c["reasons"]))

    ptx_lines = []
    for c in report["configuration_comparison"]:
        for pair, result in c["ptx_comparisons"].items():
            if result["text_identical"]:
                verdict = "identical PTX text"
            elif result["identical_statistics"]:
                verdict = "PTX text differs, statistics identical"
            else:
                changed = result["differences"].get("total_instructions")
                verdict = (f"instructions {changed}" if changed
                           else f"{len(result['differences'])} statistic(s) differ")
            ptx_lines.append(f"  {c['test_id']:<16} {pair:<10} {verdict}")
    if ptx_lines:
        lines += ["", "PTX comparison between configurations:"] + ptx_lines
    return "\n".join(lines)


def _one_line_detail(record: dict) -> str:
    status = record["status"]
    if status == pipeline.PASS or status == pipeline.OUTPUT_MISMATCH:
        return record["verification"]["detail"]
    if status == pipeline.SKIPPED_NO_CUDA:
        if record["compile"]["status"] == pipeline.SKIPPED_NO_CUDA:
            return record["compile"]["output"]       # nvcc missing
        return record["run"].get("message", "")      # compiled, but no GPU
    if status == pipeline.COMPILE_ERROR or record["run"]["status"] == "NOT_RUN":
        first_line = (record["compile"]["output"] or "").strip().splitlines()
        return first_line[0][:80] if first_line else "compilation failed"
    return record["run"].get("message", "")


def make_charts(records: list[dict], out_dir: Path) -> list[Path]:
    """Draw charts from real measurements only. Returns the files written.

    No chart is produced for a metric that has no measured values, so a
    machine without CUDA never gets an empty or invented chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")    # write files, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    charts = [
        ("compile", "duration_s", "nvcc compile time (s)", "compile_times.png"),
        ("run", "kernel_ms", "Kernel time from CUDA events (ms)", "kernel_times.png"),
    ]
    written = []
    for stage, key, label, filename in charts:
        measured = [r for r in records if r[stage].get(key) is not None
                    and (stage == "run" or r["compile"]["status"] == "SUCCESS")]
        if not measured:
            continue
        tests = list(dict.fromkeys(r["test_id"] for r in measured))
        configs = list(dict.fromkeys(r["config"] for r in measured))
        values = {(r["test_id"], r["config"]): r[stage][key] for r in measured}

        bar = 0.8 / len(configs)
        fig, ax = plt.subplots(figsize=(8, 0.3 * len(tests) * len(configs) + 1.5))
        for i, config_name in enumerate(configs):
            ys = [t + i * bar for t in range(len(tests))]
            xs = [values.get((test, config_name), 0) for test in tests]
            bars = ax.barh(ys, xs, height=bar * 0.9, label=config_name,
                           color=CONFIG_COLORS[i % len(CONFIG_COLORS)])
            ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=7, color="#333333")
        ax.set_yticks([t + bar * (len(configs) - 1) / 2 for t in range(len(tests))], tests)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.margins(x=0.12)       # room for the value labels
        ax.set_title(f"{label} by test and configuration (measured)", fontsize=10)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(title="config", frameon=False)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
