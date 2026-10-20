"""KernelCheck command-line interface.

    python main.py check-env                 what CUDA tooling is available here
    python main.py list                      show the test manifest
    python main.py validate                  check the manifest and run all references (host-only)
    python main.py run [options]             compile, run, verify, report
    python main.py analyze-ptx A.ptx [B.ptx] PTX statistics, or a comparison of two files
"""

import argparse
import sys
from pathlib import Path

import config
from src import environment, pipeline, report
from src.manifest import ManifestError, expected_values, load_manifest
from src.ptx_analyzer import analyze_ptx_file, compare_ptx, format_stats


def cmd_check_env(_args) -> int:
    print(environment.describe(environment.detect_environment()))
    return 0


def cmd_list(_args) -> int:
    tests = load_manifest(config.MANIFEST_PATH)
    for test in tests:
        print(f"{test.id:<16} {test.source.name:<18} args={' '.join(test.args) or '-':<14} "
              f"[{test.output_type}, {test.verification}]")
        print(f"{'':<16} {test.description}")
        print(f"{'':<16} expects: {test.expected_behavior}")
    print(f"\n{len(tests)} tests in {config.MANIFEST_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


def cmd_validate(_args) -> int:
    tests = load_manifest(config.MANIFEST_PATH)
    for test in tests:
        values = expected_values(test)
        print(f"OK  {test.id:<16} {len(values):>7} expected values, first: {values[:4]}")
    print(f"\nManifest valid: {len(tests)} tests, all reference outputs computed.")
    return 0


def cmd_run(args) -> int:
    for name in args.configs:
        if name not in config.COMPILER_CONFIGS:
            print(f"Unknown configuration '{name}'. Choose from: "
                  f"{', '.join(config.COMPILER_CONFIGS)}", file=sys.stderr)
            return 2

    tests = load_manifest(config.MANIFEST_PATH)
    if args.tests:
        unknown = set(args.tests) - {t.id for t in tests}
        if unknown:
            print(f"Unknown test id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        tests = [t for t in tests if t.id in args.tests]

    env = environment.detect_environment()
    print(environment.describe(env))
    print(f"\nRunning {len(tests)} tests x {len(args.configs)} configurations "
          f"({', '.join(args.configs)})\n")

    records, outputs = [], {}
    for test in tests:
        for config_name in args.configs:
            record, actual = pipeline.run_test(test, config_name, env, arch=args.arch)
            records.append(record)
            if actual is not None:
                outputs[(test.id, config_name)] = actual
            print(f"  {test.id:<16} {config_name:<4} {record['status']}")

    comparisons = pipeline.compare_configurations(records, outputs)
    configs = {name: config.COMPILER_CONFIGS[name] for name in args.configs}
    full_report = report.build_report(env, configs, args.arch, records, comparisons)

    output = Path(args.output)
    report.write_json(full_report, output)
    summary_text = report.format_summary(full_report)
    summary_path = output.with_name(output.stem + "_summary.txt")
    summary_path.write_text(summary_text + "\n")
    print("\n" + summary_text)
    print(f"\nJSON report : {output}\nSummary     : {summary_path}")

    if not args.no_charts:
        charts = report.make_charts(records, output.parent)
        if charts:
            print("Charts      : " + ", ".join(str(c) for c in charts))
        else:
            print("Charts      : none (no measured compile or kernel times, "
                  "or matplotlib not installed)")

    return 1 if full_report["summary"]["failed"] else 0


def cmd_analyze_ptx(args) -> int:
    first = analyze_ptx_file(Path(args.ptx[0]))
    if len(args.ptx) == 1:
        print(format_stats(first))
        return 0
    second = analyze_ptx_file(Path(args.ptx[1]))
    label_a, label_b = Path(args.ptx[0]).name, Path(args.ptx[1]).name
    print(f"== {label_a}\n{format_stats(first)}\n\n== {label_b}\n{format_stats(second)}\n")
    result = compare_ptx(first, second, "A", "B")
    if result["identical_statistics"]:
        print("Statistics are identical.")
    else:
        print(f"Differences (A = {label_a}, B = {label_b}):")
        for name, values in result["differences"].items():
            print(f"  {name:<36} A={values['A']:<6} B={values['B']}")
    print("\nNote: static PTX counts describe the compiler's output, not GPU performance.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelcheck", description=
                                     "Automated CUDA compiler testing and verification")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env", help="report nvcc / GPU availability").set_defaults(func=cmd_check_env)
    sub.add_parser("list", help="list the test manifest").set_defaults(func=cmd_list)
    sub.add_parser("validate", help="validate manifest and references").set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="compile, run and verify the tests")
    run.add_argument("--configs", type=lambda s: s.split(","), default=config.DEFAULT_CONFIGS,
                     help=f"comma-separated, from {list(config.COMPILER_CONFIGS)} "
                          f"(default: {','.join(config.DEFAULT_CONFIGS)})")
    run.add_argument("--tests", type=lambda s: s.split(","), default=None,
                     help="comma-separated test ids (default: all)")
    run.add_argument("--arch", default=None,
                     help="GPU architecture passed to nvcc, e.g. sm_75 (default: nvcc's default)")
    run.add_argument("--output", default=str(config.RESULTS_DIR / "report.json"))
    run.add_argument("--no-charts", action="store_true")
    run.set_defaults(func=cmd_run)

    ptx = sub.add_parser("analyze-ptx", help="PTX statistics for one file, or compare two")
    ptx.add_argument("ptx", nargs="+", metavar="FILE.ptx")
    ptx.set_defaults(func=cmd_analyze_ptx)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "ptx", None) and len(args.ptx) > 2:
        print("analyze-ptx takes one or two files", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except ManifestError as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
