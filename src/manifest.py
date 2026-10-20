"""Load and validate test_cases/manifest.json.

The manifest is the single list of test definitions. It contains only *what*
to test; results are written elsewhere (results/), so definitions and
outcomes never get mixed up.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from reference.references import REFERENCES

REQUIRED_FIELDS = ["id", "description", "source", "expected_behavior",
                   "verification", "output_type"]
VERIFICATION_STRATEGIES = ["reference", "expected_output"]
OUTPUT_TYPES = ["int", "float"]


class ManifestError(Exception):
    """Raised when the manifest is missing or invalid. Lists every problem found."""


@dataclass
class TestCase:
    __test__ = False           # tell pytest this is not a test class
    id: str
    description: str
    source: Path
    expected_behavior: str
    verification: str          # "reference" or "expected_output"
    output_type: str           # "int" or "float"
    args: list[str] = field(default_factory=list)
    reference: str | None = None
    expected_output: list[float] | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None


def _check_entry(entry: dict, index: int, test_dir: Path) -> list[str]:
    """Return a list of problems with one manifest entry (empty if valid)."""
    where = f"test #{index} ({entry.get('id', '<no id>')})"
    problems = [f"{where}: missing field '{name}'" for name in REQUIRED_FIELDS
                if name not in entry]
    if problems:
        return problems

    if not (test_dir / entry["source"]).is_file():
        problems.append(f"{where}: source file '{entry['source']}' not found in {test_dir}")
    if entry["output_type"] not in OUTPUT_TYPES:
        problems.append(f"{where}: output_type must be one of {OUTPUT_TYPES}")
    if not all(isinstance(a, str) for a in entry.get("args", [])):
        problems.append(f"{where}: args must be a list of strings")

    strategy = entry["verification"]
    if strategy not in VERIFICATION_STRATEGIES:
        problems.append(f"{where}: verification must be one of {VERIFICATION_STRATEGIES}")
    elif strategy == "reference" and entry.get("reference") not in REFERENCES:
        problems.append(f"{where}: unknown reference function '{entry.get('reference')}'")
    elif strategy == "expected_output" and not isinstance(entry.get("expected_output"), list):
        problems.append(f"{where}: 'expected_output' must be a list of numbers")
    return problems


def load_manifest(path: Path) -> list[TestCase]:
    if not path.is_file():
        raise ManifestError(f"Manifest file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc

    entries = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ManifestError("Manifest must be an object with a 'tests' list")

    test_dir = path.parent
    problems: list[str] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"test #{index}: entry must be an object")
            continue
        problems += _check_entry(entry, index, test_dir)
        test_id = entry.get("id")
        if test_id in seen_ids:
            problems.append(f"test #{index}: duplicate id '{test_id}'")
        seen_ids.add(test_id)
    if problems:
        raise ManifestError("Invalid manifest:\n  " + "\n  ".join(problems))

    return [
        TestCase(
            id=e["id"],
            description=e["description"],
            source=test_dir / e["source"],
            expected_behavior=e["expected_behavior"],
            verification=e["verification"],
            output_type=e["output_type"],
            args=e.get("args", []),
            reference=e.get("reference"),
            expected_output=e.get("expected_output"),
            rel_tol=e.get("rel_tol"),
            abs_tol=e.get("abs_tol"),
        )
        for e in entries
    ]


def expected_values(test: TestCase) -> list[float]:
    """Compute the trusted expected output for a test (host-only, no CUDA needed)."""
    if test.verification == "expected_output":
        return list(test.expected_output or [])
    return REFERENCES[test.reference](test.args)
