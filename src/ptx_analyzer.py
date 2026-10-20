"""A deliberately simple, line-based PTX statistics tool.

PTX is NVIDIA's virtual instruction set: nvcc translates CUDA C++ into PTX,
and the ptxas assembler (or the driver, at run time) translates PTX into the
GPU's real machine code (SASS).

Limitations (by design):
  * This is not a PTX parser. It relies on the regular layout nvcc emits:
    one statement per line, directives starting with '.', labels ending ':'.
    Hand-formatted or unusual PTX can be miscounted.
  * Counts are *static*: an instruction inside a loop counts once, however
    many times it executes.
  * PTX registers are virtual and unlimited; the real register allocation is
    done later by ptxas. The register numbers here are declarations, not
    hardware usage.
  * Because ptxas optimises further, PTX statistics do not measure GPU
    performance. They only show how the compiler's PTX output changed.
"""

import re
from collections import Counter
from pathlib import Path

CATEGORIES = {
    "arithmetic": {"add", "sub", "mul", "mad", "fma", "div", "rem", "neg", "abs", "min", "max",
                   "rcp", "sqrt", "rsqrt"},
    "bitwise": {"and", "or", "xor", "not", "shl", "shr"},
    "compare": {"setp", "selp", "set", "slct"},
    "memory": {"ld", "st", "atom", "red", "ldu"},
    "branch": {"bra", "brx", "call", "ret", "exit"},
    "sync": {"bar", "barrier", "membar", "fence"},
    "move_convert": {"mov", "cvt", "cvta"},
}
STATE_SPACES = {"global", "shared", "local", "param", "const"}

_LABEL = re.compile(r"^[\w$]+:\s*")
_PREDICATE = re.compile(r"^@!?%\w+\s+")
_REG_DECL = re.compile(r"^\.reg\s+\.(\w+)\s+(.+?);")
_ENTRY = re.compile(r"\.entry\s+([\w$]+)")


def _category(opcode: str) -> str:
    for name, opcodes in CATEGORIES.items():
        if opcode in opcodes:
            return name
    return "other"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def analyze_ptx_text(text: str) -> dict:
    opcode_counts: Counter = Counter()
    category_counts: Counter = Counter({name: 0 for name in [*CATEGORIES, "other"]})
    memory_by_space: Counter = Counter()
    registers: Counter = Counter()
    entries: list[str] = []
    predicated = 0
    uses_shared = False
    in_debug_section = False      # -G output contains .section blocks of raw bytes
    continuing = False            # inside an instruction that spans several lines

    for raw_line in _strip_comments(text).splitlines():
        line = _LABEL.sub("", raw_line.strip())
        if not line:
            continue
        if in_debug_section:
            in_debug_section = line != "}"
            continue
        if line.startswith(".section"):
            in_debug_section = True
            continue

        if line.startswith("."):
            entry = _ENTRY.search(line)
            if entry:
                entries.append(entry.group(1))
            reg = _REG_DECL.match(line)
            if reg:
                # ".reg .b32 %r<9>;" declares %r0..%r8 -> 9 registers
                count = re.search(r"<(\d+)>", reg.group(2))
                registers[reg.group(1)] += int(count.group(1)) if count else len(reg.group(2).split(","))
            if line.startswith(".shared"):
                uses_shared = True
            continue
        if continuing:
            # Continuation of a multi-line instruction (e.g. 'call' with an
            # argument list); it has already been counted. Checked before the
            # bracket rule below, because the closing line is often ");".
            continuing = not line.endswith(";")
            continue
        if line[0] in "{}()":
            continue

        if _PREDICATE.match(line):
            predicated += 1
            line = _PREDICATE.sub("", line)
        token = line.split()[0].rstrip(";,")
        parts = token.split(".")
        opcode = parts[0]
        opcode_counts[opcode] += 1
        category_counts[_category(opcode)] += 1
        if _category(opcode) == "memory":
            space = next((p for p in parts[1:] if p in STATE_SPACES), "generic")
            memory_by_space[f"{opcode}.{space}"] += 1
            uses_shared = uses_shared or space == "shared"
        continuing = not line.endswith(";")

    return {
        "total_instructions": sum(opcode_counts.values()),
        "category_counts": dict(category_counts),
        "memory_by_space": dict(sorted(memory_by_space.items())),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "predicated_instructions": predicated,
        "register_declarations": dict(sorted(registers.items())),
        "has_register_declarations": bool(registers),
        "uses_shared_memory": uses_shared,
        "entries": entries,
    }


def analyze_ptx_file(path: Path) -> dict:
    """Analyse a PTX file. Raises FileNotFoundError if it does not exist."""
    return analyze_ptx_text(Path(path).read_text())


def compare_ptx(stats_a: dict, stats_b: dict, label_a: str, label_b: str) -> dict:
    """List every statistic that differs between two analyses.

    A difference only says the compiler emitted different PTX; it says
    nothing on its own about correctness or speed.
    """
    differences = {}

    def check(name, a, b):
        if a != b:
            differences[name] = {label_a: a, label_b: b}

    check("total_instructions", stats_a["total_instructions"], stats_b["total_instructions"])
    check("predicated_instructions", stats_a["predicated_instructions"],
          stats_b["predicated_instructions"])
    for group in ("category_counts", "memory_by_space", "register_declarations"):
        for key in sorted(set(stats_a[group]) | set(stats_b[group])):
            check(f"{group}.{key}", stats_a[group].get(key, 0), stats_b[group].get(key, 0))
    return {"identical_statistics": not differences, "differences": differences}


def format_stats(stats: dict) -> str:
    lines = [f"kernels           : {', '.join(stats['entries']) or '(none found)'}",
             f"instructions      : {stats['total_instructions']} "
             f"({stats['predicated_instructions']} predicated)"]
    lines += [f"  {name:<16}: {count}" for name, count in stats["category_counts"].items()]
    if stats["memory_by_space"]:
        lines.append("memory operations : " + ", ".join(
            f"{k}={v}" for k, v in stats["memory_by_space"].items()))
    regs = stats["register_declarations"]
    lines.append("virtual registers : " + (", ".join(f"{k}={v}" for k, v in regs.items())
                                           if regs else "none declared"))
    lines.append(f"shared memory     : {'yes' if stats['uses_shared_memory'] else 'no'}")
    return "\n".join(lines)
