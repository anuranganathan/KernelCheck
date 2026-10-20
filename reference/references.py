"""Trusted CPU reference implementations, written in plain Python.

Each function takes the test's command-line arguments (as strings, exactly as
listed in manifest.json) and returns the list of numbers the CUDA program is
expected to print, in the same order.

The input-generation formulas here MUST match the ones in the .cu files.
They are kept deliberately simple (small integers, values exactly
representable in float32) so that the reference itself is easy to trust.
If a test fails, a wrong reference is one of the first things to rule out.
"""

import math


def int_inputs(n: int) -> tuple[list[int], list[int]]:
    """Same formulas as make_int_inputs() in kc_common.h.

    a is in [-50, 50] and b in [-18, 18], so a*b and 3*a - b can never
    overflow a 32-bit int (signed overflow would be undefined behaviour in C++).
    """
    a = [(i * 7) % 101 - 50 for i in range(n)]
    b = [(i * 13) % 37 - 18 for i in range(n)]
    return a, b


def vec_add_int(args: list[str]) -> list[int]:
    a, b = int_inputs(int(args[0]))
    return [x + y for x, y in zip(a, b)]


def vec_mul_int(args: list[str]) -> list[int]:
    a, b = int_inputs(int(args[0]))
    return [x * y for x, y in zip(a, b)]


def vec_size(args: list[str]) -> list[int]:
    a, b = int_inputs(int(args[0]))
    return [3 * x - y for x, y in zip(a, b)]


def c_div(a: int, d: int) -> int:
    """C/C++ integer division truncates toward zero; Python's // floors.

    For example -7 / 2 is -3 in C but -7 // 2 is -4 in Python. Using // here
    would make the *reference* wrong for negative numbers.
    """
    q = abs(a) // abs(d)
    return q if (a >= 0) == (d >= 0) else -q


def int_divmod(args: list[str]) -> list[int]:
    """Output is q0 r0 q1 r1 ... where q = a / d and r = a % d in C semantics."""
    n = int(args[0])
    a, _ = int_inputs(n)
    out = []
    for i in range(n):
        d = (i % 9) - 4
        if d == 0:
            d = 5
        q = c_div(a[i], d)
        out += [q, a[i] - q * d]
    return out


def float_inputs(n: int) -> tuple[list[float], list[float]]:
    """x in [-12.5, 12.25], y in [1, 25.5]; all exactly representable in float32."""
    x = [(i % 100) * 0.25 - 12.5 for i in range(n)]
    y = [((i * 3) % 50) * 0.5 + 1.0 for i in range(n)]
    return x, y


def float_arith(args: list[str]) -> list[float]:
    """out = x*y + x/y - 0.5*y, computed in double precision.

    The GPU computes in float32, so results differ in the last few bits;
    the manifest gives this test an explicit tolerance.
    """
    x, y = float_inputs(int(args[0]))
    return [xi * yi + xi / yi - 0.5 * yi for xi, yi in zip(x, y)]


def saxpy(args: list[str]) -> list[float]:
    n = int(args[0])
    alpha = 2.5
    x = [(i % 64) * 0.125 for i in range(n)]
    y = [1.0 - (i % 32) * 0.0625 for i in range(n)]
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def collatz_steps(start: int, max_steps: int = 10000) -> int:
    n, steps = start, 0
    while n != 1 and steps < max_steps:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def collatz(args: list[str]) -> list[int]:
    return [collatz_steps(i + 1) for i in range(int(args[0]))]


def boundary_guard(args: list[str]) -> list[int]:
    """First n values are 2*i + 1; the guard elements must still hold the sentinel -1."""
    n, guard = int(args[0]), int(args[1])
    return [2 * i + 1 for i in range(n)] + [-1] * guard


def reduce_int(args: list[str]) -> list[int]:
    return [sum(i % 10 for i in range(int(args[0])))]


def reduce_float_inputs(n: int) -> list[float]:
    return [1.0 / (1 + (i % 1000)) for i in range(n)]


def reduce_float(args: list[str]) -> list[float]:
    # math.fsum is an exactly-rounded sum, so the reference has no
    # accumulation error of its own.
    return [math.fsum(reduce_float_inputs(int(args[0])))]


# Name used in manifest.json -> function.
REFERENCES = {
    "vec_add_int": vec_add_int,
    "vec_mul_int": vec_mul_int,
    "vec_size": vec_size,
    "int_divmod": int_divmod,
    "float_arith": float_arith,
    "saxpy": saxpy,
    "collatz": collatz,
    "boundary_guard": boundary_guard,
    "reduce_int": reduce_int,
    "reduce_float": reduce_float,
}
