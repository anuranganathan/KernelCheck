"""Checks on the reference implementations themselves.

A wrong reference produces a false "OUTPUT_MISMATCH", so the references get
their own tests against hand-checked values.
"""

import random
import struct
from array import array

import config
from reference import references as ref
from src.manifest import load_manifest
from src.verifier import compare


def f32(value: float) -> float:
    """Round a Python float (double) to the nearest float32."""
    return struct.unpack("f", struct.pack("f", value))[0]


def test_c_division_truncates_toward_zero():
    assert ref.c_div(7, 2) == 3
    assert ref.c_div(-7, 2) == -3      # Python's -7 // 2 would give -4
    assert ref.c_div(7, -2) == -3
    assert ref.c_div(-7, -2) == 3


def test_int_divmod_satisfies_c_identity():
    out = ref.int_divmod(["200"])
    a, _ = ref.int_inputs(200)
    for i in range(200):
        q, r = out[2 * i], out[2 * i + 1]
        d = (i % 9) - 4 or 5
        assert q * d + r == a[i]
        assert abs(r) < abs(d)
        assert r == 0 or (r > 0) == (a[i] > 0)   # remainder takes the sign of a in C


def test_int_inputs_stay_small():
    a, b = ref.int_inputs(10000)
    assert min(a) == -50 and max(a) == 50
    assert min(b) == -18 and max(b) == 18


def test_known_values():
    assert ref.vec_add_int(["3"]) == [-50 + -18, -43 + -5, -36 + 8]
    assert ref.collatz(["1"]) == [0]
    assert ref.collatz(["27"])[26] == 111       # well-known: 27 takes 111 steps
    assert ref.boundary_guard(["3", "2"]) == [1, 3, 5, -1, -1]
    assert ref.reduce_int(["25"]) == [45 + 45 + 0 + 1 + 2 + 3 + 4]


def test_output_lengths_match_sizes():
    for n in ["1", "257", "65537"]:
        assert len(ref.vec_size([n])) == int(n)
    assert len(ref.int_divmod(["10"])) == 20


def test_float_inputs_are_exact_in_float32():
    x, y = ref.float_inputs(1000)
    assert all(f32(v) == v for v in x + y)


def test_float32_emulation_of_float_arith_is_within_default_tolerance():
    """Imitate what the GPU does (float32 rounding after every operation, with
    and without a fused multiply-add) and check the documented default
    tolerance accepts it. This justifies the tolerance; it does not replace
    running on a GPU."""
    n = 4096
    x, y = ref.float_inputs(n)
    expected = ref.float_arith([str(n)])
    plain = [f32(f32(f32(a * b) + f32(a / b)) - f32(0.5 * b)) for a, b in zip(x, y)]
    # FMA: a*b is computed exactly and rounded only once together with the add.
    fused = [f32(f32(a * b + f32(a / b)) - f32(0.5 * b)) for a, b in zip(x, y)]
    assert compare(expected, plain, "float", rel_tol=1e-5, abs_tol=1e-6).passed
    assert compare(expected, fused, "float", rel_tol=1e-5, abs_tol=1e-6).passed


def _reduce_float_block_sums(n: int) -> list[float]:
    """float32 tree reduction of each 256-element block, as reduce_float.cu does."""
    values = ref.reduce_float_inputs(n)
    sums = []
    for start in range(0, n, 256):
        block = array("f", values[start:start + 256])
        stride = 128
        while stride > 0:
            for i in range(stride):
                block[i] += block[i + stride]
            stride //= 2
        sums.append(block[0])
    return sums


def test_reduce_float_tolerance_accepts_rounding_but_catches_a_lost_block():
    """The GPU adds the block sums with atomicAdd in no fixed order, so emulate
    several random orders in float32. The manifest tolerance must accept all
    of them, yet reject the result if any single block were lost."""
    n = 1 << 20
    test = next(t for t in load_manifest(config.MANIFEST_PATH) if t.id == "reduce_float")
    expected = ref.reduce_float([str(n)])
    block_sums = _reduce_float_block_sums(n)
    rng = random.Random(0)
    for _ in range(20):
        order = block_sums[:]
        rng.shuffle(order)
        total = array("f", [0.0])
        for value in order:
            total[0] += value
        assert total[0] != expected[0]      # float32 really does lose precision
        assert compare(expected, [total[0]], "float", test.rel_tol, test.abs_tol).passed

    smallest_block = min(block_sums)
    assert not compare(expected, [expected[0] - smallest_block], "float",
                       test.rel_tol, test.abs_tol).passed
