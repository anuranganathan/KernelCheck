from pathlib import Path

import pytest

from src.ptx_analyzer import analyze_ptx_file, analyze_ptx_text, compare_ptx, format_stats

DATA = Path(__file__).parent / "data"


def test_add_kernel_statistics():
    stats = analyze_ptx_file(DATA / "sample_add.ptx")
    assert stats["entries"] == ["_Z10add_kernelPKiS0_Pii"]
    assert stats["total_instructions"] == 22
    assert stats["category_counts"]["memory"] == 7
    assert stats["category_counts"]["arithmetic"] == 6     # mad, mul, 4x add
    assert stats["category_counts"]["branch"] == 2         # bra, ret
    assert stats["category_counts"]["sync"] == 0
    assert stats["memory_by_space"] == {"ld.global": 2, "ld.param": 4, "st.global": 1}
    assert stats["predicated_instructions"] == 1
    assert stats["register_declarations"] == {"b32": 9, "b64": 11, "pred": 2}
    assert stats["has_register_declarations"]
    assert not stats["uses_shared_memory"]


def test_reduction_kernel_statistics():
    stats = analyze_ptx_file(DATA / "sample_reduce.ptx")
    assert stats["total_instructions"] == 24
    assert stats["category_counts"]["sync"] == 2           # two bar.sync
    assert stats["category_counts"]["bitwise"] == 2        # shl, shr
    assert stats["memory_by_space"]["atom.global"] == 1
    assert stats["memory_by_space"]["ld.shared"] == 2
    assert stats["memory_by_space"]["st.shared"] == 2
    assert stats["uses_shared_memory"]
    assert stats["opcode_counts"]["bra"] == 3


def test_comments_labels_and_directives_are_not_instructions():
    text = """
    .version 8.4   // a comment
    /* a block
       comment with add.s32 inside */
    .reg .f32 %f1, %f2;
    $L__BB0_1:
    add.f32 %f1, %f1, %f2;   // add.s32 in a comment
    """
    stats = analyze_ptx_text(text)
    assert stats["total_instructions"] == 1
    assert stats["opcode_counts"] == {"add": 1}
    assert stats["register_declarations"] == {"f32": 2}


def test_debug_sections_and_loc_lines_are_skipped():
    # -G output has .loc/.file directives and .section blocks of raw bytes.
    text = """
    .file 1 "prog.cu"
    .loc 1 5 3
    mov.u32 %r1, %tid.x;
    ret;
    .section .debug_abbrev
    {
    .b8 1
    .b8 17
    }
    """
    stats = analyze_ptx_text(text)
    assert stats["total_instructions"] == 2


def test_multi_line_call_counts_once():
    text = """
    call.uni (retval0),
    vprintf,
    (
    param0,
    param1
    );
    ret;
    """
    stats = analyze_ptx_text(text)
    assert stats["opcode_counts"] == {"call": 1, "ret": 1}


def test_generic_memory_access():
    stats = analyze_ptx_text("ld.u32 %r1, [%rd1];\nst.volatile.shared.u32 [%r2], %r1;\n")
    assert stats["memory_by_space"] == {"ld.generic": 1, "st.shared": 1}


def test_empty_ptx():
    stats = analyze_ptx_text("")
    assert stats["total_instructions"] == 0
    assert not stats["has_register_declarations"]
    assert "none declared" in format_stats(stats)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        analyze_ptx_file(DATA / "missing.ptx")


def test_compare_identical():
    stats = analyze_ptx_file(DATA / "sample_add.ptx")
    result = compare_ptx(stats, stats, "O0", "O2")
    assert result["identical_statistics"]
    assert result["differences"] == {}


def test_compare_different():
    add = analyze_ptx_file(DATA / "sample_add.ptx")
    reduce = analyze_ptx_file(DATA / "sample_reduce.ptx")
    result = compare_ptx(add, reduce, "add", "reduce")
    assert not result["identical_statistics"]
    assert result["differences"]["total_instructions"] == {"add": 22, "reduce": 24}
    assert result["differences"]["category_counts.sync"] == {"add": 0, "reduce": 2}
    assert result["differences"]["memory_by_space.atom.global"] == {"add": 0, "reduce": 1}
