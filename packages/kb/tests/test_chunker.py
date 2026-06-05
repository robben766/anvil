from anvil_kb.ingest.chunker import chunk_markdown

DOC = """# 条款
## 等待期
等待期为90天。
意外不受限。
## 表格
| 项目 | 值 |
|------|----|
| 等待期 | 90天 |
说明行。
"""


def test_invariant_offsets_slice_back():
    for c in chunk_markdown(DOC, size=20, overlap=5):
        assert DOC[c.start_offset:c.end_offset] == c.content


def test_header_path_and_table_atomic():
    chunks = chunk_markdown(DOC, size=20, overlap=5)
    waiting = [c for c in chunks if "90天。" in c.content]
    assert waiting and waiting[0].header_path == "条款 > 等待期"
    tables = [c for c in chunks if c.content.startswith("| 项目")]
    assert len(tables) == 1                      # 表格原子,3 行一个 chunk
    assert "| 等待期 | 90天 |" in tables[0].content


def test_window_split_with_overlap():
    text = "a" * 50 + "\n" + "b" * 50 + "\n" + "c" * 50 + "\n"
    chunks = chunk_markdown(text, size=60, overlap=10)
    assert len(chunks) >= 2                      # 153 字符,size 60 必然切多块
    assert all(len(c.content) <= 61 for c in chunks)  # 行对齐允许 +1 容差(换行)
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_no_headers_whole_doc_one_section():
    chunks = chunk_markdown("纯文本一行。", size=600, overlap=100)
    assert len(chunks) == 1 and chunks[0].header_path == ""


def test_seq_monotonic():
    chunks = chunk_markdown(DOC, size=20, overlap=5)
    assert [c.seq for c in chunks] == list(range(len(chunks)))


# ── 边界测试 ──────────────────────────────────────────────────────────────────

def test_empty_section_no_chunk():
    """连续两个标题中间无内容，空 section 不产 chunk。"""
    text = "# A\n## B\n# C\n内容行。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    # A 和 B 都是空 section，不产 chunk；C 有内容
    assert all("内容行。" in c.content for c in chunks)
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_text_before_first_header():
    """文首无标题文本后接标题，无标题段仍正确产 chunk，header_path 为空串。"""
    text = "前言行。\n# 标题\n正文行。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    prelude = [c for c in chunks if "前言行。" in c.content]
    assert prelude and prelude[0].header_path == ""
    body = [c for c in chunks if "正文行。" in c.content]
    assert body and body[0].header_path == "标题"
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_table_at_section_start():
    """表格出现在 section 的第一行，依然原子成 chunk。"""
    text = "# 表格节\n| 列1 | 列2 |\n|-----|-----|\n| v1 | v2 |\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    tables = [c for c in chunks if c.content.startswith("| 列1")]
    assert len(tables) == 1
    assert "| v1 | v2 |" in tables[0].content
    assert tables[0].header_path == "表格节"
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_single_long_line_becomes_own_chunk():
    """单行超过 size 时允许该行独立成超长 chunk，且不变量仍成立。"""
    long_line = "x" * 200 + "\n"
    text = long_line + "短行。\n"
    chunks = chunk_markdown(text, size=60, overlap=10)
    long_chunks = [c for c in chunks if "x" * 10 in c.content]
    assert long_chunks
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_h3_header_path_three_levels():
    """### 时 header_path 应为「一级 > 二级 > 三级」。"""
    text = "# A\n## B\n### C\n内容。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    assert chunks and chunks[0].header_path == "A > B > C"
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_seq_starts_at_zero():
    """seq 从 0 开始连续。"""
    chunks = chunk_markdown("行1。\n行2。\n", size=600, overlap=100)
    assert chunks[0].seq == 0


def test_overlap_does_not_exceed_section_boundary():
    """overlap 回退不能跨越 section 边界（新 section 从 0 开始窗口）。"""
    text = "# S1\n" + "a" * 30 + "\n" + "# S2\n" + "b" * 30 + "\n"
    chunks = chunk_markdown(text, size=40, overlap=10)
    s1 = [c for c in chunks if "a" in c.content]
    s2 = [c for c in chunks if "b" in c.content]
    # S2 的内容不应出现在 S1 的任何 chunk 里
    for c in s1:
        assert "b" not in c.content
    for c in s2:
        assert "a" not in c.content
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


def test_real_overlap_shares_line_content():
    # 10 字符/行;size=25 时窗口含两行(22<25),overlap=10 回退到第二行行首
    text = "a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10 + "\n" + "d" * 10 + "\n"
    chunks = chunk_markdown(text, size=25, overlap=10)
    assert len(chunks) >= 2
    # 相邻 chunk 必须共享至少一行内容(overlap 生效)
    assert chunks[0].content[-11:] == chunks[1].content[:11]
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content


# ── 空白 chunk 过滤测试 (KB-D2 review) ────────────────────────────────────────

def test_no_blank_chunks_from_consecutive_blank_lines():
    """含连续空行的 markdown 不应产生 content.strip()=='' 的 chunk。

    场景：两个节之间有连续空行，chunker 必须将全空白内容过滤掉，
    不输出空 chunk。
    """
    # Consecutive blank lines between sections
    text = "# 第一节\n\n\n\n内容行A。\n\n\n# 第二节\n\n\n内容行B。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    for c in chunks:
        assert c.content.strip() != "", (
            f"Blank chunk found: repr={c.content!r}, seq={c.seq}"
        )


def test_seq_continuous_after_blank_filter():
    """空白 chunk 过滤后 seq 必须仍是从 0 开始的连续整数序列。"""
    # Multiple blank lines that could otherwise produce blank chunks
    text = "# A\n\n\n\n行1。\n\n\n# B\n\n\n行2。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    assert len(chunks) > 0
    assert [c.seq for c in chunks] == list(range(len(chunks))), (
        f"seq not contiguous after blank filter: {[c.seq for c in chunks]}"
    )


def test_blank_only_section_produces_no_chunk():
    """仅含空行的 section（无实际文本）不产生任何 chunk。"""
    # Section body is entirely blank lines
    text = "# 空节\n\n\n\n# 有内容节\n内容行。\n"
    chunks = chunk_markdown(text, size=600, overlap=100)
    # Must have exactly 1 chunk (the content line), none for the blank section
    assert len(chunks) == 1
    assert "内容行。" in chunks[0].content
    assert chunks[0].seq == 0
