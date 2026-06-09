from anvil_code_agent.repomap.build import build_repo_map


def test_build_ranks_defining_file_and_lists_symbols(tmp_path):
    # core.py 定义 helper;a.py、b.py 都调用 helper → core.py 应排在最前
    (tmp_path / "core.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "a.py").write_text("from core import helper\n\ndef fa():\n    return helper(1)\n")
    (tmp_path / "b.py").write_text("from core import helper\n\ndef fb():\n    return helper(2)\n")
    text = build_repo_map(str(tmp_path), ["core.py", "a.py", "b.py"], max_chars=2000)
    # core.py 在 a.py / b.py 之前出现
    assert text.index("core.py") < text.index("a.py")
    assert text.index("core.py") < text.index("b.py")
    # 列出了定义符号
    assert "helper" in text


def test_build_respects_char_budget(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text(f"def func{i}():\n    return {i}\n")
    files = [f"f{i}.py" for i in range(50)]
    text = build_repo_map(str(tmp_path), files, max_chars=300)
    assert len(text) <= 400  # 预算 + 截断提示的宽松上界


def test_build_handles_empty_filelist(tmp_path):
    assert build_repo_map(str(tmp_path), [], max_chars=2000) == ""
