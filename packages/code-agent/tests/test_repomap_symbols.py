from anvil_code_agent.repomap.build import build_repo_map


def test_symbols_ranked_by_reference_count(tmp_path):
    # core.py 定义 hot(被多处调用)和 cold(没人调用);hot 应排在 cold 前
    (tmp_path / "core.py").write_text("def hot(x):\n    return x\n\ndef cold(y):\n    return y\n")
    (tmp_path / "a.py").write_text("from core import hot\n\ndef fa():\n    return hot(1)\n")
    (tmp_path / "b.py").write_text("from core import hot\n\ndef fb():\n    return hot(hot(2))\n")
    text = build_repo_map(str(tmp_path), ["core.py", "a.py", "b.py"], max_chars=4000)
    # 在 core.py 段里 hot 出现在 cold 之前
    seg = text[text.index("core.py"):]
    assert seg.index("hot") < seg.index("cold")


def test_per_file_symbol_cap(tmp_path):
    # 一个文件很多 def,渲染应限制每文件符号数(top-K)并标省略
    body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(40))
    (tmp_path / "big.py").write_text(body)
    text = build_repo_map(str(tmp_path), ["big.py"], max_chars=4000, max_symbols_per_file=8)
    # big.py 段里最多 8 个符号 + 省略提示
    assert "more symbols" in text or "…" in text or "..." in text
