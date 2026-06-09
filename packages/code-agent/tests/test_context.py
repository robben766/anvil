from anvil_code_agent.harness.context import compact, estimate_tokens


def _msgs():
    return [
        {"role": "system", "content": "you are a coder"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "X" * 4000},  # 老的大工具输出
        {"role": "assistant", "content": "still working"},
        {"role": "user", "content": "keep going"},
    ]


def test_estimate_tokens_grows_with_content():
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "hi" * 1000}]
    assert estimate_tokens(big) > estimate_tokens(small)
    assert estimate_tokens([]) == 0


def test_compact_noop_under_budget():
    m = _msgs()
    assert compact(m, max_tokens=10_000) == m  # 未超预算原样返回


def test_compact_truncates_old_tool_output_but_keeps_structure():
    m = _msgs()
    out = compact(m, max_tokens=100, keep_recent=2, tool_cap=120)
    # 消息条数、role 序列、tool_call_id 全保留(不破坏 tool_use 配对)
    assert len(out) == len(m)
    assert [x["role"] for x in out] == [x["role"] for x in m]
    assert out[3]["tool_call_id"] == "c1"
    # 老的大工具输出被截断 → 总 token 下降
    assert estimate_tokens(out) < estimate_tokens(m)
    assert "truncated" in out[3]["content"]


def test_compact_protects_system_task_and_recent():
    m = _msgs()
    out = compact(m, max_tokens=1, keep_recent=2, tool_cap=50)
    assert out[0] == m[0]  # system 不动
    assert out[1] == m[1]  # 原始任务不动
    assert out[-1] == m[-1]  # 最近窗口不动
