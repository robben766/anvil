from anvil_code_agent.harness.context import compact, estimate_tokens


def _convo():
    # system, task, 然后多个"回合":assistant(tool_calls)+tool ... 最后 user
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 500})
    msgs.append({"role": "user", "content": "recent ask"})
    return msgs


def test_summarizer_replaces_middle_and_preserves_pairing():
    m = _convo()
    summary_calls = {}

    def fake_summarizer(middle):
        summary_calls["n"] = len(middle)
        return "did some reading"

    out = compact(m, max_tokens=50, keep_recent=3, tool_cap=80, summarizer=fake_summarizer)
    # 摘要被调用、生成了一条摘要消息
    assert summary_calls.get("n", 0) > 0
    assert any("did some reading" in (x.get("content") or "") for x in out)
    # system + task 仍在最前
    assert out[0] == m[0] and out[1] == m[1]
    # 最近窗口仍在最后
    assert out[-1] == m[-1]
    # tool_use 配对完好:每个 role==tool 的前面都有带该 tool_call_id 的 assistant
    open_ids = set()
    for x in out:
        if x["role"] == "assistant" and x.get("tool_calls"):
            for tc in x["tool_calls"]:
                open_ids.add(tc["id"])
        if x["role"] == "tool":
            assert x["tool_call_id"] in open_ids  # 不孤儿
    # 总 token 下降
    assert estimate_tokens(out) < estimate_tokens(m)


def test_no_summarizer_falls_back_to_truncation_only():
    m = _convo()
    out = compact(m, max_tokens=50, keep_recent=3, tool_cap=80)  # 无 summarizer
    # 行为同 M3:条数不变(只截断,不替换)
    assert len(out) == len(m)


def test_summarizer_not_called_when_under_budget():
    m = _convo()
    called = {"v": False}

    def fake(middle):
        called["v"] = True
        return "x"

    out = compact(m, max_tokens=10_000, keep_recent=3, summarizer=fake)
    assert called["v"] is False
    assert out == m
