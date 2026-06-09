from anvil_ai_employee.mcp.client import McpToolSpec


def _spec(name, annotations, props=None):
    return McpToolSpec(
        name=name,
        description=f"{name} desc",
        input_schema={"type": "object", "properties": props or {}, "required": []},
        annotations=annotations,
    )


def test_mcp_risk_mapping():
    from anvil_ai_employee.mcp.adapter import mcp_risk

    assert mcp_risk(_spec("read", {"readOnlyHint": True})) == "low"
    assert mcp_risk(_spec("send", {"destructiveHint": True})) == "high"
    assert mcp_risk(_spec("none", {})) == "high"  # no annotations → conservative
    assert mcp_risk(_spec("idem", {"idempotentHint": True})) == "medium"


class _FakeClient:
    connector = "gmail"

    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"called {name} with {arguments}"


def test_mcp_tools_namespacing_and_schema():
    from anvil_ai_employee.mcp.adapter import mcp_tools

    client = _FakeClient()
    specs = [_spec("send_email", {"destructiveHint": True}, props={"to": {"type": "string"}})]
    tools = mcp_tools(client, specs)
    assert tools[0].name == "gmail__send_email"
    assert tools[0].schema["function"]["parameters"]["properties"] == {"to": {"type": "string"}}


def test_mcp_tools_fn_dispatches_to_client_with_unqualified_name():
    from anvil_ai_employee.mcp.adapter import mcp_tools
    from anvil_code_agent.tools.base import ToolContext

    client = _FakeClient()
    specs = [
        _spec("send_email", {"destructiveHint": True}),
        _spec("list_events", {"readOnlyHint": True}),
    ]
    tools = {t.name: t for t in mcp_tools(client, specs)}
    res = tools["gmail__send_email"]({"to": "x"}, ToolContext(workdir="/tmp"))
    assert res.ok is True
    assert "called send_email" in res.content
    assert client.calls == [("send_email", {"to": "x"})]  # unqualified name to server
