from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool


def test_tool_decorator_builds_schema():
    @tool(
        name="echo",
        description="echo text",
        params={"text": {"type": "string"}},
        required=["text"],
    )
    def echo(args, ctx):
        return ToolResult(content=args["text"], ok=True)

    assert echo.name == "echo"
    sch = echo.schema
    assert sch["type"] == "function"
    assert sch["function"]["name"] == "echo"
    assert sch["function"]["parameters"]["required"] == ["text"]


def test_registry_collects_schemas_and_dispatches():
    @tool(name="echo", description="d", params={"text": {"type": "string"}}, required=["text"])
    def echo(args, ctx):
        return ToolResult(content=args["text"].upper(), ok=True)

    reg = ToolRegistry([echo])
    assert [s["function"]["name"] for s in reg.schemas()] == ["echo"]
    res = reg.dispatch("echo", {"text": "hi"}, ToolContext(workdir="/tmp"))
    assert res.content == "HI" and res.ok


def test_registry_unknown_tool_returns_error_result():
    reg = ToolRegistry([])
    res = reg.dispatch("nope", {}, ToolContext(workdir="/tmp"))
    assert res.ok is False
    assert "unknown tool" in res.content.lower()


def test_dispatch_catches_tool_exception_as_error_result():
    @tool(name="boom", description="d", params={}, required=[])
    def boom(args, ctx):
        raise RuntimeError("kaboom")

    reg = ToolRegistry([boom])
    res = reg.dispatch("boom", {}, ToolContext(workdir="/tmp"))
    assert res.ok is False
    assert "kaboom" in res.content
