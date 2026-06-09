import sys

import pytest

pytestmark = pytest.mark.asyncio

_ECHO = (
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    line = line.strip()\n"
    "    if not line:\n"
    "        continue\n"
    "    msg = json.loads(line)\n"
    "    sys.stdout.write(json.dumps({'echo': msg}) + '\\n')\n"
    "    sys.stdout.flush()\n"
)


async def test_send_receive_roundtrip():
    from anvil_ai_employee.mcp.transport import StdioTransport

    t = StdioTransport(command=sys.executable, args=["-c", _ECHO], env={})
    await t.start()
    try:
        await t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        reply = await t.receive()
        assert reply == {"echo": {"jsonrpc": "2.0", "id": 1, "method": "ping"}}
    finally:
        await t.close()


async def test_env_is_passed_to_subprocess():
    from anvil_ai_employee.mcp.transport import StdioTransport

    prog = (
        "import sys, os, json\n"
        "sys.stdout.write(json.dumps({'tok': os.environ.get('MY_TOKEN')}) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    t = StdioTransport(command=sys.executable, args=["-c", prog], env={"MY_TOKEN": "abc"})
    await t.start()
    try:
        assert await t.receive() == {"tok": "abc"}
    finally:
        await t.close()


async def test_receive_on_eof_raises():
    from anvil_ai_employee.mcp.transport import McpTransportError, StdioTransport

    t = StdioTransport(command=sys.executable, args=["-c", "import sys; sys.exit(0)"], env={})
    await t.start()
    try:
        with pytest.raises(McpTransportError):
            await t.receive()
    finally:
        await t.close()
