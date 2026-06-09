import json
import subprocess
import sys


def _run_sequence(messages, env=None):
    """Feed newline-delimited JSON messages to the mock server, collect its responses."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "anvil_ai_employee.mcp.mock_servers.email_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    payload = "".join(json.dumps(m) + "\n" for m in messages)
    out, _ = proc.communicate(payload, timeout=10)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_initialize_and_tools_list():
    replies = _run_sequence(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
    )
    # initialized is a notification → no reply; so 2 replies for ids 1 and 2.
    assert replies[0]["id"] == 1
    assert replies[0]["result"]["protocolVersion"]
    tools = {t["name"]: t for t in replies[1]["result"]["tools"]}
    assert set(tools) == {"list_events", "send_email"}
    assert tools["list_events"]["annotations"]["readOnlyHint"] is True
    assert tools["send_email"]["annotations"]["destructiveHint"] is True


def test_send_email_requires_token():
    import os

    base = dict(os.environ)
    base.pop("GMAIL_TOKEN", None)
    replies = _run_sequence(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": "x", "subject": "s", "body": "b"},
                },
            }
        ],
        env=base,
    )
    assert replies[0]["result"]["isError"] is True


def test_send_email_with_token_echoes_redactable_token():
    import os

    base = dict(os.environ)
    base["GMAIL_TOKEN"] = "TOKVAL"
    replies = _run_sequence(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": "x", "subject": "s", "body": "b"},
                },
            }
        ],
        env=base,
    )
    text = replies[0]["result"]["content"][0]["text"]
    assert "TOKVAL" in text  # client-side redaction (Task 4) will scrub this


def test_unknown_method_is_jsonrpc_error():
    replies = _run_sequence([{"jsonrpc": "2.0", "id": 9, "method": "no/such"}])
    assert replies[0]["error"]["code"] == -32601
