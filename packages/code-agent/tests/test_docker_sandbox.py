# packages/code-agent/tests/test_docker_sandbox.py
import pytest
from anvil_code_agent.sandbox import DockerSandbox, has_docker

pytestmark = pytest.mark.skipif(not has_docker(), reason="docker daemon unavailable")


def test_docker_executor_runs_in_container(tmp_path):
    (tmp_path / "hello.txt").write_text("hi\n")
    with DockerSandbox(str(tmp_path)) as box:
        rc, out = box.exec("cat hello.txt")
        assert rc == 0
        assert "hi" in out
        # 容器隔离:容器里能跑命令
        rc2, out2 = box.exec("python -c 'print(2+2)'")
        assert rc2 == 0 and "4" in out2


def test_docker_executor_nonzero_exit(tmp_path):
    with DockerSandbox(str(tmp_path)) as box:
        rc, out = box.exec("exit 7")
        assert rc == 7


def test_docker_sandbox_cleans_up_container(tmp_path):
    import subprocess

    with DockerSandbox(str(tmp_path)) as box:
        name = box.name
    # 退出后容器应已删除
    res = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"],
                         capture_output=True, text=True)
    assert res.stdout.strip() == ""
