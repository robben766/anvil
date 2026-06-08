import os

import pytest
from anvil_council.agreement import compare_jury
from anvil_council.cli import _calibrate as calibrate_entry  # noqa: F401  (import-existence)


def test_compare_jury_used_by_cli_path():
    rep = compare_jury(
        {"deepseek-chat": [1.0, 0.0], "qwen-plus": [1.0, 0.0]},
        [1.0, 0.0],
        [1.0, 0.0],
    )
    assert rep.jury_vs_human == 1.0
    assert rep.n == 2


@pytest.mark.live
async def test_calibrate_live_smoke():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("no key")
