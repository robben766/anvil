# Council-Jury Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-model evaluation jury (`packages/core/council`) where several different-model jurors independently score a candidate answer against a shared functional rubric, aggregate into a calibrated verdict with explicit disagreement, and measure whether the jury beats the best single juror against human labels.

**Architecture:** A new universal package `anvil-council` reuses `gateway` (each juror is a model), `guard.structured_chat` (each juror returns structured scores), and `eval.calibration` (Cohen's κ / quantize). Jurors fan out in parallel and score independently (no cross-talk — evaluation wants independence, which also sidesteps sycophancy/homogenization). Aggregation takes per-criterion medians and flags criteria where jurors diverge. Calibration extends the existing pairwise Cohen's κ with a hand-rolled Fleiss' κ (≥2 raters) and a jury-vs-human-vs-best-single comparison.

**Tech Stack:** Python 3.12, uv workspace, pytest (`asyncio_mode=auto`), respx (mock juror HTTP calls), anvil-gateway / anvil-guard / anvil-eval (workspace deps), stdlib `statistics`/`asyncio`. No new third-party deps.

---

## Context for the implementer (read once)

You are working in the `anvil` monorepo. Key facts:

- **Workspace:** `[tool.uv.workspace] members = ["packages/core/*", "packages/kb", "apps/kb-api"]` (root `pyproject.toml`). `packages/core/*` is a glob, so a new `packages/core/council` auto-joins once it has a `pyproject.toml`. No root edit needed.
- **Test config** (root `pyproject.toml`): `asyncio_mode = "auto"` (so `async def test_*` needs NO decorator), marker `live` for real API calls. CI runs `uv run pytest -m "not live" -q`.
- **Reused interfaces (verified):**
  - `from anvil_guard import structured_chat` — `async def structured_chat(model, messages, *, schema=None, max_retries=1, **chat_kwargs) -> dict`. With `schema={"required":[...]}` it enforces those keys are present, retries once, raises `StructuredOutputError` on failure. Default `response_format={"type":"json_object"}`. Callers must mention "JSON" in the prompt.
  - `from anvil_eval.calibration import quantize, cohen_kappa, CalibrationCase, load_calibration` — `quantize(score: float) -> int` maps [0,1]→{0,1,2}; `cohen_kappa(a: list[int], b: list[int]) -> float`; `CalibrationCase{id, question, reference, answer, human_score}`; `load_calibration(path) -> list[CalibrationCase]`.
  - `from anvil_gateway import chat, configure`. Models registered: `deepseek-chat` (provider deepseek, env `DEEPSEEK_API_KEY`) and `qwen-plus` (provider dashscope, env `DASHSCOPE_API_KEY`).
- **Mocking pattern** (copy from `packages/core/eval/tests/test_judge.py`): `@respx.mock` against `https://api.deepseek.com/v1/chat/completions`; autouse `env` fixture sets `DEEPSEEK_API_KEY` and calls `anvil_gateway.configure(database_url=os.environ.get("ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"), retry_base_delay=0)`. DashScope's OpenAI-compatible URL is `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` — mock that for qwen-plus jurors.
- **Commands:** `uv sync --all-packages`; `uv run pytest -m "not live" -q`; lint `uv run ruff check .`. ruff config selects `["E","F","I","UP","B"]` (B = bugbear: use `zip(a, b, strict=True)`, no unused imports).
- **Branch:** all work on `feat/council-jury` (already created, spec already committed there). Do NOT switch branches.
- **Commit rules:** conventional commits, English subject, author email is the pinned GitHub noreply (do not touch git config). `.env` never committed.
- **Public-repo red lines:** no company names (one-policy/forge/iic/zbcloud/TAPD/radium), no machine absolute paths (`/home/itachi/...`), no real personal email in committed files. Golden cases use the fictional 星辉人寿 universe.
- **Live experiment dependency:** CJ-M2's headline numbers require BOTH `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY` in `.env` (the 百炼 key activates the second juror, the long-pending task #22). Unit tests are fully mocked and need neither. If `DASHSCOPE_API_KEY` is absent at experiment time, run the jury DeepSeek-only and record that the second juror is pending — do NOT fabricate numbers.

---

## File Structure

**New package `packages/core/council/`:**
- `pyproject.toml` — `anvil-council`, deps `anvil-gateway` / `anvil-guard` / `anvil-eval`
- `src/anvil_council/__init__.py` — public exports
- `src/anvil_council/rubric.py` — `Criterion`, `Rubric`, `DEFAULT_RUBRIC` (functional scoring dimensions, NOT hand-written personas)
- `src/anvil_council/verdict.py` — `JurorScore`, `Verdict` dataclasses
- `src/anvil_council/aggregate.py` — `aggregate()` (per-criterion median + disagreement flag + confidence)
- `src/anvil_council/seats.py` — `score_one()`, `score_case()` (parallel fan-out via `structured_chat`)
- `src/anvil_council/agreement.py` — `fleiss_kappa()`, `jurors_fleiss()`, `compare_jury()` (jury vs human vs best-single)
- `src/anvil_council/cli.py` — `anvil-council judge` / `anvil-council calibrate`
- `tests/test_guard_sanity.py`-style sanity + per-module tests

**Extended:**
- `packages/core/eval/golden/calibration.jsonl` — expand 14 → ~30 labeled cases (CJ-M2)
- `examples/05-council/README.md` — experiment writeup (CJ-M3)
- `CLAUDE.md` — anvil-council section (CJ-M3)

---

## CJ-M1: Core jury engine

### Task 1: `anvil-council` package skeleton

**Files:**
- Create: `packages/core/council/pyproject.toml`
- Create: `packages/core/council/src/anvil_council/__init__.py`
- Create: `packages/core/council/tests/test_council_sanity.py`

- [ ] **Step 1: Write `packages/core/council/pyproject.toml`**

```toml
[project]
name = "anvil-council"
version = "0.1.0"
description = "anvil: multi-model evaluation jury — independent juror scoring, aggregation, disagreement, calibration"
requires-python = ">=3.12"
dependencies = [
    "anvil-gateway",
    "anvil-guard",
    "anvil-eval",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "respx>=0.21",
    "httpx>=0.27",
    "ruff>=0.6",
]

[tool.uv.sources]
anvil-gateway = { workspace = true }
anvil-guard = { workspace = true }
anvil-eval = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/anvil_council"]
```

- [ ] **Step 2: Write `packages/core/council/src/anvil_council/__init__.py`**

```python
"""anvil-council: multi-model evaluation jury (圈1 universal)."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write `packages/core/council/tests/test_council_sanity.py`**

```python
def test_package_imports():
    import anvil_council

    assert anvil_council.__version__ == "0.1.0"
```

- [ ] **Step 4: Sync and test**

Run: `uv sync --all-packages && uv run pytest packages/core/council -q`
Expected: PASS (1 test); `anvil-council` discovered as a new member.

- [ ] **Step 5: Commit**

```bash
git add packages/core/council/pyproject.toml packages/core/council/src/anvil_council/__init__.py packages/core/council/tests/test_council_sanity.py uv.lock
git commit -m "feat(council): scaffold anvil-council package (multi-model eval jury)"
```

---

### Task 2: `rubric.py` — functional scoring dimensions

**Files:**
- Create: `packages/core/council/src/anvil_council/rubric.py`
- Create: `packages/core/council/tests/test_rubric.py`

- [ ] **Step 1: Write the failing test** — `packages/core/council/tests/test_rubric.py`

```python
from anvil_council.rubric import DEFAULT_RUBRIC, Criterion, Rubric


def test_default_rubric_has_functional_criteria():
    assert isinstance(DEFAULT_RUBRIC, Rubric)
    keys = DEFAULT_RUBRIC.keys()
    assert keys == ["correctness", "evidence", "completeness", "relevance"]
    # criteria are functional dimensions, not personas
    for c in DEFAULT_RUBRIC.criteria:
        assert isinstance(c, Criterion)
        assert c.description


def test_rubric_keys_helper():
    r = Rubric(name="t", criteria=(Criterion("a", "desc a"), Criterion("b", "desc b")))
    assert r.keys() == ["a", "b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_rubric.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'anvil_council.rubric'`

- [ ] **Step 3: Implement `packages/core/council/src/anvil_council/rubric.py`**

```python
"""Scoring rubric = a set of FUNCTIONAL criteria (correctness/evidence/...), NOT
hand-written personas. Jurors all score the same rubric independently."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    description: str


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[Criterion, ...]

    def keys(self) -> list[str]:
        return [c.key for c in self.criteria]


DEFAULT_RUBRIC = Rubric(
    name="general-answer-quality",
    criteria=(
        Criterion("correctness", "答案是否与参考一致、事实正确,无错误或矛盾"),
        Criterion("evidence", "答案是否有充分依据/证据支撑,而非凭空断言"),
        Criterion("completeness", "答案是否完整覆盖问题的关键要点"),
        Criterion("relevance", "答案是否切题,无离题或冗余内容"),
    ),
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_rubric.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/council/src/anvil_council/rubric.py packages/core/council/tests/test_rubric.py
git commit -m "feat(council): functional scoring rubric (no hand-written personas)"
```

---

### Task 3: `verdict.py` + `aggregate.py` — verdict model and aggregation

**Files:**
- Create: `packages/core/council/src/anvil_council/verdict.py`
- Create: `packages/core/council/src/anvil_council/aggregate.py`
- Create: `packages/core/council/tests/test_aggregate.py`

- [ ] **Step 1: Write the failing test (hand-computed)** — `packages/core/council/tests/test_aggregate.py`

```python
import pytest
from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.verdict import JurorScore, Verdict


def _js(model, correctness, evidence, completeness, relevance, overall):
    return JurorScore(
        model=model,
        per_criterion={
            "correctness": correctness,
            "evidence": evidence,
            "completeness": completeness,
            "relevance": relevance,
        },
        overall=overall,
        reason="r",
    )


def test_aggregate_medians_and_disagreement():
    # juror A scores correctness 1.0, juror B scores 0.0 → spread 1.0 > 0.5 → disagreement
    a = _js("deepseek-chat", 1.0, 1.0, 1.0, 1.0, 1.0)
    b = _js("qwen-plus", 0.0, 1.0, 1.0, 1.0, 0.5)
    v = aggregate([a, b], DEFAULT_RUBRIC, disagreement_threshold=0.5)
    assert isinstance(v, Verdict)
    # median of [1.0, 0.0] = 0.5
    assert v.per_criterion["correctness"] == pytest.approx(0.5)
    assert v.per_criterion["evidence"] == pytest.approx(1.0)
    # overall median of [1.0, 0.5] = 0.75
    assert v.overall == pytest.approx(0.75)
    # only correctness diverged beyond threshold
    assert v.disagreements == ["correctness"]
    # confidence = 1 - mean(spreads); spreads = [1.0, 0, 0, 0] → mean 0.25 → conf 0.75
    assert v.confidence == pytest.approx(0.75)
    assert len(v.jurors) == 2


def test_aggregate_unanimous_no_disagreement_full_confidence():
    a = _js("deepseek-chat", 1.0, 1.0, 1.0, 1.0, 1.0)
    b = _js("qwen-plus", 1.0, 1.0, 1.0, 1.0, 1.0)
    v = aggregate([a, b], DEFAULT_RUBRIC)
    assert v.disagreements == []
    assert v.confidence == pytest.approx(1.0)
    assert v.overall == pytest.approx(1.0)


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([], DEFAULT_RUBRIC)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_aggregate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'anvil_council.aggregate'`

- [ ] **Step 3: Implement `packages/core/council/src/anvil_council/verdict.py`**

```python
"""Verdict data model: a juror's scores and the aggregated jury verdict."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JurorScore:
    model: str
    per_criterion: dict[str, float]  # criterion key -> score in [0,1]
    overall: float
    reason: str


@dataclass
class Verdict:
    overall: float
    per_criterion: dict[str, float]
    confidence: float  # 1 - mean per-criterion spread, clamped to [0,1]
    disagreements: list[str]  # criterion keys where jurors diverged beyond threshold
    jurors: list[JurorScore] = field(default_factory=list)
```

- [ ] **Step 4: Implement `packages/core/council/src/anvil_council/aggregate.py`**

```python
"""Aggregate independent juror scores into a verdict.

Per-criterion median (robust to one outlier juror) + disagreement flag when the
spread (max-min) on a criterion exceeds a threshold. Confidence = 1 - mean spread.
Pure function, no network."""

from __future__ import annotations

import statistics

from anvil_council.rubric import DEFAULT_RUBRIC, Rubric
from anvil_council.verdict import JurorScore, Verdict


def aggregate(
    scores: list[JurorScore],
    rubric: Rubric = DEFAULT_RUBRIC,
    disagreement_threshold: float = 0.5,
) -> Verdict:
    if not scores:
        raise ValueError("cannot aggregate an empty juror list")
    per_criterion: dict[str, float] = {}
    disagreements: list[str] = []
    spreads: list[float] = []
    for c in rubric.criteria:
        vals = [s.per_criterion[c.key] for s in scores]
        per_criterion[c.key] = statistics.median(vals)
        spread = max(vals) - min(vals)
        spreads.append(spread)
        if spread > disagreement_threshold:
            disagreements.append(c.key)
    overall = statistics.median([s.overall for s in scores])
    confidence = max(0.0, 1.0 - (sum(spreads) / len(spreads)))
    return Verdict(
        overall=overall,
        per_criterion=per_criterion,
        confidence=confidence,
        disagreements=disagreements,
        jurors=list(scores),
    )
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_aggregate.py -q`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add packages/core/council/src/anvil_council/verdict.py packages/core/council/src/anvil_council/aggregate.py packages/core/council/tests/test_aggregate.py
git commit -m "feat(council): verdict model + median aggregation with disagreement flag"
```

---

### Task 4: `seats.py` — parallel juror scoring

**Files:**
- Create: `packages/core/council/src/anvil_council/seats.py`
- Create: `packages/core/council/tests/test_seats.py`
- Modify: `packages/core/council/src/anvil_council/__init__.py`

- [ ] **Step 1: Write the failing test** — `packages/core/council/tests/test_seats.py`

```python
import json
import os

import httpx
import pytest
import respx
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.seats import score_case, score_one
from anvil_council.verdict import JurorScore

DS_URL = "https://api.deepseek.com/v1/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

CASE = {
    "question": "等待期是多少天?",
    "reference": "本合同等待期为90天。",
    "answer": "等待期为90天。",
}


def _resp(obj: dict, model: str):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _score_obj(score: float):
    return {
        "reason": "符合参考",
        "per_criterion": {
            "correctness": score,
            "evidence": score,
            "completeness": score,
            "relevance": score,
        },
        "overall": score,
    }


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k2")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@respx.mock
async def test_score_one_returns_juror_score():
    respx.post(DS_URL).mock(return_value=_resp(_score_obj(1.0), "deepseek-chat"))
    js = await score_one("deepseek-chat", CASE, DEFAULT_RUBRIC)
    assert isinstance(js, JurorScore)
    assert js.model == "deepseek-chat"
    assert js.per_criterion["correctness"] == 1.0
    assert js.overall == 1.0


@respx.mock
async def test_score_one_defaults_missing_criterion_to_zero():
    # model omits "relevance" → defaults to 0.0, does not crash
    obj = {"reason": "r", "per_criterion": {"correctness": 1.0}, "overall": 0.5}
    respx.post(DS_URL).mock(return_value=_resp(obj, "deepseek-chat"))
    js = await score_one("deepseek-chat", CASE, DEFAULT_RUBRIC)
    assert js.per_criterion["relevance"] == 0.0
    assert js.per_criterion["correctness"] == 1.0


@respx.mock
async def test_score_case_fans_out_to_all_models():
    respx.post(DS_URL).mock(return_value=_resp(_score_obj(1.0), "deepseek-chat"))
    respx.post(QWEN_URL).mock(return_value=_resp(_score_obj(0.5), "qwen-plus"))
    scores = await score_case(CASE, ["deepseek-chat", "qwen-plus"], DEFAULT_RUBRIC)
    assert len(scores) == 2
    models = {s.model for s in scores}
    assert models == {"deepseek-chat", "qwen-plus"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_seats.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'anvil_council.seats'`

- [ ] **Step 3: Implement `packages/core/council/src/anvil_council/seats.py`**

```python
"""Jurors score a case independently and in parallel via guard.structured_chat.

Each juror is a model. Jurors do NOT see each other's scores — evaluation wants
independence (this also avoids sycophancy/anchoring). The prompt mentions JSON so
DeepSeek's json_object mode engages."""

from __future__ import annotations

import asyncio
from typing import Any

from anvil_guard import structured_chat

from anvil_council.rubric import DEFAULT_RUBRIC, Rubric
from anvil_council.verdict import JurorScore

_SYSTEM = (
    "你是严格的评测陪审员。仅依据给定资料,对候选答案按每个维度打 0~1 分"
    "(1=完全满足,0=完全不满足)。先写 reason 再给分,只输出一个 JSON 对象。"
)


def _build_messages(case: dict[str, Any], rubric: Rubric) -> list[dict[str, str]]:
    crit_lines = "\n".join(f"- {c.key}: {c.description}" for c in rubric.criteria)
    keys = ", ".join(rubric.keys())
    user = (
        f"问题:{case['question']}\n"
        f"参考答案:{case['reference']}\n"
        f"候选答案:{case['answer']}\n\n"
        f"评分维度:\n{crit_lines}\n\n"
        f'只输出 JSON,字段:reason(简短理由)、per_criterion(对象,键为 [{keys}],'
        f"值为 0~1 的数字)、overall(0~1 的总分)。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


async def score_one(
    model: str, case: dict[str, Any], rubric: Rubric = DEFAULT_RUBRIC
) -> JurorScore:
    obj = await structured_chat(
        model,
        _build_messages(case, rubric),
        schema={"required": ["per_criterion", "overall"]},
        temperature=0.0,
        session_id="anvil-council",
    )
    raw = obj.get("per_criterion") or {}
    per_criterion = {c.key: float(raw.get(c.key, 0.0)) for c in rubric.criteria}
    return JurorScore(
        model=model,
        per_criterion=per_criterion,
        overall=float(obj.get("overall", 0.0)),
        reason=str(obj.get("reason", "")),
    )


async def score_case(
    case: dict[str, Any], models: list[str], rubric: Rubric = DEFAULT_RUBRIC
) -> list[JurorScore]:
    """Fan out: every juror scores the same case in parallel, independently."""
    return list(await asyncio.gather(*(score_one(m, case, rubric) for m in models)))
```

- [ ] **Step 4: Export from `__init__.py`** — replace `packages/core/council/src/anvil_council/__init__.py`

```python
"""anvil-council: multi-model evaluation jury (圈1 universal)."""

from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC, Criterion, Rubric
from anvil_council.seats import score_case, score_one
from anvil_council.verdict import JurorScore, Verdict

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_RUBRIC",
    "Criterion",
    "JurorScore",
    "Rubric",
    "Verdict",
    "aggregate",
    "score_case",
    "score_one",
]
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest packages/core/council -q`
Expected: PASS (sanity + rubric + aggregate + seats). Also `uv run ruff check packages/core/council` clean.

- [ ] **Step 6: Commit**

```bash
git add packages/core/council/src/anvil_council/seats.py packages/core/council/src/anvil_council/__init__.py packages/core/council/tests/test_seats.py
git commit -m "feat(council): parallel independent juror scoring via structured_chat"
```

---

### Task 5: `cli.py` — `anvil-council judge`

**Files:**
- Create: `packages/core/council/src/anvil_council/cli.py`
- Modify: `packages/core/council/pyproject.toml` (add `[project.scripts]`)
- Create: `packages/core/council/tests/test_cli_judge.py`

- [ ] **Step 1: Add the console script to `pyproject.toml`** — insert after the `[project]` `dependencies` block (top-level table):

```toml
[project.scripts]
anvil-council = "anvil_council.cli:main"
```

- [ ] **Step 2: Write the failing test** — `packages/core/council/tests/test_cli_judge.py`

```python
import json
import os

import httpx
import pytest
import respx
from anvil_council.cli import _run_judge

DS_URL = "https://api.deepseek.com/v1/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _resp(score: float, model: str):
    obj = {
        "reason": "r",
        "per_criterion": {k: score for k in ("correctness", "evidence", "completeness", "relevance")},
        "overall": score,
    }
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(obj)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k2")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@respx.mock
async def test_run_judge_prints_verdict(tmp_path, capsys):
    respx.post(DS_URL).mock(return_value=_resp(1.0, "deepseek-chat"))
    respx.post(QWEN_URL).mock(return_value=_resp(0.0, "qwen-plus"))
    ds = tmp_path / "cases.jsonl"
    ds.write_text(
        json.dumps({"question": "q", "reference": "r", "answer": "a"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    await _run_judge(str(ds), ["deepseek-chat", "qwen-plus"])
    out = capsys.readouterr().out
    # disagreement expected: correctness spread 1.0 → flagged
    assert "correctness" in out
    assert "分歧" in out or "disagree" in out.lower()
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_cli_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'anvil_council.cli'`

- [ ] **Step 4: Implement `packages/core/council/src/anvil_council/cli.py`**

```python
"""CLI: anvil-council judge / calibrate.

judge: run the jury over a JSONL of {question, reference, answer} cases and print
each verdict (overall, per-criterion, confidence, disagreements)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.seats import score_case

DEFAULT_MODELS = ["deepseek-chat", "qwen-plus"]


def _load_cases(path: str) -> list[dict]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


async def _run_judge(dataset: str, models: list[str]) -> None:
    cases = _load_cases(dataset)
    for i, case in enumerate(cases, 1):
        scores = await score_case(case, models, DEFAULT_RUBRIC)
        v = aggregate(scores, DEFAULT_RUBRIC)
        print(f"[{i}] {case.get('question', '')[:40]}")
        print(f"    总分={v.overall:.2f}  置信度={v.confidence:.2f}")
        print(f"    分项={ {k: round(x, 2) for k, x in v.per_criterion.items()} }")
        if v.disagreements:
            print(f"    ⚠ 分歧维度: {', '.join(v.disagreements)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anvil-council", description="Multi-model evaluation jury")
    sub = parser.add_subparsers(dest="command")
    j = sub.add_parser("judge", help="Run the jury over a JSONL of {question,reference,answer}")
    j.add_argument("--dataset", required=True, help="Path to cases JSONL")
    j.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated juror models (default: deepseek-chat,qwen-plus)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "judge":
        asyncio.run(_run_judge(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()]))
    else:
        parser.print_help()
        sys.exit(2)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv sync --all-packages && uv run pytest packages/core/council/tests/test_cli_judge.py -q`
Expected: PASS (1 test). Also `uv run pytest packages/core/council -q` all green; `uv run ruff check packages/core/council` clean. Confirm `uv run anvil-council --help` lists the `judge` subcommand.

- [ ] **Step 6: Commit**

```bash
git add packages/core/council/pyproject.toml packages/core/council/src/anvil_council/cli.py packages/core/council/tests/test_cli_judge.py uv.lock
git commit -m "feat(council): anvil-council judge CLI (jury verdict over a dataset)"
```

---

## CJ-M2: Multi-juror agreement, calibration, headline experiment

### Task 6: `agreement.py` — hand-rolled Fleiss' κ

**Files:**
- Create: `packages/core/council/src/anvil_council/agreement.py`
- Create: `packages/core/council/tests/test_agreement.py`

- [ ] **Step 1: Write the failing test (hand-computed κ)** — `packages/core/council/tests/test_agreement.py`

```python
import pytest
from anvil_council.agreement import fleiss_kappa, jurors_fleiss


def test_fleiss_kappa_hand_computed():
    # 2 items, 3 raters, 2 categories.
    # item1 = [3,0] (all raters cat0); item2 = [1,2].
    # p = (0.667, 0.333); P_e = 0.5556
    # P_i = [1.0, 0.333]; P_bar = 0.6667
    # kappa = (0.6667-0.5556)/(1-0.5556) = 0.25
    k = fleiss_kappa([[3, 0], [1, 2]])
    assert abs(k - 0.25) < 0.01


def test_fleiss_perfect_agreement():
    assert fleiss_kappa([[3, 0], [0, 3]]) == 1.0


def test_fleiss_rejects_ragged_rows():
    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [1, 1, 1]])


def test_jurors_fleiss_builds_matrix():
    # 2 items; each item has labels from 3 jurors over 2 categories (0/1)
    # item1 jurors all 0 → [3,0]; item2 jurors [0,1,1] → [1,2]  → same as hand-computed 0.25
    k = jurors_fleiss([[0, 0, 0], [0, 1, 1]], num_categories=2)
    assert abs(k - 0.25) < 0.01
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_agreement.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'anvil_council.agreement'`

- [ ] **Step 3: Implement `packages/core/council/src/anvil_council/agreement.py`**

```python
"""Inter-juror agreement: hand-rolled Fleiss' κ (≥2 raters, no sklearn).

Fleiss' κ generalizes Cohen's κ to more than two raters. Input is a matrix where
row i holds, for item i, the count of raters who chose each category."""

from __future__ import annotations


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """matrix[i][j] = number of raters who assigned item i to category j.
    Every row must sum to the same n (number of raters)."""
    if not matrix:
        raise ValueError("empty matrix")
    n = sum(matrix[0])
    if n < 2:
        raise ValueError("need at least 2 raters")
    if any(sum(row) != n for row in matrix):
        raise ValueError("all rows must sum to the same number of raters")
    big_n = len(matrix)
    k = len(matrix[0])
    p_j = [sum(matrix[i][j] for i in range(big_n)) / (big_n * n) for j in range(k)]
    p_e = sum(pj * pj for pj in p_j)
    p_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in matrix]
    p_bar = sum(p_i) / big_n
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def jurors_fleiss(item_labels: list[list[int]], num_categories: int = 3) -> float:
    """item_labels[i] = the category label each juror gave item i. Builds the count
    matrix and returns Fleiss' κ. Labels must be ints in [0, num_categories)."""
    matrix: list[list[int]] = []
    for labels in item_labels:
        row = [0] * num_categories
        for lab in labels:
            row[lab] += 1
        matrix.append(row)
    return fleiss_kappa(matrix)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_agreement.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/council/src/anvil_council/agreement.py packages/core/council/tests/test_agreement.py
git commit -m "feat(council): hand-rolled Fleiss' kappa for inter-juror agreement"
```

---

### Task 7: `compare_jury` — jury vs human vs best-single

**Files:**
- Modify: `packages/core/council/src/anvil_council/agreement.py`
- Create: `packages/core/council/tests/test_compare.py`

- [ ] **Step 1: Write the failing test** — `packages/core/council/tests/test_compare.py`

```python
from anvil_council.agreement import CompareReport, compare_jury


def test_compare_jury_perfect_alignment():
    # 4 cases; per-juror overall scores + jury overall + human scores
    juror_overalls = {
        "deepseek-chat": [1.0, 0.0, 0.5, 1.0],
        "qwen-plus": [1.0, 0.0, 0.5, 1.0],
    }
    jury_overalls = [1.0, 0.0, 0.5, 1.0]
    human = [1.0, 0.0, 0.5, 1.0]
    rep = compare_jury(juror_overalls, jury_overalls, human)
    assert isinstance(rep, CompareReport)
    assert rep.jury_vs_human == 1.0
    assert rep.single_vs_human["deepseek-chat"] == 1.0
    assert rep.best_single_kappa == 1.0
    assert rep.n == 4
    md = rep.to_markdown()
    assert "jury" in md.lower()


def test_compare_jury_flags_when_jury_not_better():
    # jury aligns worse than the best single juror
    juror_overalls = {
        "good": [1.0, 0.0, 1.0, 0.0],   # matches human exactly
        "bad": [0.0, 1.0, 0.0, 1.0],    # inverted
    }
    jury_overalls = [0.5, 0.5, 0.5, 0.5]  # median of the two → all mid, poor
    human = [1.0, 0.0, 1.0, 0.0]
    rep = compare_jury(juror_overalls, jury_overalls, human)
    assert rep.best_single_kappa >= rep.jury_vs_human  # honest: jury did NOT beat best single
    assert rep.jury_beats_best_single is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_compare.py -q`
Expected: FAIL — `ImportError: cannot import name 'CompareReport'`

- [ ] **Step 3: Append to `packages/core/council/src/anvil_council/agreement.py`** (add the import at the top with the others)

Add at the top of the file:

```python
from dataclasses import dataclass

from anvil_eval.calibration import cohen_kappa, quantize
```

Append at the end:

```python
@dataclass
class CompareReport:
    n: int
    jury_vs_human: float
    single_vs_human: dict[str, float]
    best_single_kappa: float
    jury_beats_best_single: bool

    def to_markdown(self) -> str:
        lines = [
            "## Jury Calibration",
            "",
            f"- n = {self.n}",
            f"- jury vs human κ = {self.jury_vs_human:.3f}",
        ]
        for model, k in self.single_vs_human.items():
            lines.append(f"- {model} vs human κ = {k:.3f}")
        verdict = "YES" if self.jury_beats_best_single else "NO"
        lines.append(f"- best single κ = {self.best_single_kappa:.3f} — jury beats best single: {verdict}")
        return "\n".join(lines) + "\n"


def compare_jury(
    juror_overalls: dict[str, list[float]],
    jury_overalls: list[float],
    human_scores: list[float],
) -> CompareReport:
    """Quantize every overall score to ordinal buckets, then compute Cohen's κ vs human
    for the jury and for each single juror. Reports whether the jury beats the best single."""
    human_labels = [quantize(s) for s in human_scores]
    jury_labels = [quantize(s) for s in jury_overalls]
    jury_k = cohen_kappa(jury_labels, human_labels)
    single = {
        model: cohen_kappa([quantize(s) for s in scores], human_labels)
        for model, scores in juror_overalls.items()
    }
    best_single = max(single.values()) if single else 0.0
    return CompareReport(
        n=len(human_scores),
        jury_vs_human=jury_k,
        single_vs_human=single,
        best_single_kappa=best_single,
        jury_beats_best_single=jury_k > best_single,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_compare.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/council/src/anvil_council/agreement.py packages/core/council/tests/test_compare.py
git commit -m "feat(council): jury-vs-human-vs-best-single comparison report"
```

---

### Task 8: Expand calibration golden 14 → ~30

**Files:**
- Modify: `packages/core/eval/golden/calibration.jsonl`
- Create: `packages/core/council/tests/test_calibration_golden.py`

- [ ] **Step 1: Write the failing test** — `packages/core/council/tests/test_calibration_golden.py`

```python
from pathlib import Path

from anvil_eval.calibration import load_calibration

CALIB = (
    Path(__file__).resolve().parents[3]
    / "core"
    / "eval"
    / "golden"
    / "calibration.jsonl"
)


def test_calibration_set_is_large_enough():
    cases = load_calibration(str(CALIB))
    assert len(cases) >= 30, f"expected >=30 calibration cases, got {len(cases)}"


def test_calibration_scores_span_three_buckets():
    cases = load_calibration(str(CALIB))
    scores = {c.human_score for c in cases}
    # need good (1.0), partial (0.5), and wrong (0.0) examples for a meaningful kappa
    assert 1.0 in scores and 0.5 in scores and 0.0 in scores


def test_calibration_ids_unique():
    cases = load_calibration(str(CALIB))
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_calibration_golden.py -q`
Expected: FAIL on `test_calibration_set_is_large_enough` (currently 14 < 30).

- [ ] **Step 3: Append ~16 new cases (cal-15..cal-30) to `packages/core/eval/golden/calibration.jsonl`**

Keep the existing 14 lines unchanged. Append new lines, all in the fictional 星辉人寿 universe, each `{id, question, reference, answer, human_score}` with `human_score ∈ {0.0, 0.5, 1.0}`. Cover good/partial/wrong, varying the question topics (等待期/犹豫期/重疾/理赔/免责/受益人/保费). Balance roughly across the three score buckets. Example shapes (DERIVE real ones, keep them self-consistent — the answer's correctness vs the reference must match its human_score):

```jsonl
{"id": "cal-15", "question": "免责条款包含哪些?", "reference": "免责包含战争、自杀(2年内)、犯罪、酒驾等情形。", "answer": "战争、2年内自杀、犯罪、酒驾等属于免责。", "human_score": 1.0}
{"id": "cal-16", "question": "免责条款包含哪些?", "reference": "免责包含战争、自杀(2年内)、犯罪、酒驾等情形。", "answer": "什么都赔,没有免责。", "human_score": 0.0}
{"id": "cal-17", "question": "受益人可以变更吗?", "reference": "投保人经被保险人同意可书面申请变更受益人。", "answer": "可以,投保人书面申请即可变更。", "human_score": 0.5}
```

After writing, verify it loads and spans buckets:

Run: `uv run python -c "from anvil_eval.calibration import load_calibration; c=load_calibration('packages/core/eval/golden/calibration.jsonl'); print(len(c), sorted({x.human_score for x in c}))"`
Expected: `30 [0.0, 0.5, 1.0]` (or more than 30). Fix until it loads cleanly with all three buckets present.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_calibration_golden.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/core/eval/golden/calibration.jsonl packages/core/council/tests/test_calibration_golden.py
git commit -m "test(council): expand calibration golden 14->30 for jury calibration"
```

---

### Task 9: `anvil-council calibrate` CLI + headline experiment

**Files:**
- Modify: `packages/core/council/src/anvil_council/cli.py`
- Create: `packages/core/council/tests/test_cli_calibrate.py`

- [ ] **Step 1: Write the failing test** — `packages/core/council/tests/test_cli_calibrate.py`

```python
import os

import pytest
from anvil_council.agreement import compare_jury
from anvil_council.cli import _calibrate as calibrate_entry  # noqa: F401  (import-existence)


def test_compare_jury_used_by_cli_path():
    # The non-live unit check: the comparison the CLI relies on works end to end.
    rep = compare_jury(
        {"deepseek-chat": [1.0, 0.0], "qwen-plus": [1.0, 0.0]},
        [1.0, 0.0],
        [1.0, 0.0],
    )
    assert rep.jury_vs_human == 1.0
    assert rep.n == 2


@pytest.mark.live
async def test_calibrate_live_smoke():
    # Real jury over a couple of calibration cases; needs DEEPSEEK_API_KEY (+ DASHSCOPE for qwen).
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("no key")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_cli_calibrate.py -m "not live" -q`
Expected: FAIL — `ImportError: cannot import name '_calibrate'`

- [ ] **Step 3: Add `calibrate` to `cli.py`** — add the import at the top:

```python
from anvil_council.agreement import compare_jury
from anvil_eval.calibration import load_calibration
```

Add the calibration runner and entry (after `_run_judge`):

```python
async def _run_calibrate(dataset: str, models: list[str]) -> int:
    cases = load_calibration(dataset)
    juror_overalls: dict[str, list[float]] = {m: [] for m in models}
    jury_overalls: list[float] = []
    human_scores: list[float] = []
    for case in cases:
        payload = {"question": case.question, "reference": case.reference, "answer": case.answer}
        scores = await score_case(payload, models, DEFAULT_RUBRIC)
        v = aggregate(scores, DEFAULT_RUBRIC)
        jury_overalls.append(v.overall)
        human_scores.append(case.human_score)
        for s in scores:
            juror_overalls[s.model].append(s.overall)
    report = compare_jury(juror_overalls, jury_overalls, human_scores)
    print(report.to_markdown())
    if not report.jury_beats_best_single:
        print(
            "⚠️  陪审团未跑赢最佳单评委 —— 多模型集成在本集上没有增益,如实记录"
            "(可能是强/弱模型平均拉低,或样本太小)。"
        )
    return 0  # diagnostic, never blocks CI


def _calibrate(args) -> int:
    return asyncio.run(
        _run_calibrate(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()])
    )
```

Register the subparser in `_build_parser` (after the `judge` parser):

```python
    c = sub.add_parser("calibrate", help="Compare jury vs human vs best-single (Cohen's κ)")
    c.add_argument("--dataset", required=True, help="Path to calibration JSONL (with human_score)")
    c.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated juror models")
```

Dispatch it in `main` — change the dispatch block to:

```python
    if args.command == "judge":
        asyncio.run(_run_judge(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()]))
    elif args.command == "calibrate":
        sys.exit(_calibrate(args))
    else:
        parser.print_help()
        sys.exit(2)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/core/council/tests/test_cli_calibrate.py -m "not live" -q`
Expected: PASS (1 test; live deselected). Also `uv run pytest packages/core/council -m "not live" -q` all green; `uv run ruff check packages/core/council` clean.

- [ ] **Step 5: Run the headline experiment live (records real numbers)**

If `.env` has `DEEPSEEK_API_KEY` (and ideally `DASHSCOPE_API_KEY`):

Run: `uv run anvil-council calibrate --dataset packages/core/eval/golden/calibration.jsonl`
Capture the EXACT κ table it prints (jury vs human, each juror vs human, best-single, beats-best-single). If `DASHSCOPE_API_KEY` is absent, run DeepSeek-only: `--models deepseek-chat` and record that the second juror (百炼 qwen-plus) is pending the key — do NOT fabricate the qwen numbers. These real numbers go into the CJ-M3 README. (This is the milestone's honest experiment — the jury may or may not beat the single juror; record whatever is true.)

- [ ] **Step 6: Commit**

```bash
git add packages/core/council/src/anvil_council/cli.py packages/core/council/tests/test_cli_calibrate.py
git commit -m "feat(council): anvil-council calibrate — jury vs human vs best-single experiment"
```

---

## CJ-M3: Dogfood on anvil's own answers + docs + PR

### Task 10: Dogfood — jury over anvil KB answers

**Files:**
- Create: `packages/core/council/golden/kb_answers.jsonl`
- Create: `packages/core/council/tests/test_dogfood.py`

- [ ] **Step 1: Create a small dogfood dataset** — `packages/core/council/golden/kb_answers.jsonl`

A handful of {question, reference, answer} rows representing anvil KB outputs (fictional 星辉人寿 domain), including at least one deliberately weak answer so the jury has something to flag. ~6 rows:

```jsonl
{"question": "等待期是多少天?", "reference": "本合同等待期为90天。", "answer": "等待期为90天,自合同生效起算。"}
{"question": "犹豫期多长?", "reference": "本合同设有15天犹豫期。", "answer": "犹豫期是15天。"}
{"question": "重疾赔付比例?", "reference": "确诊重大疾病按保险金额的100%给付。", "answer": "赔保额的一半。"}
{"question": "意外在等待期赔吗?", "reference": "意外伤害不受等待期限制。", "answer": "意外不受等待期限制,合同生效后即保障。"}
{"question": "理赔第一步?", "reference": "出险后应及时报案。", "answer": "先报案。"}
{"question": "犹豫期退保扣费吗?", "reference": "犹豫期内退保全额退还,不收费用。", "answer": "犹豫期退保不收任何费用。"}
```

- [ ] **Step 2: Write the failing test** — `packages/core/council/tests/test_dogfood.py`

```python
import json
import os
from pathlib import Path

import httpx
import pytest
import respx
from anvil_council.aggregate import aggregate
from anvil_council.seats import score_case

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DOGFOOD = Path(__file__).resolve().parents[1] / "golden" / "kb_answers.jsonl"


def _resp(score: float):
    obj = {
        "reason": "r",
        "per_criterion": {k: score for k in ("correctness", "evidence", "completeness", "relevance")},
        "overall": score,
    }
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(obj)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


def test_dogfood_dataset_loads():
    rows = [json.loads(x) for x in DOGFOOD.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 6
    for r in rows:
        assert r["question"] and r["reference"] and r["answer"]


@respx.mock
async def test_jury_scores_a_dogfood_case():
    respx.post(DS_URL).mock(return_value=_resp(1.0))
    rows = [json.loads(x) for x in DOGFOOD.read_text(encoding="utf-8").splitlines() if x.strip()]
    scores = await score_case(rows[0], ["deepseek-chat"], )
    v = aggregate(scores)
    assert 0.0 <= v.overall <= 1.0
    assert v.confidence == 1.0  # single juror → no spread
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/core/council/tests/test_dogfood.py -q`
Expected: FAIL — `kb_answers.jsonl` missing or test import error if the file wasn't created. (If Step 1 created it, the failure is the assertion/structure until both pass; ensure the dataset exists and the jury path runs.)

- [ ] **Step 4: Make it pass** — the implementation already exists (seats/aggregate from CJ-M1); this task only adds the dataset + dogfood test. Run:

Run: `uv run pytest packages/core/council/tests/test_dogfood.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the dogfood live (optional, records a real verdict sample)**

If keys present: `uv run anvil-council judge --dataset packages/core/council/golden/kb_answers.jsonl`
Capture a sample verdict (the weak "赔保额的一半" row should score low on correctness / be flagged). Record for the README.

- [ ] **Step 6: Commit**

```bash
git add packages/core/council/golden/kb_answers.jsonl packages/core/council/tests/test_dogfood.py
git commit -m "test(council): dogfood jury over anvil KB answer samples"
```

---

### Task 11: Docs + full regression + PR

**Files:**
- Create: `examples/05-council/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write `examples/05-council/README.md`** using the REAL numbers captured in Task 9 Step 5 / Task 10 Step 5 (replace every `<填...>` with the measured value; if the qwen juror was pending the key, say so honestly):

```markdown
# 示例 05:多模型评测陪审团(Council-Jury)

把质量竖梁从「单 LLM-judge + Cohen's κ」升级到「校准过的多模型评测陪审团」:多个不同
模型的陪审员**独立**按统一 rubric 打分 → 聚合(分项中位数 + 标分歧)→ 用 Cohen's κ /
Fleiss' κ 校准,并回答「陪审团是否真的打得过最好的单个评委」。

## 设计要点(正面修两个已知坑)

- 角色 = **功能性评分维度**(correctness/evidence/completeness/relevance),库内固定,
  **不让用户手写人格 prompt**(persona 脆弱是上一版失败主因)。
- 陪审员**独立、互不可见**——评测要独立性,天然避开谄媚锚定与同质化。
- 陪审员**喂参考/证据接地**,不空对空。

## 命令

- `anvil-council judge --dataset cases.jsonl --models deepseek-chat,qwen-plus` — 跑陪审团,打印裁决
- `anvil-council calibrate --dataset calibration.jsonl` — 陪审团 vs 人工 vs 最佳单评委 κ 对比

## 核心实验(真实数据)

校准集 N=<填> 条。结果:

- jury vs human κ = <填>
- deepseek-chat vs human κ = <填>
- qwen-plus(百炼)vs human κ = <填,或"百炼 key 待补,本轮 DeepSeek 单评委">
- best single κ = <填> —— **陪审团是否跑赢最佳单评委:<填 YES/NO>**

诚实结论:<一句话,照实写——陪审团赢了/没赢/样本太小不足以下结论。对照 KB Contextual
Retrieval 的负向结果口径,不修饰>。

## 复用底座

structured_chat(guard)/ judge 思路(eval)/ Cohen's κ(eval)/ gateway 多 provider /
golden 数据集 —— 几乎全是现成,council 只新增「多 seat 并行 + 聚合 + 分歧 + Fleiss' κ」。
```

- [ ] **Step 2: Update `CLAUDE.md`** — after the `## anvil-eval 校准(packages/core/eval)` section, add:

```markdown
## anvil-council (packages/core/council)

多模型评测陪审团(圈1 普适):多个不同模型独立按 rubric 打分 → 聚合 → 标分歧 → 校准。

- `score_case(case, models)` — 多陪审员并行独立打分(走 structured_chat)
- `aggregate(scores)` — 分项中位数 + 分歧探测 + 置信度
- `fleiss_kappa` / `compare_jury` — 多评委一致性 + 陪审团 vs 人工 vs 最佳单评委
- CLI: `anvil-council judge --dataset … --models deepseek-chat,qwen-plus` / `anvil-council calibrate --dataset …`
- 测试: `uv run pytest packages/core/council -q`(respx mock,无需 key);live 实验需 DEEPSEEK + 百炼 DASHSCOPE key
- 复用 gateway/guard/eval,编排原语供 P3 复用
```

- [ ] **Step 3: Full regression across the workspace**

Run: `uv sync --all-packages && uv run pytest -m "not live" -q`
Expected: all PASS (gateway/obs/eval/guard/council/kb/kb-api). Note the new council tests raise the total.

- [ ] **Step 4: Lint the whole repo**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 5: Red-line self-check on the diff**

Run: `git diff main...feat/council-jury | grep -iE "one-policy|forge|iic|zbcloud|tapd|radium|/home/itachi|chenzhixi@gmail" || echo "CLEAN"`
Expected: `CLEAN` (or only benign matches like the word "forge" inside other words / the red-line rule text — inspect to confirm no real leak).

- [ ] **Step 6: Commit, push, open PR**

```bash
git add examples/05-council/README.md CLAUDE.md
git commit -m "docs(council): example 05 writeup + CLAUDE section (real experiment numbers)"
git push -u origin feat/council-jury
gh pr create --title "Council-Jury: multi-model evaluation jury (P2)" --body "$(cat <<'EOF'
## What

P2, converged from "multi-model chat council" to a **multi-model evaluation jury** (`packages/core/council`) — the place where ensembles genuinely beat single models (jury > single judge) and where it dogfoods anvil's own eval.

- **anvil-council** (new 圈1 package): independent parallel juror scoring (`score_case` via `structured_chat`), median aggregation with disagreement flag (`aggregate`), hand-rolled **Fleiss' κ** inter-juror agreement, and a **jury-vs-human-vs-best-single** comparison (`compare_jury`).
- CLI: `anvil-council judge` / `anvil-council calibrate`.
- Calibration golden expanded 14 → 30; dogfood dataset over anvil KB answers.
- Reuses gateway (multi-provider jurors: DeepSeek + 百炼 Qwen) / guard (`structured_chat`) / eval (Cohen's κ). Orchestration primitives are reused by P3.

## Design (fixes two known failure modes)

Functional rubric criteria instead of brittle hand-written personas; jurors score independently (evaluation wants independence, sidestepping sycophancy/homogenization).

## Experiment

Jury vs human vs best-single Cohen's κ recorded honestly in `examples/05-council/README.md` (jury may or may not beat the best single juror — recorded as measured).

## Tests

Full `pytest -m "not live"` green; `ruff check .` clean. No new migrations.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Wait for CI green**

Run: `gh pr checks --watch`
Expected: all checks pass. If red, read the failing log, fix on the branch, push, re-watch.

---

## Self-Review (completed by plan author)

**Spec coverage** (against `2026-06-08-council-jury-design.md`):
- CJ-M1 核心引擎 (seats/rubric/aggregate/verdict/cli judge) → Tasks 1-5 ✓
- CJ-M2 Fleiss' κ + 校准 + 核心实验 → Tasks 6 (Fleiss) + 7 (compare) + 8 (golden 14→30) + 9 (calibrate CLI + live experiment, activates 百炼/task #22) ✓
- CJ-M3 接入/吃狗粮 + 文档 + PR → Tasks 10 (dogfood) + 11 (docs + regression + PR) ✓
- 复用底座 (structured_chat / cohen_kappa / quantize / gateway multi-provider / golden) ✓
- 不做边界 (no debate/cross-exam, no co-creation, no domain-briefing, no web view) — none introduced ✓
- Architecture: `packages/core/council` glob-included, deps gateway+guard+eval ✓

**Type consistency:** `JurorScore{model, per_criterion, overall, reason}` and `Verdict{overall, per_criterion, confidence, disagreements, jurors}` used identically in Tasks 3/4/5/9/10. `score_case(case, models, rubric=DEFAULT_RUBRIC) -> list[JurorScore]` and `aggregate(scores, rubric=DEFAULT_RUBRIC, disagreement_threshold=0.5) -> Verdict` consistent in Tasks 4/5/9/10. `fleiss_kappa(matrix)` / `jurors_fleiss(item_labels, num_categories)` / `compare_jury(juror_overalls, jury_overalls, human_scores) -> CompareReport` consistent in Tasks 6/7/9. `CompareReport{n, jury_vs_human, single_vs_human, best_single_kappa, jury_beats_best_single}` consistent in Tasks 7/9. Reused `quantize`/`cohen_kappa`/`load_calibration`/`CalibrationCase` match the verified anvil_eval.calibration signatures.

**Placeholder scan:** no TBD/TODO; the only fill-at-runtime values are the experiment's measured κ numbers (Task 9 Step 5, Task 11 Step 1), explicitly flagged as real-measurements-not-fabricated, with an honest fallback when the 百炼 key is absent.

**Hand-computed anchors:** Fleiss' κ = 0.25 (Task 6) and aggregate medians/disagreement/confidence (Task 3) are arithmetic facts; never adjust the test to a buggy implementation.
