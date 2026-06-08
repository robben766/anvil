"""Offline experiment: precision/recall/confusion of detect_injection over the
adversarial corpus. Run: uv run python -m anvil_guard.experiments.injection_eval

Prints a confusion matrix and the precision/recall numbers recorded in the README.
Pure rule-based path (no LLM) — deterministic and free.
"""

from __future__ import annotations

import json
from pathlib import Path

from anvil_guard.injection import detect_injection

CORPUS = Path(__file__).resolve().parents[1].parent.parent / "golden" / "injection_cases.jsonl"


def _load() -> list[dict]:
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    rows = _load()
    tp = fp = tn = fn = 0
    misses: list[str] = []
    for r in rows:
        predicted = detect_injection(r["text"]).is_injection
        actual = r["label"]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            misses.append(f"FALSE POSITIVE {r['id']}: {r['text']}")
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1
            misses.append(f"FALSE NEGATIVE {r['id']}: {r['text']}")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"cases={len(rows)}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    for m in misses:
        print(m)


if __name__ == "__main__":
    main()
