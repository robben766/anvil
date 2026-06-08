"""Inter-juror agreement: hand-rolled Fleiss' κ (≥2 raters, no sklearn).

Fleiss' κ generalizes Cohen's κ to more than two raters. Input is a matrix where
row i holds, for item i, the count of raters who chose each category."""

from __future__ import annotations

from dataclasses import dataclass

from anvil_eval.calibration import cohen_kappa, quantize


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """matrix[i][j] = number of raters who assigned item i to category j.
    Every row must sum to the same n (number of raters)."""
    if not matrix:
        raise ValueError("empty matrix")
    n = sum(matrix[0])
    if n < 2:
        raise ValueError("need at least 2 raters")
    k = len(matrix[0])
    if any(len(row) != k for row in matrix):
        raise ValueError("all rows must have the same number of categories")
    if any(sum(row) != n for row in matrix):
        raise ValueError("all rows must sum to the same number of raters")
    big_n = len(matrix)
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


@dataclass
class CompareReport:
    n: int
    jury_vs_human: float
    single_vs_human: dict[str, float]
    best_single_kappa: float
    jury_beats_best_single: bool
    inter_juror_fleiss: float  # agreement AMONG jurors (Fleiss' κ); nan if < 2 jurors

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
        lines.append(
            f"- best single κ = {self.best_single_kappa:.3f} — jury beats best single: {verdict}"
        )
        fleiss = "n/a (<2 jurors)" if self.inter_juror_fleiss != self.inter_juror_fleiss else (
            f"{self.inter_juror_fleiss:.3f}"
        )
        lines.append(f"- inter-juror agreement (Fleiss' κ) = {fleiss}")
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
    # inter-juror agreement: per item, the quantized label each juror gave → Fleiss' κ
    models = list(juror_overalls.keys())
    if len(models) >= 2 and human_scores:
        item_labels = [
            [quantize(juror_overalls[m][i]) for m in models] for i in range(len(human_scores))
        ]
        inter_juror_fleiss = jurors_fleiss(item_labels, num_categories=3)
    else:
        inter_juror_fleiss = float("nan")  # undefined with a single juror
    return CompareReport(
        n=len(human_scores),
        jury_vs_human=jury_k,
        single_vs_human=single,
        best_single_kappa=best_single,
        jury_beats_best_single=jury_k > best_single,
        inter_juror_fleiss=inter_juror_fleiss,
    )
