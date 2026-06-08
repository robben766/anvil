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
