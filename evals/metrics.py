"""Dependency-free metrics shared by manifest verifiers."""

import math
from collections.abc import Callable


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(gold: list[float], predictions: list[float]) -> float:
    """Spearman correlation with average ranks for ties."""
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("Spearman inputs must have the same non-trivial length")
    left = _average_ranks(gold)
    right = _average_ranks(predictions)
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Spearman input is constant")
    return numerator / (left_norm * right_norm)


def pearson(gold: list[float], predictions: list[float]) -> float:
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("Pearson inputs must have the same non-trivial length")
    gold_mean = math.fsum(gold) / len(gold)
    prediction_mean = math.fsum(predictions) / len(predictions)
    numerator = math.fsum(
        (x - gold_mean) * (y - prediction_mean)
        for x, y in zip(gold, predictions)
    )
    gold_norm = math.sqrt(math.fsum((value - gold_mean) ** 2 for value in gold))
    prediction_norm = math.sqrt(
        math.fsum((value - prediction_mean) ** 2 for value in predictions)
    )
    if gold_norm == 0.0 or prediction_norm == 0.0:
        raise ValueError("Pearson input is constant")
    return numerator / (gold_norm * prediction_norm)


def accuracy(gold: list[float], predictions: list[float]) -> float:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("accuracy inputs must have the same non-zero length")
    correct = sum(
        expected == actual for expected, actual in zip(gold, predictions)
    )
    return correct / len(gold)


def binary_auroc(positive: list[float], negative: list[float]) -> float:
    """AUROC from positive and negative score lists; larger means positive."""
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    merged = sorted(
        [(value, 1) for value in positive] + [(value, 0) for value in negative]
    )
    ranks = _average_ranks([value for value, _label in merged])
    positive_rank_sum = math.fsum(
        rank for rank, (_value, label) in zip(ranks, merged) if label == 1
    )
    n_pos, n_neg = len(positive), len(negative)
    return (
        positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)


def auroc(gold: list[float], predictions: list[float]) -> float:
    """Binary AUROC where larger prediction scores indicate the positive class."""
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("AUROC inputs must have the same non-trivial length")
    if any(label not in (0.0, 1.0) for label in gold):
        raise ValueError("AUROC gold labels must be binary")
    positive = [score for label, score in zip(gold, predictions) if label == 1.0]
    negative = [score for label, score in zip(gold, predictions) if label == 0.0]
    return binary_auroc(positive, negative)


MetricFn = Callable[[list[float], list[float]], float]
METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "top1_accuracy": accuracy,
    "robust_accuracy": accuracy,
    "auroc": auroc,
    "pearson": pearson,
    "spearman": spearman,
}
