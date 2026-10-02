"""Small shared helper functions used across all five mini-projects."""
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error between two equal-length arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error between two equal-length arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Days with zero actual value are skipped."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers, starting 1, 1, 2, 3, 5, ..."""
    if n <= 0:
        return []
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib[:n]
