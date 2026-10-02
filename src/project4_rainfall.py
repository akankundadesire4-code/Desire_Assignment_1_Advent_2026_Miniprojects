"""Rainfall pattern and crop suitability analyser (Mini-Project 4)."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


class Region:
    """Stores one region's 12 monthly rainfall totals in millimetres."""

    def __init__(self, name: str, monthly_mm: list[float]) -> None:
        values = np.asarray(monthly_mm, dtype=float)
        if len(values) != 12:
            raise ValueError("monthly_mm must have exactly 12 values")
        if np.any(values < 0):
            raise ValueError("rainfall cannot be negative")
        self.name = name
        self.monthly_mm = values

    def annual_total(self) -> float:
        return float(np.sum(self.monthly_mm))

    def annual_mean(self) -> float:
        return float(np.mean(self.monthly_mm))

    def wettest_month(self) -> int:
        """Index (0=Jan) of the wettest month."""
        return int(np.argmax(self.monthly_mm))

    def driest_month(self) -> int:
        """Index (0=Jan) of the driest month."""
        return int(np.argmin(self.monthly_mm))

    def coefficient_of_variation(self) -> float:
        return float(np.std(self.monthly_mm, ddof=1) / np.mean(self.monthly_mm))

    def is_bimodal(self) -> bool:
        """True if two separate rainy-season peaks are detected."""
        peaks, _ = find_peaks(self.monthly_mm, prominence=15, distance=2)
        return len(peaks) >= 2


class CropRule:
    """Defines one crop's suitable monthly rainfall range (mm), with a cited source."""

    def __init__(self, crop: str, min_mm: float, max_mm: float, source: str) -> None:
        self.crop = crop
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.source = source

    def classify_month(self, rainfall_mm: float) -> str:
        """Label one month's rainfall for this crop."""
        if rainfall_mm < self.min_mm:
            return "Drought risk"
        if rainfall_mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors: dot product over product of norms."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
