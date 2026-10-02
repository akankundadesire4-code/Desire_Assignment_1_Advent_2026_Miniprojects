"""Taxi route revenue, pricing and fleet planner (Mini-Project 5)."""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod

import numpy as np


class Route:
    """Stores daily passenger counts and the fixed fare for one taxi route."""

    def __init__(self, name: str, passengers: list[float], fare: float) -> None:
        values = np.asarray(passengers, dtype=float)
        if len(values) == 0:
            raise ValueError("passengers cannot be empty")
        if np.any(values < 0):
            raise ValueError("passenger counts cannot be negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name = name
        self.passengers = values
        self.fare = fare

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(np.sum(self.daily_revenue()))

    def describe(self) -> dict[str, float]:
        revenue = self.daily_revenue()
        return {
            "mean": statistics.mean(revenue),
            "variance": statistics.variance(revenue),
            "stdev": statistics.stdev(revenue),
        }


class RouteForecaster(ABC):
    """Base class for the three simple passenger-demand forecasters."""

    @abstractmethod
    def fit(self, series: np.ndarray) -> None:
        """Learn from the historical daily passenger series."""

    @abstractmethod
    def predict_next(self) -> float:
        """Predict the next day's passenger count."""


class MovingAverageForecaster(RouteForecaster):
    """Average of the last `window` days."""

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def fit(self, series: np.ndarray) -> None:
        self.series = series

    def predict_next(self) -> float:
        return float(np.mean(self.series[-self.window :]))


class ExponentialSmoothingForecaster(RouteForecaster):
    """Simple exponential smoothing with a fixed weight alpha (0 < alpha <= 1)."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def fit(self, series: np.ndarray) -> None:
        level = series[0]
        for value in series[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        self.level = level

    def predict_next(self) -> float:
        return float(self.level)


class LinearTrendForecaster(RouteForecaster):
    """Straight-line trend fitted with np.polyfit (degree 1)."""

    def fit(self, series: np.ndarray) -> None:
        days = np.arange(len(series))
        self.slope, self.intercept = np.polyfit(days, series, 1)
        self.next_day = len(series)

    def predict_next(self) -> float:
        return float(self.slope * self.next_day + self.intercept)


def vehicles_needed(forecast_passengers: float, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15) -> int:
    """Number of vehicles needed for one day's forecast passengers, with a safety buffer."""
    capacity_per_vehicle = trips_per_day * seats
    return math.ceil(forecast_passengers * (1 + buffer) / capacity_per_vehicle)
