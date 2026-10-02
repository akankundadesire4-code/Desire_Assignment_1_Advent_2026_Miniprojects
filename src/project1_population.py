"""District population storage and simple forecasting models (Mini-Project 1)."""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod

import numpy as np

from utils import fibonacci


class DistrictPopulation:
    """Stores yearly population estimates (in thousands) for one district."""

    def __init__(self, name: str, years: list[int], populations: list[float]) -> None:
        years_arr = np.asarray(years, dtype=int)
        pop_arr = np.asarray(populations, dtype=float)
        if len(years_arr) != len(pop_arr):
            raise ValueError("years and populations must have the same length")
        if len(pop_arr) == 0:
            raise ValueError("populations cannot be empty")
        if np.any(pop_arr < 0):
            raise ValueError("populations cannot be negative")
        self.name = name
        self.years = years_arr
        self.populations = pop_arr

    def __repr__(self) -> str:
        return f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-{self.years[-1]}, n={len(self)})"

    def __len__(self) -> int:
        return len(self.populations)

    def describe_statistics(self) -> dict[str, float]:
        """Mean, median, variance and standard deviation, computed two ways."""
        pop = self.populations
        return {
            "statistics_mean": statistics.mean(pop),
            "statistics_median": statistics.median(pop),
            "statistics_variance": statistics.variance(pop),  # sample variance, ddof=1
            "statistics_stdev": statistics.stdev(pop),
            "numpy_mean": float(np.mean(pop)),
            "numpy_var_ddof0": float(np.var(pop)),  # population variance, ddof=0
            "numpy_var_ddof1": float(np.var(pop, ddof=1)),  # matches statistics.variance
            "numpy_std_ddof1": float(np.std(pop, ddof=1)),
        }

    def year_on_year_growth(self) -> np.ndarray:
        """Percentage growth rate from one year to the next."""
        return (self.populations[1:] - self.populations[:-1]) / self.populations[:-1] * 100

    def cagr(self) -> float:
        """Compound Annual Growth Rate over the full stored period."""
        n_periods = len(self) - 1
        start, end = self.populations[0], self.populations[-1]
        return (end / start) ** (1 / n_periods) - 1


class Forecaster(ABC):
    """Base class for all population forecasting models."""

    @abstractmethod
    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        """Learn model parameters from historical data."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict the next `horizon` years after the training data."""


class LinearTrendForecaster(Forecaster):
    """Straight-line trend fitted with np.polyfit (degree 1)."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        self.last_year = years[-1]
        self.slope, self.intercept = np.polyfit(years, populations, 1)

    def predict(self, horizon: int) -> np.ndarray:
        future_years = self.last_year + np.arange(1, horizon + 1)
        return self.slope * future_years + self.intercept


class CAGRForecaster(Forecaster):
    """Exponential growth using the Compound Annual Growth Rate."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        n_periods = len(populations) - 1
        self.last_value = populations[-1]
        self.rate = (populations[-1] / populations[0]) ** (1 / n_periods) - 1

    def predict(self, horizon: int) -> np.ndarray:
        return self.last_value * (1 + self.rate) ** np.arange(1, horizon + 1)


class FibonacciRatioForecaster(Forecaster):
    """Scales the last known value by successive Fibonacci ratios F(n+1)/F(n)."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        self.last_value = populations[-1]

    def predict(self, horizon: int) -> np.ndarray:
        fib = fibonacci(horizon + 2)
        ratios = np.array([fib[i + 1] / fib[i] for i in range(1, horizon + 1)])
        return self.last_value * ratios
