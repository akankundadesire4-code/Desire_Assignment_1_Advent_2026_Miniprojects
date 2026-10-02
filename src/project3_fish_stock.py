"""Lake Victoria fish stock, price and risk model (Mini-Project 3)."""
from __future__ import annotations

import numpy as np


class FishStock:
    """Discrete logistic growth model with weekly harvesting."""

    def __init__(self, r: float, k: float, n0: float, harvest_rate: float) -> None:
        if k <= 0:
            raise ValueError("carrying capacity k must be positive")
        if n0 < 0:
            raise ValueError("initial stock n0 cannot be negative")
        self.r = r
        self.k = k
        self.n0 = n0
        self.harvest_rate = harvest_rate

    def simulate(self, weeks: int) -> np.ndarray:
        """Simulate weekly stock levels, starting at week 0. Stock never goes below zero."""
        stock = np.zeros(weeks + 1)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            growth = self.r * n * (1 - n / self.k)
            harvest = self.harvest_rate * n
            stock[t + 1] = max(n + growth - harvest, 0.0)
        return stock

    def harvested_amounts(self, stock: np.ndarray) -> np.ndarray:
        """Tonnes harvested each week, computed from a stock trajectory."""
        return self.harvest_rate * stock[:-1]

    def maximum_sustainable_yield(self) -> float:
        """Theoretical maximum sustainable yield MSY = r*K/4."""
        return self.r * self.k / 4


class PriceModel:
    """Weekly UGX/kg price as a bounded, seeded random walk."""

    def __init__(self, start_price: float, low: float, high: float, seed: int = 0) -> None:
        self.start_price = start_price
        self.low = low
        self.high = high
        self.seed = seed

    def simulate(self, weeks: int) -> np.ndarray:
        """Simulate one weekly price path, clipped to [low, high]."""
        rng = np.random.default_rng(self.seed)
        steps = rng.normal(0, 300, weeks)
        prices = np.zeros(weeks + 1)
        prices[0] = self.start_price
        for t in range(weeks):
            prices[t + 1] = np.clip(prices[t] + steps[t], self.low, self.high)
        return prices


class RiskAssessor:
    """Classifies revenue risk from its coefficient of variation and runs Monte Carlo VaR."""

    def __init__(self, cv_low: float = 0.15, cv_high: float = 0.35) -> None:
        self.cv_low = cv_low
        self.cv_high = cv_high

    def coefficient_of_variation(self, revenue: np.ndarray) -> float:
        """Standard deviation over mean; unitless, unlike raw variance."""
        revenue = np.asarray(revenue, dtype=float)
        if revenue.size == 0:
            raise ValueError("revenue cannot be empty")
        mean = np.mean(revenue)
        if mean == 0:
            raise ValueError("mean revenue is zero; coefficient of variation is undefined")
        return float(np.std(revenue, ddof=1) / mean)

    def classify(self, revenue: np.ndarray) -> str:
        """Low/Medium/High risk label based on the coefficient of variation."""
        cv = self.coefficient_of_variation(revenue)
        if cv < self.cv_low:
            return "Low risk"
        if cv < self.cv_high:
            return "Medium risk"
        return "High risk"

    def monte_carlo_annual_revenue(
        self, fish_stock: FishStock, price_model: PriceModel, weeks: int, n_paths: int, seed: int = 1
    ) -> np.ndarray:
        """Simulate n_paths independent annual revenues using different price seeds."""
        stock = fish_stock.simulate(weeks)
        harvested_kg = fish_stock.harvested_amounts(stock) * 1000  # tonnes -> kg
        rng = np.random.default_rng(seed)
        revenues = np.zeros(n_paths)
        for i in range(n_paths):
            path_seed = int(rng.integers(0, 1_000_000))
            path_prices = PriceModel(price_model.start_price, price_model.low, price_model.high, seed=path_seed)
            prices = path_prices.simulate(weeks)[:-1]
            revenues[i] = float(np.sum(harvested_kg * prices))
        return revenues

    def value_at_risk(self, revenues: np.ndarray, confidence: float = 0.95) -> float:
        """5% Value-at-Risk: the revenue level below which only 5% of outcomes fall."""
        return float(np.quantile(revenues, 1 - confidence))
