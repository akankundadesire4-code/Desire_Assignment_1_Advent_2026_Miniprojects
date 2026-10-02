"""Solar micro-grid dispatch planner: solves x (solar) and y (battery) per day (Mini-Project 2)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls

SOLAR_COST_PER_KWH = 150  # UGX, illustrative
BATTERY_COST_PER_KWH = 450  # UGX, illustrative


class MicroGrid:
    """Solves the daily 2x2 solar/battery dispatch system for one health centre."""

    def __init__(self) -> None:
        # 3x + 2y = D1 (daytime load), 4x + y = D2 (critical-equipment load)
        self.coefficients = np.array([[3.0, 2.0], [4.0, 1.0]])

    def determinant(self) -> float:
        """Determinant of the coefficient matrix; near zero means an ill-posed system."""
        return float(np.linalg.det(self.coefficients))

    def condition_number(self) -> float:
        """Condition number: how much solution error grows from small input error."""
        return float(np.linalg.cond(self.coefficients))

    def solve_day(self, d1: float, d2: float) -> tuple[float, float]:
        """Solve for (solar_kwh, battery_kwh) given one day's two demand values."""
        x, y = solve(self.coefficients, np.array([d1, d2]))
        return float(x), float(y)

    def solve_many_days_loop(self, demands: np.ndarray) -> np.ndarray:
        """Solve day by day in a plain Python loop. demands has shape (n_days, 2)."""
        results = np.zeros((len(demands), 2))
        for i, (d1, d2) in enumerate(demands):
            results[i] = solve(self.coefficients, np.array([d1, d2]))
        return results

    def solve_many_days_vectorised(self, demands: np.ndarray) -> np.ndarray:
        """Solve all days at once with a single 2xN right-hand side."""
        rhs = demands.T  # shape (2, n_days)
        return solve(self.coefficients, rhs).T

    def fix_infeasible_day(self, d1: float, d2: float) -> tuple[float, float]:
        """Use non-negative least squares when the direct solve gives negative x or y."""
        x, y = nnls(self.coefficients, np.array([d1, d2]))[0]
        return float(x), float(y)

    def daily_cost(self, solar_kwh: float, battery_kwh: float) -> float:
        """Daily energy cost in UGX."""
        return solar_kwh * SOLAR_COST_PER_KWH + battery_kwh * BATTERY_COST_PER_KWH


def read_positive_number(prompt: str) -> float:
    """Ask the user for a positive number, re-prompting until the input is valid."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if value <= 0:
            print("Please enter a positive number.")
            continue
        return value


def generate_demand_csv(path: str, n_days: int = 30, seed: int = 42) -> None:
    """Create n_days of synthetic demand with a weekly pattern and noise, saved to CSV."""
    rng = np.random.default_rng(seed)
    day_of_week = np.arange(n_days) % 7
    weekend_boost = np.where(day_of_week >= 5, 1.15, 1.0)
    d1 = 100 * weekend_boost + rng.normal(0, 5, n_days)
    d2 = 80 * weekend_boost + rng.normal(0, 4, n_days)
    with open(path, "w", encoding="utf-8") as f:
        f.write("day,d1,d2\n")
        for day in range(n_days):
            f.write(f"{day + 1},{d1[day]:.2f},{d2[day]:.2f}\n")
