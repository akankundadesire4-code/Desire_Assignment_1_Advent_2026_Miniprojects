import numpy as np
import pytest

from project2_microgrid import MicroGrid


def test_solve_day_matches_known_solution():
    grid = MicroGrid()
    # 3x + 2y = 8, 4x + y = 9 -> x=2, y=1
    x, y = grid.solve_day(8, 9)
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(1.0)


def test_determinant_is_nonzero_for_this_system():
    grid = MicroGrid()
    assert grid.determinant() != 0


def test_vectorised_matches_loop_solution():
    grid = MicroGrid()
    demands = np.array([[8, 9], [16, 18]])
    loop_result = grid.solve_many_days_loop(demands)
    vector_result = grid.solve_many_days_vectorised(demands)
    assert np.allclose(loop_result, vector_result)


def test_fix_infeasible_day_returns_nonnegative_values():
    grid = MicroGrid()
    x, y = grid.fix_infeasible_day(1, 100)  # direct solve would give negative y here
    assert x >= 0
    assert y >= 0


def test_daily_cost_is_positive_for_positive_usage():
    grid = MicroGrid()
    cost = grid.daily_cost(10, 5)
    assert cost > 0
