import numpy as np
import pytest

from project5_taxi import MovingAverageForecaster, Route, vehicles_needed


def test_route_revenue_calculation():
    route = Route("Test", [10, 20, 30], fare=1000)
    assert route.total_revenue() == 60_000


def test_route_rejects_negative_passengers():
    with pytest.raises(ValueError):
        Route("Test", [10, -5], fare=1000)


def test_route_rejects_nonpositive_fare():
    with pytest.raises(ValueError):
        Route("Test", [10, 20], fare=0)


def test_moving_average_forecaster_predicts_recent_mean():
    model = MovingAverageForecaster(window=2)
    model.fit(np.array([10.0, 20.0, 30.0]))
    assert model.predict_next() == pytest.approx(25.0)


def test_vehicles_needed_rounds_up_with_buffer():
    assert vehicles_needed(90, trips_per_day=8, seats=14, buffer=0.15) == 1
    assert vehicles_needed(200, trips_per_day=8, seats=14, buffer=0.15) == 3
