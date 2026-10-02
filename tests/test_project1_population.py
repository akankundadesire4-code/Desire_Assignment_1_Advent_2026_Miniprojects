import numpy as np
import pytest

from project1_population import CAGRForecaster, DistrictPopulation, LinearTrendForecaster


def test_district_population_basic_stats():
    district = DistrictPopulation("Test", [2020, 2021, 2022], [100, 110, 121])
    assert len(district) == 3
    assert district.cagr() == pytest.approx(0.1, rel=1e-6)


def test_district_population_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DistrictPopulation("Test", [2020, 2021], [100, 110, 120])


def test_district_population_rejects_negative_values():
    with pytest.raises(ValueError):
        DistrictPopulation("Test", [2020, 2021], [100, -5])


def test_linear_trend_forecaster_predicts_correct_length():
    model = LinearTrendForecaster()
    model.fit(np.array([2020, 2021, 2022]), np.array([100.0, 110.0, 120.0]))
    forecast = model.predict(5)
    assert len(forecast) == 5


def test_cagr_forecaster_grows_by_fixed_rate():
    model = CAGRForecaster()
    model.fit(np.array([2020, 2021]), np.array([100.0, 110.0]))
    forecast = model.predict(1)
    assert forecast[0] == pytest.approx(121.0, rel=1e-6)
