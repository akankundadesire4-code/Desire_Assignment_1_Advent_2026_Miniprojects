import numpy as np
import pytest

from project3_fish_stock import FishStock, PriceModel, RiskAssessor


def test_stock_grows_when_harvest_rate_is_low():
    stock_model = FishStock(r=0.4, k=10_000, n0=4_000, harvest_rate=0.05)
    stock = stock_model.simulate(52)
    assert stock[-1] > stock[0]


def test_stock_never_goes_negative_even_with_high_harvest():
    stock_model = FishStock(r=0.4, k=10_000, n0=4_000, harvest_rate=0.9)
    stock = stock_model.simulate(52)
    assert np.all(stock >= 0)


def test_msy_formula():
    stock_model = FishStock(r=0.4, k=10_000, n0=4_000, harvest_rate=0.1)
    assert stock_model.maximum_sustainable_yield() == pytest.approx(1000.0)


def test_price_model_stays_within_bounds():
    price_model = PriceModel(start_price=12_000, low=9_000, high=16_000, seed=1)
    prices = price_model.simulate(52)
    assert prices.min() >= 9_000
    assert prices.max() <= 16_000


def test_risk_assessor_rejects_empty_revenue_edge_case():
    assessor = RiskAssessor()
    with pytest.raises(ValueError):
        assessor.coefficient_of_variation(np.array([]))
