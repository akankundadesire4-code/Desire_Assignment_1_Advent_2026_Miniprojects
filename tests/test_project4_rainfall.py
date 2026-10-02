import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from project4_rainfall import CropRule, Region, cosine_similarity


def test_region_rejects_wrong_length():
    with pytest.raises(ValueError):
        Region("Test", [10] * 11)


def test_region_rejects_negative_rainfall():
    with pytest.raises(ValueError):
        Region("Test", [-5] + [10] * 11)


def test_region_wettest_and_driest_month():
    region = Region("Test", [10, 20, 30, 5, 10, 10, 10, 10, 10, 10, 10, 10])
    assert region.wettest_month() == 2
    assert region.driest_month() == 3


def test_crop_rule_classifies_three_bands():
    rule = CropRule("maize", min_mm=60, max_mm=180, source="FAO")
    assert rule.classify_month(30) == "Drought risk"
    assert rule.classify_month(200) == "Waterlogging risk"
    assert rule.classify_month(100) == "Good for maize"


def test_cosine_similarity_matches_scipy():
    a = np.array([120.0, 140.0, 180.0])
    b = np.array([8.0, 25.0, 75.0])
    ours = cosine_similarity(a, b)
    scipy_similarity = 1 - scipy_cosine(a, b)
    assert ours == pytest.approx(scipy_similarity, rel=1e-6)
