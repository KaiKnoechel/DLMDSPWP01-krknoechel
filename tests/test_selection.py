"""Unit tests for find_ideal_func.selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from find_ideal_func.selection import IdealFunctionSelector, SelectedIdealFunction


@pytest.fixture
def ideal_functions() -> pd.DataFrame:
    """Fixture providing a sample set of ideal functions."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return pd.DataFrame(
        {
            "x": x,
            "y1": x,
            "y2": x**2,
            "y3": -x,
            "y4": 3 * x,
            "y5": x + 10,
            "y6": 2 * x,
        }
    )


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Fixture providing a sample set of training data."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    noise = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
    return pd.DataFrame(
        {
            "x": x,
            "y1": -x + noise,  # closest to ideal y3
            "y2": x + noise,  # closest to ideal y1
            "y3": 2 * x + noise,  # closest to ideal y6
            "y4": x**2 + noise,  # closest to ideal y2
        }
    )


def test_select_ideal_function_picks_expected_pairing(training_data, ideal_functions):
    """Test that the ideal function selector picks the expected pairing for each training column."""
    selector = IdealFunctionSelector(training_data, ideal_functions)
    selected = selector.select_ideal_function()

    picked = {sf.t_column: sf.ideal_column for sf in selected}
    assert picked == {"y1": "y3", "y2": "y1", "y3": "y6", "y4": "y2"}


def test_select_ideal_function_reports_max_deviation(training_data, ideal_functions):
    """Test that the ideal function selector reports the maximum deviation correctly."""
    selector = IdealFunctionSelector(training_data, ideal_functions)
    selected = selector.select_ideal_function()

    for sf in selected:
        assert isinstance(sf, SelectedIdealFunction)
        assert sf.max_delta_y == pytest.approx(0.02, abs=1e-9)
        assert sf.sse >= 0


def test_select_ideal_function_returns_one_result_per_training_column(
    training_data, ideal_functions
):
    """Test that the ideal function selector returns exactly one result per training column."""
    selector = IdealFunctionSelector(training_data, ideal_functions)
    selected = selector.select_ideal_function()

    assert len(selected) == 4
    assert {sf.t_column for sf in selected} == {"y1", "y2", "y3", "y4"}
