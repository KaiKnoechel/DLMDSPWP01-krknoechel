"""Unit tests for find_ideal_func.mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from find_ideal_func.database import DbManager, Mappings
from find_ideal_func.mapping import MappingResult, TestDataMapper
from find_ideal_func.selection import SelectedIdealFunction


@pytest.fixture
def ideal_functions() -> pd.DataFrame:
    """Fixture providing a sample set of ideal functions."""
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y1": [0.0, 1.0, 2.0, 3.0, 4.0],  # y1 = x
            "y2": [5.0, 6.0, 7.0, 8.0, 9.0],  # y2 = x + 5
        }
    )


@pytest.fixture
def selected_functions() -> list[SelectedIdealFunction]:
    """Fixture providing a sample set of selected ideal functions."""
    return [
        SelectedIdealFunction(
            ideal_column="y1", t_column="ta", idf_number=1, sse=0.0, max_delta_y=1.0
        ),
        SelectedIdealFunction(
            ideal_column="y2", t_column="tb", idf_number=2, sse=0.0, max_delta_y=0.5
        ),
    ]


def _mapper(
    test_data: pd.DataFrame, ideal_functions: pd.DataFrame, selected_functions
) -> TestDataMapper:
    return TestDataMapper(test_data, ideal_functions, selected_functions, session=None)


def test_map_data_matches_within_tolerance(ideal_functions, selected_functions):
    """Test that mapping data points within tolerance correctly assigns ideal functions."""
    test_data = pd.DataFrame(
        {"x": [0.0], "y": [0.5]}
    )  # |0.5 - 0| = 0.5 <= 1.0 * sqrt(2)

    results = _mapper(test_data, ideal_functions, selected_functions).map_data()

    assert len(results) == 1
    assert results[0].ideal_function_number == 1
    assert results[0].delta_y == pytest.approx(0.5)


def test_map_data_rejects_deviation_beyond_tolerance(
    ideal_functions, selected_functions
):
    """Test that mapping data points beyond tolerance results in no match."""
    test_data = pd.DataFrame({"x": [2.0], "y": [100.0]})

    results = _mapper(test_data, ideal_functions, selected_functions).map_data()

    assert results[0].ideal_function_number is None
    assert results[0].delta_y is None


def test_map_data_ignores_x_missing_from_ideal_functions(
    ideal_functions, selected_functions
):
    """Test that mapping data points with x missing from ideal functions results in no match."""
    test_data = pd.DataFrame({"x": [99.0], "y": [0.0]})

    results = _mapper(test_data, ideal_functions, selected_functions).map_data()

    assert results[0].ideal_function_number is None
    assert results[0].delta_y is None


def test_map_data_picks_closest_match_when_multiple_qualify():
    """Test that mapping data points picks the closest match when multiple ideal functions qualify."""
    ideal_functions = pd.DataFrame({"x": [0.0], "y1": [0.0], "y3": [0.1]})
    selected_functions = [
        SelectedIdealFunction(
            ideal_column="y1", t_column="ta", idf_number=1, sse=0.0, max_delta_y=1.0
        ),
        SelectedIdealFunction(
            ideal_column="y3", t_column="tc", idf_number=3, sse=0.0, max_delta_y=1.0
        ),
    ]
    # y=0.03 is within tolerance of BOTH y1 (dev 0.03) and y3 (dev 0.07) -> closest one must win
    test_data = pd.DataFrame({"x": [0.0], "y": [0.03]})

    results = _mapper(test_data, ideal_functions, selected_functions).map_data()

    assert results[0].ideal_function_number == 1
    assert results[0].delta_y == pytest.approx(0.03, abs=1e-9)


def test_save_results_skips_unmapped_points(tmp_path):
    """Test that saving mapping results skips points that were not mapped."""
    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    mapper = TestDataMapper(
        pd.DataFrame({"x": []}), pd.DataFrame({"x": []}), [], session
    )
    results = [
        MappingResult(delta_y=0.1, x=1.0, y=2.0, ideal_function_number=5),
        MappingResult(delta_y=None, x=3.0, y=4.0, ideal_function_number=None),
    ]

    mapper.save_results(results)

    saved = session.query(Mappings).all()
    assert len(saved) == 1
    assert saved[0].ideal_function_number == 5
    session.close()


@pytest.mark.parametrize("replacement", ["same", "changed", "empty"])
def test_save_results_replaces_previous_mappings(tmp_path, replacement):
    """Replaces old mappings rather than accumulating rows."""
    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    with db.session_scope() as session:
        mapper = TestDataMapper(
            pd.DataFrame({"x": []}), pd.DataFrame({"x": []}), [], session
        )
        og = MappingResult(0.1, 1.0, 2.0, 5)
        mapper.save_results([og])
        results = {
            "same": [og],
            "changed": [MappingResult(0.2, 3.0, 4.0, 7)],
            "empty": [],
        }[replacement]
        mapper.save_results(results)
        saved = session.query(Mappings).all()
        assert [(row.delta_y, row.x, row.y, row.ideal_function_number) for row in saved] == [
            (result.delta_y, result.x, result.y, result.ideal_function_number)
            for result in results
        ]
