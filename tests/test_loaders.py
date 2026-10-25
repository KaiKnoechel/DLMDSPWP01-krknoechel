"""Unit tests for find_ideal_func.loaders."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect

from find_ideal_func.database import DbManager
from find_ideal_func.errors import DatavalidatorError
from find_ideal_func.loaders import (
    IFCSVLoader,
    TestDataCSVLoader,
    TrainingDataCSVLoader,
)


def test_read_csv_raises_when_file_missing(tmp_path):
    """Test that reading a missing CSV file raises a DatavalidatorError."""
    loader = TrainingDataCSVLoader(tmp_path / "missing.csv")

    with pytest.raises(DatavalidatorError):
        loader.read_csv()


def test_load_df_raises_on_missing_columns(tmp_path):
    """Test that loading a CSV with missing columns raises a DatavalidatorError."""
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"x": [0, 1], "y1": [0, 1]}).to_csv(
        csv_path, index=False
    )  # missing y2..y4

    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    with pytest.raises(DatavalidatorError):
        TrainingDataCSVLoader(csv_path).load_df(session)
    session.close()


def test_load_df_writes_table(tmp_path):
    """Test that loading a CSV writes the data to the correct table."""
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}).to_csv(csv_path, index=False)

    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    df = TestDataCSVLoader(csv_path).load_df(session)

    assert list(df["y"]) == [1.0, 2.0]
    assert inspect(db.engine).has_table("TestData")
    session.close()


def test_load_df_rejects_missing_values(tmp_path):
    """Test that loading a CSV with missing values raises a DatavalidatorError."""
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, None]}).to_csv(
        csv_path,
        index=False,
    )

    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    with pytest.raises(DatavalidatorError):
        TestDataCSVLoader(csv_path).load_df(session)

    session.close()


def test_load_df_rejects_duplicate_x_values(tmp_path):
    """Test that loading a CSV with duplicate x values raises a DatavalidatorError."""
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "x": [0.0, 0.0],
            "y1": [1.0, 2.0],
            "y2": [1.0, 2.0],
            "y3": [1.0, 2.0],
            "y4": [1.0, 2.0],
        }
    ).to_csv(
        csv_path,
        index=False,
    )

    db = DbManager(f"sqlite:///{tmp_path / 'db.sqlite'}")
    db.init_db()
    session = db.get_session()

    with pytest.raises(DatavalidatorError):
        TrainingDataCSVLoader(csv_path).load_df(session)

    session.close()


def test_ideal_function_loader_expects_51_columns():
    """Test that the ideal function CSV loader expects 51 columns, with the first being 'x'."""
    assert IFCSVLoader.expected_columns[0] == "x"
    assert len(IFCSVLoader.expected_columns) == 51
