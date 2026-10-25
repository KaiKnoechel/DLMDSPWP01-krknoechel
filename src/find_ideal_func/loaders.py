"""Load the CSV data into the database."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
from sqlalchemy.orm import Session

from find_ideal_func.errors import DatavalidatorError


class CSVLoader:
    """Class to load CSV data into the database."""

    table_name: ClassVar[str]
    expected_columns: ClassVar[list[str]]
    require_unique_x: ClassVar[bool] = False

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def read_csv(self) -> pd.DataFrame:
        """Read the CSV file into a pandas DataFrame and handle errors."""
        if not self.csv_path.exists():
            raise DatavalidatorError(f"CSV file does not exist: {self.csv_path}")
        try:
            return pd.read_csv(self.csv_path)
        except (OSError, pd.errors.ParserError) as error:
            raise DatavalidatorError(
                f"Error reading CSV file: {error}"
            ) from error

    def validate(self, df: pd.DataFrame) -> None:
        """Validate the DataFrame against the expected columns."""
        missing_columns = set(self.expected_columns) - set(df.columns)

        if df.empty:
            raise DatavalidatorError(f"CSV file is empty: {self.csv_path}")
        if self.require_unique_x and df["x"].duplicated().any():
            raise DatavalidatorError(
                f"CSV file contains duplicate values in column 'x': {self.csv_path}"
            )
        if missing_columns:
            raise DatavalidatorError(
                f"CSV file is missing the following columns: {missing_columns}"
            )
        if df.isna().any().any():
            raise DatavalidatorError(
                f"CSV file contains missing values: {self.csv_path}"
            )

        for column in self.expected_columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise DatavalidatorError(
                    f"Numeric expected in column '{column}' in CSV file: {self.csv_path}"
                )

    def load_df(self, session: Session) -> pd.DataFrame:
        """Read and validate the CSV, store it, replacing the destination table."""
        df = self.read_csv()
        self.validate(df)
        df.to_sql(
            self.table_name, con=session.get_bind(), if_exists="replace", index=False
        )
        return df


class IFCSVLoader(CSVLoader):
    """Class to load ideal function CSV data into the database."""

    table_name: ClassVar[str] = "IdealFunction"
    require_unique_x: ClassVar[bool] = True
    expected_columns: ClassVar[list[str]] = ["x"] + [
        f"y{i}" for i in range(1, 51)
    ]


class TestDataCSVLoader(CSVLoader):
    """Class to load raw test data CSV into the database."""

    __test__ = False
    table_name: ClassVar[str] = "TestData"
    expected_columns: ClassVar[list[str]] = ["x", "y"]


class TrainingDataCSVLoader(CSVLoader):
    """Class to load training data CSV into the database."""

    table_name: ClassVar[str] = "TrainingData"
    require_unique_x: ClassVar[bool] = True
    expected_columns: ClassVar[list[str]] = ["x", "y1", "y2", "y3", "y4"]
