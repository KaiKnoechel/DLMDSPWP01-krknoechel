"""Select the ideal function with the lowest SSE for each training column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from find_ideal_func.errors import (
    DatavalidatorError,
)


@dataclass(frozen=True)
class SelectedIdealFunction:
    """Dataclass to hold the selected ideal function for a given training column."""

    ideal_column: str
    t_column: str
    idf_number: int
    sse: float
    max_delta_y: float


class IdealFunctionSelector:
    """Class to select the ideal function for each training data column."""

    def __init__(
        self, training_data: pd.DataFrame, ideal_functions: pd.DataFrame
    ) -> None:
        self.training_data = self._prepare(training_data)
        self.ideal_functions = self._prepare(ideal_functions)
        self._validate_x()

    @staticmethod
    def _prepare(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame by setting 'x' as the index and sorting it."""
        if "x" not in dataframe.columns:
            raise DatavalidatorError("DataFrame must contain 'x' column")
        return dataframe.set_index("x").sort_index()

    def _validate_x(self) -> None:
        """Validate that both training data and ideal functions contain the same 'x' values."""
        if not self.training_data.index.equals(self.ideal_functions.index):
            raise DatavalidatorError(
                "Training data and ideal functions must have the same 'x' values"
            )

    def _calculate_sse(self) -> pd.DataFrame:
        """Calculate the sum of squared errors (SSE) between training data and ideal functions."""
        idel_values = self.ideal_functions.to_numpy()
        training_values = self.training_data.to_numpy()

        diff = training_values[:, :, None] - idel_values[:, None, :]
        sse = np.square(diff).sum(axis=0)
        return pd.DataFrame(
            sse, index=self.training_data.columns, columns=self.ideal_functions.columns
        )

    def select_ideal_function(self) -> list[SelectedIdealFunction]:
        """Return the ideal function minimizing SSE for each training column."""
        sse_df = self._calculate_sse()
        selected_funcs = []

        for training_col in self.training_data.columns:
            row = sse_df.loc[training_col]
            ideal_col = str(row.idxmin())
            train_val = self.training_data[training_col].to_numpy()
            ideal_val = self.ideal_functions[ideal_col].to_numpy()
            delta_y = np.abs(train_val - ideal_val)
            max_delta_y = np.max(delta_y)

            selected_funcs.append(
                SelectedIdealFunction(
                    ideal_column=ideal_col,
                    t_column=training_col,
                    idf_number=int(ideal_col[1:]),
                    sse=cast(float, row[ideal_col]),
                    max_delta_y=max_delta_y,
                )
            )

        return selected_funcs
