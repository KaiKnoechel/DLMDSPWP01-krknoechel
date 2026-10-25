"""Map the selected ideal functions to the test data points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from find_ideal_func.database import Mappings
from find_ideal_func.selection import SelectedIdealFunction


@dataclass(frozen=True)
class MappingResult:
    """Result for a given test data point."""

    delta_y: float | None
    x: float
    y: float
    ideal_function_number: int | None


class TestDataMapper:
    """Class to map the selected ideal functions to the test data points."""

    __test__ = False

    def __init__(
        self,
        t_data: pd.DataFrame,
        ideal_functions: pd.DataFrame,
        select_functions: list[SelectedIdealFunction],
        session: Session | None = None,
    ) -> None:
        self.t_data = t_data
        self.ideal_functions = ideal_functions.set_index("x").sort_index()
        self.select_functions = select_functions
        self.session = session

    def map_data(self) -> list[MappingResult]:
        """Map test data points to selected ideal functions."""

        def _threshold() -> dict[str, tuple[int, float]]:
            """Build a dictionary mapping ideal columns to their allowed delta_y thresholds."""
            return {
                sf.ideal_column: (sf.idf_number, sf.max_delta_y * math.sqrt(2))
                for sf in self.select_functions
            }

        results = []
        treshold_dict = _threshold()

        for row in self.t_data.itertuples(index=False):
            x = float(cast(float, row.x))
            y = float(cast(float, row.y))

            if x not in self.ideal_functions.index:
                results.append(MappingResult(None, x, y, None))
                continue

            idf_number: int | None = None
            delta_y: float | None = None

            # Iterate over each ideal function and check if the test data point falls within the allowed delta_y threshold

            for ideal_column, (idf_num, allowed_delta_y) in treshold_dict.items():
                ideal_y = cast(float, self.ideal_functions.at[x, ideal_column])
                current_delta_y = abs(y - ideal_y)

                if current_delta_y <= allowed_delta_y and (
                    delta_y is None or current_delta_y < delta_y
                ):
                    idf_number = idf_num
                    delta_y = current_delta_y
            results.append(MappingResult(delta_y, x, y, idf_number))

        return results

    def save_results(self, results: list[MappingResult]) -> None:
        """Save the mapping results using the configured Session"""
        if self.session is None:
            raise ValueError("No session configured for saving results")

        save_results(self.session, results)


def save_results(session: Session, results: list[MappingResult]) -> None:
    """Replace saved mappings with successfully mapped points and commit."""
    session.execute(delete(Mappings))
    session.add_all(
        Mappings(
            x=result.x,
            y=result.y,
            delta_y=result.delta_y,
            ideal_function_number=result.ideal_function_number,
        )
        for result in results
        if result.ideal_function_number is not None
    )
    session.commit()
