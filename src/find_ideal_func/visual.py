"""Bokeh visualization for training data, ideal functions, and the test-data mapping."""

from __future__ import annotations

import math

import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.annotations import ColorBar
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure, output_file, save
from bokeh.transform import linear_cmap

from find_ideal_func.mapping import MappingResult
from find_ideal_func.selection import SelectedIdealFunction


class Vizualizer:
    """Builds Bokeh plots for the training/ideal data and the test-data mapping."""

    def __init__(
        self,
        training_data: pd.DataFrame,
        test_data: pd.DataFrame,
        ideal_functions: pd.DataFrame,
        selected_functions: list[SelectedIdealFunction],
        mapping_results: list[MappingResult],
    ) -> None:
        """
        Initialize the Vizualizer with training data, test data, ideal functions,
        the four selected ideal functions, and the test-data mapping results.

        Args:
            training_data (pd.DataFrame): The training data with 'x' and 'y1'..'y4' columns.
            test_data (pd.DataFrame): The test data with 'x' and 'y' columns.
            ideal_functions (pd.DataFrame): The ideal functions with 'x' and 'y1'..'y50' columns.
            selected_functions (list[SelectedIdealFunction]): The four chosen ideal functions.
            mapping_results (list[MappingResult]): The test-point mapping outcomes.
        """
        self.training_data = training_data
        self.test_data = test_data
        self.ideal_functions = ideal_functions
        self.selected_functions = selected_functions
        self.mapping_results = mapping_results

    def _plot_ideal_function(
        self, plot: figure, selected_function: SelectedIdealFunction, color: str
    ) -> None:
        """Plot an ideal curve with a band of max_delta_y * sqrt(2) on each side."""
        ideal = self.ideal_functions.sort_values("x")
        ideal_x = ideal["x"].to_numpy(dtype=float)
        ideal_y = ideal[selected_function.ideal_column].to_numpy(dtype=float)
        tolerance = selected_function.max_delta_y * math.sqrt(2)
        plot.varea(
            x=ideal_x,
            y1=ideal_y - tolerance,
            y2=ideal_y + tolerance,
            fill_color=color,
            fill_alpha=0.15,
            legend_label=f"{selected_function.ideal_column} tolerance",
        )
        plot.line(
            ideal_x,
            ideal_y,
            color=color,
            legend_label=selected_function.ideal_column,
        )

    def _training_vs_ideal_plots(self):
        """Overlay each training column with its ideal curve and acceptance band."""
        colors = Category10[4]
        plots = []

        for color, sf in zip(colors, self.selected_functions):
            p = figure(
                title=f"{sf.t_column} vs {sf.ideal_column} (SSE={sf.sse:.2f})",
                x_axis_label="x",
                y_axis_label="y",
                width=400,
                height=300,
            )
            self._plot_ideal_function(p, sf, "black")
            p.scatter(
                self.training_data["x"],
                self.training_data[sf.t_column],
                size=4,
                color=color,
                legend_label="training",
                marker="circle",
            )
            plots.append(p)

        return gridplot(plots, ncols=2)

    def _test_mapping_plot(self):
        """
        Build the test-point plot.
        Mapped points are colored by their deviation from the assigned ideal function, 
        with assignment details on hover, while unmapped points are shown in gray.
        Selected ideal curves are shown with their acceptance bands.
        """
        mapped = [
            r for r in self.mapping_results if r.ideal_function_number is not None
        ]
        unmapped = [r for r in self.mapping_results if r.ideal_function_number is None]

        p = figure(
            title="Test data mapping (color = deviation from assigned ideal function)",
            x_axis_label="x",
            y_axis_label="y",
            width=800,
            height=400,
        )

        for color, selected_function in zip(Category10[4], self.selected_functions):
            self._plot_ideal_function(p, selected_function, color)

        if mapped:
            source = ColumnDataSource(
                data={
                    "x": [r.x for r in mapped],
                    "y": [r.y for r in mapped],
                    "delta_y": [r.delta_y for r in mapped],
                    "idf_number": [
                        r.ideal_function_number for r in mapped
                    ],
                }
            )
            mapper = linear_cmap(
                "delta_y",
                Viridis256,
                low=min(source.data["delta_y"]),
                high=max(source.data["delta_y"]),
            )
            mapped_renderer = p.scatter(
                "x",
                "y",
                source=source,
                size=8,
                color=mapper,
                marker="circle",
                legend_label="mapped",
            )
            p.add_tools(
                HoverTool(
                    renderers=[mapped_renderer],
                    tooltips=[
                        ("x", "@x"),
                        ("y", "@y"),
                        ("Ideal function", "y@idf_number"),
                        ("Deviation", "@delta_y{0.000000}"),
                    ],
                )
            )
            p.add_layout(
                ColorBar(color_mapper=mapper["transform"], title="delta_y"), "right"
            )

        if unmapped:
            p.scatter(
                [r.x for r in unmapped],
                [r.y for r in unmapped],
                size=6,
                color="gray",
                marker="x",
                legend_label="unmapped",
            )

        p.legend.click_policy = "hide"
        return p

    def render(self, output_html: str) -> None:
        """
        Render plots and save them to a HTML file.
        """
        output_file(output_html)
        save(column(self._training_vs_ideal_plots(), self._test_mapping_plot()))
