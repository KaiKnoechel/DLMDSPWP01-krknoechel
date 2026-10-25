"""Unit tests for find_ideal_func.visual.

Tests verify mapping hover details and rendering to a non-empty HTML file.
"""

from __future__ import annotations

import pandas as pd
from bokeh.models import HoverTool

from find_ideal_func.mapping import MappingResult
from find_ideal_func.selection import SelectedIdealFunction
from find_ideal_func.visual import Vizualizer


def test_render_writes_html_file(tmp_path):
    training_data = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0]})
    test_data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.1, 0.9]})
    ideal_functions = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0]})
    selected_functions = [
        SelectedIdealFunction(
            ideal_column="y1", t_column="y1", idf_number=1, sse=0.01, max_delta_y=0.1
        )
    ]
    mapping_results = [
        MappingResult(delta_y=0.1, x=0.0, y=0.1, ideal_function_number=1),
        MappingResult(delta_y=None, x=1.0, y=0.9, ideal_function_number=None),
    ]

    out_file = tmp_path / "viz.html"
    visualizer = Vizualizer(
        training_data, test_data, ideal_functions, selected_functions, mapping_results
    )
    mapping_plot = visualizer._test_mapping_plot()
    hover = mapping_plot.select_one(HoverTool)
    assert hover is not None
    assert ("Ideal function", "y@idf_number") in hover.tooltips
    assert ("Deviation", "@delta_y{0.000000}") in hover.tooltips
    assert len(hover.renderers) == 1
    source = hover.renderers[0].data_source
    assert source.data["idf_number"] == [1]
    assert source.data["delta_y"] == [0.1]
    visualizer.render(str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0
