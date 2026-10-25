# Ideal Functions

Select the best-fitting ideal function for each of four training functions,
map qualifying test points, store the results in SQLite, and visualize them
with Bokeh.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup and Run

From the repository directory:

```powershell
uv sync
uv run python main.py
```

Input CSV files belong in `data/`: `train.csv`, `ideal.csv`, and `test.csv`.
The application creates:

- `output/ideal_functions.db`, an SQLite database
- `output/visualization.html`, an interactive Bokeh visualization

Each run replaces the saved mappings. Open the HTML file in a browser to view
the plots; hover over mapped points for their assigned function and deviation.

## Algorithm

For each training column, select the ideal function with the lowest sum of
squared errors (SSE) across all training points.

For each selected function, the program calculates the largest absolute
training deviation. A test point is accepted when its deviation is no greater
than that value multiplied by `sqrt(2)`:

```text
allowed deviation = maximum training deviation * sqrt(2)
```

If more than one selected function accepts a test point, the function with the
smallest deviation is used.

## Tests and Quality Checks

```powershell
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## Git Workflow

Example team workflow, assuming the remote repository has a `develop` branch:

```powershell
git clone --branch develop <repository-url>
cd <repository-directory>
git switch -c feature/new-function
```

After implementing and testing the change:

```powershell
git add .
git commit -m "Add new function"
git push -u origin feature/new-function
```

Open a pull request from `feature/new-function` into `develop` for team review
before merging.
