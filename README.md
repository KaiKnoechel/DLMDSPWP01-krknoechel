 # Ideal Functions

Python implementation of the assignment. The program uses four
training functions to select the best four functions from 50 ideal
functions. It maps the test data to those functions. All results are stored in a
SQLite db. Using the Bokeh lib we visualize our results.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Project Structure

```text
main.py                         Application entry point
src/find_ideal_func/            Application package
data/train.csv                  Training data
data/ideal.csv                  Fifty ideal functions
data/test.csv                   Test data
tests/                          Unit tests
output/                         Generated database and visualization
pyproject.toml                  Project configuration
```

## Installation

Install the project and its development dependencies with:

```powershell
uv sync
```

## Run the Pipeline

Run the application with either command:

```powershell
uv run python main.py
uv run ideal-functions
```

The application creates:

- `output/ideal_functions.db`, an SQLite database
- `output/visualization.html`, an interactive Bokeh visualization

## Algorithm

For each training function, the program calculates the sum of squared errors
against every ideal function. So the ideal function with the smallest value is
selected.

For each selected function, the program calculates the largest absolute
training deviation. A test point is accepted when its deviation is no greater
than that value multiplied by `sqrt(2)`:

```text
allowed deviation = maximum training deviation * sqrt(2)
```

If more than one selected function accepts a test point, the function with the
smallest deviation is used.

## Database Tables

The SQLite database contains the following tables:

- `TrainingData`: `x`, `y1`, `y2`, `y3`, and `y4`
- `IdealFunction`: `x` and the fifty ideal-function columns
- `Mappings`: test `x`, test `y`, deviation, and selected ideal-function number

## Visualization

The visualization contains:

- Each training function plotted against its selected ideal function
- Mapped test points colored by deviation
- Unmapped test points shown separately

## Tests and Quality Checks

Run the unit tests with:

```powershell
uv run pytest
```

Run the static checks with:

```powershell
uv run ruff check .
uv run mypy src tests
```

## Git Workflow
Each adjustment, fix or new feature can be added following workflow

Clone the `release` branch taht is based on the `main` branch if ist not already cloned:

```powershell
git clone <repository-url>
cd <repository-directory>
```

otherwise pull from your `release` branch with:

```powershell
git switch <release branch>
git pull
```

Create a feature branch, commit the changes, and push it:

```powershell
git switch -c feature/new-function
git add .
git commit -m "Add new function"
git push -u origin feature/new-function
```

After pushing, open a pull request (PR) from the feature branch to `release branch` for
review and merging. Afterwards you can create pre-relase canidate and after testing it on a UAT system the release-branch is then merged with an PR into the main branch.
