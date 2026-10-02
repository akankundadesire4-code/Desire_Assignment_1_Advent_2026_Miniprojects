# OOP with Python - Assignment 2 (Advent 2026)

Five small mini-projects, each with one Jupyter notebook, one `src/` module, and pytest
tests. Code is written at a beginner level: short files, simple classes, no frameworks
beyond what the assignment asks for.

## Repository structure

```
src/                    reusable code (one small file per mini-project, plus utils.py)
notebooks/              one notebook per mini-project - the analysis, plots and findings
tests/                  pytest unit tests (3+ per mini-project)
data/                   generated/downloaded CSV data used by the notebooks
notes/                  plain-text answers to the assignment's "explain this" questions
references/             citations for the one real dataset and the crop thresholds used
```

| Mini-project | Notebook | Module |
|---|---|---|
| 1. UBOS District Population Forecaster | `notebooks/project1_population.ipynb` | `src/project1_population.py` |
| 2. Solar Micro-Grid Dispatch Planner | `notebooks/project2_microgrid.ipynb` | `src/project2_microgrid.py` |
| 3. Lake Victoria Fish Stock & Export Risk Model | `notebooks/project3_fish_stock.ipynb` | `src/project3_fish_stock.py` |
| 4. Rainfall Pattern & Crop Suitability Analyser | `notebooks/project4_rainfall.ipynb` | `src/project4_rainfall.py` |
| 5. Taxi Route Revenue, Pricing & Fleet Planner | `notebooks/project5_taxi.ipynb` | `src/project5_taxi.py` |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running the notebooks

Open any notebook in `notebooks/` in VS Code or Jupyter and choose **Restart & Run All**.
Each notebook adds `../src` to `sys.path` so it can import the matching module directly -
no packaging or installation step is needed. Notebooks that read/write CSV files
(Projects 2 and 4) expect to be run from the `notebooks/` folder, which is the default
working directory when you open them from there.

## Running the tests

```powershell
python -m pytest
```

This runs 25 tests across the five mini-projects (5 per project), covering normal cases
and at least one edge case each (invalid input, empty data, or a zero/negative value).

## Data sources

- Population, micro-grid demand shape, fish stock/price parameters, and taxi passenger
  counts are the **illustrative** data supplied in the assignment brief (or synthetic
  data generated from it with a fixed random seed).
- Mini-Project 4's extension uses **real** NASA POWER monthly precipitation data
  (2015-2024) for Kampala, Gulu and Mbarara. Full citation, coordinates and retrieval
  date are in `references/sources.txt`.
- Crop rainfall thresholds are simplified, illustrative bands informed by general FAO
  crop-water guidance; also cited in `references/sources.txt`.

## AI use declaration

AI coding assistance (GitHub Copilot) was used to help scaffold the `src/` modules,
notebooks, and tests for this assignment, and to help draft the plain-language
explanation notes. All code was reviewed, run, and can be explained line-by-line by the
student. Each notebook restates this declaration and states where illustrative data was
added or changed.

## Summary of findings (see each notebook's own Findings & Limitations section for more)

1. **Population**: the CAGR growth model beat linear trend and Fibonacci-ratio on every
   district's 2022-2024 test data. Wakiso grew fastest in relative terms (CAGR 6.5%).
   Wakiso and Kampala need the most additional classrooms by 2029.
2. **Micro-grid**: the daily dispatch system is well-posed (determinant -5, condition
   number ~5.8). Vectorised solving was about 17x faster than a day-by-day loop over 30
   days. No infeasible (negative) days occurred in this seeded dataset, but the
   non-negative least-squares fallback is implemented and tested.
3. **Fish stock**: revenue peaked around a harvest rate of h=0.20, close to the
   theoretical MSY harvest rate (r/2=0.20) - pushing to h=0.30 crashed the stock and
   reduced revenue, a classic overfishing pattern. The Fibonacci "stock" baseline was
   shown to be biologically indefensible (unbounded growth, no carrying capacity).
4. **Rainfall**: cosine similarity between regions was high (0.75-0.89) even where
   Pearson correlation was weak or negative, showing why relying on cosine similarity
   alone can be misleading. Peak detection on the illustrative data did not match real
   Uganda climate zones, but the real NASA POWER data (the extension) did show the
   expected bimodal Kampala / unimodal Gulu pattern.
5. **Taxi routes**: the Ntinda fare (UGX 2,000) sits below the calculated market
   equilibrium (UGX 2,200), implying excess demand. Exponential smoothing with a high
   alpha (0.9) beat the moving average and linear trend in walk-forward backtesting for
   every route. Each route needs just one vehicle to cover its day-11 forecast with a
   15% buffer.
