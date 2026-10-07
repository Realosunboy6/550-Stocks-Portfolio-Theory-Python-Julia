# 550 Stocks Portfolio Theory

Portfolio optimization with matrix algebra on ~400 US stocks across 11 GICS
sectors, in Python and Julia. Data comes **straight from Yahoo Finance** —
no CSVs, no local paths.

## Overview

The project implements Modern Portfolio Theory (MPT) concepts including:
- Log returns calculation
- Correlation and covariance matrix analysis (with positive-semidefinite check)
- Portfolio risk and return calculations
- Sharpe ratio optimization (L-BFGS-B and SLSQP / IPNewton)
- Train/test validation of optimized portfolios

## Files

### Main notebooks (start here)
- `portfolio_theory_python.ipynb` — the full pipeline in Python: Yahoo
  download → log-return matrix → correlation/covariance → PSD check →
  equal-weight metrics → Sharpe optimization → 60/40 train/test validation.
  Verified end-to-end on 2026-10-06 (403 tickers, 2020 → today).
- `portfolio_theory_julia.ipynb` — the same pipeline in Julia with Optim.jl.
  Reads the shared `prices_long.csv` cache written by the Python notebook
  (run it once first). Verified end-to-end on 2026-10-06 with Julia 1.11.7:
  identical 1697×403 return matrix, correlation eigenvalues match Python to
  5 decimals, equal-weight Sharpe 0.0255 in both. The optimizers land on
  different local optima (max-Sharpe is non-convex): Python SLSQP 0.1042 vs
  Julia L-BFGS 0.0715–0.0772 in-sample.

**Global-optimum check (2026-10-06/07):** max-Sharpe has a convex QP
reformulation (min y'Σy s.t. μ'y = 1, y ≥ 0, then w = y/Σy), solved with the
AMPL+Gurobi course license (`gurobi_check.py`): certified global daily Sharpe
**0.1043** (barrier, 11 iterations, max weight 0.21 so the w ≤ 1 cap never
binds). Python's SLSQP (0.1042, 13 iterations) had found the global optimum.
Julia's L-BFGS variants stopped at inferior local optima (0.0715–0.0772), so
the Julia notebook gained a JuMP + AmplNLWriter section driving the same
AMPL-bundled Gurobi driver — it certifies **0.1042** in Julia too (verified
2026-10-07).

### Reference
- `Portfolio_Optimization_COLAB.ipynb` — the detailed full production
  pipeline this work grew from (backtest engine, CVaR sweep, stress tests,
  dashboard). Ancestor of the
  [portlab](https://github.com/Realosunboy6/free-portfolio-visualizer) package.
- `Portfolio_Optimization_SuperPrompt.docx` — prompt doc

## Data

Downloaded live via `yfinance`: 409 curated tickers (11 GICS sectors),
2020-01-01 → today, adjusted closes. Tickers that fail to download
(delisted, e.g. CTRA/BK/HOLX in the Oct-2026 run) are dropped automatically;
prices cache to `prices_2020_latest.parquet` so re-runs skip the download.

Expected columns after the download step:
- `Date` - Trading date
- `Ticker` - Stock ticker symbol
- `Close` - Closing price
- `Sector` - Market sector

## Requirements

### Python
```bash
pip install yfinance pandas numpy scipy plotly pyarrow jupyter
```

### Julia
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "LinearAlgebra", "Statistics", "Dates", "PlotlyJS", "Optim", "JuMP", "AmplNLWriter"])
```
The JuMP/AmplNLWriter section also needs the AMPL course license: set
`AMPL_BIN`/`GUROBI_BIN` env vars to your amplpy install (see the notebook
cell) so the Gurobi driver finds `ampl.lic`.

## Usage

### Python
```bash
jupyter notebook portfolio_theory_python.ipynb
```

### Julia
```bash
# run the Python notebook once first to write prices_long.csv, then:
jupyter notebook portfolio_theory_julia.ipynb
```

## Key Concepts

### Log Returns
Log returns are calculated as: `ln(P_t / P_{t-1})`

### Sharpe Ratio
Risk-adjusted return metric: `(E[R] - R_f) / σ` where R_f = 0

### Portfolio Optimization
Maximize Sharpe ratio subject to:
- Weights sum to 1
- No short selling (weights ≥ 0)

## Conversion Notes

The Julia implementation provides equivalent functionality to the Python version with the following key library mappings:

| Python | Julia |
|--------|-------|
| pandas | DataFrames.jl |
| numpy | LinearAlgebra, Statistics |
| scipy.optimize | Optim.jl |
| plotly | PlotlyJS.jl |
| | |

## License

This project is provided for educational and research purposes.
