"""Global max-Sharpe check via AMPL + Gurobi (course license).

Reformulation (Sharpe is scale-invariant): with y = w / (mu'w),
    max  mu'w / sqrt(w' Sigma w)   <=>   min  y' Sigma y
    s.t. mu'y = 1, y >= 0,  sum(w) = 1
then w* = y* / sum(y*). Convex QP -> Gurobi certifies global optimum.
Data: same 1697x403 log-return matrix as the notebooks.
"""
import numpy as np
import pandas as pd
from amplpy import AMPL

# --- rebuild the exact return matrix used by both notebooks ---
long = pd.read_csv("prices_long.csv", parse_dates=["Date"])
long = long.sort_values(["Ticker", "Date"])
long["LogReturn"] = np.log(long.groupby("Ticker")["Close"].transform(
    lambda x: x / x.shift(1)))
rets = long.pivot(index="Date", columns="Ticker", values="LogReturn")
rets = rets.loc[:, rets.count() > 500]               # same ticker filter
rets = rets.dropna()                                 # drop gappy rows
R = rets.values
tickers = rets.columns.tolist()
n = R.shape[1]
mu = R.mean(axis=0)
Sigma = np.cov(R, rowvar=False)
print(f"R: {R.shape}, tickers: {n}")

ampl = AMPL()
ampl.eval("set A; param mu{A}; param Sigma{A,A}; var y{A} >= 0; "
          "minimize quad: sum{i in A, j in A} y[i]*Sigma[i,j]*y[j]; "
          "subject to ret1: sum{i in A} mu[i]*y[i] = 1;")
ampl.set["A"] = tickers
ampl.param["mu"] = dict(zip(tickers, mu))
ampl.param["Sigma"] = {(t1, t2): Sigma[i, j]
                       for i, t1 in enumerate(tickers)
                       for j, t2 in enumerate(tickers)}
ampl.option["solver"] = "gurobi"
ampl.solve()

y = np.array([ampl.var["y"][t].value() for t in tickers])
w = y / y.sum()
port_r = R @ w
sharpe = port_r.mean() / port_r.std()
print(f"Gurobi status : {ampl.get_value('solve_result')}")
print(f"Global max daily Sharpe : {sharpe:.4f}")
print(f"sum(w)={w.sum():.6f}  max(w)={w.max():.4f}  n>0.001: {(w > 0.001).sum()}")
top = np.argsort(w)[::-1][:10]
print("Top holdings:", ", ".join(f"{tickers[i]}:{w[i]:.3f}" for i in top))
# sanity: Sharpe of the QP solution measured the same way as the notebooks
print(f"Check mu'y = {mu @ y:.6f} (must be 1)")
