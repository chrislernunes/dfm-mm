#!/usr/bin/env python3

# UAE / Dubai Efficient Frontier Strategy – yfinance only

# - Same Markowitz / efficient-frontier / tangency logic as original
# - Data: top 20 Dubai Financial Market (DFM) stocks via yfinance


import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# Top 20 Dubai (DFM) stocks 
# Selected by market-cap / liquidity (2025-26)
# ----------------------------------------------------------------------
DUBAI_TOP20 = [
    "EMIRATESNBD.AE",   # Emirates NBD
    "DEWA.AE",          # Dubai Electricity & Water
    "EMAAR.AE",         # Emaar Properties
    "DIB.AE",           # Dubai Islamic Bank
    "EMAARDEV.AE",      # Emaar Development
    "DU.AE",            # Emirates Integrated Telecom (du)
    "SALIK.AE",         # Salik
    "CBD.AE",           # Commercial Bank of Dubai
    "AIRARABIA.AE",     # Air Arabia
    "DIC.AE",           # Dubai Investments
    "EMPOWER.AE",       # Emirates Central Cooling
    "DFM.AE",           # Dubai Financial Market
    "TABREED.AE",       # National Central Cooling
    "ARMX.AE",          # Aramex
    "AJMANBANK.AE",     # Ajman Bank
    "AMANAT.AE",        # Amanat Holdings
    "UPP.AE",           # Union Properties
    "SHUAA.AE",         # Shuaa Capital
    "NCC.AE",           # National Cement
    "GFH.AE",           # GFH Financial Group
]


def download_dubai_prices(symbols, start="2015-01-01"):
    print(f"Downloading {len(symbols)} Dubai stocks from yfinance...")
    data = {}
    failed = []
    for i, sym in enumerate(symbols, 1):
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            if len(df) > 300:          # need enough history for monthly returns
                data[sym] = df
                print(f"  [{i:2d}/{len(symbols)}] {sym:18s}  {len(df):5d} rows")
            else:
                failed.append(sym)
                print(f"  [{i:2d}/{len(symbols)}] {sym:18s}  too short – skipped")
        except Exception as e:
            failed.append(sym)
            print(f"  [{i:2d}/{len(symbols)}] {sym:18s}  error: {e}")
    print(f"Successfully loaded {len(data)} stocks")
    return data


# Core optimizers 

def project_simplex(v, z=1.0):
    v = np.asarray(v, dtype=float)
    z = float(z)
    if v.size == 0:
        return v
    if not (math.isfinite(z) and z > 0):
        return np.zeros_like(v)
    v = np.where(np.isfinite(v), v, 0.0)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, len(u) + 1, dtype=float)
    cond = u * j > (cssv - z)
    rho_idx = np.nonzero(cond)[0]
    if rho_idx.size == 0:
        theta = (cssv[-1] - z) / float(len(u))
    else:
        rho = int(rho_idx[-1])
        theta = (cssv[rho] - z) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if math.isfinite(s) and s > 0:
        w *= (z / s)
    else:
        w = np.full_like(v, z / float(v.size))
    return w

def solve_mvp_long_only(cov, ridge=1e-5, max_iter=1200, tol=1e-10):
    cov = np.asarray(cov, dtype=float) + np.eye(cov.shape[0]) * ridge
    n = cov.shape[0]
    if n < 2:
        return None
    w = np.full(n, 1.0 / n)
    try:
        maxeig = float(np.max(np.linalg.eigvalsh(cov)))
        L = 2.0 * max(maxeig, 1e-12)
    except Exception:
        L = 2.0
    step = 0.9 / L
    for _ in range(max_iter):
        grad = 2.0 * cov.dot(w)
        w_new = project_simplex(w - step * grad, z=1.0)
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new
    w = np.clip(w, 0.0, np.inf)
    s = float(w.sum())
    if not (math.isfinite(s) and s > 0):
        return None
    return w / s

def solve_frontier_long_only(mu, cov, target_sigma, ridge=1e-5, max_iter=2000, tol=1e-10):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float) + np.eye(len(mu)) * ridge
    n = mu.size
    if n < 2:
        return None
    target_var = target_sigma ** 2
    w = solve_mvp_long_only(cov, ridge=0.0)   # already ridged
    if w is None:
        w = np.full(n, 1.0 / n)
    mu_scale = float(np.max(np.abs(mu))) or 1.0
    mu_n = mu / mu_scale
    try:
        maxeig = float(np.max(np.linalg.eigvalsh(cov)))
    except Exception:
        maxeig = 1.0
    step = 0.15 / (1.0 + 2.0 * maxeig)
    lam = 10.0
    for _ in range(max_iter):
        var = float(w.dot(cov).dot(w))
        viol = max(0.0, var - target_var)
        if viol > target_var * 0.002:
            lam = min(lam * 1.05, 1e6)
        else:
            lam = max(lam * 0.995, 1e-6)
        grad = mu_n.copy()
        if viol > 0:
            grad = grad - (lam * 2.0 * viol) * (2.0 * cov.dot(w))
        w_new = project_simplex(w + step * grad, z=1.0)
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new
    # safety blend toward MVP if still over risk
    var = float(w.dot(cov).dot(w))
    if math.isfinite(var) and var > target_var * 1.01:
        w_mvp = solve_mvp_long_only(cov, ridge=0.0)
        if w_mvp is not None:
            lo, hi = 0.0, 1.0
            best = w_mvp
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                cand = project_simplex((1 - mid) * w_mvp + mid * w, z=1.0)
                v = float(cand.dot(cov).dot(cand))
                if math.isfinite(v) and v <= target_var:
                    best = cand
                    lo = mid
                else:
                    hi = mid
            w = best
    return w

def solve_tangency_long_only(mu_m, cov_m, ridge=1e-5, frontier_points=25, rf_annual=0.0):
    mu_m = np.asarray(mu_m, dtype=float)
    cov_m = np.asarray(cov_m, dtype=float)
    n = mu_m.size
    if n < 2:
        return None
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    w_mvp = solve_mvp_long_only(cov_m, ridge=ridge)
    if w_mvp is None:
        w_mvp = np.full(n, 1.0 / n)
    var_mvp = float(w_mvp.dot(cov_m).dot(w_mvp))
    if not (math.isfinite(var_mvp) and var_mvp > 0):
        return None
    sigma_min = math.sqrt(var_mvp)
    diag = np.diag(cov_m)
    diag = diag[np.isfinite(diag) & (diag > 0)]
    sigma_max = max(sigma_min * 1.15, math.sqrt(float(np.max(diag))) * 1.05) if diag.size else sigma_min * 2
    if not (math.isfinite(sigma_max) and sigma_max > sigma_min):
        sigma_max = sigma_min * 2
    best_w, best_sharpe = None, -1e99
    for target_sigma in np.linspace(sigma_min, sigma_max, max(frontier_points, 5)):
        w = solve_frontier_long_only(mu_m, cov_m, float(target_sigma), ridge=ridge)
        if w is None:
            continue
        er = float(np.dot(mu_m, w))
        var = float(w.dot(cov_m).dot(w))
        if not (math.isfinite(er) and math.isfinite(var) and var > 0):
            continue
        sigma = math.sqrt(var)
        sharpe = (er - rf_m) / sigma
        if math.isfinite(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w
    return best_w

# ----------------------------------------------------------------------
# Monthly return matrix 

def build_monthly_returns(price_dict, min_months=12):
    """Return aligned monthly simple returns (columns = assets)."""
    series_list = []
    names = []
    for sym, df in price_dict.items():
        if "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 50:
            continue
        monthly = close.resample("ME").last().dropna()
        rets = monthly.pct_change().dropna()
        if len(rets) < min_months:
            continue
        if float(rets.abs().sum()) <= 0:
            continue
        series_list.append(rets)
        names.append(sym)
    if len(series_list) < 2:
        return None, None
    # intersection of dates
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.intersection(s.index)
    if len(idx) < min_months:
        return None, None
    cols, final_names = [], []
    for s, name in zip(series_list, names):
        aligned = s.reindex(idx)
        if aligned.isna().any():
            continue
        x = aligned.to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            continue
        cols.append(x)
        final_names.append(name)
    if len(cols) < 2:
        return None, None
    Rm = np.column_stack(cols)
    return Rm, final_names


# Backtest 

def run_uae_efficient_frontier(
    symbols=None,
    start_date="2015-01-01",
    cash=100_000.0,
    leverage=1.0,
    min_months=12,
    max_assets=20,
    ridge=1e-5,
    frontier_points=25,
    risk_free_rate_annual=0.0,
    use_tangency=True,
):
    if symbols is None:
        symbols = DUBAI_TOP20

    price_data = download_dubai_prices(symbols, start=start_date)
    if len(price_data) < 2:
        raise RuntimeError("Not enough Dubai stocks downloaded")

    # Build full monthly return matrix once
    Rm_full, names = build_monthly_returns(price_data, min_months=min_months)
    if Rm_full is None:
        raise RuntimeError("Could not build monthly returns")

    print(f"Universe after filters: {len(names)} stocks, {Rm_full.shape[0]} months")

    # Cap at max_assets
    if len(names) > max_assets:
        names = names[:max_assets]
        Rm_full = Rm_full[:, :max_assets]

    # Month-end dates
    # We need the actual month-end timestamps that correspond to the rows
    # Rebuild a clean monthly price frame for equity tracking
    monthly_prices = {}
    for sym in names:
        close = price_data[sym]["Close"].dropna()
        monthly_prices[sym] = close.resample("ME").last()

    common_idx = None
    for s in monthly_prices.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()

    capital = float(cash)
    equity_curve = []
    weights_history = []

    # Walk forward: at each month-end use only past data
    for t in range(min_months, len(common_idx)):
        # Past window for estimation (no look-ahead)
        window = Rm_full[:t]          # rows 0 … t-1
        if window.shape[0] < min_months:
            continue

        mu_m = np.mean(window, axis=0)
        cov_m = np.cov(window, rowvar=False)

        if use_tangency:
            w = solve_tangency_long_only(
                mu_m, cov_m,
                ridge=ridge,
                frontier_points=frontier_points,
                rf_annual=risk_free_rate_annual,
            )
            if w is None:
                w = solve_mvp_long_only(cov_m, ridge=ridge)
        else:
            w = solve_mvp_long_only(cov_m, ridge=ridge)

        if w is None or not np.all(np.isfinite(w)):
            continue

        # Apply leverage & normalise
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s <= 0:
            continue
        w = (w / s) * leverage

        # Realised return over next month
        if t + 1 >= len(common_idx):
            break
        next_ret = []
        for i, sym in enumerate(names):
            p0 = monthly_prices[sym].loc[common_idx[t]]
            p1 = monthly_prices[sym].loc[common_idx[t + 1]]
            if p0 > 0 and math.isfinite(p0) and math.isfinite(p1):
                next_ret.append(p1 / p0 - 1.0)
            else:
                next_ret.append(0.0)
        next_ret = np.array(next_ret)
        port_ret = float(np.dot(w, next_ret))
        capital *= (1.0 + port_ret)

        equity_curve.append({
            "date": common_idx[t + 1],
            "equity": capital,
            "port_ret": port_ret,
        })
        weights_history.append({"date": common_idx[t], **{names[i]: w[i] for i in range(len(names))}})

    eq = pd.DataFrame(equity_curve).set_index("date")
    return eq, pd.DataFrame(weights_history).set_index("date"), names

# ----------------------------------------------------------------------
# Main

if __name__ == "__main__":
    print("=" * 65)
    print("UAE / Dubai Efficient Frontier Strategy  –  yfinance version")
    print("Top 20 DFM stocks | Monthly rebalance | Long-only tangency / MVP")
    print("=" * 65)

    equity, weights, universe = run_uae_efficient_frontier(
        symbols=DUBAI_TOP20,
        start_date="2015-01-01",
        cash=100_000.0,
        leverage=1.0,
        min_months=12,
        max_assets=20,
        ridge=1e-5,
        frontier_points=25,
        risk_free_rate_annual=0.0,
        use_tangency=True,
    )

    e = equity["equity"]
    total = e.iloc[-1] / e.iloc[0] - 1
    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / max(years, 0.01)) - 1
    monthly_rets = e.pct_change().dropna()
    sharpe = (monthly_rets.mean() * 12) / (monthly_rets.std() * np.sqrt(12) + 1e-12)
    mdd = (e / e.cummax() - 1).min()

    print(f"\nUniverse used     : {len(universe)} stocks")
    print(f"Period            : {e.index[0].date()} → {e.index[-1].date()} ({years:.1f} yrs)")
    print(f"Total Return      : {total*100:7.2f}%")
    print(f"CAGR              : {cagr*100:7.2f}%")
    print(f"Sharpe (ann.)     : {sharpe:7.2f}")
    print(f"Max Drawdown      : {mdd*100:7.2f}%")
    print(f"Final Equity      : {e.iloc[-1]:,.0f}")

    # Save
    equity.to_csv("/home/workdir/artifacts/uae_ef_equity.csv")
    weights.to_csv("/home/workdir/artifacts/uae_ef_weights.csv")
    pd.Series(universe).to_csv("/home/workdir/artifacts/uae_ef_universe.csv", index=False, header=["symbol"])

    fig, ax = plt.subplots(figsize=(12, 5))
    e.plot(ax=ax, color="darkgreen", lw=1.5)
    ax.set_title("UAE Efficient Frontier (Top 20 Dubai) – Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/workdir/artifacts/uae_ef_equity.png", dpi=140)

    print("\nFiles saved:")
    print("  uae_ef_equity.csv / .png")
    print("  uae_ef_weights.csv")
    print("  uae_ef_universe.csv")
    print("Done.")
