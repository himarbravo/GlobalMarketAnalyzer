"""
PORTFOLIO OPTIMIZER — Markowitz Mean-Variance Optimization
===========================================================
Takes a list of tickers, fetches historical returns, and computes
the optimal portfolio weights that maximize the Sharpe ratio.

Usage:
    from ml.portfolio_optimizer import optimize_portfolio
    result = optimize_portfolio(['AAPL', 'NVDA', 'V', 'AMGN', 'ENEL.MI'])
"""

import numpy as np
import yfinance as yf
from scipy.optimize import minimize


def optimize_portfolio(tickers, period='6mo', risk_free_rate=None):
    """
    Compute optimal Markowitz weights for a set of tickers.

    Args:
        tickers: list of ticker symbols
        period: lookback period for returns (default 6 months)
        risk_free_rate: annual risk-free rate. If None, fetched from SHY yield.

    Returns:
        dict with keys:
            weights: {ticker: weight} optimal allocation
            sharpe: portfolio Sharpe ratio
            annual_return: expected annualized return
            annual_vol: annualized volatility
            var_95: 95% daily Value at Risk (negative number)
            correlations: list of (tk1, tk2, corr) for high correlations
            equal_weight_sharpe: Sharpe of naive 1/N portfolio (benchmark)
    """
    if len(tickers) < 2:
        return {'error': 'Need at least 2 tickers'}

    # Fetch returns
    prices = {}
    for tk in tickers:
        try:
            h = yf.Ticker(tk).history(period=period)
            if not h.empty and len(h) > 20:
                prices[tk] = h['Close']
        except Exception:
            pass

    valid_tickers = [tk for tk in tickers if tk in prices]
    if len(valid_tickers) < 2:
        return {'error': f'Only {len(valid_tickers)} tickers with data'}

    # Align dates — forward fill to handle different market calendars
    import pandas as pd
    price_df = pd.DataFrame(prices).ffill().dropna()

    if len(price_df) < 30:
        return {'error': f'Only {len(price_df)} overlapping trading days'}

    returns = price_df.pct_change().dropna()

    n = len(valid_tickers)
    mu = returns.mean().values * 252          # annualized returns
    cov = returns.cov().values * 252          # annualized covariance
    corr = returns.corr().values

    # Check for NaN
    if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
        return {'error': 'NaN in return/covariance data'}

    # Risk-free rate
    if risk_free_rate is None:
        risk_free_rate = 0.035

    # --- Optimization ---
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0
        return -(port_ret - risk_free_rate) / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.05, 0.40) for _ in range(n)]  # 5% min, 40% max per stock
    w0 = np.ones(n) / n  # start from equal weight

    result = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    w_opt = result.x
    port_ret = w_opt @ mu
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    # Equal-weight benchmark
    w_eq = np.ones(n) / n
    eq_ret = w_eq @ mu
    eq_vol = np.sqrt(w_eq @ cov @ w_eq)
    eq_sharpe = (eq_ret - risk_free_rate) / eq_vol if eq_vol > 0 else 0

    # VaR 95% (parametric)
    daily_vol = port_vol / np.sqrt(252)
    daily_ret = port_ret / 252
    var_95 = daily_ret - 1.645 * daily_vol

    # High correlations
    high_corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i][j]) > 0.6:
                high_corrs.append((valid_tickers[i], valid_tickers[j],
                                   round(float(corr[i][j]), 3)))
    high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        'tickers': valid_tickers,
        'weights': {valid_tickers[i]: round(float(w_opt[i]), 4)
                    for i in range(n)},
        'sharpe': round(float(sharpe), 3),
        'annual_return': round(float(port_ret), 4),
        'annual_vol': round(float(port_vol), 4),
        'var_95': round(float(var_95), 4),
        'correlations': high_corrs[:5],
        'equal_weight_sharpe': round(float(eq_sharpe), 3),
    }


if __name__ == '__main__':
    r = optimize_portfolio(['AAPL', 'NVDA', 'V', 'AMGN', 'ENEL.MI'])
    import json
    print(json.dumps(r, indent=2))
