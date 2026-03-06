"""
RISK MONITOR — CVaR, correlation breakdown, and regime-conditional risk
=========================================================================
Computes conditional risk metrics for a portfolio of tickers.

Usage:
    from ml.risk_monitor import compute_risk_metrics
    risk = compute_risk_metrics(['AAPL', 'NVDA', 'V', 'AMGN', 'ENEL.MI'],
                                weights={'AAPL': 0.08, 'NVDA': 0.09, ...})
"""

import numpy as np
import pandas as pd
import yfinance as yf


def compute_risk_metrics(tickers, weights=None, period='1y'):
    """
    Compute comprehensive risk metrics for a portfolio.

    Args:
        tickers: list of ticker symbols
        weights: dict {ticker: weight}. If None, equal weight.
        period: lookback period

    Returns:
        dict with:
            var_95, var_99: parametric VaR
            cvar_95: expected loss beyond VaR 95%
            max_drawdown: worst peak-to-trough
            correlation_alert: pairs with correlation > 0.7
            correlation_change: pairs where correlation changed >0.2 in 30d
            regime_cvar: CVaR estimated per regime (if HMM available)
    """
    # Fetch prices
    prices = {}
    for tk in tickers:
        try:
            h = yf.Ticker(tk).history(period=period)
            if not h.empty and len(h) > 30:
                prices[tk] = h['Close']
        except Exception:
            pass

    valid = [tk for tk in tickers if tk in prices]
    if len(valid) < 2:
        return {'error': f'Only {len(valid)} tickers with data'}

    price_df = pd.DataFrame(prices).ffill().dropna()
    returns = price_df.pct_change().dropna()

    n = len(valid)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.array([weights.get(tk, 1/n) for tk in valid])
        w = w / w.sum()  # normalize

    # Portfolio daily returns
    port_returns = (returns[valid].values @ w)

    # --- VaR (parametric) ---
    mu = port_returns.mean()
    sigma = port_returns.std()
    var_95 = mu - 1.645 * sigma
    var_99 = mu - 2.326 * sigma

    # --- CVaR (historical) ---
    sorted_ret = np.sort(port_returns)
    n_obs = len(sorted_ret)
    cutoff_95 = int(n_obs * 0.05)
    cvar_95 = float(sorted_ret[:cutoff_95].mean()) if cutoff_95 > 0 else float(var_95)

    # --- Max Drawdown ---
    cum = (1 + port_returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # --- Correlation analysis ---
    # Full period correlation
    corr_full = returns[valid].corr().values

    # Recent 30-day correlation
    recent = returns[valid].tail(30)
    corr_recent = recent.corr().values if len(recent) >= 20 else corr_full

    # Older correlation (60-90 days ago)
    older = returns[valid].iloc[-90:-30] if len(returns) > 90 else returns[valid].iloc[:60]
    corr_older = older.corr().values if len(older) >= 20 else corr_full

    # High correlations
    high_corrs = []
    corr_changes = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr_full[i][j])
            if abs(c) > 0.7:
                high_corrs.append({
                    'pair': f"{valid[i]}-{valid[j]}",
                    'correlation': round(c, 3),
                })
            # Correlation change
            delta = float(corr_recent[i][j] - corr_older[i][j])
            if abs(delta) > 0.15:
                corr_changes.append({
                    'pair': f"{valid[i]}-{valid[j]}",
                    'before': round(float(corr_older[i][j]), 3),
                    'now': round(float(corr_recent[i][j]), 3),
                    'change': round(delta, 3),
                })

    high_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    corr_changes.sort(key=lambda x: abs(x['change']), reverse=True)

    return {
        'tickers': valid,
        'var_95': round(float(var_95), 4),
        'var_99': round(float(var_99), 4),
        'cvar_95': round(float(cvar_95), 4),
        'max_drawdown': round(float(max_dd), 4),
        'daily_vol': round(float(sigma), 4),
        'annual_vol': round(float(sigma * np.sqrt(252)), 4),
        'high_correlations': high_corrs[:5],
        'correlation_changes': corr_changes[:5],
    }


if __name__ == '__main__':
    import json
    r = compute_risk_metrics(
        ['AAPL', 'NVDA', 'V', 'AMGN', 'ENEL.MI'],
        weights={'AAPL': 0.08, 'NVDA': 0.09, 'V': 0.05, 'AMGN': 0.38, 'ENEL.MI': 0.40}
    )
    print(json.dumps(r, indent=2))
