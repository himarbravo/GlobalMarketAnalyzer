"""
BAYESIAN ONLINE CHANGEPOINT DETECTION (BOCD)
=============================================
Detects regime changes in real-time using the Adams & MacKay (2007) algorithm.
Does NOT require pre-specifying the number of regimes.

Usage:
    from ml.changepoint import detect_changepoints
    result = detect_changepoints()
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def _bocd(data, hazard_lambda=60, mu0=0, kappa0=1, alpha0=1, beta0=1):
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Args:
        data: 1D array of observations
        hazard_lambda: expected run length (higher = fewer changepoints)
        mu0, kappa0, alpha0, beta0: Normal-Inverse-Gamma prior params

    Returns:
        run_length_probs: (T x T) matrix of run-length probabilities
        changepoint_probs: (T,) array of P(changepoint at t)
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    changepoint_probs = np.zeros(T)

    for t in range(T):
        x = data[t]

        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred_probs = stats.t.pdf(x, df=df, loc=mu, scale=scale)

        H = 1.0 / hazard_lambda

        R[1:t + 2, t + 1] = R[:t + 1, t] * pred_probs * (1 - H)
        R[0, t + 1] = np.sum(R[:t + 1, t] * pred_probs * H)

        evidence = R[:t + 2, t + 1].sum()
        if evidence > 0:
            R[:t + 2, t + 1] /= evidence

        changepoint_probs[t] = R[0, t + 1]

        new_mu = np.append([mu0], (kappa * mu + x) / (kappa + 1))
        new_kappa = np.append([kappa0], kappa + 1)
        new_alpha = np.append([alpha0], alpha + 0.5)
        new_beta = np.append([beta0], beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1)))

        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    return R, changepoint_probs


def detect_changepoints(period='1y', threshold=0.15, hazard_lambda=60):
    """
    Detect regime changepoints in SPY using z-scored 5-day returns.

    Returns:
        dict with changepoints, run_length, current probability
    """
    spy = yf.Ticker('SPY').history(period=period)
    if spy.empty or len(spy) < 50:
        return {'error': 'Not enough SPY data'}

    close = spy['Close']

    # Z-score of 5-day returns (amplifies regime shifts)
    ret_5d = close.pct_change(5).dropna()
    rolling_mean = ret_5d.rolling(20).mean()
    rolling_std = ret_5d.rolling(20).std()
    z_score = ((ret_5d - rolling_mean) / rolling_std.replace(0, 1)).dropna()

    data = z_score.values
    dates = z_score.index
    prices = close.reindex(dates).values

    # Run BOCD
    _, cp_probs = _bocd(data, hazard_lambda=hazard_lambda)

    # Find changepoints
    changepoints = []
    for i in range(len(cp_probs)):
        if cp_probs[i] > threshold:
            if i >= 5 and i < len(data) - 5:
                before = data[max(0, i - 5):i].mean()
                after = data[i:min(len(data), i + 5)].mean()
                if before > 0 and after < 0:
                    desc = "Bull → Bear"
                elif before < 0 and after > 0:
                    desc = "Bear → Bull"
                elif after > before:
                    desc = "Aceleración alcista"
                else:
                    desc = "Aceleración bajista"
            else:
                desc = "Cambio detectado"

            changepoints.append({
                'date': str(dates[i].date()) if i < len(dates) else '?',
                'probability': round(float(cp_probs[i]), 3),
                'spy_price': round(float(prices[i]), 2) if i < len(prices) else 0,
                'description': desc,
            })

    current_cp_prob = float(cp_probs[-1]) if len(cp_probs) > 0 else 0

    # Run length: days since last changepoint
    last_cp_idx = 0
    for i in range(len(cp_probs) - 1, -1, -1):
        if cp_probs[i] > threshold:
            last_cp_idx = i
            break
    run_length = len(cp_probs) - last_cp_idx

    recent = [round(float(p), 3) for p in cp_probs[-20:]]

    return {
        'changepoints': changepoints[-10:],
        'total_changepoints': len(changepoints),
        'current_run_length': int(run_length),
        'changepoint_prob_now': round(current_cp_prob, 3),
        'recent_probs': recent,
    }


if __name__ == '__main__':
    import json
    print("Running BOCD on SPY (1 year)...")
    result = detect_changepoints()
    print(json.dumps(result, indent=2))
