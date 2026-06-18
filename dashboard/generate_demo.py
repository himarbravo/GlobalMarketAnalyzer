"""
Generate demo simulation: model signals vs realized prices + animated graph.
Downloads ~90 days of real prices via yfinance, simulates O-U signals,
and outputs a standalone HTML with interactive Plotly charts + vis-network graph.
"""
import json
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pathlib

LOOKBACK = 120
SIGNAL_WINDOW = 90
Z_ENTRY = 1.2

# ── Real ticker universe from config.py — 4 monetary zones + FX/macro ──────
# Only tickers reliably available on yfinance (exclude illiquid local exchanges)
UNIVERSE = [
    # USD zone — banks
    {"t": "JPM",       "zone": "USD",   "role": "bank",       "country": "US"},
    {"t": "BAC",       "zone": "USD",   "role": "bank",       "country": "US"},
    {"t": "GS",        "zone": "USD",   "role": "bank",       "country": "US"},
    {"t": "MS",        "zone": "USD",   "role": "bank",       "country": "US"},
    # USD zone — productive
    {"t": "AAPL",      "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "MSFT",      "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "NVDA",      "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "META",      "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "AMZN",      "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "V",         "zone": "USD",   "role": "productive", "country": "US"},
    {"t": "XOM",       "zone": "USD",   "role": "productive", "country": "US"},
    # USD zone — bonds / macro
    {"t": "TLT",       "zone": "USD",   "role": "bond",       "country": "US"},
    {"t": "GLD",       "zone": "USD",   "role": "commodity",  "country": "US"},
    {"t": "SPY",       "zone": "USD",   "role": "index",      "country": "US"},

    # EUR zone — banks
    {"t": "HSBC",      "zone": "EUR",   "role": "bank",       "country": "UK"},
    {"t": "SAN",       "zone": "EUR",   "role": "bank",       "country": "ES"},
    {"t": "ING",       "zone": "EUR",   "role": "bank",       "country": "NL"},
    {"t": "BBVA",      "zone": "EUR",   "role": "bank",       "country": "ES"},
    # EUR zone — productive
    {"t": "SAP",       "zone": "EUR",   "role": "productive", "country": "DE"},
    {"t": "ASML",      "zone": "EUR",   "role": "productive", "country": "NL"},
    {"t": "NVO",       "zone": "EUR",   "role": "productive", "country": "DK"},
    {"t": "AZN",       "zone": "EUR",   "role": "productive", "country": "UK"},
    {"t": "LVMHF",     "zone": "EUR",   "role": "productive", "country": "FR"},
    {"t": "EWG",       "zone": "EUR",   "role": "index",      "country": "DE"},

    # ASIA zone — banks
    {"t": "MUFG",      "zone": "ASIA",  "role": "bank",       "country": "JP"},
    {"t": "SMFG",      "zone": "ASIA",  "role": "bank",       "country": "JP"},
    # ASIA zone — productive
    {"t": "TSM",       "zone": "ASIA",  "role": "productive", "country": "TW"},
    {"t": "TM",        "zone": "ASIA",  "role": "productive", "country": "JP"},
    {"t": "SONY",      "zone": "ASIA",  "role": "productive", "country": "JP"},
    {"t": "BHP",       "zone": "ASIA",  "role": "productive", "country": "AU"},
    {"t": "BABA",      "zone": "ASIA",  "role": "productive", "country": "CN"},
    {"t": "EWJ",       "zone": "ASIA",  "role": "index",      "country": "JP"},

    # EM zone — banks
    {"t": "ITUB",      "zone": "EM",    "role": "bank",       "country": "BR"},
    {"t": "HDB",       "zone": "EM",    "role": "bank",       "country": "IN"},
    # EM zone — productive
    {"t": "PBR",       "zone": "EM",    "role": "productive", "country": "BR"},
    {"t": "VALE",      "zone": "EM",    "role": "productive", "country": "BR"},
    {"t": "EWZ",       "zone": "EM",    "role": "index",      "country": "BR"},
    {"t": "INDA",      "zone": "EM",    "role": "index",      "country": "IN"},
    {"t": "FXI",       "zone": "EM",    "role": "index",      "country": "CN"},

    # FX cross-zone connectors
    {"t": "EURUSD=X",  "zone": "FX",    "role": "fx",         "country": "FX"},
    {"t": "JPY=X",     "zone": "FX",    "role": "fx",         "country": "FX"},
    {"t": "DX-Y.NYB",  "zone": "FX",    "role": "fx",         "country": "FX"},

    # Macro — sovereign debt / rates proxies
    {"t": "^VIX",      "zone": "MACRO", "role": "macro",      "country": "US"},
    {"t": "^TNX",      "zone": "MACRO", "role": "macro",      "country": "US"},
]

TICKERS = [u["t"] for u in UNIVERSE]
TMETA = {u["t"]: u for u in UNIVERSE}

# Only generate O-U signals on equity/bond tickers
SIGNAL_TICKERS = [u["t"] for u in UNIVERSE
                  if u["role"] in ("bank", "productive", "bond", "commodity", "index")
                  and not u["t"].startswith("^")
                  and "=" not in u["t"]
                  and u["t"] not in ("SPY",)]

ROOT = pathlib.Path(__file__).parent.parent


def download_prices(tickers, days=LOOKBACK):
    end = datetime.today()
    start = end - timedelta(days=days)
    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    closes = raw["Close"].dropna(how="all")
    return closes


# ── Regime-adaptive Bayesian parameters (from config.py calibration) ────────
# These mirror the README table: γ·m'' + m' = -α·L^s·m + f(t) + Ω(t) + v(t)
REGIME_PARAMS = {
    #           alpha   gamma    s     S_fear  label
    "calm":   (0.020,  10.0,  0.90,   1.10,  "calm"),
    "normal": (0.030,   5.0,  0.70,   1.00,  "normal"),
    "stress": (0.045,   2.0,  0.45,   0.85,  "stress"),
    "panic":  (0.060,   1.0,  0.20,   0.65,  "panic"),
}

def get_regime(vix: float) -> str:
    if vix < 15:  return "calm"
    if vix < 25:  return "normal"
    if vix < 35:  return "stress"
    return "panic"


def compute_ou_signals(closes, window=20):
    """
    O-U signal generator with Bayesian-adaptive parameters.

    Equation (discrete, per spectral mode k):
        m_k(t+1) = m_k(t) · exp(-α · λ_k^s) + f_k/λ_k^s · (1 - exp(-α · λ_k^s))

    where:
        λ_k^s  = fractional Laplacian eigenvalue (connectivity degree ^ s)
        α      = diffusion speed   → adapts with VIX regime
        γ      = inertia           → dampens velocity term
        s      = fractional order  → 0.9 (local) → 0.2 (global panic)
        S_fear = VIX sentiment     → scales confidence and expected return

    δ = m_pred - m_real  is the mispricing signal.
    Signal fires when |δ| > threshold (which also adapts to S_fear).
    """
    # Extract VIX time series aligned to closes index
    vix_col = "^VIX"
    equity_tickers = [t for t in SIGNAL_TICKERS if t in closes.columns]
    recent_eq = closes[equity_tickers].iloc[-SIGNAL_WINDOW:]

    # VIX as Series aligned to same index
    if vix_col in closes.columns:
        vix_series = closes[vix_col].reindex(recent_eq.index).ffill().bfill()
    else:
        vix_series = pd.Series(20.0, index=recent_eq.index)

    signals = []
    prev_signal = {t: None for t in equity_tickers}

    for i in range(window, len(recent_eq)):
        date = recent_eq.index[i]
        date_str = date.strftime("%Y-%m-%d")

        vix_val = float(vix_series.iloc[i]) if not np.isnan(vix_series.iloc[i]) else 20.0
        regime = get_regime(vix_val)
        alpha, gamma, s, S_fear, _ = REGIME_PARAMS[regime]

        # Window slice for this step
        wdata = recent_eq.iloc[max(0, i - window):i]
        mu = wdata.mean()
        sigma = wdata.std().replace(0, np.nan)
        current = recent_eq.iloc[i]

        # Simplified correlation-based Laplacian degree per node
        corr = wdata.corr().fillna(0)

        # Velocity (dz/dt proxy) for γ term — needs previous step
        if i > 0:
            prev = recent_eq.iloc[i - 1]
            vel = (current - prev) / (sigma + 1e-9)
        else:
            vel = pd.Series(0.0, index=equity_tickers)

        for ticker in equity_tickers:
            price = current.get(ticker, np.nan)
            if np.isnan(price):
                continue

            z_raw = (current[ticker] - mu[ticker]) / (sigma[ticker] + 1e-9) \
                    if not np.isnan(sigma[ticker]) else 0.0
            if np.isnan(z_raw):
                continue

            # λ_k: effective Laplacian eigenvalue (graph degree of node)
            lambda_k = max(0.1, corr[ticker].abs().sum() - 1.0)
            # Fractional Laplacian: λ^s
            lambda_s = float(lambda_k ** s)

            # O-U mean reversion step in spectral space
            # m_pred = m_real · exp(-α·λ^s) + m_eq·(1 - exp(-α·λ^s))
            # δ = m_pred - m_real = -(m_real - m_eq)·(1 - exp(-α·λ^s))
            decay = float(np.exp(-alpha * lambda_s))
            delta_ou = -z_raw * (1.0 - decay)

            # γ momentum: inertia dampens δ when price is trending strongly
            v_i = float(vel.get(ticker, 0.0))
            momentum_term = gamma * v_i * 0.05
            delta_ou += momentum_term

            # Source term f_i: simple fundamental proxy (mean-reversion strength)
            # f_i ∝ distance from long-run mean (120-day vs 20-day)
            long_mu = closes[ticker].iloc[max(0, -LOOKBACK):-SIGNAL_WINDOW + i].mean() \
                      if ticker in closes.columns else mu[ticker]
            f_i = float((long_mu - current[ticker]) / (sigma[ticker] + 1e-9)) * 0.02 \
                  if not np.isnan(long_mu) else 0.0
            delta_ou += f_i

            # Confidence = p_reversion adjusted by S_fear (Bayesian posterior)
            half_life = max(2, int(np.log(2) / (alpha * lambda_s + 1e-9)))
            p_rev = min(0.92, 0.45 + abs(z_raw) * 0.12) * S_fear
            expected_ret = -z_raw * 0.018 * S_fear

            # Adaptive threshold in delta_ou space.
            # delta_ou ≈ -z*(1-exp(-α·λ^s)); typical scale ~0.02-0.15.
            # Base threshold proportional to decay; stricter in panic (S_fear↓).
            decay_base = 1.0 - float(np.exp(-alpha * max(0.3, lambda_s)))
            threshold = max(0.02, decay_base * Z_ENTRY * 0.6) / S_fear

            if delta_ou < -threshold and prev_signal[ticker] != "BUY":
                sig = "BUY"
                prev_signal[ticker] = "BUY"
            elif delta_ou > threshold and prev_signal[ticker] != "SELL":
                sig = "SELL"
                prev_signal[ticker] = "SELL"
            else:
                sig = None
                if abs(delta_ou) < threshold * 0.4:
                    prev_signal[ticker] = None

            if sig:
                signals.append({
                    "ticker": ticker,
                    "date": date_str,
                    "signal": sig,
                    "price": round(float(price), 2),
                    "z_score": round(float(z_raw), 3),
                    "delta_ou": round(float(delta_ou), 4),
                    "confidence": round(p_rev * 100, 1),
                    "expected_return_5d": round(expected_ret * 100, 2),
                    "half_life_days": half_life,
                    "regime": regime,
                    "alpha": round(alpha, 3),
                    "gamma": int(gamma),
                    "s": round(s, 2),
                    "S_fear": round(S_fear, 2),
                    "vix": round(vix_val, 1),
                    "lambda_s": round(lambda_s, 3),
                })

    return signals


def compute_realized_returns(signals, closes):
    enriched = []
    spy = closes["SPY"]
    for s in signals:
        ticker = s["ticker"]
        if ticker not in closes.columns:
            continue
        try:
            idx = closes.index.get_loc(pd.Timestamp(s["date"]))
        except KeyError:
            continue
        series = closes[ticker]
        p0 = series.iloc[idx]

        def fwd_ret(n):
            if idx + n < len(series):
                return round((series.iloc[idx + n] / p0 - 1) * 100, 2)
            return None

        def spy_ret(n):
            if idx + n < len(spy):
                return round((spy.iloc[idx + n] / spy.iloc[idx] - 1) * 100, 2)
            return None

        r5 = fwd_ret(5)
        win = None
        if r5 is not None:
            win = bool(r5 > 0) if s["signal"] == "BUY" else bool(r5 < 0)

        enriched.append({**s, "realized_5d": r5, "realized_10d": fwd_ret(10),
                         "spy_5d": spy_ret(5), "win": win})
    return enriched


def build_price_series(closes, tickers, days=SIGNAL_WINDOW):
    out = {}
    for t in tickers:
        if t in closes.columns:
            s = closes[t].iloc[-days:]
            out[t] = {
                "dates": [d.strftime("%Y-%m-%d") for d in s.index],
                "prices": [round(float(v), 2) for v in s.values],
            }
    return out


def build_bayesian_timeline(closes):
    """Time series of α, γ, s, S_fear, VIX regime for the Bayesian State chart."""
    vix_col = "^VIX"
    idx = closes.index[-SIGNAL_WINDOW:]
    if vix_col in closes.columns:
        vix_s = closes[vix_col].reindex(idx).ffill().bfill()
    else:
        vix_s = pd.Series(20.0, index=idx)

    records = []
    for date, vix_val in vix_s.items():
        v = float(vix_val) if not np.isnan(vix_val) else 20.0
        regime = get_regime(v)
        alpha, gamma, s, S_fear, _ = REGIME_PARAMS[regime]
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "vix": round(v, 1),
            "regime": regime,
            "alpha": alpha,
            "gamma": gamma,
            "s": s,
            "S_fear": S_fear,
        })
    return records


def build_graph_snapshots(closes, window=20, step=3):
    """
    Build graph snapshots every `step` trading days.
    Node color = δ_ou (O-U model divergence from reality), not raw z-score.
    Bayesian parameters (α, γ, s) adapt to VIX regime and are embedded per snapshot.
    """
    graph_tickers = [t for t in closes.columns if t in TMETA]
    equity_tickers = [t for t in graph_tickers if TMETA[t]["role"] not in ("fx", "macro", "index")]
    recent = closes[graph_tickers].iloc[-SIGNAL_WINDOW:]

    vix_col = "^VIX"
    if vix_col in closes.columns:
        vix_s = closes[vix_col].reindex(recent.index).ffill().bfill()
    else:
        vix_s = pd.Series(20.0, index=recent.index)

    snapshots = []
    for i in range(window, len(recent), step):
        date_str = recent.index[i].strftime("%Y-%m-%d")
        window_data = recent.iloc[max(0, i - window):i]

        # Bayesian regime for this timestep
        vix_val = float(vix_s.iloc[i]) if not np.isnan(vix_s.iloc[i]) else 20.0
        regime = get_regime(vix_val)
        alpha, gamma, s, S_fear, _ = REGIME_PARAMS[regime]

        mu = window_data.mean()
        sigma = window_data.std().replace(0, np.nan)
        current = recent.iloc[i]

        # Correlation → Laplacian degree per node
        eq_data = window_data[equity_tickers] if equity_tickers else window_data
        corr = eq_data.corr().fillna(0)

        # Velocity (momentum term for γ)
        prev = recent.iloc[i - 1] if i > 0 else current

        # Compute δ_ou per equity node (O-U model divergence from reality)
        delta_ou = {}
        for t in equity_tickers:
            if t not in current.index or np.isnan(current[t]):
                delta_ou[t] = 0.0
                continue
            sig_t = sigma.get(t, np.nan)
            if np.isnan(sig_t) or sig_t == 0:
                delta_ou[t] = 0.0
                continue
            z_raw = (current[t] - mu[t]) / sig_t
            if np.isnan(z_raw):
                delta_ou[t] = 0.0
                continue
            lambda_k = max(0.1, corr[t].abs().sum() - 1.0) if t in corr else 0.5
            lambda_s = float(lambda_k ** s)
            decay = float(np.exp(-alpha * lambda_s))
            d = -z_raw * (1.0 - decay)
            # γ momentum term
            v_i = float((current[t] - prev[t]) / sig_t) if t in prev.index else 0.0
            d += gamma * v_i * 0.05
            # f_i source term (long-run vs short-run equilibrium gap)
            long_mu = closes[t].iloc[:-SIGNAL_WINDOW].mean() if t in closes.columns else mu[t]
            if not np.isnan(long_mu):
                d += float((long_mu - current[t]) / sig_t) * 0.02
            delta_ou[t] = float(d) if not np.isnan(d) else 0.0

        # Edges — weighted by correlation
        # Rules:
        #   intra-zone: any pair with |corr| > 0.25
        #   cross-zone: at least one node must be bank, FX, or macro; |corr| > 0.40
        #   FX/macro nodes connect to all zones (they are global)
        all_edge_tickers = equity_tickers + [t for t in graph_tickers if TMETA[t]["role"] in ("fx", "macro")]
        full_corr = window_data[[t for t in all_edge_tickers if t in window_data.columns]].corr().fillna(0)

        edges = []
        for a_idx, a in enumerate(all_edge_tickers):
            for b in all_edge_tickers[a_idx + 1:]:
                za, zb = TMETA[a]["zone"], TMETA[b]["zone"]
                ra, rb = TMETA[a]["role"], TMETA[b]["role"]
                same_zone = za == zb
                # cross-zone only allowed via bank, fx, or macro nodes
                if not same_zone and not any(r in ("bank", "fx", "macro") for r in (ra, rb)):
                    continue
                c = full_corr.loc[a, b] if (a in full_corr.index and b in full_corr.columns) else 0.0
                if np.isnan(c):
                    c = 0.0
                threshold = 0.25 if same_zone else 0.40
                if abs(c) > threshold:
                    edges.append({
                        "from": a, "to": b,
                        "weight": round(float(c), 3),
                        "cross_zone": not same_zone,
                    })

        nodes = []
        for t in graph_tickers:
            m = TMETA[t]
            # For equity: use δ_ou; for FX/macro: use raw z-score
            if m["role"] in ("fx", "macro"):
                sig_t = sigma.get(t, 1.0)
                d = float((current.get(t, mu.get(t, 0)) - mu.get(t, 0)) / (sig_t if not np.isnan(sig_t) else 1.0))
                d = 0.0 if np.isnan(d) else d
            else:
                d = delta_ou.get(t, 0.0)
            nodes.append({
                "id": t,
                "delta": round(d, 3),       # O-U divergence (model vs reality)
                "price": round(float(current.get(t, 0)), 2),
                "zone": m["zone"],
                "role": m["role"],
                "country": m["country"],
            })

        snapshots.append({
            "date": date_str,
            "nodes": nodes,
            "edges": edges,
            "regime": regime,
            "alpha": round(alpha, 3),
            "gamma": int(gamma),
            "s": round(s, 2),
            "S_fear": round(S_fear, 2),
            "vix": round(vix_val, 1),
        })

    return snapshots


def build_html(signals, price_series, closes):
    spy_series = price_series.get("SPY", {})
    dates_portfolio = spy_series.get("dates", [])
    spy_prices = spy_series.get("prices", [])
    spy_norm = [round(100 * p / spy_prices[0], 2) for p in spy_prices] if spy_prices else []

    # Simple strategy equity curve
    port_curve_map = {}
    for sig in signals:
        if sig.get("realized_5d") is None or sig["date"] not in dates_portfolio:
            continue
        i = dates_portfolio.index(sig["date"])
        ret = sig["realized_5d"] / 100
        if sig["signal"] == "SELL":
            ret = -ret
        port_curve_map.setdefault(i, []).append(ret)

    v = 100.0
    port_vals = []
    for i in range(len(dates_portfolio)):
        rets = port_curve_map.get(i, [])
        if rets:
            v = v * (1 + sum(rets) / len(rets))
        port_vals.append(round(v, 2))

    graph_snapshots = build_graph_snapshots(closes)
    bayesian_timeline = build_bayesian_timeline(closes)

    cal_data = [(s["expected_return_5d"], s["realized_5d"], s["signal"],
                 s["ticker"], s["confidence"])
                for s in signals if s.get("realized_5d") is not None]

    win_count = sum(1 for s in signals if s.get("win") is True)
    total_w = sum(1 for s in signals if s.get("win") is not None)
    win_rate = round(win_count / total_w * 100, 1) if total_w else 0
    avg_conf = round(sum(s["confidence"] for s in signals) / len(signals), 1) if signals else 0
    tickers_no_spy = [u["t"] for u in UNIVERSE if u["role"] in ("bank", "productive", "bond", "commodity")
                      and not u["t"].startswith("^") and "=" not in u["t"]]

    wr_class = "green" if win_rate >= 50 else "red"
    n_tickers = len(tickers_no_spy)

    # Build HTML using explicit replacements to avoid f-string/JS brace conflicts
    template = open(ROOT / "dashboard" / "_demo_template.html").read()
    return (template
        .replace("__SIGNALS__", json.dumps(signals))
        .replace("__PRICES__", json.dumps(price_series))
        .replace("__SPY_NORM__", json.dumps(spy_norm))
        .replace("__PORT__", json.dumps(port_vals))
        .replace("__DATES_PORT__", json.dumps(dates_portfolio))
        .replace("__CAL__", json.dumps(cal_data))
        .replace("__TICKERS__", json.dumps(tickers_no_spy))
        .replace("__GRAPH__", json.dumps(graph_snapshots))
        .replace("__BAYES__", json.dumps(bayesian_timeline))
        .replace("__WIN_RATE__", str(win_rate))
        .replace("__WR_CLASS__", wr_class)
        .replace("__AVG_CONF__", str(avg_conf))
        .replace("__N_SIGNALS__", str(len(signals)))
        .replace("__N_TICKERS__", str(n_tickers))
        .replace("__Z_ENTRY__", str(Z_ENTRY))
    )


def main():
    print("Downloading prices...")
    closes = download_prices(TICKERS, LOOKBACK)
    print(f"Got {len(closes)} days for {len(closes.columns)} tickers")

    print("Computing O-U signals...")
    signals = compute_ou_signals(closes)
    print(f"Generated {len(signals)} signals")

    print("Computing realized returns...")
    signals = compute_realized_returns(signals, closes)

    price_series = build_price_series(closes, TICKERS, SIGNAL_WINDOW)

    print("Building graph snapshots...")
    # graph_snapshots computed inside build_html

    print("Building HTML...")
    html = build_html(signals, price_series, closes)

    out = ROOT / "output" / "demo_simulation.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"Saved: {out}")

    wins = sum(1 for s in signals if s.get("win") is True)
    total = sum(1 for s in signals if s.get("win") is not None)
    print(f"Win rate: {wins}/{total} = {wins/total*100:.1f}%" if total else "No outcomes yet")


if __name__ == "__main__":
    main()
