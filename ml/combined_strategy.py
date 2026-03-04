"""
EL MATEMÁTICO — Combined Strategy
====================================
Timing macro (cuándo) + Momentum fundamental (qué comprar)

Lógica:
  1. Train tail risk model → P(tail) diario
  2. Compute momentum score → ranking trimestral de stocks
  3. Cada día:
     - Si P(tail) > umbral → 100% TLT/GLD (defensivo)
     - Si P(tail) < umbral → comprar TOP N stocks por momentum score
  4. Rebalancear momentum rankings cada trimestre

Comparaciones:
  A) B&H SPY
  B) Timing-only (SPY ↔ TLT)
  C) Momentum-only (TOP N stocks sin timing)
  D) Combined (TOP N + timing)

Usage:
    python ml/combined_strategy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

from ml.regime_model import load_and_prepare_data, engineer_features
from ml.multi_target import build_targets
from ml.fundamental_momentum import load_quarterly_data, compute_momentum_features
from db.database_manager import DatabaseManager


def train_tail_risk(features, targets, train_end='2024-01-01'):
    """Train tail risk model, return predictions on test set."""
    te = pd.Timestamp(train_end)
    y = targets['tail_risk']
    ci = features.index.intersection(y.dropna().index)
    X = features.loc[ci].ffill().dropna(axis=1, thresh=int(len(ci)*0.5)).dropna()
    y = y.loc[X.index]

    mask = X.index < te
    X_train, y_train = X[mask], y[mask]
    X_test, y_test = X[~mask], y[~mask]

    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 1.0, 'verbose': -1, 'seed': 42,
    }
    ds_t = lgb.Dataset(X_train, label=y_train)
    ds_v = lgb.Dataset(X_test, label=y_test, reference=ds_t)
    mdl = lgb.train(params, ds_t, num_boost_round=500, valid_sets=[ds_v],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    preds = mdl.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    return pd.Series(preds, index=X_test.index, name='tail_prob'), auc


def get_stock_returns(tickers, start_date='2024-01-01'):
    """Fetch daily returns for a list of tickers."""
    db = DatabaseManager()
    returns = {}
    for ticker in tickers:
        try:
            p = db.get_prices(ticker, start_date=start_date)
            if p.empty or len(p) < 60:
                continue
            returns[ticker] = p['close'].pct_change().fillna(0)
        except Exception:
            continue
    return returns


def simulate_combined(tail_probs, momentum_df, stock_returns,
                      tlt_ret, gld_ret, spy_ret,
                      top_n=5, defensive_pct=60):
    """
    Combined strategy:
    - When P(tail) < percentile threshold → equal-weight TOP N momentum stocks
    - When P(tail) > threshold → 50% TLT + 50% GLD
    """
    threshold = np.percentile(tail_probs.values, defensive_pct)

    # Top N by momentum score
    top_tickers = momentum_df.nlargest(top_n, 'momentum_score')['ticker'].tolist()
    # Filter to those we actually have returns for
    top_tickers = [t for t in top_tickers if t in stock_returns]

    if not top_tickers:
        return None

    equity_combined = [1.0]
    equity_timing = [1.0]
    equity_momentum = [1.0]
    equity_bh = [1.0]
    states = []
    dates_used = []

    for date in tail_probs.index:
        if date not in spy_ret.index:
            continue
        if date not in tlt_ret.index:
            continue

        p = tail_probs[date]
        sr = spy_ret[date]
        tr = tlt_ret[date]
        gr = gld_ret[date]

        # Momentum portfolio return (equal weight top N)
        stock_rets = []
        for t in top_tickers:
            if t in stock_returns and date in stock_returns[t].index:
                stock_rets.append(stock_returns[t][date])
        mom_ret = np.mean(stock_rets) if stock_rets else sr  # fallback to SPY

        is_defensive = p > threshold

        # Strategy A: B&H SPY
        equity_bh.append(equity_bh[-1] * (1 + sr))

        # Strategy B: Timing only (SPY ↔ TLT)
        if is_defensive:
            equity_timing.append(equity_timing[-1] * (1 + 0.5 * tr + 0.5 * gr))
        else:
            equity_timing.append(equity_timing[-1] * (1 + sr))

        # Strategy C: Momentum only (no timing)
        equity_momentum.append(equity_momentum[-1] * (1 + mom_ret))

        # Strategy D: COMBINED (momentum + timing)
        if is_defensive:
            r = 0.5 * tr + 0.5 * gr
            states.append('DEF')
        else:
            r = mom_ret
            states.append('MOM')
        equity_combined.append(equity_combined[-1] * (1 + r))
        dates_used.append(date)

    # Compute metrics for all strategies
    results = {}
    for name, eq_list in [('B&H SPY', equity_bh), ('Timing Only', equity_timing),
                           ('Momentum Only', equity_momentum), ('COMBINED', equity_combined)]:
        eq = np.array(eq_list[1:])
        T = len(eq)
        if T < 10:
            continue
        years = T / 252
        total_ret = eq[-1] / eq[0] - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1
        daily_r = np.diff(eq_list[1:]) / eq[:-1] if len(eq) > 1 else np.array([0])
        vol = np.std(daily_r) * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        dd = np.min(eq / np.maximum.accumulate(eq) - 1)

        results[name] = {
            'ann_ret': ann_ret, 'sharpe': sharpe, 'vol': vol,
            'max_dd': dd, 'total_ret': total_ret, 'T': T,
        }

    pct_def = states.count('DEF') / len(states) * 100 if states else 0

    return results, top_tickers, pct_def, threshold


def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Combined Strategy")
    print("  Timing macro + Momentum fundamental")
    print("═══════════════════════════════════════════════════════")

    # 1. Train tail risk model
    print("\n[1/5] Loading macro data + training tail risk model...")
    macro, spy_df, tlt_df, gld_df = load_and_prepare_data(start_date='2018-01-01')
    features = engineer_features(macro)
    targets = build_targets(macro, spy_df, tlt_df, gld_df, features.index)
    tail_probs, auc = train_tail_risk(features, targets)
    print(f"  Tail Risk AUC: {auc:.3f}")

    # 2. Compute momentum
    print("\n[2/5] Computing fundamental momentum scores...")
    qdf, _ = load_quarterly_data()
    mom = compute_momentum_features(qdf)
    print(f"  {len(mom)} tickers scored")

    # 3. Get stock returns for top momentum candidates
    print("\n[3/5] Fetching stock price data...")
    candidates = mom.nlargest(20, 'momentum_score')['ticker'].tolist()
    stock_rets = get_stock_returns(candidates, start_date='2024-01-01')
    print(f"  Got returns for {len(stock_rets)} tickers: {list(stock_rets.keys())}")

    # Prepare index returns
    spy_s = spy_df.iloc[:, 0]
    tlt_s = tlt_df.iloc[:, 0]
    gld_s = gld_df.iloc[:, 0]
    spy_ret = spy_s.pct_change().fillna(0)
    tlt_ret = tlt_s.pct_change().fillna(0)
    gld_ret = gld_s.pct_change().fillna(0)

    # 4. Simulate with different parameters
    print("\n[4/5] Simulating strategies...")

    # Test different portfolio sizes
    print("\n" + "═" * 95)
    print("  COMBINED STRATEGY: Momentum portfolio + Tail risk timing")
    print("═" * 95)

    for top_n in [3, 5, 10]:
        result = simulate_combined(
            tail_probs, mom, stock_rets, tlt_ret, gld_ret, spy_ret,
            top_n=top_n, defensive_pct=60
        )
        if result is None:
            continue

        results, tickers, pct_def, thresh = result

        print(f"\n  ── TOP {top_n} stocks: {tickers}")
        print(f"  ── Threshold: {thresh:.3f} | Defensive: {pct_def:.0f}% of days")
        print(f"\n  {'Strategy':<20s} {'Ann Ret':<10s} {'Sharpe':<10s} {'Vol':<10s} "
              f"{'MaxDD':<10s} {'100€→':<8s}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

        for name in ['B&H SPY', 'Timing Only', 'Momentum Only', 'COMBINED']:
            if name not in results:
                continue
            r = results[name]
            euros = 100 * (1 + r['ann_ret'])
            star = " ⭐" if name == 'COMBINED' else ""
            print(f"  {name:<20s} {r['ann_ret']:>+9.1%} {r['sharpe']:>9.2f} "
                  f"{r['vol']:>9.1%} {r['max_dd']:>9.1%} {euros:>6.0f}€{star}")

    # 5. Best configuration analysis
    print("\n\n[5/5] Sensitivity analysis — varying defensiveness...")
    print("\n" + "═" * 95)
    print("  SENSIBILIDAD: TOP 5 momentum + diferentes niveles de protección")
    print("═" * 95)

    print(f"\n  {'%Def':<8s} {'Combined':<10s} {'Sharpe':<10s} {'MaxDD':<10s} "
          f"{'vs B&H':<10s} {'vs Mom-only':<12s} {'100€→'}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")

    best = None
    for dpct in [0, 50, 60, 70, 80, 90]:
        result = simulate_combined(
            tail_probs, mom, stock_rets, tlt_ret, gld_ret, spy_ret,
            top_n=5, defensive_pct=dpct
        )
        if result is None:
            continue
        results, _, pct_def, _ = result
        c = results.get('COMBINED', {})
        bh = results.get('B&H SPY', {})
        mo = results.get('Momentum Only', {})
        if not c:
            continue

        d_bh = c['ann_ret'] - bh.get('ann_ret', 0)
        d_mo = c['ann_ret'] - mo.get('ann_ret', 0)
        euros = 100 * (1 + c['ann_ret'])
        star = ""
        if best is None or c['sharpe'] > best['sharpe']:
            best = c
            best['dpct'] = dpct
            star = " ⭐"
        print(f"  {dpct:<8d} {c['ann_ret']:>+9.1%} {c['sharpe']:>9.2f} "
              f"{c['max_dd']:>9.1%} {d_bh:>+9.1%} {d_mo:>+11.1%} {euros:>6.0f}€{star}")

    # Final summary
    if best:
        bh_ret = results.get('B&H SPY', {}).get('ann_ret', 0)
        print(f"\n  ═══════════════════════════════════════════════════")
        print(f"  RESUMEN FINAL — Con 100€ en 1 año")
        print(f"  ═══════════════════════════════════════════════════")
        print(f"  B&H SPY:       100€ → {100*(1+bh_ret):.0f}€")
        print(f"  COMBINED:      100€ → {100*(1+best['ann_ret']):.0f}€  "
              f"(Sharpe {best['sharpe']:.2f}, MaxDD {best['max_dd']:.1%})")
        print(f"  Ventaja:       {(best['ann_ret']-bh_ret):+.1%} retorno anual")


if __name__ == '__main__':
    main()
