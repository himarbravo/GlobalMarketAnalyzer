"""
EL MATEMÁTICO — Market Timing Benchmark
==========================================
Uses tail risk model to rotate between assets.
Percentile-based threshold (adapts to signal distribution).

Usage:
    python ml/market_timing.py
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


def train_and_predict(X_train, y_train, X_test, y_test):
    """Train LightGBM, return predictions."""
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
    return preds, auc, mdl


def simulate(preds, dates, spy_ret, tlt_ret, gld_ret, label, defensive_pct=75):
    """
    Market timing using PERCENTILE threshold.
    When prediction > percentile threshold → go defensive.
    """
    threshold = np.percentile(preds, defensive_pct)

    equity = [1.0]
    equity_bh = [1.0]
    states = []

    for i, date in enumerate(dates):
        if date not in spy_ret.index:
            continue
        p = preds[i]
        sr = spy_ret[date]
        tr = tlt_ret[date]
        gr = gld_ret[date]

        if p > threshold:
            r = 0.20 * sr + 0.40 * tr + 0.40 * gr  # Defensive
            states.append('DEF')
        else:
            r = 0.90 * sr + 0.05 * tr + 0.05 * gr  # Risk-on
            states.append('ON')

        equity.append(equity[-1] * (1 + r))
        equity_bh.append(equity_bh[-1] * (1 + sr))

    eq = np.array(equity[1:])
    bh = np.array(equity_bh[1:])
    T = len(eq)
    years = T / 252

    if T < 10:
        return None

    # Strategy metrics
    ret_s = eq[-1] / eq[0] - 1
    ann_s = (1 + ret_s) ** (1 / years) - 1
    daily_ret_s = np.diff(equity[1:]) / eq[:-1] if len(eq) > 1 else np.array([0])
    vol_s = np.std(daily_ret_s) * np.sqrt(252)
    sharpe_s = ann_s / vol_s if vol_s > 0 else 0
    dd_s = np.min(eq / np.maximum.accumulate(eq) - 1)

    # B&H metrics
    ret_bh = bh[-1] / bh[0] - 1
    ann_bh = (1 + ret_bh) ** (1 / years) - 1
    daily_ret_bh = np.diff(equity_bh[1:]) / bh[:-1] if len(bh) > 1 else np.array([0])
    vol_bh = np.std(daily_ret_bh) * np.sqrt(252)
    sharpe_bh = ann_bh / vol_bh if vol_bh > 0 else 0
    dd_bh = np.min(bh / np.maximum.accumulate(bh) - 1)

    pct_def = states.count('DEF') / len(states) * 100

    return {
        'label': label, 'ann_s': ann_s, 'sharpe_s': sharpe_s,
        'vol_s': vol_s, 'dd_s': dd_s, 'ann_bh': ann_bh,
        'sharpe_bh': sharpe_bh, 'dd_bh': dd_bh, 'vol_bh': vol_bh,
        'pct_def': pct_def, 'threshold': threshold, 'T': T,
    }


def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Market Timing Benchmark")
    print("═══════════════════════════════════════════════════════")

    # Load
    print("\n[1/3] Loading data...")
    macro, spy_df, tlt_df, gld_df = load_and_prepare_data(start_date='2018-01-01')
    features = engineer_features(macro)
    targets = build_targets(macro, spy_df, tlt_df, gld_df, features.index)

    # Prepare
    y = targets['tail_risk']
    ci = features.index.intersection(y.dropna().index)
    X = features.loc[ci].ffill().dropna(axis=1, thresh=int(len(ci)*0.5)).dropna()
    y = y.loc[X.index]

    spy_s = spy_df.iloc[:, 0]; tlt_s = tlt_df.iloc[:, 0]; gld_s = gld_df.iloc[:, 0]
    spy_ret = spy_s.pct_change().fillna(0)
    tlt_ret = tlt_s.pct_change().fillna(0)
    gld_ret = gld_s.pct_change().fillna(0)

    # Train with temporal split
    te = pd.Timestamp('2024-01-01')
    mask = X.index < te
    X_train, y_train = X[mask], y[mask]
    X_test, y_test = X[~mask], y[~mask]

    print(f"\n[2/3] Training tail risk model...")
    print(f"  Train: {len(X_train)} days | Test: {len(X_test)} days")
    print(f"  Tail events: train {y_train.mean():.1%} | test {y_test.mean():.1%}")

    preds, auc, mdl = train_and_predict(X_train, y_train, X_test, y_test)
    print(f"  AUC: {auc:.3f}")
    print(f"  Preds: min={preds.min():.3f} median={np.median(preds):.3f} max={preds.max():.3f}")

    imp = pd.Series(mdl.feature_importance(importance_type='gain'),
                    index=X_train.columns).sort_values(ascending=False)
    print(f"  Top: {list(imp.head(3).index)}")

    # Simulate with different percentile thresholds
    print(f"\n[3/3] Simulating strategies...")
    print("\n" + "═" * 95)
    print("  MARKET TIMING: SPY → Defensive cuando P(tail) alto")
    print("═" * 95)
    print(f"\n  {'%ile':<6s} {'Thresh':<8s} {'Ann Ret':<10s} {'Sharpe':<10s} {'Vol':<10s} "
          f"{'MaxDD':<10s} {'%Def':<8s} {'vs B&H':<10s}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    best = None
    for pct in [50, 60, 70, 75, 80, 85, 90]:
        r = simulate(preds, X_test.index, spy_ret, tlt_ret, gld_ret,
                     f"p{pct}", defensive_pct=pct)
        if r is None:
            continue
        delta = r['ann_s'] - r['ann_bh']
        star = " ⭐" if best is None or r['sharpe_s'] > best['sharpe_s'] else ""
        print(f"  {pct:<6d} {r['threshold']:<8.3f} {r['ann_s']:>+9.1%} {r['sharpe_s']:>9.2f} "
              f"{r['vol_s']:>9.1%} {r['dd_s']:>9.1%} {r['pct_def']:>6.0f}% "
              f"{delta:>+9.1%}{star}")
        if best is None or r['sharpe_s'] > best['sharpe_s']:
            best = r

    if best:
        print(f"\n  B&H SPY:  {best['ann_bh']:>+9.1%}  {best['sharpe_bh']:>9.2f}  "
              f"{best['vol_bh']:>9.1%}  {best['dd_bh']:>9.1%}")

    # Volatile assets
    print("\n\n" + "═" * 95)
    print("  ACTIVOS VOLÁTILES — Mismo modelo, más volatilidad")
    print("═" * 95)

    from db.database_manager import DatabaseManager
    db = DatabaseManager()

    best_pct = 75  # use default good value
    print(f"\n  {'Activo':<8s} {'Timing':<10s} {'Sharpe':<10s} {'MaxDD':<10s} "
          f"{'B&H':<10s} {'B&H DD':<10s} {'Δ Ret':<10s} {'Δ DD':<10s} {'100€→'}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for ticker in ['SPY', 'QQQ', 'ARKK', 'MSTR', 'COIN', 'XLE', 'GDX', 'TQQQ', 'SOXL']:
        try:
            if ticker == 'SPY':
                close = spy_s
            else:
                p = db.get_prices(ticker, start_date='2018-01-01')
                if p.empty or len(p) < 100:
                    continue
                close = p['close'].rename(ticker)

            asset_ret = close.pct_change().fillna(0)
            r = simulate(preds, X_test.index, asset_ret, tlt_ret, gld_ret,
                         ticker, defensive_pct=best_pct)
            if r is None:
                continue

            # B&H for this asset
            common = X_test.index.intersection(close.index)
            c = close.loc[common]
            bh_total = c.iloc[-1] / c.iloc[0] - 1
            bh_ann = (1 + bh_total) ** (252/len(common)) - 1

            d_ret = r['ann_s'] - r['ann_bh']
            d_dd = r['dd_s'] - r['dd_bh']
            euros = 100 * (1 + r['ann_s'])
            marker = " ✅" if r['sharpe_s'] > r['sharpe_bh'] else ""
            print(f"  {ticker:<8s} {r['ann_s']:>+9.1%} {r['sharpe_s']:>9.2f} "
                  f"{r['dd_s']:>9.1%} {r['ann_bh']:>+9.1%} {r['dd_bh']:>9.1%} "
                  f"{d_ret:>+9.1%} {d_dd:>+9.1%} {euros:>6.0f}€{marker}")
        except Exception as e:
            print(f"  {ticker:<8s} Error: {e}")

    # Final summary
    if best:
        print(f"\n  ═══ Con 100€ en 1 año (SPY timing, umbral p{75}) ═══")
        print(f"  Timing: 100€ → {100*(1+best['ann_s']):.0f}€ (Sharpe {best['sharpe_s']:.2f})")
        print(f"  B&H:    100€ → {100*(1+best['ann_bh']):.0f}€ (Sharpe {best['sharpe_bh']:.2f})")

    return best


if __name__ == '__main__':
    main()
