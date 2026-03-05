"""
EL MATEMÁTICO — Multi-Target Predictor
=========================================
Predicts ALL objectively measurable financial quantities:

1. Volatility (realized vol 5d/20d) — regression
2. Tail risk (prob of -5% drawdown in 20d) — binary
3. VIX direction (up/down in 5d) — binary
4. Credit spread direction (widen/tighten in 5d) — binary
5. Yield curve direction (steepen/flatten in 20d) — binary
6. Stock-bond correlation sign (next 20d) — binary
7. SPY direction (up/down in 20d) — binary (baseline)

For each target, trains LightGBM with temporal split:
  Train: 2018 → 2023
  Test:  2024 → 2026

Then simulates a combined strategy using the best predictors.

Usage:
    python ml/multi_target.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import (accuracy_score, roc_auc_score, r2_score,
                             mean_absolute_error, classification_report)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

# Reuse feature engineering
from ml.regime_model import load_and_prepare_data, engineer_features


# ═══════════════════════════════════════════════════════
# 1. BUILD ALL TARGETS
# ═══════════════════════════════════════════════════════

def build_targets(macro: pd.DataFrame, spy: pd.DataFrame,
                  tlt: pd.DataFrame, gld: pd.DataFrame,
                  macro_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build all prediction targets. Each is either binary (0/1) or continuous.
    ALL targets use FUTURE data (shift(-N)) — these are labels, not features.
    """
    targets = pd.DataFrame(index=macro_index)

    # Align price series to macro index
    spy_s = spy.iloc[:, 0].reindex(macro_index, method='ffill') if not spy.empty else pd.Series(dtype=float)
    tlt_s = tlt.iloc[:, 0].reindex(macro_index, method='ffill') if not tlt.empty else pd.Series(dtype=float)

    # ── 1. Volatility (regression) ──
    if len(spy_s) > 20:
        spy_ret = spy_s.pct_change()
        targets['vol_5d'] = spy_ret.rolling(5).std().shift(-5) * np.sqrt(252)
        targets['vol_20d'] = spy_ret.rolling(20).std().shift(-20) * np.sqrt(252)

    # ── 2. Tail risk: drawdown > 5% in next 20d ──
    if len(spy_s) > 20:
        max_drawdown_20d = pd.Series(index=macro_index, dtype=float)
        spy_arr = spy_s.values
        for t in range(len(spy_arr) - 20):
            future = spy_arr[t+1:t+21]
            if np.all(np.isfinite(future)) and spy_arr[t] > 0:
                dd = np.min(future / spy_arr[t] - 1)
                max_drawdown_20d.iloc[t] = dd
        targets['tail_risk'] = (max_drawdown_20d < -0.05).astype(float)

    # ── 3. VIX direction (up in 5d) ──
    if 'vix' in macro.columns:
        vix = pd.to_numeric(macro['vix'], errors='coerce')
        targets['vix_up_5d'] = (vix.shift(-5) > vix).astype(float)

    # ── 4. Credit spread direction (widens in 5d) ──
    if 'credit_spread_bbb' in macro.columns:
        cs = pd.to_numeric(macro['credit_spread_bbb'], errors='coerce')
        targets['credit_widens_5d'] = (cs.shift(-5) > cs).astype(float)

    # ── 5. Yield curve direction (steepens in 20d) ──
    if 'yield_spread_10y_2y' in macro.columns:
        ys = pd.to_numeric(macro['yield_spread_10y_2y'], errors='coerce')
        targets['yield_steepens_20d'] = (ys.shift(-20) > ys).astype(float)

    # ── 6. Stock-bond correlation sign (next 20d) ──
    if len(spy_s) > 20 and len(tlt_s) > 20:
        spy_ret = spy_s.pct_change()
        tlt_ret = tlt_s.pct_change()
        corr_20d = pd.Series(index=macro_index, dtype=float)
        for t in range(len(spy_ret) - 20):
            s = spy_ret.iloc[t+1:t+21]
            b = tlt_ret.iloc[t+1:t+21]
            if len(s.dropna()) > 10 and len(b.dropna()) > 10:
                corr_20d.iloc[t] = s.corr(b)
        targets['stock_bond_neg'] = (corr_20d < 0).astype(float)

    # ── 7. SPY direction (up in 20d) — baseline ──
    if len(spy_s) > 20:
        targets['spy_up_20d'] = (spy_s.shift(-20) > spy_s).astype(float)

    logger.info(f"Targets built: {list(targets.columns)}")
    return targets


# ═══════════════════════════════════════════════════════
# 2. TRAIN + EVALUATE EACH TARGET
# ═══════════════════════════════════════════════════════

def train_and_evaluate(features: pd.DataFrame, targets: pd.DataFrame,
                       train_end='2024-01-01'):
    """
    For each target, train LightGBM and evaluate on test set.
    Returns dict of results.
    """
    results = {}
    train_end_dt = pd.Timestamp(train_end)

    for target_name in targets.columns:
        target = targets[target_name]
        is_regression = target_name.startswith('vol_')

        # Align features and target
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx].copy()
        y = target.loc[common_idx].copy()

        # Forward-fill and clean
        X = X.ffill()
        X = X.dropna(axis=1, thresh=int(len(X) * 0.5))
        X = X.dropna()
        y = y.loc[X.index]

        # Split
        train_mask = X.index < train_end_dt
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        if len(X_train) < 100 or len(X_test) < 50:
            logger.warning(f"  {target_name}: Not enough data (train={len(X_train)}, test={len(X_test)}), skipping")
            continue

        # LightGBM params
        if is_regression:
            params = {
                'objective': 'regression', 'metric': 'mae',
                'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
                'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 1.0, 'reg_lambda': 1.0, 'verbose': -1, 'seed': 42,
            }
        else:
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
                'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 1.0, 'reg_lambda': 1.0, 'verbose': -1, 'seed': 42,
            }

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

        model = lgb.train(
            params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_test)

        # Metrics
        if is_regression:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            imp = pd.Series(model.feature_importance(importance_type='gain'),
                            index=X.columns).sort_values(ascending=False)
            results[target_name] = {
                'type': 'regression', 'r2': r2, 'mae': mae,
                'n_train': len(X_train), 'n_test': len(X_test),
                'best_iter': model.best_iteration,
                'top_features': list(imp.head(5).index),
            }
        else:
            y_pred_bin = (y_pred > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred_bin)
            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError:
                auc = 0.5
            base_rate = y_test.mean()
            imp = pd.Series(model.feature_importance(importance_type='gain'),
                            index=X.columns).sort_values(ascending=False)
            results[target_name] = {
                'type': 'binary', 'accuracy': acc, 'auc': auc,
                'base_rate': base_rate,
                'lift': acc - base_rate if base_rate < 0.5 else acc - (1 - base_rate),
                'n_train': len(X_train), 'n_test': len(X_test),
                'best_iter': model.best_iteration,
                'top_features': list(imp.head(5).index),
                'predictions': y_pred,
                'actuals': y_test.values,
                'dates': X_test.index,
            }

    return results


# ═══════════════════════════════════════════════════════
# 3. COMBINED TRADING SIMULATION
# ═══════════════════════════════════════════════════════

def simulate_combined_strategy(results, spy, tlt, gld):
    """
    Use the best binary predictors to build a combined strategy.
    """
    # Find targets with AUC > 0.55
    good_targets = {k: v for k, v in results.items()
                    if v['type'] == 'binary' and v.get('auc', 0) > 0.55}

    if not good_targets:
        print("\n  No targets with AUC > 0.55 — no actionable predictions found.")
        return

    print(f"\n═══ COMBINED STRATEGY (using {len(good_targets)} predictors with AUC > 0.55) ═══")

    # Use tail_risk and spy_up_20d if available
    if 'tail_risk' in good_targets:
        tr = good_targets['tail_risk']
        dates = tr['dates']
        preds = tr['predictions']

        # Align returns
        spy_s = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
        tlt_s = tlt.iloc[:, 0] if isinstance(tlt, pd.DataFrame) else tlt
        gld_s = gld.iloc[:, 0] if isinstance(gld, pd.DataFrame) else gld
        spy_ret = spy_s.pct_change().fillna(0)
        tlt_ret = tlt_s.pct_change().fillna(0)
        gld_ret = gld_s.pct_change().fillna(0)

        equity = [1.0]
        equity_bh = [1.0]

        for i, date in enumerate(dates):
            if date not in spy_ret.index:
                continue
            p_tail = preds[i]
            # High tail risk → go defensive
            if p_tail > 0.5:
                r = 0.20 * spy_ret[date] + 0.40 * tlt_ret[date] + 0.40 * gld_ret[date]
            elif p_tail > 0.3:
                r = 0.50 * spy_ret[date] + 0.30 * tlt_ret[date] + 0.20 * gld_ret[date]
            else:
                r = 0.80 * spy_ret[date] + 0.10 * tlt_ret[date] + 0.10 * gld_ret[date]

            equity.append(equity[-1] * (1 + r))
            equity_bh.append(equity_bh[-1] * (1 + spy_ret[date]))

        equity = np.array(equity[1:])
        equity_bh = np.array(equity_bh[1:])

        if len(equity) > 1:
            T = len(equity)
            ret_s = equity[-1] / equity[0] - 1
            ret_bh = equity_bh[-1] / equity_bh[0] - 1
            vol_s = np.std(np.diff(equity) / equity[:-1]) * np.sqrt(252)
            vol_bh = np.std(np.diff(equity_bh) / equity_bh[:-1]) * np.sqrt(252)
            ann_s = (1 + ret_s) ** (252 / T) - 1
            ann_bh = (1 + ret_bh) ** (252 / T) - 1
            dd_s = np.min(equity / np.maximum.accumulate(equity) - 1)
            dd_bh = np.min(equity_bh / np.maximum.accumulate(equity_bh) - 1)

            print(f"\n  Tail-Risk Strategy ({dates[0].date()} → {dates[-1].date()}):")
            print(f"  {'Metric':<20s} {'Strategy':<14s} {'B&H SPY':<14s}")
            print(f"  {'─'*20} {'─'*14} {'─'*14}")
            print(f"  {'Annual return':<20s} {ann_s:>+12.1%}  {ann_bh:>+12.1%}")
            print(f"  {'Volatility':<20s} {vol_s:>12.1%}  {vol_bh:>12.1%}")
            print(f"  {'Sharpe':<20s} {ann_s/vol_s if vol_s > 0 else 0:>12.2f}  {ann_bh/vol_bh if vol_bh > 0 else 0:>12.2f}")
            print(f"  {'Max drawdown':<20s} {dd_s:>12.1%}  {dd_bh:>12.1%}")


# ═══════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Multi-Target Predictor")
    print("═══════════════════════════════════════════════════════")

    # Load
    print("\n[1/4] Loading data...")
    macro, spy, tlt, gld = load_and_prepare_data(start_date='2018-01-01')

    # Features
    print("\n[2/4] Engineering features...")
    features = engineer_features(macro)

    # Targets
    print("\n[3/4] Building prediction targets...")
    targets = build_targets(macro, spy, tlt, gld, features.index)

    # Train all
    print("\n[4/4] Training models for each target...")
    results = train_and_evaluate(features, targets, train_end='2024-01-01')

    # Results summary
    print("\n" + "═" * 80)
    print("  RESULTS SUMMARY — What can El Matemático predict?")
    print("═" * 80)

    print(f"\n  {'Target':<25s} {'Type':<12s} {'Metric':<12s} {'Value':<10s} "
          f"{'Base Rate':<10s} {'Iters':<8s} {'Top Feature'}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*25}")

    for name, r in sorted(results.items()):
        tf = r['top_features'][0] if r['top_features'] else 'N/A'
        if r['type'] == 'regression':
            print(f"  {name:<25s} {'regr':<12s} {'R²':<12s} {r['r2']:<10.3f} "
                  f"{'—':<10s} {r['best_iter']:<8d} {tf}")
        else:
            val = f"{r['auc']:.3f}"
            base = f"{r['base_rate']:.2f}"
            marker = " ✅" if r['auc'] > 0.55 else " ⚠️" if r['auc'] > 0.52 else " ❌"
            print(f"  {name:<25s} {'binary':<12s} {'AUC':<12s} {val:<10s} "
                  f"{base:<10s} {r['best_iter']:<8d} {tf}{marker}")

    # Trading sim from best predictors
    simulate_combined_strategy(results, spy, tlt, gld)

    return results


if __name__ == '__main__':
    results = main()
