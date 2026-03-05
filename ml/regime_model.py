"""
EL MATEMÁTICO — Capa 1: Regime Detection + Trading Simulation
==================================================================
End-to-end script:
1. Load macro + price data from Supabase
2. Engineer features (market-level, no per-asset)
3. Label regimes by SPY forward 20d returns
4. Train LightGBM classifier (temporal split)
5. Simulate allocation strategy: equity/bonds/gold by predicted regime
6. Compare vs buy-and-hold SPY

Usage:
    python ml/regime_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


# ═══════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def load_and_prepare_data(start_date='2018-01-01', end_date=None):
    """Load macro + SPY/TLT/GLD prices from Supabase."""
    from db.database_manager import DatabaseManager
    db = DatabaseManager()

    # Macro (paginated — get_macro handles it)
    macro = db.get_macro(start_date=start_date, end_date=end_date)
    logger.info(f"Macro: {len(macro)} rows, {len(macro.columns)} columns")

    # Prices for benchmark assets (single ticker each)
    spy_df = db.get_prices('SPY', start_date=start_date, end_date=end_date)
    tlt_df = db.get_prices('TLT', start_date=start_date, end_date=end_date)
    gld_df = db.get_prices('GLD', start_date=start_date, end_date=end_date)

    spy = spy_df[['close']].rename(columns={'close': 'SPY'}) if not spy_df.empty else pd.DataFrame()
    tlt = tlt_df[['close']].rename(columns={'close': 'TLT'}) if not tlt_df.empty else pd.DataFrame()
    gld = gld_df[['close']].rename(columns={'close': 'GLD'}) if not gld_df.empty else pd.DataFrame()

    logger.info(f"SPY: {len(spy)} rows, TLT: {len(tlt)} rows, GLD: {len(gld)} rows")

    return macro, spy, tlt, gld


def engineer_features(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime-detection features from macro data.
    All features are backward-looking (no data leakage).
    """
    # Cast all columns to numeric (Supabase may return None/strings)
    for col in macro.columns:
        if col not in ('updated_at',):
            macro[col] = pd.to_numeric(macro[col], errors='coerce')

    feat = pd.DataFrame(index=macro.index)

    # ── Volatility features ──
    if 'vix' in macro.columns:
        feat['vix'] = macro['vix']
        feat['vix_ma20'] = macro['vix'].rolling(20).mean()
        feat['vix_std20'] = macro['vix'].rolling(20).std()
        feat['vix_zscore'] = (feat['vix'] - feat['vix_ma20']) / feat['vix_std20'].clip(lower=0.1)
        feat['vix_5d_chg'] = macro['vix'].pct_change(5)
        feat['vix_20d_chg'] = macro['vix'].pct_change(20)

    if 'vix_3m' in macro.columns and 'vix' in macro.columns:
        feat['vix_term'] = macro['vix'] - macro['vix_3m']  # >0 = backwardation = stress

    if 'vvix' in macro.columns:
        feat['vvix'] = macro['vvix']  # vol of vol

    # ── Yields / Rates ──
    if 'yield_10y' in macro.columns:
        feat['yield_10y'] = macro['yield_10y']
        feat['yield_10y_5d_chg'] = macro['yield_10y'].diff(5)
        feat['yield_10y_20d_chg'] = macro['yield_10y'].diff(20)

    if 'yield_spread_10y_2y' in macro.columns:
        feat['yield_spread'] = macro['yield_spread_10y_2y']
        feat['yield_spread_5d_chg'] = macro['yield_spread_10y_2y'].diff(5)
        feat['yield_curve_inverted'] = (macro['yield_spread_10y_2y'] < 0).astype(float)

    if 'fed_rate' in macro.columns:
        feat['fed_rate'] = macro['fed_rate']
        feat['fed_rate_60d_chg'] = macro['fed_rate'].diff(60)

    # ── Credit ──
    if 'credit_spread_bbb' in macro.columns:
        feat['credit_spread'] = macro['credit_spread_bbb']
        feat['credit_spread_5d_chg'] = macro['credit_spread_bbb'].diff(5)
        feat['credit_spread_20d_chg'] = macro['credit_spread_bbb'].diff(20)

    # ── Dollar ──
    if 'dxy' in macro.columns:
        feat['dxy'] = macro['dxy']
        feat['dxy_20d_chg'] = macro['dxy'].pct_change(20)

    # ── Commodities ──
    if 'oil_wti' in macro.columns:
        feat['oil_20d_chg'] = macro['oil_wti'].pct_change(20)
    if 'copper' in macro.columns and 'gold' in macro.columns:
        feat['copper_gold_ratio'] = macro['copper'] / macro['gold'].clip(lower=1)
        feat['copper_gold_20d_chg'] = feat['copper_gold_ratio'].pct_change(20)

    # ── Equity indices ──
    if 'sp500' in macro.columns:
        sp = macro['sp500']
        feat['sp500_ret_5d'] = sp.pct_change(5)
        feat['sp500_ret_20d'] = sp.pct_change(20)
        feat['sp500_ret_60d'] = sp.pct_change(60)
        feat['sp500_ma50'] = sp.rolling(50).mean()
        feat['sp500_above_ma50'] = (sp > feat['sp500_ma50']).astype(float)
        feat['sp500_drawdown'] = sp / sp.rolling(252).max() - 1  # drawdown from 1y high
        feat['sp500_vol_20d'] = sp.pct_change().rolling(20).std() * np.sqrt(252)

    if 'russell_2000' in macro.columns and 'sp500' in macro.columns:
        feat['small_vs_large'] = (macro['russell_2000'].pct_change(20) -
                                   macro['sp500'].pct_change(20))

    # ── Cross-asset momentum ──
    if 'nasdaq' in macro.columns and 'sp500' in macro.columns:
        feat['tech_vs_broad'] = (macro['nasdaq'].pct_change(20) -
                                  macro['sp500'].pct_change(20))

    # ── PMI / Economic ──
    for col in ['pmi_us', 'pmi_eu', 'pmi_jp', 'pmi_cn']:
        if col in macro.columns:
            feat[col] = macro[col]

    if 'unemp_us' in macro.columns:
        feat['unemp_us'] = macro['unemp_us']

    # Drop incomplete rows
    feat = feat.dropna(how='all')

    logger.info(f"Features: {len(feat)} rows × {len(feat.columns)} features")
    return feat


# ═══════════════════════════════════════════════════════
# 2. REGIME LABELING
# ═══════════════════════════════════════════════════════

def label_regimes(spy: pd.DataFrame, macro_index: pd.DatetimeIndex,
                  forward_days=20,
                  bull_threshold=0.02, bear_threshold=-0.02) -> pd.Series:
    """
    Label each day by SPY forward return:
      return > +2%  → 0 = risk-on  (bull)
      return < -2%  → 2 = risk-off (bear)
      else          → 1 = transition

    Reindexes SPY to macro_index (forward-fill) so labels align with features.
    """
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]

    # Reindex SPY to macro dates (forward-fill weekends/holidays)
    spy_aligned = spy.reindex(macro_index, method='ffill')

    forward_ret = spy_aligned.pct_change(forward_days).shift(-forward_days)

    labels = pd.Series(1, index=macro_index, name='regime')  # default: transition
    labels[forward_ret > bull_threshold] = 0    # risk-on
    labels[forward_ret < bear_threshold] = 2    # risk-off

    # Drop days where we can't compute forward return
    labels = labels[forward_ret.notna()]

    counts = labels.value_counts().sort_index()
    logger.info(f"Regime labels: risk-on={counts.get(0,0)}, "
                f"transition={counts.get(1,0)}, risk-off={counts.get(2,0)}")

    return labels


# ═══════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════

def train_regime_model(features: pd.DataFrame, labels: pd.Series,
                       train_end='2024-01-01'):
    """
    Train LightGBM classifier with temporal split.
    """
    import lightgbm as lgb
    from sklearn.metrics import classification_report, accuracy_score

    # Align
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx].copy()
    y = labels.loc[common_idx].copy()

    # Forward-fill sparse features (PMI=monthly, unemployment=monthly)
    X = X.ffill()
    # Drop rows still missing (early rows before first available data)
    X = X.dropna(axis=1, thresh=int(len(X) * 0.5))  # drop features with >50% NaN
    X = X.dropna()  # then drop remaining rows with any NaN
    y = y.loc[X.index]

    logger.info(f"After alignment: {len(X)} rows × {len(X.columns)} features")

    # Temporal split
    train_end_dt = pd.Timestamp(train_end)
    train_mask = X.index < train_end_dt
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    logger.info(f"Train: {len(X_train)} rows ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    logger.info(f"Test:  {len(X_test)} rows ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    # LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    model = lgb.train(
        params, train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    regime_names = {0: 'risk-on', 1: 'transition', 2: 'risk-off'}
    print(f"\n═══ TEST SET RESULTS ({X_test.index[0].date()} → {X_test.index[-1].date()}) ═══")
    print(f"  Accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred,
          target_names=[regime_names[i] for i in sorted(regime_names)]))

    # Feature importance
    imp = pd.Series(model.feature_importance(importance_type='gain'),
                    index=X.columns).sort_values(ascending=False)
    print("  Top 10 features:")
    for fname, fval in imp.head(10).items():
        print(f"    {fname:<30s} {fval:.0f}")

    return model, X_test, y_test, y_pred, y_pred_prob


# ═══════════════════════════════════════════════════════
# 4. TRADING SIMULATION
# ═══════════════════════════════════════════════════════

def simulate_regime_strategy(dates, predictions, spy, tlt, gld):
    """
    Simulate allocation strategy based on predicted regime.

    Allocations:
      risk-on (0):     80% SPY, 10% TLT, 10% GLD
      transition (1):  50% SPY, 30% TLT, 20% GLD
      risk-off (2):    20% SPY, 40% TLT, 40% GLD
    """
    allocations = {
        0: {'spy': 0.80, 'tlt': 0.10, 'gld': 0.10},  # risk-on
        1: {'spy': 0.50, 'tlt': 0.30, 'gld': 0.20},  # transition
        2: {'spy': 0.20, 'tlt': 0.40, 'gld': 0.40},  # risk-off
    }

    # Get daily returns for each asset
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    if isinstance(tlt, pd.DataFrame):
        tlt = tlt.iloc[:, 0]
    if isinstance(gld, pd.DataFrame):
        gld = gld.iloc[:, 0]

    spy_ret = spy.pct_change().fillna(0)
    tlt_ret = tlt.pct_change().fillna(0)
    gld_ret = gld.pct_change().fillna(0)

    # Strategy equity curve
    equity_strategy = [1.0]
    equity_bh = [1.0]     # buy & hold SPY
    equity_6040 = [1.0]   # 60/40 SPY/TLT

    regime_log = []

    for i, date in enumerate(dates):
        if date not in spy_ret.index:
            continue

        regime = predictions[i]
        alloc = allocations[regime]

        # Daily return of strategy
        r_strategy = (alloc['spy'] * spy_ret.loc[date] +
                      alloc['tlt'] * tlt_ret.loc[date] +
                      alloc['gld'] * gld_ret.loc[date])

        r_bh = spy_ret.loc[date]
        r_6040 = 0.60 * spy_ret.loc[date] + 0.40 * tlt_ret.loc[date]

        equity_strategy.append(equity_strategy[-1] * (1 + r_strategy))
        equity_bh.append(equity_bh[-1] * (1 + r_bh))
        equity_6040.append(equity_6040[-1] * (1 + r_6040))

        regime_log.append({'date': date, 'regime': regime,
                           'spy_wt': alloc['spy'], 'r': r_strategy})

    equity_strategy = np.array(equity_strategy[1:])
    equity_bh = np.array(equity_bh[1:])
    equity_6040 = np.array(equity_6040[1:])

    def metrics(eq, name):
        T = len(eq)
        total_ret = eq[-1] / eq[0] - 1
        ann_ret = (1 + total_ret) ** (252 / max(T, 1)) - 1
        daily_ret = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])
        vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 0 else 0
        sharpe = ann_ret / vol if vol > 0 else 0
        max_dd = np.min(eq / np.maximum.accumulate(eq) - 1)
        return {
            'name': name,
            'total_return': total_ret,
            'annual_return': ann_ret,
            'volatility': vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }

    m_strat = metrics(equity_strategy, 'Regime Strategy')
    m_bh = metrics(equity_bh, 'Buy & Hold SPY')
    m_6040 = metrics(equity_6040, '60/40 SPY/TLT')

    print(f"\n═══ TRADING SIMULATION ═══")
    print(f"  Period: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")

    regime_counts = pd.Series(predictions).value_counts().sort_index()
    regime_names = {0: 'risk-on', 1: 'transition', 2: 'risk-off'}
    for r, count in regime_counts.items():
        print(f"  Predicted {regime_names[r]}: {count} days ({count/len(predictions):.0%})")

    print(f"\n  {'Metric':<20s} {'Regime':<14s} {'B&H SPY':<14s} {'60/40':<14s}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*14}")
    print(f"  {'Total return':<20s} {m_strat['total_return']:>+12.1%}  {m_bh['total_return']:>+12.1%}  {m_6040['total_return']:>+12.1%}")
    print(f"  {'Annual return':<20s} {m_strat['annual_return']:>+12.1%}  {m_bh['annual_return']:>+12.1%}  {m_6040['annual_return']:>+12.1%}")
    print(f"  {'Volatility':<20s} {m_strat['volatility']:>12.1%}  {m_bh['volatility']:>12.1%}  {m_6040['volatility']:>12.1%}")
    print(f"  {'Sharpe':<20s} {m_strat['sharpe']:>12.2f}  {m_bh['sharpe']:>12.2f}  {m_6040['sharpe']:>12.2f}")
    print(f"  {'Max drawdown':<20s} {m_strat['max_drawdown']:>12.1%}  {m_bh['max_drawdown']:>12.1%}  {m_6040['max_drawdown']:>12.1%}")

    return m_strat, m_bh, m_6040, regime_log


# ═══════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Capa 1: Regime Detection")
    print("═══════════════════════════════════════════════════════")

    # Load data
    print("\n[1/4] Loading data from Supabase...")
    macro, spy, tlt, gld = load_and_prepare_data(start_date='2018-01-01')

    # Engineer features
    print("\n[2/4] Engineering features...")
    features = engineer_features(macro)

    # Label regimes
    print("\n[3/4] Labeling regimes...")
    labels = label_regimes(spy, macro_index=features.index, forward_days=20)

    # Train model with temporal split
    print("\n[4/4] Training LightGBM...")
    model, X_test, y_test, y_pred, y_pred_prob = train_regime_model(
        features, labels, train_end='2024-01-01'
    )

    # Trading simulation on test period
    test_dates = X_test.index
    simulate_regime_strategy(test_dates, y_pred, spy, tlt, gld)

    # Also test: what if we used PERFECT regime knowledge? (upper bound)
    print("\n═══ PERFECT HINDSIGHT (upper bound) ═══")
    simulate_regime_strategy(test_dates, y_test.values, spy, tlt, gld)

    return model


if __name__ == '__main__':
    model = main()
