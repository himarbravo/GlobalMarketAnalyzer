"""
EL MATEMÁTICO — Macro vs Macro+Graph Comparison
===================================================
Compares multi-target predictions using:
  A) Solo macro features (VIX, yields, etc.)
  B) Macro + Graph features (entropy, eigenvalues, z-score dispersion, etc.)

If (B) > (A) → the O-U equation adds predictive power.
If (B) ≈ (A) → the equation is redundant with raw macro.

Usage:
    python ml/graph_vs_macro.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

from ml.regime_model import load_and_prepare_data, engineer_features
from ml.multi_target import build_targets


# ═══════════════════════════════════════════════════════
# GRAPH FEATURES — extracted from rolling correlation matrices
# (same math as graph_builder, but computed on a rolling basis)
# ═══════════════════════════════════════════════════════

def compute_graph_features(start_date='2018-01-01', end_date=None,
                           window=60, step=5):
    """
    Compute graph-derived features on a rolling basis.
    Instead of rebuilding the full graph (slow), we compute
    the eigenvalue spectrum of the rolling correlation matrix.

    Returns DataFrame with graph features indexed by date.
    """
    from db.database_manager import DatabaseManager
    db = DatabaseManager()

    # Load prices for most liquid assets (faster than all 171)
    liquid = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'XLK', 'XLF',
              'XLE', 'XLV', 'XLI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
              'META', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX',
              'JNJ', 'PFE', 'UNH', 'DIS', 'NFLX', 'TSLA', 'BTC-USD', 'ETH-USD']

    print(f"  Loading prices for {len(liquid)} liquid assets...")
    all_prices = {}
    for ticker in liquid:
        try:
            df = db.get_prices(ticker, start_date=start_date, end_date=end_date)
            if not df.empty and 'close' in df.columns:
                all_prices[ticker] = df['close']
        except Exception:
            pass

    if len(all_prices) < 10:
        logger.warning(f"Only {len(all_prices)} tickers loaded, need at least 10")
        return pd.DataFrame()

    prices = pd.DataFrame(all_prices)
    prices = prices.ffill().dropna(axis=1, thresh=int(len(prices) * 0.5))
    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    N = returns.shape[1]
    print(f"  Returns matrix: {len(returns)} days × {N} assets")

    # Rolling graph features
    graph_features = []

    for t in range(window, len(returns), step):
        date = returns.index[t]
        ret_window = returns.iloc[t-window:t].values

        # Correlation matrix
        corr = np.corrcoef(ret_window.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Eigenvalues of correlation matrix
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending
        eigenvalues = np.clip(eigenvalues, 1e-10, None)

        # === GRAPH FEATURES ===

        # 1. Von Neumann entropy (same as reversibility.py)
        lam_pos = eigenvalues[eigenvalues > 1e-10]
        p = lam_pos / lam_pos.sum()
        entropy = -np.sum(p * np.log(p))

        # 2. Eigenvalue concentration (top mode explains how much?)
        top1_ratio = eigenvalues[0] / eigenvalues.sum()
        top3_ratio = eigenvalues[:3].sum() / eigenvalues.sum()

        # 3. Effective dimensionality (how many modes matter?)
        eff_dim = np.exp(entropy)  # e^S

        # 4. Mean correlation (overall connectedness)
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = np.mean(upper)

        # 5. Correlation dispersion (some very correlated, some not?)
        std_corr = np.std(upper)

        # 6. Market mode strength (1st eigenvalue vs RMT prediction)
        # Random matrix theory: largest eigenvalue ~ (1 + sqrt(N/T))^2
        T_w = window
        rmt_max = (1 + np.sqrt(N / T_w)) ** 2
        market_mode_excess = eigenvalues[0] / rmt_max

        # 7. Cross-sectional return dispersion
        last_day_ret = ret_window[-1]
        cs_dispersion = np.std(last_day_ret)

        # 8. Cross-sectional momentum (abs mean return)
        cs_momentum = np.mean(np.abs(last_day_ret))

        # 9. Rolling volatility of eigenvalues (graph stability)
        if t >= window + 20:
            ret_prev = returns.iloc[t-window-20:t-20].values
            corr_prev = np.corrcoef(ret_prev.T)
            corr_prev = np.nan_to_num(corr_prev, nan=0.0)
            eig_prev = np.sort(np.linalg.eigvalsh(corr_prev))[::-1]
            eig_prev = np.clip(eig_prev, 1e-10, None)
            p_prev = eig_prev[eig_prev > 1e-10] / eig_prev[eig_prev > 1e-10].sum()
            entropy_prev = -np.sum(p_prev * np.log(p_prev))
            delta_entropy = entropy - entropy_prev
        else:
            delta_entropy = 0.0

        graph_features.append({
            'date': date,
            'graph_entropy': entropy,
            'graph_top1_ratio': top1_ratio,
            'graph_top3_ratio': top3_ratio,
            'graph_eff_dim': eff_dim,
            'graph_mean_corr': mean_corr,
            'graph_std_corr': std_corr,
            'graph_market_excess': market_mode_excess,
            'graph_cs_dispersion': cs_dispersion,
            'graph_cs_momentum': cs_momentum,
            'graph_delta_entropy': delta_entropy,
        })

    gf = pd.DataFrame(graph_features).set_index('date')
    print(f"  Graph features: {len(gf)} rows × {len(gf.columns)} features")
    return gf


# ═══════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════

def train_single_target(X_train, y_train, X_test, y_test, is_regression=False):
    """Train LightGBM and return metrics."""
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

    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_test, label=y_test, reference=ds_train)
    model = lgb.train(params, ds_train, num_boost_round=500,
                      valid_sets=[ds_val],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    y_pred = model.predict(X_test)
    imp = pd.Series(model.feature_importance(importance_type='gain'),
                    index=X_train.columns).sort_values(ascending=False)

    if is_regression:
        return {'r2': r2_score(y_test, y_pred), 'best_iter': model.best_iteration,
                'top3': list(imp.head(3).index)}
    else:
        y_bin = (y_pred > 0.5).astype(int)
        try:
            auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            auc = 0.5
        return {'auc': auc, 'acc': accuracy_score(y_test, y_bin),
                'best_iter': model.best_iteration, 'top3': list(imp.head(3).index)}


def compare_macro_vs_graph(macro_features, graph_features, targets, train_end='2024-01-01'):
    """Compare each target with macro-only vs macro+graph features."""
    train_end_dt = pd.Timestamp(train_end)

    # Merge graph features into macro (forward-fill since computed every 5 days)
    combined = macro_features.join(graph_features, how='left').ffill()

    results = {}

    for target_name in targets.columns:
        target = targets[target_name]
        is_regression = target_name.startswith('vol_')

        # --- Macro only ---
        common = macro_features.index.intersection(target.dropna().index)
        Xm = macro_features.loc[common].ffill()
        Xm = Xm.dropna(axis=1, thresh=int(len(Xm) * 0.5)).dropna()
        ym = target.loc[Xm.index]

        mask_m = Xm.index < train_end_dt
        if mask_m.sum() < 100 or (~mask_m).sum() < 50:
            continue

        res_macro = train_single_target(
            Xm[mask_m], ym[mask_m], Xm[~mask_m], ym[~mask_m], is_regression)

        # --- Macro + Graph ---
        common_g = combined.index.intersection(target.dropna().index)
        Xg = combined.loc[common_g].ffill()
        Xg = Xg.dropna(axis=1, thresh=int(len(Xg) * 0.5)).dropna()
        yg = target.loc[Xg.index]

        mask_g = Xg.index < train_end_dt
        if mask_g.sum() < 100 or (~mask_g).sum() < 50:
            continue

        res_graph = train_single_target(
            Xg[mask_g], yg[mask_g], Xg[~mask_g], yg[~mask_g], is_regression)

        results[target_name] = {
            'type': 'regression' if is_regression else 'binary',
            'macro': res_macro,
            'graph': res_graph,
        }

    return results


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Macro vs Macro+Graph Comparison")
    print("═══════════════════════════════════════════════════════")

    # 1. Load data
    print("\n[1/5] Loading macro + prices...")
    macro, spy, tlt, gld = load_and_prepare_data(start_date='2018-01-01')

    # 2. Macro features
    print("\n[2/5] Engineering macro features...")
    macro_features = engineer_features(macro)

    # 3. Graph features (rolling eigenvalue analysis)
    print("\n[3/5] Computing graph features (rolling eigenvalue spectrum)...")
    graph_features = compute_graph_features(start_date='2018-01-01')

    # 4. Build targets
    print("\n[4/5] Building prediction targets...")
    targets = build_targets(macro, spy, tlt, gld, macro_features.index)

    # 5. Compare
    print("\n[5/5] Training models: Macro-only vs Macro+Graph...")
    results = compare_macro_vs_graph(macro_features, graph_features, targets)

    # Results
    print("\n" + "═" * 90)
    print("  ¿APORTA LA ECUACIÓN? — Macro vs Macro+Graph")
    print("═" * 90)

    metric_name = lambda r: 'AUC' if r['type'] == 'binary' else 'R²'
    val_macro = lambda r: r['macro']['auc'] if r['type'] == 'binary' else r['macro']['r2']
    val_graph = lambda r: r['graph']['auc'] if r['type'] == 'binary' else r['graph']['r2']

    print(f"\n  {'Target':<25s} {'Metric':<8s} {'Macro':<10s} {'Macro+Graph':<12s} "
          f"{'Δ':<8s} {'Veredicto':<12s} {'Top Graph Feature'}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*12} {'─'*25}")

    for name, r in sorted(results.items()):
        m = metric_name(r)
        vm = val_macro(r)
        vg = val_graph(r)
        delta = vg - vm

        # Check if any graph feature is in top 3
        graph_top = [f for f in r['graph']['top3'] if f.startswith('graph_')]
        top_gf = graph_top[0] if graph_top else '—'

        if delta > 0.02:
            verdict = "✅ MEJORA"
        elif delta > -0.02:
            verdict = "⚠️ IGUAL"
        else:
            verdict = "❌ PEOR"

        print(f"  {name:<25s} {m:<8s} {vm:<10.3f} {vg:<12.3f} "
              f"{delta:>+7.3f} {verdict:<12s} {top_gf}")

    # Summary
    improvements = sum(1 for r in results.values()
                       if val_graph(r) > val_macro(r) + 0.01)
    total = len(results)
    print(f"\n  Resumen: La ecuación mejora {improvements}/{total} targets")

    if improvements > total / 2:
        print("  → LA ECUACIÓN APORTA VALOR PREDICTIVO ✅")
    else:
        print("  → LA ECUACIÓN NO APORTA VALOR PREDICTIVO CLARO ⚠️")

    return results


if __name__ == '__main__':
    results = main()
