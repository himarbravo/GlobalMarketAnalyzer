"""
EL MATEMÁTICO — Macro vs Macro+Graph (PROPER GraphBuilder)
=============================================================
Uses the real graph_builder with all dimensional corrections.

Approach: load data ONCE, then walk-forward graph.build() (no Supabase).

Usage:
    python ml/graph_vs_macro_v2.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import roc_auc_score, r2_score

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

from ml.regime_model import engineer_features
from ml.multi_target import build_targets


def extract_graph_features(start_date='2018-01-01', rebuild_every=40):
    """Load once via graph_builder, then walk-forward build()."""
    from db.database_manager import DatabaseManager
    from core.graph_builder import GraphBuilder

    db = DatabaseManager()
    gb = GraphBuilder(db)

    print("  Loading data (one-time Supabase call)...")
    gb.load_data(start_date=start_date)
    print(f"  Loaded: {gb.N} assets, {len(gb.prices)} price days")

    # Initial build to populate returns
    print("  Initial graph build...")
    gb.build()
    print(f"  Returns: {len(gb.returns)} days")

    dates = gb.returns.index
    features = []
    rebuild_dates = dates[120::rebuild_every]
    total = len(rebuild_dates)
    print(f"  Walk-forward: {total} rebuilds (every {rebuild_every}d)...")

    for i, ref_date in enumerate(rebuild_dates):
        try:
            gb.build(reference_date=str(ref_date.date()))
        except Exception as e:
            continue

        eig = gb.eigenvalues
        eig_pos = eig[eig > 1e-10]
        eig_sum = eig.sum() if eig.sum() > 0 else 1

        # Entropy
        p = eig_pos / eig_pos.sum() if len(eig_pos) > 0 else np.array([1])
        entropy = -np.sum(p * np.log(p + 1e-15))

        # Concentration
        top1 = eig[-1] / eig_sum
        top3 = eig[-3:].sum() / eig_sum if len(eig) >= 3 else top1

        # s (fractional exponent)
        s = float(gb.s) if hasattr(gb, 's') and gb.s is not None else 1.0

        # Connectivity from W
        W = gb.W
        upper = W[np.triu_indices_from(W, k=1)]
        mean_w = np.mean(np.abs(upper))
        neg_frac = np.mean(upper < 0) if len(upper) > 0 else 0

        # Spectral gap
        nonzero = eig[eig > 1e-6]
        gap = nonzero[0] if len(nonzero) > 0 else 0.0

        features.append({
            'date': ref_date,
            'gb_entropy': entropy,
            'gb_top1': top1,
            'gb_top3': top3,
            'gb_eff_dim': np.exp(entropy),
            'gb_s': s,
            'gb_mean_w': mean_w,
            'gb_neg_frac': neg_frac,
            'gb_gap': gap,
        })

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"    {i+1}/{total} ({ref_date.date()})")

    gf = pd.DataFrame(features).set_index('date')
    print(f"  Done: {len(gf)} graph snapshots × {len(gf.columns)} features")

    # Also fetch macro + ETFs for comparison
    macro = db.get_macro(start_date=start_date)
    spy = db.get_prices('SPY', start_date=start_date)
    tlt = db.get_prices('TLT', start_date=start_date)
    gld = db.get_prices('GLD', start_date=start_date)

    spy = spy[['close']].rename(columns={'close': 'SPY'}) if not spy.empty else pd.DataFrame()
    tlt = tlt[['close']].rename(columns={'close': 'TLT'}) if not tlt.empty else pd.DataFrame()
    gld = gld[['close']].rename(columns={'close': 'GLD'}) if not gld.empty else pd.DataFrame()

    return gf, macro, spy, tlt, gld


def train_single(X_tr, y_tr, X_te, y_te, is_reg=False):
    params = {
        'objective': 'regression' if is_reg else 'binary',
        'metric': 'mae' if is_reg else 'binary_logloss',
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 1.0, 'verbose': -1, 'seed': 42,
    }
    ds_t = lgb.Dataset(X_tr, label=y_tr)
    ds_v = lgb.Dataset(X_te, label=y_te, reference=ds_t)
    mdl = lgb.train(params, ds_t, num_boost_round=500, valid_sets=[ds_v],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    yp = mdl.predict(X_te)
    imp = pd.Series(mdl.feature_importance(importance_type='gain'),
                    index=X_tr.columns).sort_values(ascending=False)
    if is_reg:
        return {'r2': r2_score(y_te, yp), 'it': mdl.best_iteration, 'top': list(imp.head(3).index)}
    try:
        auc = roc_auc_score(y_te, yp)
    except:
        auc = 0.5
    return {'auc': auc, 'it': mdl.best_iteration, 'top': list(imp.head(3).index)}


def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Macro vs GraphBuilder (corrected)")
    print("═══════════════════════════════════════════════════════")

    print("\n[1/4] Graph walk-forward...")
    gf, macro, spy, tlt, gld = extract_graph_features(rebuild_every=40)

    print("\n[2/4] Macro features...")
    for c in macro.columns:
        if c != 'updated_at':
            macro[c] = pd.to_numeric(macro[c], errors='coerce')
    mf = engineer_features(macro)

    print("\n[3/4] Targets...")
    targets = build_targets(macro, spy, tlt, gld, mf.index)

    print("\n[4/4] Training: Macro vs Macro+GraphBuilder...")
    te = pd.Timestamp('2024-01-01')
    combined = mf.join(gf, how='left').ffill()

    results = {}
    for tgt in targets.columns:
        y = targets[tgt]
        is_reg = tgt.startswith('vol_')

        # Macro only
        ci = mf.index.intersection(y.dropna().index)
        Xm = mf.loc[ci].ffill().dropna(axis=1, thresh=int(len(ci)*0.5)).dropna()
        ym = y.loc[Xm.index]; mm = Xm.index < te
        if mm.sum() < 100 or (~mm).sum() < 50: continue
        rm = train_single(Xm[mm], ym[mm], Xm[~mm], ym[~mm], is_reg)

        # Macro + GraphBuilder
        ci2 = combined.index.intersection(y.dropna().index)
        Xg = combined.loc[ci2].ffill().dropna(axis=1, thresh=int(len(ci2)*0.5)).dropna()
        yg = y.loc[Xg.index]; mg = Xg.index < te
        if mg.sum() < 100 or (~mg).sum() < 50: continue
        rg = train_single(Xg[mg], yg[mg], Xg[~mg], yg[~mg], is_reg)

        results[tgt] = {'type': 'reg' if is_reg else 'bin', 'macro': rm, 'graph': rg}

    # Results
    print("\n" + "═" * 95)
    print("  ¿APORTA EL GRAFO CORREGIDO? — Macro vs Macro+GraphBuilder")
    print("═" * 95)

    vm = lambda r: r['macro']['auc'] if r['type']=='bin' else r['macro']['r2']
    vg = lambda r: r['graph']['auc'] if r['type']=='bin' else r['graph']['r2']

    print(f"\n  {'Target':<25s} {'M':<8s} {'Macro':<10s} {'+ Graph':<12s} "
          f"{'Δ':<8s} {'Veredicto':<12s} {'Top Graph Feature'}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*12} {'─'*25}")

    for n, r in sorted(results.items()):
        m = 'AUC' if r['type']=='bin' else 'R²'
        a, b = vm(r), vg(r); d = b - a
        gt = [f for f in r['graph']['top'] if f.startswith('gb_')]
        tf = gt[0] if gt else '—'
        v = "✅ MEJORA" if d > 0.02 else "⚠️ IGUAL" if d > -0.02 else "❌ PEOR"
        print(f"  {n:<25s} {m:<8s} {a:<10.3f} {b:<12.3f} {d:>+7.3f} {v:<12s} {tf}")

    imps = sum(1 for r in results.values() if vg(r) > vm(r) + 0.01)
    print(f"\n  Resultado: Grafo corregido mejora {imps}/{len(results)} targets")


if __name__ == '__main__':
    main()
