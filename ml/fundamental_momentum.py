"""
EL MATEMÁTICO — Fundamental Momentum
======================================
Calcula features de momentum fundamental por empresa:
- Revenue acceleration (QoQ growth trend)
- Margin improvement (operating margin Δ)
- EPS momentum (trend over quarters)  
- ROIC trend
- FCF improvement

Luego testea si las acciones con mejor momentum fundamental
superan en retorno a las que empeoran.

Usage:
    python ml/fundamental_momentum.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

from db.database_manager import DatabaseManager


def load_quarterly_data():
    """Load all quarterly fundamentals from Supabase."""
    db = DatabaseManager()
    res = db.client.table('fundamentals').select('*').order('report_date').execute()
    df = pd.DataFrame(res.data)

    if df.empty:
        raise ValueError("No fundamentals data found in Supabase")

    # Parse dates
    df['report_date'] = pd.to_datetime(df['report_date'])

    # Remove old snapshot rows (those without valid fiscal_quarter like "2025Q1")
    valid_q = df['fiscal_quarter'].str.match(r'^\d{4}Q[1-4]$', na=False)
    df = df[valid_q].copy()

    print(f"  Loaded: {len(df)} quarter-records for {df['ticker'].nunique()} tickers")
    return df, db


def compute_momentum_features(df):
    """
    For each ticker, compute momentum features from quarterly history.
    Returns a DataFrame with one row per ticker and momentum metrics.
    """
    results = []

    for ticker, group in df.groupby('ticker'):
        g = group.sort_values('report_date').copy()
        n = len(g)

        if n < 3:  # Need at least 3 quarters for meaningful momentum
            continue

        record = {'ticker': ticker, 'n_quarters': n}

        # === Revenue Momentum ===
        rev = pd.to_numeric(g['revenue'], errors='coerce')
        if rev.notna().sum() >= 3:
            rev_growth = rev.pct_change()
            record['rev_last_qoq'] = rev_growth.iloc[-1]  # Latest QoQ growth
            record['rev_avg_qoq'] = rev_growth.iloc[1:].mean()  # Avg QoQ growth
            record['rev_accel'] = rev_growth.iloc[-1] - rev_growth.iloc[-2] if n >= 3 else np.nan  # Acceleration
            record['rev_trend'] = np.polyfit(range(len(rev.dropna())), rev.dropna().values, 1)[0] / rev.dropna().mean()  # Normalized slope

        # === EPS Momentum ===
        eps = pd.to_numeric(g['eps'], errors='coerce')
        if eps.notna().sum() >= 3:
            record['eps_last'] = eps.iloc[-1]
            record['eps_first'] = eps.iloc[0]
            record['eps_growth_total'] = (eps.iloc[-1] / eps.iloc[0] - 1) if eps.iloc[0] != 0 else np.nan
            record['eps_trend'] = np.polyfit(range(len(eps.dropna())), eps.dropna().values, 1)[0]

        # === Margin Improvement ===
        for col, name in [('gross_margin', 'gm'), ('operating_margin', 'om'), ('net_margin', 'nm')]:
            m = pd.to_numeric(g[col], errors='coerce')
            if m.notna().sum() >= 2:
                record[f'{name}_latest'] = m.iloc[-1]
                record[f'{name}_delta'] = m.iloc[-1] - m.iloc[0]  # Total change
                record[f'{name}_improving'] = int(m.iloc[-1] > m.iloc[-2])  # Improved last quarter?

        # === ROIC Trend ===
        roic = pd.to_numeric(g['roic'], errors='coerce')
        if roic.notna().sum() >= 2:
            record['roic_latest'] = roic.iloc[-1]
            record['roic_delta'] = roic.iloc[-1] - roic.iloc[0]
            record['roic_trend'] = np.polyfit(range(len(roic.dropna())), roic.dropna().values, 1)[0]

        # === FCF Quality ===
        fcf = pd.to_numeric(g['free_cash_flow'], errors='coerce')
        rev_v = pd.to_numeric(g['revenue'], errors='coerce')
        if fcf.notna().sum() >= 2 and rev_v.notna().sum() >= 2:
            fcf_yield = fcf / rev_v.replace(0, np.nan)
            record['fcf_yield_latest'] = fcf_yield.iloc[-1]
            record['fcf_improving'] = int(fcf_yield.iloc[-1] > fcf_yield.iloc[-2]) if len(fcf_yield) >= 2 else np.nan

        # === Debt Trajectory ===
        debt = pd.to_numeric(g['debt_to_equity'], errors='coerce')
        if debt.notna().sum() >= 2:
            record['debt_delta'] = debt.iloc[-1] - debt.iloc[0]
            record['deleveraging'] = int(debt.iloc[-1] < debt.iloc[0])

        # === Composite Momentum Score ===
        score = 0
        count = 0
        if 'rev_last_qoq' in record and pd.notna(record.get('rev_last_qoq')):
            score += 1 if record['rev_last_qoq'] > 0 else -1
            count += 1
        if 'eps_growth_total' in record and pd.notna(record.get('eps_growth_total')):
            score += 1 if record['eps_growth_total'] > 0 else -1
            count += 1
        if 'om_improving' in record:
            score += 1 if record['om_improving'] else -1
            count += 1
        if 'roic_delta' in record and pd.notna(record.get('roic_delta')):
            score += 1 if record['roic_delta'] > 0 else -1
            count += 1
        if 'fcf_improving' in record and pd.notna(record.get('fcf_improving')):
            score += 1 if record['fcf_improving'] else -1
            count += 1
        if 'deleveraging' in record:
            score += 1 if record['deleveraging'] else -1
            count += 1

        record['momentum_score'] = score
        record['momentum_pct'] = score / count if count > 0 else 0
        record['last_quarter'] = g['report_date'].iloc[-1]

        results.append(record)

    return pd.DataFrame(results)


def backtest_momentum(momentum_df, db, lookback_months=6):
    """
    Simple backtest: after the last reporting quarter, do high-momentum stocks
    outperform low-momentum stocks over the next N months?
    """
    # Get forward returns for each ticker
    results = []

    for _, row in momentum_df.iterrows():
        ticker = row['ticker']
        last_q = row['last_quarter']

        try:
            prices = db.get_prices(ticker, start_date='2024-01-01')
            if prices.empty or len(prices) < 60:
                continue

            close = prices['close']
            # Find the closest trading day after last_quarter
            after = close[close.index >= last_q]
            if len(after) < 20:
                continue

            # Forward returns: 1m, 3m, 6m
            start_price = after.iloc[0]
            ret_1m = after.iloc[min(20, len(after)-1)] / start_price - 1 if len(after) > 20 else np.nan
            ret_3m = after.iloc[min(60, len(after)-1)] / start_price - 1 if len(after) > 60 else np.nan

            results.append({
                'ticker': ticker,
                'momentum_score': row['momentum_score'],
                'momentum_pct': row['momentum_pct'],
                'rev_last_qoq': row.get('rev_last_qoq'),
                'om_delta': row.get('om_delta'),
                'eps_growth_total': row.get('eps_growth_total'),
                'fwd_ret_1m': ret_1m,
                'fwd_ret_3m': ret_3m,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def main():
    print("═══════════════════════════════════════════════════════")
    print("  EL MATEMÁTICO — Fundamental Momentum")
    print("═══════════════════════════════════════════════════════")

    # 1. Load quarterly data
    print("\n[1/4] Loading quarterly fundamentals...")
    qdf, db = load_quarterly_data()

    # 2. Compute momentum features
    print("\n[2/4] Computing momentum features...")
    mom = compute_momentum_features(qdf)
    print(f"  Computed for {len(mom)} tickers")

    # 3. Display rankings
    print("\n" + "═" * 90)
    print("  TOP 15 MOMENTUM — Empresas mejorando")
    print("═" * 90)
    top = mom.nlargest(15, 'momentum_score')
    print(f"\n  {'Ticker':<8} {'Score':<7} {'Rev QoQ':<10} {'EPS Grow':<10} "
          f"{'OM Δ':<10} {'ROIC Δ':<10} {'FCF↑':<6} {'Delev':<6}")
    print(f"  {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*6}")

    for _, r in top.iterrows():
        rev = f"{r.get('rev_last_qoq', np.nan):+.1%}" if pd.notna(r.get('rev_last_qoq')) else "—"
        eps = f"{r.get('eps_growth_total', np.nan):+.1%}" if pd.notna(r.get('eps_growth_total')) else "—"
        om = f"{r.get('om_delta', np.nan):+.3f}" if pd.notna(r.get('om_delta')) else "—"
        roic = f"{r.get('roic_delta', np.nan):+.4f}" if pd.notna(r.get('roic_delta')) else "—"
        fcf = "✅" if r.get('fcf_improving') == 1 else "❌" if r.get('fcf_improving') == 0 else "—"
        dlv = "✅" if r.get('deleveraging') == 1 else "❌" if r.get('deleveraging') == 0 else "—"
        print(f"  {r['ticker']:<8} {r['momentum_score']:>5}   {rev:>9} {eps:>9} "
              f"{om:>9} {roic:>9}  {fcf:>4}  {dlv:>4}")

    print("\n" + "═" * 90)
    print("  BOTTOM 10 MOMENTUM — Empresas empeorando")
    print("═" * 90)
    bot = mom.nsmallest(10, 'momentum_score')
    print(f"\n  {'Ticker':<8} {'Score':<7} {'Rev QoQ':<10} {'EPS Grow':<10} "
          f"{'OM Δ':<10} {'ROIC Δ':<10}")
    print(f"  {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for _, r in bot.iterrows():
        rev = f"{r.get('rev_last_qoq', np.nan):+.1%}" if pd.notna(r.get('rev_last_qoq')) else "—"
        eps = f"{r.get('eps_growth_total', np.nan):+.1%}" if pd.notna(r.get('eps_growth_total')) else "—"
        om = f"{r.get('om_delta', np.nan):+.3f}" if pd.notna(r.get('om_delta')) else "—"
        roic = f"{r.get('roic_delta', np.nan):+.4f}" if pd.notna(r.get('roic_delta')) else "—"
        print(f"  {r['ticker']:<8} {r['momentum_score']:>5}   {rev:>9} {eps:>9} "
              f"{om:>9} {roic:>9}")

    # 4. Backtest: do momentum stocks outperform?
    print("\n\n[3/4] Backtesting momentum vs forward returns...")
    bt = backtest_momentum(mom, db)
    print(f"  Got forward returns for {len(bt)} tickers")

    if len(bt) > 10:
        print("\n" + "═" * 90)
        print("  ¿LAS EMPRESAS CON MEJOR MOMENTUM GENERAN MÁS RETORNO?")
        print("═" * 90)

        # Split into quintiles
        bt['quintile'] = pd.qcut(bt['momentum_score'], q=5, labels=['Q1 (worst)', 'Q2', 'Q3', 'Q4', 'Q5 (best)'],
                                  duplicates='drop')

        print(f"\n  {'Quintile':<15} {'N':<5} {'Fwd 1m':<12} {'Fwd 3m':<12} {'Avg MomScore'}")
        print(f"  {'─'*15} {'─'*5} {'─'*12} {'─'*12} {'─'*12}")

        for q in ['Q1 (worst)', 'Q2', 'Q3', 'Q4', 'Q5 (best)']:
            sub = bt[bt['quintile'] == q]
            if len(sub) == 0:
                continue
            r1m = sub['fwd_ret_1m'].mean()
            r3m = sub['fwd_ret_3m'].mean()
            ms = sub['momentum_score'].mean()
            print(f"  {q:<15} {len(sub):<5} {r1m:>+10.1%}  {r3m:>+10.1%}  {ms:>10.1f}")

        # Long-short spread
        q5 = bt[bt['quintile'] == 'Q5 (best)']['fwd_ret_1m'].mean()
        q1 = bt[bt['quintile'] == 'Q1 (worst)']['fwd_ret_1m'].mean()
        if pd.notna(q5) and pd.notna(q1):
            spread = q5 - q1
            print(f"\n  ═══ SPREAD Q5-Q1 (1 mes): {spread:+.1%} ═══")
            if spread > 0.02:
                print("  ✅ Momentum fundamental FUNCIONA — las empresas que mejoran superan a las que empeoran")
            elif spread > 0:
                print("  ⚠️  Señal débil — hay diferencia pero no muy significativa")
            else:
                print("  ❌ No hay edge — momentum fundamental no predice retornos forward")

    # 5. Correlation analysis
    print("\n\n[4/4] Correlación features vs retorno forward...")
    if len(bt) > 10:
        feat_cols = ['momentum_score', 'momentum_pct', 'rev_last_qoq', 'om_delta', 'eps_growth_total']
        for col in feat_cols:
            if col in bt.columns:
                corr_1m = bt[col].corr(bt['fwd_ret_1m'])
                corr_3m = bt[col].corr(bt['fwd_ret_3m'])
                if pd.notna(corr_1m):
                    sig = "✅" if abs(corr_1m) > 0.15 else "⚠️" if abs(corr_1m) > 0.05 else "❌"
                    print(f"  {col:<20} vs 1m: {corr_1m:+.3f}  vs 3m: {corr_3m:+.3f}  {sig}")

    return mom


if __name__ == '__main__':
    main()
