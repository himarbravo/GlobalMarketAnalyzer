"""
WALK-FORWARD BACKTEST — VIX Gate on historical periods
========================================================
Step 1: Check data availability
Step 2: Run VIX gate strategy on all available periods

Usage:
    python strategy/walk_forward_backtest.py
"""
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import numpy as np
import pandas as pd
import sys

VIX_TH = 19

def main():
    # Step 1: What data is available?
    print("═══════════════════════════════════════════════════", flush=True)
    print("  STEP 1: Checking data availability", flush=True)
    print("═══════════════════════════════════════════════════", flush=True)

    assets = {
        'SPY':  'S&P 500 ETF',
        'TLT':  '20+ Year Treasury Bond',
        'GLD':  'Gold ETF',
        '^VIX': 'CBOE Volatility Index',
    }

    availability = {}
    for ticker, desc in assets.items():
        try:
            data = yf.download(ticker, start='1990-01-01', progress=False)
            if not data.empty:
                start = data.index[0].date()
                end = data.index[-1].date()
                n = len(data)
                availability[ticker] = {'start': start, 'end': end, 'n': n}
                print(f"  {ticker:<6} {desc:<30} {start} → {end} ({n} days)", flush=True)
            else:
                print(f"  {ticker:<6} {desc:<30} NO DATA", flush=True)
        except Exception as e:
            print(f"  {ticker:<6} {desc:<30} ERROR: {e}", flush=True)

    # Step 2: Run VIX gate on all overlapping periods
    print("\n═══════════════════════════════════════════════════", flush=True)
    print("  STEP 2: VIX Gate Backtest", flush=True)
    print("═══════════════════════════════════════════════════", flush=True)

    # Download core data
    data = yf.download(['SPY', 'TLT', 'GLD', '^VIX'], start='1993-01-01', progress=False)
    prices = data['Close'].ffill().dropna()
    returns = prices.pct_change().fillna(0)
    vix = prices['^VIX']

    # Find first date with ALL instruments
    first_all = returns.dropna().index[0]
    print(f"\n  First date with SPY+TLT+GLD+VIX: {first_all.date()}", flush=True)
    print(f"  Last date: {returns.index[-1].date()}", flush=True)
    years_total = (returns.index[-1] - first_all).days / 365.25
    print(f"  Total: {years_total:.1f} years available", flush=True)

    # Define test periods
    periods = [
        ('Post dot-com bear',   '2004-01-01', '2006-12-31'),
        ('Pre-GFC bull',        '2005-01-01', '2007-10-01'),
        ('GFC crash',           '2007-10-01', '2009-03-09'),
        ('GFC recovery',        '2009-03-09', '2011-01-01'),
        ('Euro debt scare',     '2011-04-01', '2012-01-01'),
        ('QE bull run',         '2012-01-01', '2014-12-31'),
        ('China/Oil scare',     '2015-06-01', '2016-03-01'),
        ('Low vol bull',        '2016-07-01', '2017-12-31'),
        ('Volmageddon+trade',   '2018-01-01', '2019-01-01'),
        ('Pre-COVID rally',     '2019-01-01', '2020-02-19'),
        ('COVID crash',         '2020-02-19', '2020-03-23'),
        ('COVID V-recovery',    '2020-03-23', '2020-12-31'),
        ('Meme stock mania',    '2021-01-01', '2021-12-31'),
        ('2022 bear market',    '2022-01-01', '2022-10-12'),
        ('2023 recovery',       '2022-10-12', '2023-07-31'),
        ('Rate scare Q3 2023',  '2023-08-01', '2023-10-31'),
        ('Soft landing rally',  '2023-11-01', '2024-07-31'),
        ('Japan selloff+recov', '2024-08-01', '2024-12-31'),
        ('Tariff shock 2025',   '2025-01-01', '2026-03-01'),
    ]

    print(f"\n  {'Periodo':<25} │{'SPY':>7} │{'Gate':>7} │{'  Δ':>6} │{'SPY DD':>7} │{'GatDD':>7} │{'VIX̄':>5} │{'%Ref':>4} │ Quién", flush=True)
    print(f"  {'─'*25} │{'─'*7} │{'─'*7} │{'─'*6} │{'─'*7} │{'─'*7} │{'─'*5} │{'─'*4} │ {'─'*6}", flush=True)

    wins_gate = 0
    wins_dd = 0
    total_p = 0

    for name, start, end in periods:
        mask = (returns.index >= start) & (returns.index <= end)
        if mask.sum() < 10:
            continue

        pr = returns[mask]
        pv = vix[mask]

        # B&H SPY
        spy_c = (1 + pr['SPY']).cumprod()
        spy_ret = spy_c.iloc[-1] - 1
        spy_dd = (spy_c / spy_c.cummax() - 1).min()

        # VIX gate
        eq = [1.0]
        n_def = 0
        for i in range(len(pr)):
            vi = pv.iloc[i]
            if vi >= VIX_TH:
                eq.append(eq[-1] * (1 + 0.5 * pr['TLT'].iloc[i] + 0.5 * pr['GLD'].iloc[i]))
                n_def += 1
            else:
                eq.append(eq[-1] * (1 + pr['SPY'].iloc[i]))
        eq = np.array(eq[1:])
        gate_ret = eq[-1] / eq[0] - 1
        gate_dd = (eq / np.maximum.accumulate(eq) - 1).min()

        delta = gate_ret - spy_ret
        vix_avg = pv.mean()
        pct_ref = n_def / len(pr) * 100

        total_p += 1
        ret_win = '✅' if delta > 0.01 else ('❌' if delta < -0.01 else '🟡')
        dd_win = '🛡️' if gate_dd > spy_dd + 0.01 else ''
        if delta > 0.01:
            wins_gate += 1
        if gate_dd > spy_dd + 0.01:
            wins_dd += 1

        print(f"  {name:<25} │{spy_ret:>+6.1%} │{gate_ret:>+6.1%} │{delta:>+5.1%} │{spy_dd:>+6.1%} │{gate_dd:>+6.1%} │{vix_avg:>4.0f} │{pct_ref:>3.0f}% │ {ret_win}{dd_win}", flush=True)

    print(f"\n  Gate gana retorno: {wins_gate}/{total_p} ({wins_gate/total_p*100:.0f}%)", flush=True)
    print(f"  Gate menor DD:    {wins_dd}/{total_p} ({wins_dd/total_p*100:.0f}%)", flush=True)

    # Full period
    print(f"\n  {'═'*55}", flush=True)
    print(f"  TOTAL ({years_total:.0f} años)", flush=True)
    print(f"  {'═'*55}", flush=True)

    mask_full = returns.index >= first_all
    rf = returns[mask_full]
    vf = vix[mask_full]

    spy_full = (1 + rf['SPY']).cumprod()
    eq_full = [1.0]
    for i in range(len(rf)):
        if vf.iloc[i] >= VIX_TH:
            eq_full.append(eq_full[-1] * (1 + 0.5 * rf['TLT'].iloc[i] + 0.5 * rf['GLD'].iloc[i]))
        else:
            eq_full.append(eq_full[-1] * (1 + rf['SPY'].iloc[i]))
    eq_full = np.array(eq_full[1:])

    for nm, eq in [('B&H SPY', spy_full.values), ('VIX>19 Gate', eq_full)]:
        t = eq[-1] / eq[0] - 1
        a = (1 + t) ** (1 / years_total) - 1
        d = np.diff(np.concatenate([[1.0], eq])) / np.concatenate([[1.0], eq[:-1]])
        sh = a / (np.std(d) * np.sqrt(252)) if np.std(d) > 0 else 0
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        s = ' ⭐' if 'Gate' in nm else ''
        print(f"  {nm:<15} {t:>+8.0%} ({a:>+5.1%}/yr)  Sharpe {sh:.2f}  MaxDD {dd:+.1%}  100€→{100*(1+t):.0f}€{s}", flush=True)


if __name__ == '__main__':
    main()
