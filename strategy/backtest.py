"""
BACKTEST — Momentum + VIX Gate
================================
Testea la estrategia sobre 5+ años de datos históricos.

Estrategia:
  - VIX < threshold → equal-weight TOP N momentum stocks
  - VIX ≥ threshold → 50% TLT + 50% GLD (refugio)

Usage:
    python strategy/backtest.py
    python strategy/backtest.py --vix 25 --top 5
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd


# Default TOP 10 momentum stocks (updated quarterly via select_stocks.py)
DEFAULT_TOP10 = ['AAPL', 'CRM', 'NVDA', 'AMGN', 'AMZN',
                 'AVGO', 'UPS', 'V', 'ASML', 'META']

CRISES = {
    'COVID Crash (Feb-Mar 2020)':       ('2020-02-15', '2020-04-01'),
    'COVID Recovery (Apr-Aug 2020)':     ('2020-04-01', '2020-09-01'),
    'Inflation Shock (Jan-Jun 2022)':    ('2022-01-01', '2022-07-01'),
    'Full 2022 Bear (Jan-Dec 2022)':     ('2022-01-01', '2023-01-01'),
    'SVB Banking Crisis (Mar 2023)':     ('2023-03-01', '2023-04-15'),
    'Oct 2023 Rate Scare':              ('2023-09-15', '2023-11-01'),
    'Aug 2024 Japan Selloff':           ('2024-07-15', '2024-08-30'),
    'Mar 2025 Tariff Shock':            ('2025-02-15', '2025-06-15'),
}


def load_data(tickers, start='2019-12-01'):
    """Download price data from yfinance."""
    all_t = list(set(tickers + ['SPY', 'TLT', 'GLD', '^VIX']))
    data = yf.download(all_t, start=start, progress=False)
    prices = data['Close'].ffill().dropna(how='all')
    returns = prices.pct_change().fillna(0)
    vix = prices['^VIX']
    return prices, returns, vix


def simulate(returns, vix, tickers, vix_threshold=20):
    """Run the strategy simulation."""
    avail = [t for t in tickers if t in returns.columns]
    if not avail:
        return None

    spy_r = returns['SPY']
    tlt_r = returns['TLT']
    gld_r = returns['GLD']
    mom_r = returns[avail].mean(axis=1)
    vix_s = vix.reindex(returns.index).ffill()

    eq_spy = [1.0]
    eq_mom = [1.0]
    eq_strat = [1.0]
    states = []

    for i, date in enumerate(returns.index):
        sr = spy_r.iloc[i]
        mr = mom_r.iloc[i]
        tr = tlt_r.iloc[i]
        gr = gld_r.iloc[i]
        v = vix_s.get(date, 20)

        eq_spy.append(eq_spy[-1] * (1 + sr))
        eq_mom.append(eq_mom[-1] * (1 + mr))

        if v >= vix_threshold:
            eq_strat.append(eq_strat[-1] * (1 + 0.5 * tr + 0.5 * gr))
            states.append('DEF')
        else:
            eq_strat.append(eq_strat[-1] * (1 + mr))
            states.append('MOM')

    return {
        'spy': np.array(eq_spy[1:]),
        'mom': np.array(eq_mom[1:]),
        'strat': np.array(eq_strat[1:]),
        'dates': returns.index,
        'states': states,
        'tickers': avail,
        'vix_threshold': vix_threshold,
    }


def compute_metrics(equity, years):
    """Compute standard performance metrics."""
    total = equity[-1] / equity[0] - 1
    ann = (1 + total) ** (1 / years) - 1
    daily = np.diff(np.concatenate([[1.0], equity])) / np.concatenate([[1.0], equity[:-1]])
    vol = np.std(daily) * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = np.min(equity / np.maximum.accumulate(equity) - 1)
    return {'total': total, 'ann': ann, 'sharpe': sharpe, 'vol': vol, 'maxdd': dd}


def print_results(result):
    """Print comprehensive backtest results."""
    years = len(result['dates']) / 252
    pct_def = result['states'].count('DEF') / len(result['states']) * 100

    print('\n═══════════════════════════════════════════════════════')
    print(f'  BACKTEST: Momentum + VIX>{result["vix_threshold"]} Gate')
    print(f'  TOP {len(result["tickers"])}: {result["tickers"]}')
    print(f'  Periodo: {result["dates"][0].date()} → {result["dates"][-1].date()} ({years:.1f} años)')
    print(f'  Tiempo en refugio: {pct_def:.0f}%')
    print('═══════════════════════════════════════════════════════')

    print(f'\n  {"Estrategia":<25} {"Ret total":>10} {"Ret/año":>10} {"Sharpe":>8} {"MaxDD":>8} {"100€→":>8}')
    print(f'  {"─"*25} {"─"*10} {"─"*10} {"─"*8} {"─"*8} {"─"*8}')

    for name, eq in [('B&H SPY', result['spy']),
                      ('Momentum puro', result['mom']),
                      ('Mom+VIX gate', result['strat'])]:
        m = compute_metrics(eq, years)
        star = ' ⭐' if name == 'Mom+VIX gate' else ''
        print(f'  {name:<25} {m["total"]:>+9.1%} {m["ann"]:>+9.1%} {m["sharpe"]:>7.2f} '
              f'{m["maxdd"]:>+7.1%} {100*(1+m["ann"]):>6.0f}€{star}')

    # Crisis breakdown
    print(f'\n  ── Comportamiento en crisis ──')
    returns_full = pd.DataFrame({
        'SPY': np.diff(np.concatenate([[1.0], result['spy']])) / np.concatenate([[1.0], result['spy'][:-1]]),
        'Mom': np.diff(np.concatenate([[1.0], result['mom']])) / np.concatenate([[1.0], result['mom'][:-1]]),
        'Strat': np.diff(np.concatenate([[1.0], result['strat']])) / np.concatenate([[1.0], result['strat'][:-1]]),
    }, index=result['dates'])

    for crisis_name, (cs, ce) in CRISES.items():
        cmask = (returns_full.index >= cs) & (returns_full.index <= ce)
        if cmask.sum() < 5:
            continue
        cr = returns_full[cmask]
        spy_c = (1 + cr['SPY']).cumprod()
        mom_c = (1 + cr['Mom']).cumprod()
        str_c = (1 + cr['Strat']).cumprod()

        spy_ret = spy_c.iloc[-1] - 1
        mom_ret = mom_c.iloc[-1] - 1
        str_ret = str_c.iloc[-1] - 1

        best = '⭐' if str_ret > spy_ret and str_ret > mom_ret else ''
        print(f'  {crisis_name:<35} SPY:{spy_ret:>+6.1%}  Mom:{mom_ret:>+6.1%}  '
              f'Strat:{str_ret:>+6.1%} {best}')


def main():
    parser = argparse.ArgumentParser(description='Backtest Momentum + VIX Gate')
    parser.add_argument('--vix', type=int, default=20, help='VIX threshold (default: 20)')
    parser.add_argument('--top', type=int, default=10, help='Number of stocks (default: 10)')
    parser.add_argument('--start', default='2019-12-01', help='Start date')
    args = parser.parse_args()

    tickers = DEFAULT_TOP10[:args.top]
    print(f'Loading data for {len(tickers)} tickers...')
    prices, returns, vix = load_data(tickers, start=args.start)

    mask = returns.index >= '2020-01-02'
    result = simulate(returns[mask], vix, tickers, vix_threshold=args.vix)

    if result:
        print_results(result)

        # Also test all VIX thresholds
        print(f'\n  ── Sensitivity: VIX thresholds ──')
        print(f'  {"Threshold":<12} {"Sharpe":>8} {"MaxDD":>8} {"Ret/año":>10}')
        print(f'  {"─"*12} {"─"*8} {"─"*8} {"─"*10}')
        years = len(returns[mask]) / 252
        for th in [0, 20, 25, 30, 35]:
            r = simulate(returns[mask], vix, tickers, vix_threshold=th)
            if r:
                m = compute_metrics(r['strat'], years)
                star = ' ⭐' if th == args.vix else ''
                label = 'Sin gate' if th == 0 else f'VIX>{th}'
                print(f'  {label:<12} {m["sharpe"]:>7.2f} {m["maxdd"]:>+7.1%} {m["ann"]:>+9.1%}{star}')


if __name__ == '__main__':
    main()
