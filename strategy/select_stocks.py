"""
SELECT STOCKS — Quarterly momentum ranking
============================================
Computes fundamental momentum scores and selects TOP N stocks.
Run this after each earnings season (quarterly).

Usage:
    python strategy/select_stocks.py
    python strategy/select_stocks.py --top 5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from ml.fundamental_momentum import load_quarterly_data, compute_momentum_features


def main():
    parser = argparse.ArgumentParser(description='Select TOP N momentum stocks')
    parser.add_argument('--top', type=int, default=10, help='Number of stocks (default: 10)')
    args = parser.parse_args()

    print('═══════════════════════════════════════════════════════')
    print(f'  MOMENTUM RANKING — TOP {args.top}')
    print('═══════════════════════════════════════════════════════')

    print('\nLoading quarterly data from Supabase...')
    qdf, _ = load_quarterly_data()
    mom = compute_momentum_features(qdf)

    print(f'\n  {len(mom)} tickers scored')

    top = mom.nlargest(args.top, 'momentum_score')

    print(f'\n  {"#":<4} {"Ticker":<10} {"Score":>8} {"Rev QoQ":>10} {"EPS growth":>12} '
          f'{"Margin Δ":>10} {"ROIC trend":>11}')
    print(f'  {"─"*4} {"─"*10} {"─"*8} {"─"*10} {"─"*12} {"─"*10} {"─"*11}')

    for i, (_, r) in enumerate(top.iterrows()):
        print(f'  {i+1:<4} {r["ticker"]:<10} {r["momentum_score"]:>7.1f} '
              f'{r.get("rev_last_qoq", 0):>+9.1%} '
              f'{r.get("eps_growth_total", 0):>+11.1%} '
              f'{r.get("margin_delta", 0):>+9.2f} '
              f'{r.get("roic_trend", 0):>+10.4f}')

    tickers = top['ticker'].tolist()
    print(f'\n  → Copy for backtest.py:')
    print(f'  DEFAULT_TOP{args.top} = {tickers}')

    print(f'\n  → Copy for daily_signal.py:')
    print(f'  PORTFOLIO = {tickers}')


if __name__ == '__main__':
    main()
