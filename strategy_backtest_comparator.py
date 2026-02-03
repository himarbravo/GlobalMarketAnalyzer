"""
STRATEGY BACKTESTING COMPARATOR
================================
Compara performance de las 3 estrategias en diferentes períodos:
- Value (Buffett)
- Growth (ARK)
- Risk-Off (Dalio)

Test scenarios:
1. COVID Crash (2020-03)
2. Bull Run 2023
3. Tech Selloff 2022
4. Overall performance 2020-2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from strategy_engine import StrategyEngine
from regime_detector import RegimeDetector
from correlation_monitor import CorrelationMonitor
import matplotlib.pyplot as plt

class StrategyBacktester:
    def __init__(self):
        self.results = []
        
    def backtest_period(self, start_date, end_date, scenario_name):
        """
        Backtest all 3 strategies for a given period.
        """
        print(f"\n{'='*70}")
        print(f"📅 {scenario_name}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"{'='*70}")
        
        # Download data
        tickers = ['SPY', 'NVDA', 'AAPL', 'MSFT', 'JPM', 'JNJ', 'TLT', 'GLD', 
                   'HYG', 'IEF', '^VIX', 'XLP', 'QQQ', 'TSLA', 'BTC-USD']
        
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Adj Close']
            
            data = data.ffill()
            
            if len(data) < 30:
                print(f"⚠️ Insufficient data ({len(data)} days)")
                return None
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
        
        # Determine regime
        spy_returns = data['SPY'].pct_change().dropna()
        regime_detector = RegimeDetector()
        regime_detector.train(spy_returns)
        regime_name, regime_conf, _ = regime_detector.predict_current_regime(spy_returns.tail(60))
        
        # Correlation
        corr_monitor = CorrelationMonitor(data)
        avg_corr_raw = corr_monitor.calculate_rolling_correlation()
        avg_corr = float(avg_corr_raw.iloc[-1] if hasattr(avg_corr_raw, 'iloc') else avg_corr_raw)
        
        # VIX
        vix = data['^VIX'].iloc[-1]
        
        # Initialize strategies
        engine = StrategyEngine(data)
        engine.load_strategies()
        engine.set_macro_state(regime=regime_name, vix=vix, correlation=avg_corr)
        
        # Test each strategy
        print(f"\nMacro: Regime={regime_name}, VIX={vix:.1f}, Corr={avg_corr:.2f}")
        print(f"\n{'Strategy':<15} {'Score':<8} {'Performance':<15} {'Notes'}")
        print("-" * 70)
        
        for strategy_name in ['value', 'growth', 'risk_off']:
            strategy = engine.strategies[strategy_name]
            
            # Score sample assets
            asset_scores = []
            for ticker in ['NVDA', 'AAPL', 'JPM', 'JNJ', 'TLT', 'GLD']:
                if ticker not in data.columns:
                    continue
                
                # Mock data
                fundamental = {'PE_Ratio': 20, 'Dividend_Yield': 2, 'Revenue_Growth': 15}
                technical = {'price': data[ticker].iloc[-1], 'volatility': 0.3, 'alpha': 5, 'beta': 1.0}
                
                score = strategy.score_asset(ticker, fundamental, technical, engine.macro_state)
                asset_scores.append((ticker, score))
            
            # Calculate "portfolio return" (weighted by score)
            total_score = sum([s[1] for s in asset_scores])
            weighted_return = 0
            
            for ticker, score in asset_scores:
                weight = score / total_score if total_score > 0 else 0
                ticker_return = (data[ticker].iloc[-1] / data[ticker].iloc[0] - 1) * 100
                weighted_return += weight * ticker_return
            
            # Store result
            result = {
                'scenario': scenario_name,
                'strategy': strategy.name,
                'regime': regime_name,
                'vix': vix,
                'correlation': avg_corr,
                'portfolio_return_pct': weighted_return,
                'avg_score': total_score / len(asset_scores) if asset_scores else 0
            }
            
            self.results.append(result)
            
            # Print
            perf_str = f"{weighted_return:+.1f}%"
            note = ""
            if strategy_name == 'value' and regime_name == 'BEAR':
                note = "✅ Ideal conditions"
            elif strategy_name == 'growth' and regime_name == 'BULL':
                note = "✅ Ideal conditions"
            elif strategy_name == 'risk_off' and vix > 25:
                note = "✅ Ideal conditions"
            
            print(f"{strategy.name:<15} {result['avg_score']:<8.1f} {perf_str:<15} {note}")
        
        return True
    
    def run_full_backtest(self):
        """
        Run backtest on multiple historical scenarios.
        """
        print("\n" + "="*70)
        print("🎯 STRATEGY BACKTESTING COMPARATOR")
        print("="*70)
        print("\nTesting 3 strategies across historical periods:")
        print("  1. Value (Buffett) - Buy cheap, hold long")
        print("  2. Growth (ARK) - Innovation, ignore valuation")
        print("  3. Risk-Off (Dalio) - Crisis alpha, safe havens")
        print("\n" + "="*70)
        
        # Test scenarios
        scenarios = [
            ("2020-02-01", "2020-04-30", "COVID Crash"),
            ("2020-05-01", "2020-12-31", "COVID Recovery"),  # Fixed space
            ("2022-09-01", "2022-12-31", "Tech Selloff 2022"),
            ("2023-01-01", "2023-12-31", "Bull Run 2023"),
            ("2024-01-01", "2024-06-30", "2024 H1"),  # Shorter period
        ]
        
        for start, end, name in scenarios:
            self.backtest_period(start, end, name)
        
        # Summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """
        Print summary statistics across all scenarios.
        """
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("📊 OVERALL PERFORMANCE SUMMARY")
        print("="*70 + "\n")
        
        # Group by strategy
        for strategy_name in df['strategy'].unique():
            subset = df[df['strategy'] == strategy_name]
            
            avg_return = subset['portfolio_return_pct'].mean()
            max_return = subset['portfolio_return_pct'].max()
            min_return = subset['portfolio_return_pct'].min()
            win_rate = (subset['portfolio_return_pct'] > 0).sum() / len(subset) * 100
            
            print(f"\n{strategy_name}")
            print("-" * 70)
            print(f"  Avg Return:    {avg_return:+.1f}%")
            print(f"  Best Period:   {max_return:+.1f}%")
            print(f"  Worst Period:  {min_return:+.1f}%")
            print(f"  Win Rate:      {win_rate:.0f}%")
            
            # Performance by regime
            print(f"\n  By Regime:")
            for regime in ['BULL', 'BEAR', 'NEUTRAL']:
                regime_subset = subset[subset['regime'] == regime]
                if len(regime_subset) > 0:
                    regime_avg = regime_subset['portfolio_return_pct'].mean()
                    print(f"    {regime:8s}: {regime_avg:+.1f}%")
        
        # Best strategy per scenario
        print(f"\n{'='*70}")
        print("🏆 WINNER PER SCENARIO")
        print(f"{'='*70}\n")
        
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            
            # Check if all values are NaN
            if scenario_data['portfolio_return_pct'].isna().all():
                print(f"{scenario:<25} → No valid data")
                continue
            
            # Drop NaN values before finding max
            valid_data = scenario_data.dropna(subset=['portfolio_return_pct'])
            if len(valid_data) == 0:
                print(f"{scenario:<25} → No valid data")
                continue
                
            winner = valid_data.loc[valid_data['portfolio_return_pct'].idxmax()]
            
            print(f"{scenario:<25} → {winner['strategy']:<25} ({winner['portfolio_return_pct']:+.1f}%)")
    
    def save_results(self):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv('strategy_backtest_results.csv', index=False)
        print(f"\n💾 Results saved to strategy_backtest_results.csv")


if __name__ == "__main__":
    backtester = StrategyBacktester()
    backtester.run_full_backtest()
