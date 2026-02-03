"""
STRATEGY SYSTEM DEMO
====================
Demonstrates the full strategy system with macro flows.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from strategy_engine import StrategyEngine
from macro_flow_analyzer import MacroFlowAnalyzer

def demo_strategy_system():
    print("\n" + "="*70)
    print("🎯 STRATEGY SYSTEM DEMONSTRATION")
    print("="*70)
    
    # 1. Download sample data
    print("\n📥 Downloading market data...")
    tickers = ['SPY', 'NVDA', 'AAPL', 'MSFT', 'JPM', 'JNJ', 
               'TLT', 'GLD', 'HYG', 'IEF', '^VIX', 'XLP']
    
    raw_data = yf.download(tickers, period='1y', progress=False, auto_adjust=False)
    
    # Handle MultiIndex
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Adj Close']
    else:
        data = raw_data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
    
    data = data.ffill()
    data = data.fillna(method='ffill')
    
    print(f"✅ Downloaded {len(data.columns)} tickers, {len(data)} days")
    
    # 2. Initialize macro analyzer
    print("\n" + "="*70)
    macro_analyzer = MacroFlowAnalyzer(data)
    flow_summary = macro_analyzer.print_flow_analysis()
    
    # 3. Create simple macro state
    vix_current = data['^VIX'].iloc[-1]
    
    # Calculate correlation (simplified)
    returns = data[['NVDA', 'AAPL', 'MSFT', 'JPM', 'JNJ']].pct_change().dropna()
    corr_matrix = returns.corr()
    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    # Determine regime (simplified - just from VIX)
    if vix_current > 25:
        regime = "BEAR"
    elif vix_current < 15:
        regime = "BULL"
    else:
        regime = "NEUTRAL"
    
    # 4. Initialize strategy engine
    print("🔧 Initializing Strategy Engine...")
    engine = StrategyEngine(data)
    engine.load_strategies()
    engine.set_macro_state(
        regime=regime,
        vix=vix_current,
        correlation=avg_correlation
    )
    
    engine.print_macro_context()
    
    # 5. Auto-select strategy
    print("🤖 Auto-selecting optimal strategy...")
    strategy = engine.select_strategy('auto')
    
    # 6. Get recommendations
    engine.get_strategy_recommendations()
    
    # 7. Show safe havens
    engine.get_safe_havens()
    
    # 8. Score sample assets
    print("\n" + "="*70)
    print("📊 SCORING ASSETS WITH CURRENT STRATEGY")
    print("="*70 + "\n")
    
    # Create sample asset data
    assets_data = {}
    for ticker in ['NVDA', 'AAPL', 'MSFT', 'JPM', 'JNJ', 'TLT', 'GLD']:
        if ticker not in data.columns:
            continue
        
        series = data[ticker].dropna()
        if len(series) < 50:
            continue
        
        price = series.iloc[-1]
        sma50 = series.tail(50).mean()
        returns = series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate RSI
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Mock fundamental data (in real system, fetch from yfinance info)
        fundamental = {
            'PE_Ratio': 20 if ticker in ['JPM', 'JNJ'] else 30,
            'PB_Ratio': 2,
            'ROE': 18,
            'Debt_Equity': 0.4,
            'Dividend_Yield': 2.5 if ticker in ['JPM', 'JNJ'] else 0.5,
            'Revenue_Growth': 10 if ticker in ['JPM', 'JNJ'] else 25,
            'Gross_Margin': 50 if ticker in ['JPM', 'JNJ'] else 65
        }
        
        technical = {
            'price': price,
            'sma50': sma50,
            'rsi': rsi_val,
            'volatility': volatility,
            'alpha': np.random.uniform(-5, 15),  # Mock
            'beta': 1.0,
            'correlation_spy': 0.8
        }
        
        assets_data[ticker] = {
            'fundamental': fundamental,
            'technical': technical
        }
    
    scored_df = engine.score_assets(assets_data)
    print(scored_df.to_string(index=False))
    
    # 9. Compare strategies
    print("\n")
    comparison_df = engine.compare_strategies(assets_data)
    
    # 10. Summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print(f"\nCurrent Market: {flow_summary['overall_regime']}")
    print(f"Selected Strategy: {strategy.name}")
    print(f"Top Recommendation: {scored_df.iloc[0]['ticker']} (score: {scored_df.iloc[0]['score']:.1f})")
    
    return engine, scored_df, comparison_df


if __name__ == "__main__":
    import numpy as np
    engine, scores, comparison = demo_strategy_system()
