"""
Comprehensive Backtest Suite
Tests algorithm on both:
1. Random dates (general accuracy)
2. Crisis dates (danger detection)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
import sys

# Critical market events to test
CRISIS_DATES = {
    "COVID Crash (Pre)": "2020-02-20",  # Before crash
    "COVID Crash (During)": "2020-03-15",  # During crash
    "COVID Recovery": "2020-04-15",  # Recovery
    "Tech Selloff 2022": "2022-11-01",  # Tech bear market
    "Fed Hikes Peak": "2022-09-20",  # Peak rate hike fears
    "SVB Crisis (Pre)": "2023-03-08",  # Before SVB collapse
    "SVB Crisis (During)": "2023-03-13",  # During crisis
    "Flash Crash Aug 2024": "2024-08-05",  # Japan carry trade unwind
    "Bull Run 2023": "2023-11-15",  # Strong bull
    "Q4 2023 Rally": "2023-12-15",  # End year rally
}

def test_crisis_detection():
    """
    Test if algorithm detects danger in crisis periods.
    Expected: BEAR regime or high correlation warnings.
    """
    print("=" * 70)
    print("🚨 CRISIS DETECTION TEST")
    print("=" * 70)
    print("\nProbando si el algoritmo detecta peligro en fechas críticas...\n")
    
    # Small set of tickers for faster testing
    test_tickers = ['SPY', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
                    'JPM', 'XOM', 'GLD', 'TLT', '^VIX', 'BTC-USD']
    
    engine = BacktestEngine(
        tickers=test_tickers,
        start_date='2019-01-01',
        end_date='2025-12-31'
    )
    
    results_summary = []
    
    for event_name, date_str in CRISIS_DATES.items():
        target_date = pd.to_datetime(date_str)
        
        print(f"\n📅 {event_name} ({date_str})")
        print("-" * 70)
        
        try:
            # Run analysis at this date
            signals, regime = engine.run_historical_analysis_at_date(target_date)
            
            # Get correlation state (would need to add this to backtest_engine)
            # For now, count signals
            buy_signals = sum(1 for s in signals.values() if s['signal'] == 'BUY')
            sell_signals = sum(1 for s in signals.values() if s['signal'] == 'SELL')
            hold_signals = sum(1 for s in signals.values() if s['signal'] == 'HOLD')
            
            avg_conviction = np.mean([s['conviction'] for s in signals.values()])
            
            # Danger indicators
            is_defensive = sell_signals > buy_signals
            low_conviction = avg_conviction < 45
            
            danger_detected = is_defensive or low_conviction or regime == "BEAR"
            
            results_summary.append({
                'event': event_name,
                'date': date_str,
                'regime': regime,
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': hold_signals,
                'avg_conviction': avg_conviction,
                'danger_detected': danger_detected
            })
            
            # Print summary
            indicator = "🔴 PELIGRO" if danger_detected else "🟢 SEGURO"
            print(f"{indicator}")
            print(f"  Régimen: {regime}")
            print(f"  Señales: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")
            print(f"  Convicción Promedio: {avg_conviction:.1f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results_summary.append({
                'event': event_name,
                'date': date_str,
                'regime': 'ERROR',
                'danger_detected': False
            })
    
    # Summary table
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE DETECCIÓN DE CRISIS")
    print("=" * 70 + "\n")
    
    df = pd.DataFrame(results_summary)
    
    if len(df) > 0:
        print(df[['event', 'regime', 'buy', 'sell', 'danger_detected']].to_string(index=False))
        
        # Crisis periods that SHOULD have danger
        crisis_periods = ["COVID Crash (During)", "Tech Selloff 2022", 
                         "SVB Crisis (During)", "Flash Crash Aug 2024"]
        crisis_df = df[df['event'].isin(crisis_periods)]
        
        if len(crisis_df) > 0:
            detection_rate = crisis_df['danger_detected'].mean()
            print(f"\n✅ Tasa de detección en crisis: {detection_rate*100:.1f}%")
            print(f"   Detectó {crisis_df['danger_detected'].sum()}/{len(crisis_df)} eventos críticos")
    
    return df

def test_random_portfolios(n_tests=10):
    """
    Test on N random dates with fictional portfolios.
    """
    print("\n" + "=" * 70)
    print("🎲 RANDOM PORTFOLIO TEST (10 carteras ficticias)")
    print("=" * 70 + "\n")
    
    test_tickers = ['SPY', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
                    'JPM', 'XOM', 'GLD', 'TLT', 'BTC-USD']
    
    engine = BacktestEngine(
        tickers=test_tickers,
        start_date='2020-01-01',
        end_date='2025-12-31'
    )
    
    # Generate random dates
    random_dates = engine.generate_random_dates(n=n_tests)
    
    all_results = []
    
    for i, date in enumerate(random_dates):
        print(f"\n[{i+1}/{n_tests}] Fecha: {date.strftime('%Y-%m-%d')}")
        
        try:
            # Run analysis
            signals, regime = engine.run_historical_analysis_at_date(date)
            
            if not signals:
                print("  ⚠️ Sin señales")
                continue
            
            # Measure accuracy 1 month later
            accuracy_results = engine.measure_accuracy(date, signals, horizon_days=21)
            
            if not accuracy_results:
                print("  ⚠️ Sin resultados de accuracy")
                continue
            
            # Calculate metrics
            hits = sum(1 for r in accuracy_results.values() if r['hit'])
            total = len(accuracy_results)
            accuracy = hits / total if total > 0 else 0
            
            avg_return = np.mean([r['actual_return'] for r in accuracy_results.values()])
            
            print(f"  Régimen: {regime}")
            print(f"  Accuracy: {hits}/{total} ({accuracy*100:.1f}%)")
            print(f"  Retorno Promedio Real: {avg_return:+.2f}%")
            
            all_results.append({
                'date': date,
                'regime': regime,
                'accuracy': accuracy,
                'hits': hits,
                'total': total,
                'avg_return': avg_return
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("📊 RESUMEN GENERAL")
        print("=" * 70)
        
        df = pd.DataFrame(all_results)
        overall_accuracy = df['hits'].sum() / df['total'].sum()
        avg_return = df['avg_return'].mean()
        
        print(f"\n🎯 Accuracy Global: {overall_accuracy*100:.1f}%")
        print(f"📈 Retorno Promedio: {avg_return:+.2f}%")
        
        # By regime
        print("\n📊 Por Régimen:")
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            regime_acc = regime_df['hits'].sum() / regime_df['total'].sum()
            regime_ret = regime_df['avg_return'].mean()
            print(f"  {regime:8s}: Accuracy {regime_acc*100:.1f}%, Retorno {regime_ret:+.2f}%")
        
        return df
    
    return None

if __name__ == "__main__":
    print("\n🔬 COMPREHENSIVE BACKTEST SUITE")
    print("="*70 + "\n")
    
    # Test 1: Crisis Detection
    crisis_results = test_crisis_detection()
    
    # Test 2: Random Portfolios
    random_results = test_random_portfolios(n_tests=10)
    
    print("\n" + "="*70)
    print("✅ TESTS COMPLETADOS")
    print("="*70 + "\n")
    
    # Save results
    if crisis_results is not None:
        crisis_results.to_csv('crisis_detection_results.csv', index=False)
        print("💾 Crisis results → crisis_detection_results.csv")
    
    if random_results is not None:
        random_results.to_csv('random_portfolio_results.csv', index=False)
        print("💾 Random results → random_portfolio_results.csv")
