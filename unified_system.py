"""
UNIFIED ANALYST + STRATEGIST SYSTEM
====================================
Combina análisis cuantitativo con pensamiento estratégico.

Flow:
1. Market Analyst: Análisis técnico/fundamental/quant → Signals & Conviction
2. Macro Flow Analyzer: Detecta flujos de dinero → Macro context
3. Strategy Engine: Filtra signals through estrategia macro-aware
4. Portfolio Optimizer: Balancea ataque/defensa → Final portfolio

Este es el "cerebro" completo del sistema.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from market_analyst import GlobalAnalyst
from macro_flow_analyzer import MacroFlowAnalyzer
from strategy_engine import StrategyEngine
from portfolio_optimizer import PortfolioOptimizer
import config

class UnifiedIntelligenceSystem:
    """
    Sistema unificado que combina:
    - Analista Cuantitativo (technical + quant)
    - Estratega Macro (philosophy + flows)
    - Optimizador (offense/defense balance)
    """
    
    def __init__(self, strategy_mode='auto', offense_weight=60):
        """
        Args:
            strategy_mode: 'auto', 'value', 'growth', 'risk_off'
            offense_weight: Base % para offense (0-100)
        """
        print("\n🧠 Initiating Unified Intelligence System...")
        print("="*70)
        
        # Initialize components
        self.analyst = GlobalAnalyst()
        self.strategy_mode = strategy_mode
        self.optimizer = PortfolioOptimizer(base_offense_weight=offense_weight)
        
        # Will be initialized after data fetch
        self.macro_analyzer = None
        self.strategy_engine = None
        self.active_strategy = None
        
    def run_full_analysis(self):
        """
        Run complete analysis pipeline.
        """
        print(f"\n{'='*70}")
        print("🚀 UNIFIED ANALYSIS PIPELINE")
        print(f"{'='*70}\n")
        
        # Step 1: Fetch data (Analyst)
        print("📊 STEP 1: QUANTITATIVE ANALYSIS")
        print("-" * 70)
        self.analyst.fetch_data()
        
        # Step 2: Initialize macro analyzers
        self.macro_analyzer = MacroFlowAnalyzer(self.analyst.data)
        self.strategy_engine = StrategyEngine(self.analyst.data)
        self.strategy_engine.load_strategies()
        
        # Step 3: Macro Flow Analysis
        print(f"\n{'='*70}")
        print("💰 STEP 2: MACRO FLOW ANALYSIS")
        print(f"{'='*70}")
        flow_summary = self.macro_analyzer.print_flow_analysis()
        
        # Step 4: Train quant models (direct instantiation)
        print("🔬 STEP 3: TRAINING QUANTITATIVE MODELS")
        print("-" * 70)
        
        # Train regime detector
        from regime_detector import RegimeDetector
        regime_detector = RegimeDetector()
        spy_returns = self.analyst.data['SPY'].pct_change().dropna()
        regime_detector.train(spy_returns)
        regime_name, regime_conf, regime_params = regime_detector.predict_current_regime(spy_returns.tail(60))
        
        print(f"🎯 Régimen: {regime_name} (Confianza: {regime_conf:.0f}%)")
        
        # Correlation monitoring
        from correlation_monitor import CorrelationMonitor
        corr_monitor = CorrelationMonitor(self.analyst.data)
        
        # Get correlation metrics
        avg_corr_raw = corr_monitor.calculate_rolling_correlation()
        # Extract scalar value (handle DataFrame, Series, or float)
        if hasattr(avg_corr_raw, 'iloc'):
            avg_corr = float(avg_corr_raw.iloc[-1] if hasattr(avg_corr_raw.iloc[-1], '__iter__') else avg_corr_raw.iloc[-1])
        elif hasattr(avg_corr_raw, 'mean'):
            avg_corr = float(avg_corr_raw.mean())
        else:
            avg_corr = float(avg_corr_raw)
        
        div_penalty = corr_monitor.get_diversification_penalty(avg_corr)
        
        # Determine state
        if avg_corr > 0.7:
            state = "ALTO RIESGO"
        elif avg_corr < 0.3:
            state = "DESACOPLADO"
        else:
            state = "MODERADO"
        
        corr_state = {
            'avg_correlation': avg_corr,
            'state': state,
            'diversification_penalty': div_penalty
        }
        
        print(f"🔗 Correlación: {corr_state['avg_correlation']:.2f} - {corr_state['state']}")
        
        # Step 5: Set macro state for strategy
        vix_current = self.analyst.data['^VIX'].iloc[-1]
        
        self.strategy_engine.set_macro_state(
            regime=regime_name,
            vix=vix_current,
            correlation=corr_state['avg_correlation']
        )
        
        # Step 6: Select strategy
        print(f"\n{'='*70}")
        print("🎯 STEP 4: STRATEGY SELECTION")
        print(f"{'='*70}")
        self.strategy_engine.print_macro_context()
        
        self.active_strategy = self.strategy_engine.select_strategy(self.strategy_mode)
        self.strategy_engine.get_strategy_recommendations()
        
        # Step 7: Run analyst signals
        print(f"{'='*70}")
        print("📈 STEP 5: GENERATING SIGNALS (Analyst)")
        print(f"{'='*70}\n")
        
        all_results = []
        target_assets = config.TICKERS["PORTFOLIO"] + config.TICKERS.get("TECH_MEGA", []) + \
                       config.TICKERS.get("HEALTHCARE_PHARMA", []) + config.TICKERS.get("FINANCIALS_BANKS", [])
        
        # Dedupe
        target_assets = list(set(target_assets))[:15]  # Limit for demo
        
        for ticker in target_assets:
            if ticker not in self.analyst.data.columns or ticker == '^VIX':
                continue
            
            try:
                signal_data = self._analyze_single_asset(ticker, regime_name, corr_state)
                if signal_data:
                    all_results.append(signal_data)
                    print(f"  ✓ {ticker:8s}: {signal_data['signal']:10s} (Conv: {signal_data['conviction']:.1f})")
            except Exception as e:
                print(f"  ✗ {ticker:8s}: Error - {e}")
        
        signals_df = pd.DataFrame(all_results)
        
        # Step 8: Portfolio Optimization (Attack/Defense Balance)
        print(f"\n{'='*70}")
        print("⚖️ STEP 6: PORTFOLIO OPTIMIZATION (Attack/Defense)")
        print(f"{'='*70}")
        
        portfolio = self.optimizer.optimize_portfolio(
            signals_df,
            self.strategy_engine.macro_state,
            self.strategy_mode if self.strategy_mode != 'auto' else self.active_strategy.name.split()[0].lower(),
            max_positions=10
        )
        
        self.optimizer.print_portfolio_summary(portfolio)
        
        # Step 9: Safe Havens
        self.strategy_engine.get_safe_havens()
        
        # Save results
        portfolio.to_csv('optimized_portfolio.csv', index=False)
        signals_df.to_csv('all_signals.csv', index=False)
        
        print(f"{'='*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"💾 Saved: optimized_portfolio.csv, all_signals.csv")
        
        return portfolio, signals_df
    
    def _analyze_single_asset(self, ticker, regime_name, corr_state):
        """
        Analyze single asset using full analyst pipeline.
        """
        series = self.analyst.data[ticker].dropna()
        if len(series) < 100:
            return None
        
        price = series.iloc[-1]
        
        # Technical
        sma = series.tail(50).mean()
        returns = series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # RSI
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Trend score
        trend_score = 100 if price > sma else 30
        rsi_score = 100 - abs(rsi_val - 50)
        
        # Factor analysis
        asset_returns = series.pct_change().dropna()
        try:
            from factor_analyzer import FactorAnalyzer
            factor_analyzer = FactorAnalyzer(self.analyst.data)
            factor_result = factor_analyzer.decompose_asset(asset_returns)
            alpha = factor_result['alpha_annual_pct'] if factor_result else 0
        except:
            alpha = 0
        
        alpha_score = 50 + (alpha * 2)
        
        # Diversification penalty
        div_penalty = corr_state.get('diversification_penalty', 1.0)
        
        # Conviction
        raw_conviction = (trend_score * 0.4 + rsi_score * 0.3 + alpha_score * 0.3)
        conviction = raw_conviction * div_penalty
        conviction = max(0, min(100, conviction))
        
        # Signal
        if conviction >= 75:
            signal = "STRONG BUY"
        elif conviction >= 60:
            signal = "BUY"
        elif conviction <= 30:
            signal = "SELL"
        elif conviction <= 45:
            signal = "AVOID"
        else:
            signal = "HOLD"
        
        # Mock fundamentals (in real system, fetch from yfinance)
        fundamental = {
            'PE_Ratio': np.random.uniform(15, 30),
            'Dividend_Yield': np.random.uniform(0, 3)
        }
        
        technical = {
            'price': price,
            'sma50': sma,
            'rsi': rsi_val,
            'volatility': volatility,
            'alpha': alpha,
            'beta': 1.0
        }
        
        return {
            'ticker': ticker,
            'signal': signal,
            'conviction': conviction,
            'price': price,
            'alpha': alpha,
            'regime': regime_name,
            'fundamental': fundamental,
            'technical': technical
        }


if __name__ == "__main__":
    # Run unified system
    print("\n" + "="*70)
    print("🎯 UNIFIED INTELLIGENCE SYSTEM - Full Demo")
    print("="*70)
    print("\nCombining:")
    print("  - Quantitative Analyst (HMM, Fama-French, Itō)")
    print("  - Macro Strategist (Flow analysis, Philosophy)")
    print("  - Portfolio Optimizer (Attack/Defense balance)")
    print("\n" + "="*70 + "\n")
    
    # Initialize with auto strategy selection and balanced offense
    system = UnifiedIntelligenceSystem(
        strategy_mode='auto',  # or 'value', 'growth', 'risk_off'
        offense_weight=60  # 60% offensive, 40% defensive baseline
    )
    
    # Run full pipeline
    portfolio, signals = system.run_full_analysis()
