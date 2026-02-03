"""
STRATEGY ENGINE
===============
Orquesta múltiples estrategias y permite seleccionar/comparar.

Usage:
    engine = StrategyEngine(data)
    engine.load_strategies()
    engine.run_strategy('value')
    engine.compare_strategies()
"""

import sys
import importlib
import pandas as pd
import numpy as np
from pathlib import Path

# Add strategies folder to path
sys.path.append(str(Path(__file__).parent / 'strategies'))

from strategies.value_strategy import ValueStrategy
from strategies.growth_momentum_strategy import GrowthMomentumStrategy
from strategies.risk_off_strategy import RiskOffStrategy
from macro_flow_analyzer import MacroFlowAnalyzer

class StrategyEngine:
    def __init__(self, data, macro_state=None):
        self.data = data
        self.macro_state = macro_state or {}
        self.strategies = {}
        self.active_strategy = None
        
        # Initialize macro analyzer
        self.macro_analyzer = MacroFlowAnalyzer(data)
        
    def load_strategies(self):
        """Load all available strategies."""
        self.strategies = {
            'value': ValueStrategy(),
            'growth': GrowthMomentumStrategy(),
            'risk_off': RiskOffStrategy(),
        }
        print(f"✅ Loaded {len(self.strategies)} strategies")
        
    def set_macro_state(self, regime, vix, correlation, **kwargs):
        """Update macro state for strategy decisions."""
        self.macro_state = {
            'regime': regime,
            'vix': vix,
            'correlation': correlation,
            **kwargs
        }
        
        # Add macro flow analysis
        flow_summary = self.macro_analyzer.get_money_flow_summary()
        self.macro_state['rates_trend'] = flow_summary['rates']['trend']
        self.macro_state['dollar_strength'] = flow_summary['dollar']['trend']
        self.macro_state['risk_mode'] = flow_summary['risk']['mode']
        self.macro_state['credit_state'] = flow_summary['credit']['state']
        self.macro_state['overall_regime'] = flow_summary['overall_regime']
        
    def select_strategy(self, strategy_name):
        """
        Select active strategy.
        
        Args:
            strategy_name: 'value', 'growth', 'risk_off', or 'auto'
        """
        if strategy_name == 'auto':
            # Auto-select based on macro conditions
            strategy_name = self._auto_select_strategy()
            print(f"🤖 Auto-selected strategy: {strategy_name}")
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(self.strategies.keys())}")
        
        self.active_strategy = self.strategies[strategy_name]
        print(f"✅ Active strategy: {self.active_strategy.name}")
        return self.active_strategy
    
    def _auto_select_strategy(self):
        """
        Auto-select strategy based on macro conditions.
        
        Logic:
        - High VIX or BEAR regime → risk_off
        - Low VIX + BULL + low rates → growth
        - Normal conditions → value
        """
        vix = self.macro_state.get('vix', 15)
        regime = self.macro_state.get('regime', 'NEUTRAL')
        credit_state = self.macro_state.get('credit_state', 'stable')
        overall_regime = self.macro_state.get('overall_regime', 'MIXED')
        
        # Crisis detection
        if vix > 25 or regime == 'BEAR' or credit_state == 'stress':
            return 'risk_off'
        
        # Growth conditions
        if vix < 18 and regime == 'BULL' and overall_regime == 'RISK-ON':
            return 'growth'
        
        # Default to value
        return 'value'
    
    def score_assets(self, assets_data):
        """
        Score all assets using active strategy.
        
        Args:
            assets_data: dict of {ticker: {'fundamental': {}, 'technical': {}}}
        
        Returns:
            DataFrame with scores and recommendations
        """
        if not self.active_strategy:
            raise ValueError("No active strategy. Call select_strategy() first.")
        
        results = []
        
        for ticker, data in assets_data.items():
            fundamental = data.get('fundamental', {})
            technical = data.get('technical', {})
            
            score = self.active_strategy.score_asset(
                ticker, fundamental, technical, self.macro_state
            )
            
            results.append({
                'ticker': ticker,
                'score': score,
                'strategy': self.active_strategy.name
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        
        return df
    
    def get_strategy_recommendations(self):
        """
        Get macro-aware recommendations from active strategy.
        """
        if not self.active_strategy:
            raise ValueError("No active strategy selected.")
        
        recommendations = self.active_strategy.get_recommendations(self.macro_state)
        
        print(f"\n📋 {self.active_strategy.name} - Recommendations")
        print("=" * 70)
        for rec in recommendations:
            print(f"  {rec}")
        print()
        
        return recommendations
    
    def get_safe_havens(self):
        """Get safe haven recommendations from active strategy."""
        if not self.active_strategy:
            return {}
        
        havens = self.active_strategy.get_safe_havens()
        
        print("\n🛡️ SAFE HAVENS (según estrategia activa)")
        print("=" * 70)
        for ticker, explanation in havens.items():
            print(f"  {ticker:6s}: {explanation}")
        print()
        
        return havens
    
    def compare_strategies(self, assets_data):
        """
        Compare all strategies on same assets.
        Shows which strategy would score each asset highest.
        """
        print("\n" + "=" * 70)
        print("⚖️ STRATEGY COMPARISON")
        print("=" * 70 + "\n")
        
        comparison_results = []
        
        for ticker, data in assets_data.items():
            fundamental = data.get('fundamental', {})
            technical = data.get('technical', {})
            
            scores = {}
            for name, strategy in self.strategies.items():
                score = strategy.score_asset(ticker, fundamental, technical, self.macro_state)
                scores[name] = score
            
            best_strategy = max(scores, key=scores.get)
            best_score = scores[best_strategy]
            
            comparison_results.append({
                'ticker': ticker,
                'value_score': scores.get('value', 0),
                'growth_score': scores.get('growth', 0),
                'risk_off_score': scores.get('risk_off', 0),
                'best_strategy': best_strategy,
                'best_score': best_score
            })
        
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('best_score', ascending=False)
        
        print(df.to_string(index=False))
        
        # Summary stats
        print("\n📊 Strategy Preferences:")
        for strategy_name in ['value', 'growth', 'risk_off']:
            count = (df['best_strategy'] == strategy_name).sum()
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"  {strategy_name:10s}: {count:2d} assets ({pct:.1f}%)")
        
        return df
    
    def print_macro_context(self):
        """Print current macro context for decision making."""
        print("\n" + "=" * 70)
        print("🌍 MACRO CONTEXT")
        print("=" * 70)
        
        print(f"\nRegime: {self.macro_state.get('regime', 'UNKNOWN')}")
        print(f"VIX: {self.macro_state.get('vix', 'N/A')}")
        print(f"Correlation: {self.macro_state.get('correlation', 'N/A'):.2f}")
        print(f"Overall Market: {self.macro_state.get('overall_regime', 'UNKNOWN')}")
        print(f"\nRates Trend: {self.macro_state.get('rates_trend', 'unknown')}")
        print(f"Dollar: {self.macro_state.get('dollar_strength', 'unknown')}")
        print(f"Credit: {self.macro_state.get('credit_state', 'unknown')}")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("Strategy Engine - Demo\n")
    print("Available strategies:")
    print("  - value: Warren Buffett style")
    print("  - growth: Cathie Wood / ARK style")
    print("  - risk_off: Ray Dalio crisis alpha")
    print("\nUsage:")
    print("  from strategy_engine import StrategyEngine")
    print("  engine = StrategyEngine(data)")
    print("  engine.load_strategies()")
    print("  engine.select_strategy('auto')  # or specific: 'value', 'growth', 'risk_off'")
