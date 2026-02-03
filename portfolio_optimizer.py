"""
PORTFOLIO OPTIMIZER
===================
Balancea ataque (returns) vs defensa (risk) usando:
- Analyst signals (quant)
- Strategy filters (macro-aware)
- Risk scoring (volatility, correlation, beta)

El tradeoff:
- Offense: Max returns (growth stocks, high conviction)
- Defense: Preserve capital (safe havens, low volatility)
- Balance: Cambia dinámicamente según macro regime
"""

import pandas as pd
import numpy as np

class PortfolioOptimizer:
    """
    Optimiza portfolio balanceando offense/defense.
    
    Offense Weight: 0-100%
    - 100% = Full attack (growth, high beta, high conviction)
    - 0% = Full defense (bonds, gold, low volatility)
    - 50% = Balanced
    
    Auto-adjusts based on:
    - VIX (high = more defense)
    - Regime (BEAR = more defense)
    - Credit spreads (stress = more defense)
    - Strategy selection (risk_off = more defense)
    """
    
    def __init__(self, base_offense_weight=60):
        """
        Args:
            base_offense_weight: Default offense % (0-100). 
                                 60 = slightly aggressive
                                 50 = balanced
                                 40 = slightly defensive
        """
        self.base_offense_weight = base_offense_weight
        
    def calculate_offense_weight(self, macro_state, strategy_name):
        """
        Calculate current offense weight based on macro conditions.
        
        Returns: float 0-100 (% allocation to offensive assets)
        """
        offense = self.base_offense_weight
        
        # VIX adjustment (more defense when fear high)
        vix = macro_state.get('vix', 15)
        if vix > 30:
            offense -= 30  # Crisis mode
        elif vix > 25:
            offense -= 20
        elif vix > 20:
            offense -= 10
        elif vix < 12:
            offense += 10  # Complacency = opportunity
        
        # Regime adjustment
        regime = macro_state.get('regime', 'NEUTRAL')
        if regime == 'BEAR':
            offense -= 20
        elif regime == 'BULL':
            offense += 10
        
        # Credit stress adjustment
        credit_state = macro_state.get('credit_state', 'stable')
        if credit_state == 'stress':
            offense -= 15
        elif credit_state == 'healthy':
            offense += 5
        
        # Risk mode adjustment
        risk_mode = macro_state.get('risk_mode', 'neutral')
        if risk_mode == 'risk-off':
            offense -= 10
        elif risk_mode == 'risk-on':
            offense += 10
        
        # Strategy preference
        if strategy_name == 'risk_off':
            offense = min(offense, 30)  # Cap at 30% offense
        elif strategy_name == 'growth':
            offense = max(offense, 60)  # Floor at 60% offense
        
        # Bounds
        offense = max(0, min(100, offense))
        
        return offense
    
    def classify_asset(self, ticker, technical_data, fundamental_data):
        """
        Classify asset as offensive or defensive.
        
        Returns: dict with {
            'type': 'offensive' | 'defensive' | 'neutral',
            'offense_score': 0-100
        }
        """
        offense_score = 50  # Start neutral
        
        # Beta (high beta = offensive)
        beta = technical_data.get('beta', 1.0)
        if beta > 1.3:
            offense_score += 20
        elif beta > 1.1:
            offense_score += 10
        elif beta < 0.7:
            offense_score -= 20
        elif beta < 0.9:
            offense_score -= 10
        
        # Volatility
        volatility = technical_data.get('volatility', 0.25)
        if volatility > 0.40:
            offense_score += 15  # High vol = offensive
        elif volatility < 0.15:
            offense_score -= 15  # Low vol = defensive
        
        # Asset type heuristics
        safe_havens = ['TLT', 'GLD', 'SHY', 'IEF', 'BNDX']
        defensive_sectors = ['XLP', 'XLU', 'XLV']
        offensive_sectors = ['growth', 'tech', 'small cap']
        
        if ticker in safe_havens:
            offense_score = 10  # Clearly defensive
        elif ticker in defensive_sectors:
            offense_score = 25
        elif any(x in ticker for x in ['BTC', 'ETH', 'ARK']):
            offense_score = 90  # Clearly offensive
        
        # Dividend yield (defensive)
        div_yield = fundamental_data.get('Dividend_Yield', 0)
        if div_yield > 3:
            offense_score -= 10
        
        # Classify
        if offense_score >= 70:
            asset_type = 'offensive'
        elif offense_score <= 30:
            asset_type = 'defensive'
        else:
            asset_type = 'neutral'
        
        return {
            'type': asset_type,
            'offense_score': offense_score
        }
    
    def optimize_portfolio(self, ranked_assets, macro_state, strategy_name, max_positions=10):
        """
        Optimize portfolio to match target offense/defense balance.
        
        Args:
            ranked_assets: DataFrame with columns [ticker, conviction, ...]
            macro_state: dict of macro conditions
            strategy_name: active strategy
            max_positions: max number of positions
        
        Returns:
            DataFrame with selected assets and allocations
        """
        target_offense = self.calculate_offense_weight(macro_state, strategy_name)
        target_defense = 100 - target_offense
        
        print(f"\n🎯 Portfolio Optimization")
        print(f"   Target: {target_offense:.0f}% Offense / {target_defense:.0f}% Defense")
        
        # Classify all assets
        for idx, row in ranked_assets.iterrows():
            ticker = row['ticker']
            classification = self.classify_asset(
                ticker,
                row.get('technical', {}),
                row.get('fundamental', {})
            )
            ranked_assets.at[idx, 'asset_type'] = classification['type']
            ranked_assets.at[idx, 'offense_score'] = classification['offense_score']
        
        # Separate offensive and defensive
        offensive_assets = ranked_assets[ranked_assets['offense_score'] >= 60].copy()
        defensive_assets = ranked_assets[ranked_assets['offense_score'] <= 40].copy()
        neutral_assets = ranked_assets[
            (ranked_assets['offense_score'] > 40) & 
            (ranked_assets['offense_score'] < 60)
        ].copy()
        
        # Calculate number of positions for each
        n_offensive = int(max_positions * target_offense / 100)
        n_defensive = int(max_positions * target_defense / 100)
        n_neutral = max_positions - n_offensive - n_defensive
        
        # Select top assets from each category
        selected = []
        
        if len(offensive_assets) > 0 and n_offensive > 0:
            top_offensive = offensive_assets.nlargest(n_offensive, 'conviction')
            for _, row in top_offensive.iterrows():
                selected.append({
                    'ticker': row['ticker'],
                    'allocation': target_offense / n_offensive if n_offensive > 0 else 0,
                    'conviction': row['conviction'],
                    'type': 'OFFENSE',
                    'reason': row.get('reason', 'High beta / Growth')
                })
        
        if len(defensive_assets) > 0 and n_defensive > 0:
            top_defensive = defensive_assets.nlargest(n_defensive, 'conviction')
            for _, row in top_defensive.iterrows():
                selected.append({
                    'ticker': row['ticker'],
                    'allocation': target_defense / n_defensive if n_defensive > 0 else 0,
                    'conviction': row['conviction'],
                    'type': 'DEFENSE',
                    'reason': row.get('reason', 'Safe haven / Low vol')
                })
        
        if len(neutral_assets) > 0 and n_neutral > 0:
            top_neutral = neutral_assets.nlargest(n_neutral, 'conviction')
            for _, row in top_neutral.iterrows():
                selected.append({
                    'ticker': row['ticker'],
                    'allocation': 100 / max_positions,  # Equal weight for neutral
                    'conviction': row['conviction'],
                    'type': 'NEUTRAL',
                    'reason': row.get('reason', 'Balanced')
                })
        
        # Create final portfolio
        portfolio_df = pd.DataFrame(selected)
        
        if len(portfolio_df) > 0:
            # Normalize allocations to sum to 100%
            total_alloc = portfolio_df['allocation'].sum()
            if total_alloc > 0:
                portfolio_df['allocation'] = (portfolio_df['allocation'] / total_alloc) * 100
            
            # Sort by allocation
            portfolio_df = portfolio_df.sort_values('allocation', ascending=False)
        
        return portfolio_df
    
    def print_portfolio_summary(self, portfolio_df):
        """Print human-readable portfolio summary."""
        if len(portfolio_df) == 0:
            print("⚠️ No portfolio selected")
            return
        
        print("\n" + "="*70)
        print("💼 OPTIMIZED PORTFOLIO")
        print("="*70)
        
        # Group by type
        for portfolio_type in ['OFFENSE', 'DEFENSE', 'NEUTRAL']:
            subset = portfolio_df[portfolio_df['type'] == portfolio_type]
            if len(subset) == 0:
                continue
            
            total_alloc = subset['allocation'].sum()
            icon = "⚔️" if portfolio_type == "OFFENSE" else ("🛡️" if portfolio_type == "DEFENSE" else "⚖️")
            print(f"\n{icon} {portfolio_type} ({total_alloc:.1f}% total)")
            print("-" * 70)
            
            for _, row in subset.iterrows():
                print(f"  {row['ticker']:8s} {row['allocation']:5.1f}%  (Conviction: {row['conviction']:.1f})")
        
        print("\n" + "="*70 + "\n")
        
        return portfolio_df
