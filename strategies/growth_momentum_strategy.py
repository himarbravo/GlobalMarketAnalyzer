"""
GROWTH MOMENTUM STRATEGY (Cathie Wood / ARK Style)
==================================================

Philosophy:
- Comprar innovación y crecimiento disruptivo
- Alta volatilidad = oportunidad
- Ignorar valuaciones tradicionales
- Mantener mientras el "story" esté intacto

Reglas de Entrada:
1. Revenue Growth > 25% YoY (crecimiento acelerado)
2. Gross Margin > 60% (negocio escalable)
3. Market Leadership en sector (Top 3)
4. RSI < 70 (no en euforia extrema)
5. Momentum positivo (precio > SMA50)
6. Alpha > 10% (outperformance clara)

Reglas de Salida:
1. Revenue Growth < 15% (desaceleración)
2. Competencia erosiona market share
3. RSI > 80 (euforia peligrosa)
4. Rompe SMA50 con volumen alto

Macro Conditions:
- MEJOR en: Bull markets + tipos bajos + liquidez alta
- PEOR en: Recesiones + tipos altos (Fed hikes)

Safe Havens (cuando esta estrategia falla):
- QQQ (diversificación tech)
- USMV (low volatility)
- SHY (cash)

Expected Returns:
- Bull market: 30-50% anual
- Bear market: -40% to -60% (drawdowns brutales)
- Sideways: -10% to +10%
"""

import pandas as pd
import numpy as np

class GrowthMomentumStrategy:
    def __init__(self):
        self.name = "Growth Momentum (ARK)"
        self.min_revenue_growth = 25
        self.min_gross_margin = 60
        self.max_rsi = 70
        self.min_alpha = 10
        
    def score_asset(self, ticker, fundamental_data, technical_data, macro_state):
        """
        Score 0-100 based on growth + momentum criteria.
        """
        score = 0
        
        # Revenue growth (max 30 points)
        rev_growth = fundamental_data.get('Revenue_Growth', 0)
        if rev_growth > 50:
            score += 30
        elif rev_growth > 25:
            score += 20
        elif rev_growth > 15:
            score += 10
        
        # Gross margin (max 20 points)
        gross_margin = fundamental_data.get('Gross_Margin', 0)
        if gross_margin > 70:
            score += 20
        elif gross_margin > 60:
            score += 15
        elif gross_margin > 50:
            score += 10
        
        # Alpha (max 25 points)
        alpha = technical_data.get('alpha', 0)
        if alpha > 20:
            score += 25
        elif alpha > 10:
            score += 20
        elif alpha > 5:
            score += 15
        
        # Momentum (max 15 points)
        price = technical_data.get('price', 0)
        sma50 = technical_data.get('sma50', 0)
        if price > sma50 * 1.10:  # 10% above SMA
            score += 15
        elif price > sma50:
            score += 10
        
        # RSI (max 10 points) - penalize if too high
        rsi = technical_data.get('rsi', 50)
        if rsi < 60:
            score += 10
        elif rsi < 70:
            score += 5
        elif rsi > 80:
            score -= 20  # Euforia peligrosa
        
        return max(0, min(100, score))
    
    def get_recommendations(self, macro_state):
        """
        Strategy-specific recommendations based on macro.
        """
        recommendations = []
        
        regime = macro_state.get('regime')
        
        if regime == 'BULL':
            recommendations.append("🚀 BULL MARKET: Momento ideal para growth. Agresividad máxima.")
        elif regime == 'BEAR':
            recommendations.append("🔴 BEAR MARKET: Growth sufre más. Considerar rotar a Quality/Defensive.")
        
        rates = macro_state.get('rates_trend')
        if rates == 'rising':
            recommendations.append("⚠️ Tipos subiendo: Penaliza growth (valuaciones futuras valen menos). Precaución.")
        elif rates == 'falling':
            recommendations.append("🟢 Tipos bajando: Favorece growth tech. Free money era.")
        
        vix = macro_state.get('vix', 15)
        if vix > 30:
            recommendations.append("🔴 VIX alto: Growth cae 2x más que mercado. Risk-off recomendado.")
        
        correlation = macro_state.get('correlation', 0)
        if correlation > 0.7:
            recommendations.append("⚠️ Alta correlación: Todo cae junto. Difícil escapar. Considerar cash.")
        
        return recommendations
    
    def get_safe_havens(self):
        """Returns safe haven tickers for this strategy."""
        return {
            "QQQ": "Nasdaq diversificado - menos riesgo que individual stocks",
            "USMV": "Low volatility ETF - defensive",
            "SHY": "Cash equivalente - preservar capital"
        }
