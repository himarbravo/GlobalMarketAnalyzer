"""
RISK-OFF / CRISIS ALPHA STRATEGY (Ray Dalio Risk Parity Style)
================================================================

Philosophy:
- Proteger capital es prioridad #1
- Diversificación across asset classes (no solo stocks)
- "All Weather" portfolio
- Ganar en crisis comprando safe havens antes que otros

Reglas de Entrada (Assets):
1. Bonos largos (TLT) - suben cuando stocks caen
2. Oro (GLD) - preservación en crisis sistémicas
3. Dollar (UUP) - safe haven currency
4. Defensive sectors (XLP, XLU, XLV)
5. Quality large caps con dividendos (JNJ, PG)

Reglas de Entrada (Timing):
- VIX > 25 (fear)
- Credit spreads widening (HYG/TLT ratio cayendo)
- SPY < SMA200 (bear confirmed)
- Correlation > 0.7 (diversificación rota)

Reglas de Salida:
- VIX < 15 (complacencia)
- Rotate back to growth cuando crisis pasa

Macro Conditions:
- MEJOR en: Recesiones, crisis bancarias, guerras
- PEOR en: Bull markets optimistas

Expected Returns:
- Crisis: +10% to +30% (mientras otros caen -20%)
- Bull market: -5% to +5% (underperformance)
- Objetivo: Sharpe ratio alto, no returns máximos
"""

import pandas as pd
import numpy as np

class RiskOffStrategy:
    def __init__(self):
        self.name = "Risk-Off / Crisis Alpha"
        self.vix_threshold = 25
        self.correlation_threshold = 0.7
        self.safe_haven_tickers = ['TLT', 'GLD', 'SHY', 'XLP', 'XLU', 'XLV', 'JNJ', 'PG', 'KO']
        
    def score_asset(self, ticker, fundamental_data, technical_data, macro_state):
        """
        Score based on defensive characteristics.
        """
        score = 0
        
        # Is it a safe haven?
        if ticker in self.safe_haven_tickers:
            score += 40
        
        # Low beta = defensive
        beta = technical_data.get('beta', 1.0)
        if beta < 0.7:
            score += 20
        elif beta < 0.9:
            score += 10
        
        # Dividend yield (defensive cash return)
        div_yield = fundamental_data.get('Dividend_Yield', 0)
        if div_yield > 3:
            score += 15
        elif div_yield > 2:
            score += 10
        
        # Low volatility
        volatility = technical_data.get('volatility', 0.3)
        if volatility < 0.15:
            score += 15
        elif volatility < 0.25:
            score += 10
        
        # Negative correlation with SPY = good hedge
        correlation_spy = technical_data.get('correlation_spy', 0.8)
        if correlation_spy < 0:  # Negative correlation
            score += 20
        elif correlation_spy < 0.5:
            score += 10
        
        return min(100, score)
    
    def detect_crisis_mode(self, macro_state):
        """
        Determine if we should be in crisis mode.
        """
        vix = macro_state.get('vix', 15)
        correlation = macro_state.get('correlation', 0.3)
        regime = macro_state.get('regime', 'NEUTRAL')
        credit_spread_change = macro_state.get('credit_spread_change_3d', 0)
        
        crisis_score = 0
        
        if vix > 30:
            crisis_score += 3
        elif vix > 25:
            crisis_score += 2
        elif vix > 20:
            crisis_score += 1
        
        if correlation > 0.7:
            crisis_score += 2
        elif correlation > 0.6:
            crisis_score += 1
        
        if regime == 'BEAR':
            crisis_score += 2
        
        if credit_spread_change < -0.03:  # -3% in 3 days
            crisis_score += 2
        
        # Crisis mode if score >= 4
        return crisis_score >= 4
    
    def get_recommendations(self, macro_state):
        """
        Strategy-specific recommendations based on macro.
        """
        recommendations = []
        
        crisis_mode = self.detect_crisis_mode(macro_state)
        
        if crisis_mode:
            recommendations.append("🚨 CRISIS MODE ACTIVATED")
            recommendations.append("📊 COMPRAR AHORA:")
            recommendations.append("  - TLT (Treasuries 20+) - Safe haven #1")
            recommendations.append("  - GLD (Oro) - Preservación de capital")
            recommendations.append("  - XLP (Consumer Staples) - Defensive sector")
            recommendations.append("  - SHY (Cash) - Liquidez máxima")
            recommendations.append("\n🔴 VENDER/EVITAR:")
            recommendations.append("  - Growth tech (NVDA, TSLA)")
            recommendations.append("  - Small caps (IWM)")
            recommendations.append("  - Crypto (BTC, ETH)")
        else:
            recommendations.append("✅ No crisis detected. Mantener posiciones normales.")
            if macro_state.get('vix', 15) < 15:
                recommendations.append("⚠️ VIX muy bajo = complacencia. Preparar hedges por si acaso.")
        
        # Macro flow insights
        if macro_state.get('rates_trend') == 'falling':
            recommendations.append("\n💡 Tipos bajando → TLT sube (bond prices inverse to yields)")
        
        dollar = macro_state.get('dollar_strength')
        if dollar == 'strong':
            recommendations.append("💡 Dollar fuerte → GLD puede sufrir, pero sigue siendo refugio")
        
        return recommendations
    
    def get_safe_havens(self):
        """Returns ranked safe havens."""
        return {
            "TLT": "Treasuries 20+ años - Refugio #1 en crisis",
            "SHY": "Treasuries 1-3 años - Cash equivalente",
            "GLD": "Oro - Crisis sistémicas",
            "XLP": "Consumer Staples - Walmart, P&G (gente sigue comprando)",
            "XLU": "Utilities - Electricidad, agua (no se cancela)",
            "XLV": "Healthcare - Sector defensivo"
        }
