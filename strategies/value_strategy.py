"""
VALUE INVESTING STRATEGY (Warren Buffett Style)
==================================================

Philosophy:
- Comprar empresas baratas con fundamentales sólidos
- Ignorar volatilidad de corto plazo
- Mantener largo plazo (años, no meses)

Reglas de Entrada:
1. P/E < 15 (barato relativo a earnings)
2. P/B < 3 (barato relativo a book value)
3. ROE > 15% (negocio rentable)
4. Debt/Equity < 0.5 (poca deuda)
5. Dividend Yield > 2% (retorna efectivo)
6. Alpha Fama-French > 0 (outperformance estructural)

Reglas de Salida:
1. P/E > 25 (sobrevalorado)
2. Fundamentales se deterioran (ROE < 10%)
3. Deuda aumenta >50% en 1 año

Macro Conditions:
- MEJOR en: Recesiones / Crisis (activos baratos)
- PEOR en: Bull markets parabólicos (todo caro)

Safe Havens (cuando esta estrategia falla):
- GLD (oro)
- TLT (bonos largos)
- SHY (cash equivalente)

Expected Returns:
- Normal: 8-12% anual
- Crisis: -5% to +20% (compra oportunidades)
- Bull extremo: 0-5% (no encuentra nada barato)
"""

import pandas as pd
import numpy as np

class ValueStrategy:
    def __init__(self):
        self.name = "Value Investing (Buffett)"
        self.min_pe = 0
        self.max_pe = 15
        self.max_pb = 3
        self.min_roe = 15
        self.max_debt_equity = 0.5
        self.min_dividend_yield = 2.0
        
    def score_asset(self, ticker, fundamental_data, technical_data, macro_state):
        """
        Score 0-100 based on value criteria.
        """
        score = 0
        
        # Fundamental checks
        pe = fundamental_data.get('PE_Ratio', 999)
        pb = fundamental_data.get('PB_Ratio', 999)
        roe = fundamental_data.get('ROE', 0)
        debt_equity = fundamental_data.get('Debt_Equity', 999)
        div_yield = fundamental_data.get('Dividend_Yield', 0)
        
        # P/E scoring (max 25 points)
        if pe < 10:
            score += 25
        elif pe < 15:
            score += 15
        elif pe < 20:
            score += 5
        
        # P/B scoring (max 15 points)
        if pb < 1.5:
            score += 15
        elif pb < 3:
            score += 10
        
        # ROE scoring (max 20 points)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        
        # Debt scoring (max 15 points)
        if debt_equity < 0.3:
            score += 15
        elif debt_equity < 0.5:
            score += 10
        
        # Dividend scoring (max 10 points)
        if div_yield > 3:
            score += 10
        elif div_yield > 2:
            score += 7
        
        # Alpha bonus (max 15 points)
        alpha = technical_data.get('alpha', 0)
        if alpha > 5:
            score += 15
        elif alpha > 2:
            score += 10
        elif alpha > 0:
            score += 5
        
        return min(100, score)
    
    def get_recommendations(self, macro_state):
        """
        Strategy-specific recommendations based on macro.
        """
        recommendations = []
        
        if macro_state.get('regime') == 'BEAR':
            recommendations.append("🟢 OPORTUNIDAD: Crisis = momento ideal para Value. Buscar empresas de calidad baratas.")
        
        if macro_state.get('rates_trend') == 'rising':
            recommendations.append("⚠️ Favor financieras (JPM, BAC) - se benefician de tipos altos")
        
        if macro_state.get('vix') > 25:
            recommendations.append("🟢 High VIX = Fear = Precios baratos. Buffett: 'Be greedy when others are fearful'")
        
        correlation = macro_state.get('correlation', 0)
        if correlation < 0.3:
            recommendations.append("✅ Baja correlación = Stock picking funciona. Buscar undervalued específicos.")
        
        return recommendations
    
    def get_safe_havens(self):
        """Returns safe haven tickers for this strategy."""
        return {
            "TLT": "Treasuries largos - refugio clásico",
            "GLD": "Oro - preservación de capital",
            "XLP": "Consumer Staples - defensive sector"
        }
