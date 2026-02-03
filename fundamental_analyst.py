import yfinance as yf
import pandas as pd
import numpy as np

class FundamentalAnalyst:
    def __init__(self):
        pass
        
    def get_fundamentals(self, ticker_symbol):
        """
        Extrae métricas clave de valoración y salud financiera.
        Retorna un diccionario y un Score Fundamental (0-100).
        """
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # 1. Extracción de Datos (con valores por defecto seguros)
            data = {
                "PE_Ratio": info.get('forwardPE', info.get('trailingPE', None)),
                "PEG_Ratio": info.get('pegRatio', None),
                "Profit_Margin": info.get('profitMargins', 0),
                "Debt_to_Equity": info.get('debtToEquity', 100), # Asumimos deuda media si no hay dato
                "Revenue_Growth": info.get('revenueGrowth', 0),
                "Target_Price": info.get('targetMeanPrice', None),
                "Sector": info.get('sector', "Unknown")
            }
            
            # 2. Cálculo de Score Fundamental
            score = self.calculate_fundamental_score(data)
            data['Fundamental_Score'] = score
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error fundamental en {ticker_symbol}: {e}")
            return {"Fundamental_Score": 50} # Retorno neutro en error

    def calculate_fundamental_score(self, data):
        """
        Algoritmo de puntuación basado en Value & Growth.
        """
        score = 50 # Base neutral
        
        # A. Valoración (P/E y PEG) - Weight 40%
        pe = data.get('PE_Ratio')
        peg = data.get('PEG_Ratio')
        
        if pe:
            if pe < 15: score += 10 # Barata
            elif pe > 50: score -= 10 # Cara (Bubble risk)
            
        if peg:
            if peg < 1.0: score += 10 # Undervalued relative to growth
            elif peg > 2.5: score -= 5 # Overvalued
            
        # B. Crecimiento (Revenue Growth) - Weight 30%
        growth = data.get('Revenue_Growth')
        if growth:
            if growth > 0.20: score += 15 # Hypergrowth (>20%)
            elif growth > 0.10: score += 5
            elif growth < 0: score -= 10 # Contracción
            
        # C. Salud (Márgenes y Deuda) - Weight 30%
        margin = data.get('Profit_Margin')
        debt = data.get('Debt_to_Equity')
        
        if margin and margin > 0.20: score += 10 # Cash machine
        if debt and debt < 50: score += 5 # Balance sheet fuerte
        if debt and debt > 200: score -= 10 # Deuda peligrosa
        
        return min(max(score, 0), 100) # Clamp 0-100
