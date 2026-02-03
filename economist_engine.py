import numpy as np
import pandas as pd
from scipy.stats import lognorm
import config

class EconomistEngine:
    """
    Motor Matemático basado en el Proceso de Ito (Lema de Ito) y Black-Scholes.
    Reemplaza la lógica de Mathematica para el cálculo de probabilidades y objetivos a 1 mes.
    """
    
    def __init__(self, data_df):
        self.data = data_df
        self.dt = 1/252 # Un día bursátil en términos de año
        self.horizon = 21/252 # 21 días (1 mes)

    def analyze_asset(self, ticker):
        """
        Calcula Drift (mu), Volatilidad (sigma) y Cuantiles Log-Normales.
        """
        if ticker not in self.data.columns:
            return None
            
        prices = self.data[ticker].dropna()
        if len(prices) < 30: return None
        
        s0 = prices.iloc[-1]
        
        # 1. Calcular Retornos Logarítmicos
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # 2. Parámetros Anualizados (Ito Process)
        sigma = log_returns.std() * np.sqrt(252)
        # mu = drift anualizado (simplemente la media anualizada de los retornos logarítmicos)
        mu = log_returns.mean() * 252
        
        # 3. Proyección a 1 mes (tProy = 21/252)
        # Parámetros para la distribución Log-Normal al final del horizonte
        mu_futuro = np.log(s0) + (mu - 0.5 * sigma**2) * self.horizon
        sigma_futuro = sigma * np.sqrt(self.horizon)
        
        # 4. Cálculo de Cuantiles (10%, 50%, 90%)
        # lognorm(s, scale=exp(mu)) -> s es la desviación estándar
        dist = lognorm(s=sigma_futuro, scale=np.exp(mu_futuro))
        
        val_bajo = dist.ppf(0.10)   # Soporte estadístico
        val_mediana = dist.ppf(0.50) # Objetivo real
        val_alto = dist.ppf(0.90)   # Resistencia estadística
        
        return {
            "mu": mu,
            "sigma": sigma,
            "target_median": val_mediana,
            "support_10": val_bajo,
            "resistance_90": val_alto,
            "current_price": s0
        }

    def analyze_macro_risk(self, tickers):
        """
        Calcula la correlación promedio del mercado (Efecto Pánico).
        """
        subset = self.data[tickers].dropna()
        if subset.empty: return "NORMAL", 0.5
        
        corr_matrix = subset.pct_change().corr()
        # Promedio de correlaciones (triángulo superior sin contar la diagonal)
        corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_corr = np.mean(corrs)
        
        if avg_corr > 0.6:
            state = "PÁNICO (Correlación Alta)"
        elif avg_corr < 0.2:
            state = "DESACOPLADO (Stock Picking)"
        else:
            state = "NORMAL (Riesgo Mixto)"
            
        return state, avg_corr
