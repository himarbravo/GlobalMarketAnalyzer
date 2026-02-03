import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class FactorAnalyzer:
    """
    Descompone los retornos de activos usando el modelo Fama-French de 3 factores:
    R_asset = α + β_market·R_MKT + β_value·R_HML + β_size·R_SMB + ε
    
    Un α positivo indica que el activo genera retornos superiores al mercado
    después de ajustar por factores de riesgo.
    """
    
    def __init__(self, factor_data):
        """
        Args:
            factor_data: DataFrame con columnas [SPY, VTV, IWM] (Market, Value, Size)
        """
        self.factor_data = factor_data
        self.factor_returns = factor_data.pct_change().dropna()
        
    def decompose_asset(self, asset_returns):
        """
        Realiza regresión lineal múltiple para extraer α y betas.
        
        Args:
            asset_returns: Series con retornos del activo
        
        Returns:
            dict con alpha (%), betas, R², y interpretación
        """
        # Alinear fechas
        aligned = pd.concat([asset_returns, self.factor_returns], axis=1, join='inner').dropna()
        
        if len(aligned) < 30:
            return None  # Datos insuficientes
        
        y = aligned.iloc[:, 0].values  # Asset returns
        X = aligned.iloc[:, 1:].values  # Factor returns
        
        # Regresión lineal
        model = LinearRegression().fit(X, y)
        
        alpha_daily = model.intercept_
        alpha_annual = alpha_daily * 252 * 100  # Convertir a % anual
        
        betas = dict(zip(self.factor_returns.columns, model.coef_))
        r_squared = model.score(X, y)
        
        # Interpretación
        if alpha_annual > 2:
            interpretation = "🌟 OUTPERFORMER - Genera alpha consistente"
        elif alpha_annual > 0.5:
            interpretation = "✅ POSITIVO - Supera al mercado ajustado por factores"
        elif alpha_annual > -0.5:
            interpretation = "➖ NEUTRAL - Retornos explicados por factores"
        else:
            interpretation = "⚠️ UNDERPERFORMER - Destruye valor vs. factores"
        
        return {
            "alpha_annual_pct": round(alpha_annual, 2),
            "betas": {k: round(v, 3) for k, v in betas.items()},
            "r_squared": round(r_squared, 3),
            "interpretation": interpretation,
            "is_diversifier": r_squared < 0.3  # Baja correlación con factores
        }
    
    def generate_alpha_bonus(self, alpha_annual_pct):
        """
        Convierte alpha en un bonus/penalty de puntos de confianza.
        
        Alpha > 2%: +15 puntos
        Alpha 0-2%: +5 puntos
        Alpha < -1%: -10 puntos
        """
        if alpha_annual_pct > 2:
            return 15
        elif alpha_annual_pct > 0.5:
            return 5
        elif alpha_annual_pct > -0.5:
            return 0
        else:
            return -10
