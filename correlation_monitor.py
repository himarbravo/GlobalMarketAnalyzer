import numpy as np
import pandas as pd
from arch import arch_model

class CorrelationMonitor:
    """
    Monitorea correlaciones dinámicas entre activos usando DCC-GARCH.
    Detecta cuándo la diversificación se pierde (correlaciones → 1).
    """
    
    def __init__(self, data):
        """
        Args:
            data: DataFrame con precios de múltiples activos
        """
        self.data = data
        self.returns = data.pct_change().dropna()
    
    def calculate_rolling_correlation(self, window=60):
        """
        Calcula correlación rolling simple (más rápido que DCC-GARCH completo).
        
        Args:
            window: Ventana para rolling correlation (días)
        
        Returns:
            DataFrame con correlación promedio por día
        """
        corr_series = []
        
        for i in range(window, len(self.returns)):
            subset = self.returns.iloc[i-window:i]
            corr_matrix = subset.corr()
            
            # Extraer triángulo superior (sin diagonal)
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            avg_corr = upper_triangle.stack().mean()
            corr_series.append({
                'date': self.returns.index[i],
                'avg_correlation': avg_corr
            })
        
        return pd.DataFrame(corr_series).set_index('date')
    
    def get_current_state(self, lookback=60):
        """
        Evalúa el estado actual de correlaciones.
        
        Returns:
            dict con estado, correlación promedio, y recomendación
        """
        # Calcular correlación reciente
        recent_returns = self.returns.tail(lookback)
        corr_matrix = recent_returns.corr()
        
        # Extraer triángulo superior
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        avg_corr = upper_triangle.stack().mean()
        max_corr = upper_triangle.stack().max()
        
        # Clasificar estado
        if avg_corr > 0.7:
            state = "PÁNICO"
            color = "🔴"
            recommendation = "DIVERSIFICACIÓN PERDIDA - Reducir exposición total"
        elif avg_corr > 0.5:
            state = "ALTO RIESGO"
            color = "🟠"
            recommendation = "Correlaciones elevadas - Monitorear de cerca"
        elif avg_corr > 0.3:
            state = "NORMAL"
            color = "🟡"
            recommendation = "Mercado saludable - Diversificación funcional"
        else:
            state = "DESACOPLADO"
            color = "🟢"
            recommendation = "Stock picking óptimo - Baja correlación sectorial"
        
        return {
            "state": state,
            "color": color,
            "avg_correlation": round(avg_corr, 3),
            "max_correlation": round(max_corr, 3),
            "recommendation": recommendation
        }
    
    def get_diversification_penalty(self, avg_corr):
        """
        Calcula un multiplicador de confianza basado en correlaciones.
        
        Correlación > 0.7: Reducir confianza 30%
        Correlación > 0.5: Reducir confianza 15%
        Correlación < 0.3: Sin penalización
        """
        if avg_corr > 0.7:
            return 0.70  # Reduce confianza a 70%
        elif avg_corr > 0.5:
            return 0.85
        else:
            return 1.0  # Sin penalización
    
    def analyze_asset_correlations(self, ticker):
        """
        Analiza las correlaciones de un activo específico con el resto.
        
        Returns:
            dict con correlaciones individuales y promedio
        """
        if ticker not in self.returns.columns:
            return None
        
        recent_returns = self.returns.tail(60)
        corr_with_asset = recent_returns.corr()[ticker].drop(ticker)
        
        return {
            "avg_correlation_with_market": round(corr_with_asset.mean(), 3),
            "max_correlation": round(corr_with_asset.max(), 3),
            "most_correlated_with": corr_with_asset.idxmax(),
            "is_diversifier": corr_with_asset.mean() < 0.3
        }
