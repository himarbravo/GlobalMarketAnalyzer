import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import config

class PortfolioManager:
    def __init__(self):
        self.tickers = config.TICKERS["PORTFOLIO"]
        
    def analyze_risk(self):
        """
        Analiza correlaciones y concentración.
        Retorna insights textuales y una matriz de correlación (figura no interactiva, pero datos sí).
        """
        if len(self.tickers) < 2:
            return "Cartera demasiado pequeña para análisis de matriz.", None
            
        print("⚖️  Auditando riesgo de cartera...")
        try:
            # Descargar historial 1 año
            # multi_level_index=False intenta evitar MultiIndex si es posible, pero yfinance cambia mucho.
            raw_data = yf.download(self.tickers, period="1y", progress=False)
            
            # Manejo robusto de columnas (yfinance a veces devuelve MultiIndex, a veces no)
            if 'Adj Close' in raw_data.columns:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                data = raw_data['Close']
            else:
                # Si falló la descarga o estructura desconocida
                return f"Error datos: Columnas encontradas: {list(raw_data.columns)}", None
                
            # Si solo hay 1 activo, 'data' es una Series. Necesitamos DataFrame para .corr()
            if isinstance(data, pd.Series):
                data = data.to_frame() 
                
            # Calcular retornos
            returns = data.pct_change().dropna()
            
            # Matriz de Correlación
            corr_matrix = returns.corr()
            
            # Análisis de concentración
            insights = []
            
            # 1. Buscar pares altamente correlacionados (>0.85)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if val > 0.85:
                        ticker_a = corr_matrix.columns[i]
                        ticker_b = corr_matrix.columns[j]
                        high_corr_pairs.append(f"{ticker_a} & {ticker_b} ({val:.2f})")
            
            if high_corr_pairs:
                insights.append(f"⚠️ **Riesgo de Duplicidad**: Tienes {len(high_corr_pairs)} pares con correlación extrema (>0.85). Si uno cae, el otro también. Pares: " + ", ".join(high_corr_pairs[:3]))
            else:
                insights.append("✅ **Diversificación Sana**: No se detectan correlaciones críticas entre activos.")
                
            # 2. Volatilidad de la Cartera (si fuera equiponderada) -- Aproximación simple
            port_vol = returns.mean(axis=1).std() * (252**0.5)
            insights.append(f"ℹ️ Volatilidad Agregada: {port_vol*100:.1f}% anual.")
            
            return "\n\n".join(insights), corr_matrix
            
        except Exception as e:
            return f"Error analizando portafolio: {e}", None
