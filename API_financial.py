# Instalar primero: pip install yfinance pandas
import yfinance as yf
import pandas as pd

# 1. Configuración
ticker = "NVDA"  # Ejemplo: NVIDIA, puedes cambiar a "AAPL", "TSLA", "BTC-USD"
periodo = "1y"   # 1 año de historia

print(f"Descargando datos para {ticker}...")

# 2. Descargar datos
data = yf.download(ticker, period=periodo, progress=False, auto_adjust=False)

# 3. Limpiar y Exportar
# Solo nos interesa el precio de cierre ajustado
df_export = data[['Adj Close']].reset_index()
df_export.columns = ['Fecha', 'Precio']

# Guardar en CSV para Mathematica
archivo = "datos_acciones.csv"
df_export.to_csv(archivo, index=False)

print(f"Datos guardados exitosamente en {archivo}")
print(f"Último precio: {df_export['Precio'].iloc[-1]:.2f}")