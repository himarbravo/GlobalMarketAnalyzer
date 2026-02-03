import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN DE ACTIVOS ESTRATÉGICOS ---
# Seleccionamos activos que representan distintas "fuerzas" económicas
tickers = {
    "NVDA": "Tu_Activo",       # El objetivo
    "SPY": "Mercado_General",  # Renta Variable (Beta)
    "TLT": "Bonos_Largo_Plazo",# Coste del dinero (Tipos de interés inversos)
    "GLD": "Oro_Refugio",      # Miedo a inflación/Colapso fiat
    "BTC-USD": "Especulacion", # Apetito de riesgo / Liquidez excedente
    "^VIX": "Indice_Miedo"     # Coste de protección (Volatilidad Implícita)
}

print("1. Descargando datos macroeconómicos...")
# Descargamos solo el cierre ajustado
data = yf.download(list(tickers.keys()), period="5y", progress=False, auto_adjust=False)['Adj Close']

# --- 2. EXPLICACIÓN DE RELACIONES (COMENTARIO ECONÓMICO) ---
"""
GUÍA DE RELACIONES PARA EL MODELO:

1. CORRELACIÓN NVDA vs TLT (Bonos):
   - NORMAL: Positiva. Si Bonos suben (Tipos bajan), Tech sube.
   - PELIGRO: Si Bonos caen (Tipos suben) y NVDA sigue subiendo, es una divergencia por euforia.

2. CORRELACIÓN SPY vs VIX:
   - NORMAL: Negativa fuerte (-0.7). Si Bolsa sube, Miedo baja.
   - PÁNICO SISTÉMICO: Si Bolsa cae Y Miedo NO sube (o ambos suben), el mercado está roto.

3. CORRELACIÓN BTC vs ORO:
   - NORMAL: Baja/Nula. Son activos distintos.
   - LIQUIDEZ GLOBAL: Si BTC y ORO suben juntos, el mercado está huyendo del Dólar.
"""

# --- 3. CÁLCULOS DE INGENIERÍA FINANCIERA ---
# Retornos logarítmicos (fuerza bruta del movimiento)
log_rets = np.log(data / data.shift(1))

# Ventana de análisis: 21 días (1 mes de trading) para reacción rápida
ventana = 21

print("2. Calculando métricas de riesgo sistémico...")

# A. Volatilidad Rodante (¿Cuánto se mueve cada activo?)
vol_rolling = log_rets.rolling(ventana).std() * np.sqrt(252)

# B. Matriz de Correlación Rodante (El corazón del análisis)
# Queremos saber: ¿Se mueven NVDA y el SPY al unísono HOY?
# Calculamos la correlación de NVDA con cada activo en ventanas móviles
correlaciones = pd.DataFrame()

target = "NVDA"
for ticker in tickers.keys():
    if ticker != target:
        # Correlación entre los retornos de NVDA y el Ticker X
        corr_series = log_rets[target].rolling(ventana).corr(log_rets[ticker])
        correlaciones[f"CORR_{target}_{ticker}"] = corr_series

# --- 4. CONSOLIDACIÓN Y EXPORTACIÓN ---
# Unimos Volatilidades y Correlaciones
df_export = pd.concat([vol_rolling.add_suffix("_VOL"), correlaciones], axis=1).dropna()
df_export.reset_index(inplace=True)

archivo = "datos_macro_complejos.csv"
df_export.to_csv(archivo, index=False)

print(f"\n3. ANÁLISIS COMPLETADO.")
print(f"   Datos guardados en: {archivo}")
print("\n   --- INSTANTÁNEA DEL MERCADO ACTUAL (Último día) ---")
print(df_export.tail(1).T) # Transponemos para leer mejor en vertical