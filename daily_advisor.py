import yfinance as yf
import pandas as pd
import numpy as np
import config
import subprocess
import os
from datetime import datetime

class DailyAdvisor:
    def __init__(self):
        self.all_tickers = list(set(config.TICKERS["PORTFOLIO"] + config.TICKERS["WATCHLIST"] + config.TICKERS["MARKET_HEALTH"]))
        self.data = pd.DataFrame()
        self.signals = []

    def execute_mathematica(self):
        """Ejecuta el notebook de Mathematica para actualizar cálculos externos."""
        if not config.EXECUTE_MATHEMATICA:
            return

        print(f"🧬 Ejecutando Mathematica: {config.NOTEBOOK_PATH}...")
        try:
            # Comando para evaluar el notebook usando wolframscript
            # Usamos UsingFrontEnd por si el notebook tiene elementos dinámicos
            cmd = [
                config.MATHEMATICA_PATH,
                "-code",
                f'UsingFrontEnd[NotebookEvaluate["{config.NOTEBOOK_PATH}"]];'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Mathematica ejecutado con éxito.")
            else:
                print(f"⚠️ Error en Mathematica: {result.stderr}")
        except Exception as e:
            print(f"❌ Fallo al intentar conectar con Mathematica: {e}")

    def fetch_data(self):
        print(f"📥 Descargando datos para {len(self.all_tickers)} activos...")
        # Descargar historial suficiente para calcular SMA 200 si fuera necesario, usamos 2y por seguridad
        self.data = yf.download(self.all_tickers, period="2y", progress=False, auto_adjust=False)['Adj Close']
        
        # Guardar historial limpio para la GUI
        history_df = self.data.reset_index()
        history_df.to_csv(config.DATA_FILE, index=False)
        print(f"✅ Datos guardados en {config.DATA_FILE}")

    def calculate_indicators(self, series):
        """Calcula los indicadores técnicos básicos para una serie de precios."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.STRATEGY["RSI_WINDOW"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.STRATEGY["RSI_WINDOW"]).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        sma = series.rolling(window=config.STRATEGY["TREND_WINDOW"]).mean()
        
        # Volatilidad anualizada (ventana corta)
        log_ret = np.log(series / series.shift(1))
        vol = log_ret.rolling(window=config.STRATEGY["VOLATILITY_WINDOW"]).std() * np.sqrt(252)
        
        return rsi.iloc[-1], sma.iloc[-1], vol.iloc[-1]

    def analyze_market_health(self):
        """Determina si es seguro operar hoy."""
        try:
            current_vix = self.data['^VIX'].iloc[-1]
            spy_price = self.data['SPY'].iloc[-1]
            spy_sma = self.data['SPY'].rolling(window=config.STRATEGY["TREND_WINDOW"]).mean().iloc[-1]
            
            status = "NEUTRAL"
            if current_vix > config.STRATEGY["MAX_VIX"]:
                status = "DANGER"
                msg = f"⛔ ALTO RIESGO (VIX {current_vix:.2f}). No abrir nuevas posiciones."
            elif spy_price < spy_sma:
                status = "CAUTION"
                msg = f"⚠️ PRECAUCIÓN. Mercado bajo su media de {config.STRATEGY['TREND_WINDOW']} dias."
            else:
                status = "SAFE"
                msg = f"✅ MERCADO SALUDABLE. Tendencia alcista y volatilidad controlada."
            
            return status, msg
        except KeyError:
            return "UNKNOWN", "Datos de mercado insuficientes para diagnóstico."

    def generate_signals(self):
        market_status, market_msg = self.analyze_market_health()
        print(f"\n🌍 {market_msg}")

        today = datetime.now().strftime("%Y-%m-%d")
        
        report = []
        
        # Analizar Portfolio y Watchlist (excluyendo índices de referencia puros que no se operan directamente si no se quiere)
        target_assets = config.TICKERS["PORTFOLIO"] + config.TICKERS["WATCHLIST"]
        
        for ticker in target_assets:
            if ticker not in self.data.columns:
                continue
                
            price = self.data[ticker].iloc[-1]
            rsi, sma, vol = self.calculate_indicators(self.data[ticker])
            
            signal = "WAIT"
            reason = "Sin señal clara"
            action_type = "NEUTRAL" # BULLISH, BEARISH, NEUTRAL

            # LÓGICA DE DECISIÓN "LEJOS DEL RUIDO"
            
            # 1. Chequeo de seguridad primero
            if market_status == "DANGER":
                signal = "AVOID"
                reason = "Mercado peligroso (VIX alto)"
                action_type = "BEARISH"
            
            # 2. Análisis de Tendencia
            elif price > sma:
                # TENDENCIA ALCISTA
                if rsi < config.STRATEGY["RSI_OVERSOLD"]:
                    signal = "BUY"
                    reason = "Tendencia alcista + Retroceso (Oversold)"
                    action_type = "BULLISH"
                elif rsi > config.STRATEGY["RSI_OVERBOUGHT"]:
                    signal = "TRIM"
                    reason = "Euforia: Tomar beneficios parciales"
                    action_type = "BEARISH"
                else:
                    signal = "HOLD"
                    reason = "Mantener tendencia"
                    action_type = "BULLISH"
            else:
                # TENDENCIA BAJISTA (Precio < SMA)
                if ticker in config.TICKERS["PORTFOLIO"]:
                    signal = "SELL/CUT"
                    reason = "Tendencia rota (Stop Loss)"
                    action_type = "BEARISH"
                else:
                    signal = "IGNORE"
                    reason = "Tendencia bajista (No tocar)"
                    action_type = "NEUTRAL"

            report.append({
                "Date": today,
                "Ticker": ticker,
                "Price": round(price, 2),
                "Signal": signal,
                "Reason": reason,
                "RSI": round(rsi, 2),
                "Trend_SMA": "UP" if price > sma else "DOWN",
                "Volatility": round(vol * 100, 1),
                "Type": action_type
            })
            
            # Imprimir en terminal cosas importantes
            if signal in ["BUY", "SELL/CUT", "TRIM"]:
                icon = "🟢" if signal == "BUY" else "🔴" if "SELL" in signal else "✂️"
                print(f"{icon} {ticker}: {signal} ({reason})")

        # Guardar reporte
        df_report = pd.DataFrame(report)
        df_report.to_csv(config.SIGNALS_FILE, index=False)
        print(f"\n📋 Reporte de señales guardado en {config.SIGNALS_FILE}")

if __name__ == "__main__":
    advisor = DailyAdvisor()
    advisor.execute_mathematica() # Nuevo: Ejecuta Mathematica primero
    advisor.fetch_data()
    advisor.generate_signals()
