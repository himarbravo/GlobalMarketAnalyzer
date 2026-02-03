import yfinance as yf
import pandas as pd
import numpy as np
import config
import subprocess
import os
import random
from datetime import datetime
from news_analyst import NewsAnalyst
from portfolio_manager import PortfolioManager
from notifier import Notifier
from database_manager import DatabaseManager 
from economist_engine import EconomistEngine
from regime_detector import RegimeDetector
from factor_analyzer import FactorAnalyzer
from correlation_monitor import CorrelationMonitor
import time

class GlobalAnalyst:
    def __init__(self):
        # Flatten all tickers into single list
        all_ticker_lists = list(config.TICKERS.values())
        self.all_tickers = list(set([t for sublist in all_ticker_lists for t in sublist]))
        
        self.data = pd.DataFrame()
        self.analysis_results = []
        self.market_narrative = ""
        self.db = DatabaseManager()
        print("\n🚀 Iniciando Protocolo de Analista Global (Quant Hedge Fund Edition)...")

    def fetch_data(self):
        print(f"📥 Recolectando inteligencia de mercado para {len(self.all_tickers)} activos...")
        
        # Download with longer history for regime detection
        raw_data = yf.download(self.all_tickers, period="2y", progress=False, auto_adjust=False)
        
        # Extract Adj Close (handles both single and multi-ticker downloads)
        if isinstance(raw_data.columns, pd.MultiIndex):
            self.data = raw_data['Adj Close']
        else:
            # Single ticker case
            self.data = raw_data[['Adj Close']].rename(columns={'Adj Close': self.all_tickers[0]})
        
        # Forward fill missing values (up to 5 days) - stocks don't trade on weekends
        self.data = self.data.fillna(method='ffill', limit=5)
        
        # Drop columns with more than 10% NaN in recent data (last 60 days)
        recent_data = self.data.tail(60)
        valid_tickers = []
        
        for col in self.data.columns:
            nan_pct = recent_data[col].isna().sum() / len(recent_data)
            if nan_pct < 0.10:  # Less than 10% missing
                valid_tickers.append(col)
            else:
                print(f"  ⚠️ Excluyendo {col}: {nan_pct*100:.1f}% datos faltantes recientes")
        
        self.data = self.data[valid_tickers]
        
        # Update all_tickers to only valid ones
        self.all_tickers = valid_tickers
        
        # Save history for GUI
        history_df = self.data.reset_index()
        history_df.to_csv(config.DATA_FILE, index=False)
        print(f"✅ Datos actualizados: {len(valid_tickers)} activos en {config.DATA_FILE}")

    def calculate_technical_mechanics(self, series):
        """Calcula los indicadores técnicos 'nut and bolts'."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.STRATEGY["RSI_WINDOW"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.STRATEGY["RSI_WINDOW"]).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        sma = series.rolling(window=config.STRATEGY["TREND_WINDOW"]).mean()
        
        # Volatilidad anualizada
        log_ret = np.log(series / series.shift(1))
        vol = log_ret.rolling(window=config.STRATEGY["VOLATILITY_WINDOW"]).std() * np.sqrt(252)
        
        return rsi.iloc[-1], sma.iloc[-1], vol.iloc[-1]

    def calculate_confidence_score(self, price, sma, rsi, vol):
        """
        Calcula un 'Score de Confianza' de 0 a 100.
        Cuanto más alto, más alineados están los astros (Tendencia, Momentum, Seguridad).
        """
        score = 0
        
        # 1. TENDENCIA (Weight 40%)
        # Si precio > SMA, ganamos puntos. Cuanto más lejos (con límite), mejor, hasta cierto punto de sobre-extensión.
        trend_diff = (price - sma) / sma
        if trend_diff > 0:
            # Tendencia alcista
            score += 40 * min(trend_diff / 0.10, 1.0) # Full points if 10% above SMA
        else:
            # Tendencia bajista - penalización masiva
            pass # 0 points

        # 2. RSI (Weight 30%)
        # Buscamos el "sweet spot". Ni muy bajo (crash) ni muy alto (explosión).
        # Ideal: 40-60 para entrar, 60-70 para mantener.
        if 40 <= rsi <= 70:
            score += 30
        elif rsi < 40:
            # Oversold - Puede ser bueno o cuchillo cayendo. Damos mitad de puntos.
            score += 15
        else: # > 70
            # Overbought - Peligroso
            score += 5
            
        # 3. VOLATILIDAD (Weight 30%)
        # Preferimos calma.
        if vol < 0.20: # Menos de 20% anual
            score += 30
        elif vol < 0.40:
            score += 15
        else:
            score += 0 # Demasiado ruido

        return round(score, 1)

    def detect_trend_strength(self, price, sma):
        diff_pct = (price - sma) / sma
        threshold = config.ANALYSIS_PARAMS["TREND_STRENGTH_THRESHOLD"]
        
        if diff_pct > threshold:
            return "STRONG"
        elif diff_pct > 0:
            return "STABLE"
        elif diff_pct > -threshold:
            return "FRAGILE"
        else:
            return "BROKEN"

    def generate_ai_narrative(self, context_type, data_dict):
        """
        Genera narrativa usando el proveedor configurado.
        Si es MOCK, devuelve textos pre-fabricados inteligentes.
        """
        provider = config.AI_CONFIG["PROVIDER"]
        
        if provider == "MOCK":
            return self._mock_ai_generator(context_type, data_dict)
        elif provider in ["LOCAL", "BRIDGE"]:
            return self._query_local_llm(context_type, data_dict)
        
        # Aquí iría la llamada real a OpenAI/Gemini
        return "AI Integration Not Yet Active (Check API Key)"

    def _query_local_llm(self, context_type, data):
        """Conecta con Ollama o Bridge con temperatura controlada."""
        try:
            import requests
            import json
            
            # Preparar el Prompt según el contexto
            system_prompt = config.PROMPTS["SYSTEM_ROLE"]
            
            if context_type == "MARKET_BRIEF":
                user_prompt = config.PROMPTS["MARKET_BRIEF"].format(**data)
            else:  # ASSET_INSIGHT
                user_prompt = config.PROMPTS["ASSET_INSIGHT"].format(**data)

            # Llamada API (Ollama o Bridge)
            provider = config.AI_CONFIG["PROVIDER"]
            port = 5050 if provider == "BRIDGE" else 11434
            url = f"http://localhost:{port}/api/generate"

            payload = {
                "model": config.AI_CONFIG["MODEL"],
                "prompt": f"System: {system_prompt}\nUser: {user_prompt}",
                "stream": False,
                "options": {
                    "temperature": config.AI_CONFIG.get("TEMPERATURE", 0.7),
                    "top_p": config.AI_CONFIG.get("TOP_P", 0.9),
                    "num_predict": config.AI_CONFIG.get("MAX_TOKENS", 200)
                }
            }
            
            print(f"  ... 🧠 Pensando ({provider} en :{port})... ", end="", flush=True)
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                print("✅")
                return response.json()['response'].strip()
            else:
                print(f"❌ Error {response.status_code}")
                return "Error conectando con la IA local."
                
        except Exception as e:
            print(f"❌ Fallo LLM: {e}")
            return "LLM no responde."

    def _mock_ai_generator(self, context_type, data):
        """Generador de texto 'falso' pero útil para cuando no hay API Key."""
        if context_type == "MARKET_BRIEF":
            vix = data.get('VIX', 20)
            if vix > 25:
                return "El mercado respira miedo. La alta volatilidad sugiere proteger capital y evitar héroes."
            elif vix < 15:
                return "Aguas tranquilas. El mercado favorece la toma de riesgos controlada. Buen momento para dejar correr ganancias."
            else:
                return "Entorno mixto. Hay oportunidades selectivas, pero el ruido de fondo requiere disciplina en los stops."
        
        elif context_type == "ASSET_INSIGHT":
            trend = data.get('Trend', 'NEUTRAL')
            conf = data.get('Confidence', 50)
            ticker = data.get('Ticker', 'Asset')
            
            if conf > 80:
                return f"Configuración técnica impecable para {ticker}. Los compradores tienen el control total."
            elif conf > 50:
                return f"{ticker} muestra constructividad, aunque le falta un catalizador fuerte para despegar."
            elif trend == "BROKEN":
                return f"Estructuralmente dañado. {ticker} necesita recuperar niveles clave antes de ser considerado."
            else:
                return f"Precaución en {ticker}. La relación riesgo/beneficio no es atractiva ahora mismo."
        
        return "Análisis no disponible."

    def run_workflow(self):
        from fundamental_analyst import FundamentalAnalyst
        from news_analyst import NewsAnalyst
        
        fund_analyst = FundamentalAnalyst()
        news_analyst = NewsAnalyst()
        
        # 1. Get Data
        self.fetch_data()
        
        # --- QUANT MODULE INITIALIZATION ---
        print("\n🔬 Inicializando Módulos Cuantitativos Avanzados...")
        
        # Ito Engine
        eco_engine = EconomistEngine(self.data)
        
        # Regime Detector
        regime_detector = RegimeDetector()
        regime_detector.load_model()  # Try loading trained model
        
        # Train if no model exists
        if regime_detector.model is None:
            print("  → Entrenando Detector de Régimen por primera vez...")
            spy_data = self.data['SPY'].pct_change().dropna()
            regime_detector.train(spy_data, save=True)
        
        # Factor Analyzer (uses SPY, VTV, IWM as factors)
        factor_data = self.data[['SPY', 'VTV', 'IWM']].dropna()
        factor_analyzer = FactorAnalyzer(factor_data)
        
        # Correlation Monitor
        corr_monitor = CorrelationMonitor(self.data)
        
        # --- REGIME DETECTION ---
        spy_recent = self.data['SPY'].pct_change().dropna().tail(40)
        regime_name, regime_conf, regime_params = regime_detector.predict_current_regime(spy_recent)
        regime_weights = regime_detector.get_regime_weights(regime_name)
        
        print(f"🎯 Régimen Detectado: {regime_name} (Confianza: {regime_conf:.0f}%)")
        print(f"   μ anual: {regime_params['mu_annual']*100:.1f}%, σ anual: {regime_params['sigma_annual']*100:.1f}%")
        
        # --- CORRELATION STATE ---
        corr_state = corr_monitor.get_current_state()
        div_penalty = corr_monitor.get_diversification_penalty(corr_state['avg_correlation'])
        
        print(f"{corr_state['color']} Estado de Correlación: {corr_state['state']} (Avg: {corr_state['avg_correlation']:.2f})")
        print(f"   {corr_state['recommendation']}")
        
        # 3. Market Health Summary (Direct Facts - NO LLM)
        try:
            vix_curr = self.data['^VIX'].iloc[-1]
            
            # Direct market summary
            market_narrative = f"""🌍 Estado del Mercado Global:
├─ VIX: {vix_curr:.1f} {'(Alta volatilidad)' if vix_curr > 20 else '(Baja volatilidad)'}
├─ Régimen: {regime_name} (confianza {regime_conf:.0f}%)
├─ Correlación: {corr_state['avg_correlation']:.2f} - {corr_state['state']}
└─ Recomendación: {corr_state['recommendation']}"""
        except:
             market_narrative = "Datos insuficientes para diagnóstico global."

        print(f"\n{market_narrative}")

        # 4. Deep Dive Analysis per Asset
        target_assets = config.TICKERS["PORTFOLIO"] + config.TICKERS.get("TECH", []) + config.TICKERS.get("HEALTHCARE", [])
        
        for ticker in target_assets:
            if ticker not in self.data.columns:
                continue
            
            # --- Technical Mechanics ---
            series = self.data[ticker]
            price = series.iloc[-1]
            rsi, sma, vol = self.calculate_technical_mechanics(series)
            tech_conf = self.calculate_confidence_score(price, sma, rsi, vol)
            trend_str = self.detect_trend_strength(price, sma)
            
            # --- Ito/GBM Analysis ---
            eco_data = eco_engine.analyze_asset(ticker)
            ito_signal = "NEUTRO"
            if eco_data:
                if price <= eco_data["support_10"]: ito_signal = "COMPRAR (Barato)"
                elif price >= eco_data["resistance_90"]: ito_signal = "VENDER (Caro)"

            # --- Fundamental Deep Dive ---
            print(f"  🔍 Analizando {ticker}...", end="")
            fund_data = fund_analyst.get_fundamentals(ticker)
            fund_score = fund_data.get('Fundamental_Score', 50)
            
            # --- Factor Alpha (Fama-French) ---
            asset_returns = series.pct_change().dropna()
            factor_result = factor_analyzer.decompose_asset(asset_returns)
            
            alpha_bonus = 0
            if factor_result:
                alpha_bonus = factor_analyzer.generate_alpha_bonus(factor_result['alpha_annual_pct'])
                print(f" | α={factor_result['alpha_annual_pct']:.1f}%",  end="")
            
            # --- News Sentiment ---
            news_data = news_analyst.get_news_sentiment(ticker)
            sent_score = news_data.get("Sentiment_Score", 0)
            sent_score_norm = max(0, min(100, (sent_score + 100) / 2)) 
            
            # Bonus Ito
            bonus_ito = 10 if "COMPRAR" in ito_signal else (-10 if "VENDER" in ito_signal else 0)
            
            # --- REGIME-ADJUSTED TOTAL CONVICTION ---
            # Regime weights are already fractions (e.g., 0.3 for 30%)
            raw_conviction = (
                tech_conf * regime_weights['technical'] +
                fund_score * regime_weights['fundamental'] +
                sent_score_norm * regime_weights['sentiment']
            )  # Already in 0-100 scale
            
            # Add factor alpha bonus
            raw_conviction += alpha_bonus
            
            # Add ito bonus (scaled by regime weight)
            raw_conviction += bonus_ito * regime_weights['ito']
            
            # Apply diversification penalty
            total_conviction = raw_conviction * div_penalty
            total_conviction = max(0, min(100, round(total_conviction, 1)))

            # --- PRE-PROCESS ALL INTERPRETATIONS (NO LLM HALLUCINATIONS) ---
            # RSI State
            if rsi < 30:
                rsi_state = " (sobreventa)"
            elif rsi > 70:
                rsi_state = " (sobrecompra)"
            else:
                rsi_state = " (neutral)"
            
            # Price vs SMA50
            price_vs_sma_pct = ((price / sma - 1) * 100)
            if price_vs_sma_pct > 0:
                price_vs_sma_str = f"+{price_vs_sma_pct:.1f}% sobre SMA50 (alcista)"
            else:
                price_vs_sma_str = f"{price_vs_sma_pct:.1f}% bajo SMA50 (bajista)"
            
            # Ito Target formatting
            ito_target_str = f"${eco_data['target_median']:.2f}" if eco_data else "N/A"

            #  Determine Action
            if total_conviction >= 75: signal = "STRONG BUY"
            elif total_conviction >= 60: signal = "BUY"
            elif total_conviction <= 30: signal = "SELL"
            elif total_conviction <= 45: signal = "AVOID"
            else: signal = "HOLD"

            # AI Insight with ONLY pre-processed facts
            ai_data_packet = {
                "Ticker": ticker,
                "Price": f"{price:.2f}",
                "Signal": signal,
                "Confidence": f"{total_conviction:.1f}",
                "Alpha": f"{factor_result['alpha_annual_pct']:.1f}" if factor_result else "0.0",
                "RSI": int(rsi),
                "RSI_State": rsi_state,
                "Price_vs_SMA": price_vs_sma_str,
                "Regime": regime_name,
                "Ito_Target": ito_target_str
            }
            
            # Direct factual report (NO LLM)
            ai_insight = f"""📊 {ticker} - ${price:.2f}
├─ Señal: {signal} (Convicción: {total_conviction:.1f}%)
├─ Alpha Fama-French: {factor_result['alpha_annual_pct']:+.1f}% anual vs factores
├─ RSI: {int(rsi)}{rsi_state}
├─ Precio vs SMA50: {price_vs_sma_str}
├─ Régimen Actual: {regime_name}
└─ Target 1M (Itō): {ito_target_str}"""
            
            # --- MEMORIA CICLO (BRAIN) ---
            self.db.log_signal(ticker, signal, total_conviction, price, ai_insight)

            self.analysis_results.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Price": round(price, 2),
                "Signal": signal,
                "Confidence_Score": total_conviction,
                "Regime": regime_name,
                "Alpha_Pct": round(factor_result['alpha_annual_pct'], 2) if factor_result else 0,
                "Ito_Signal": ito_signal,
                "Target_1M": round(eco_data["target_median"], 2) if eco_data else 0,
                "Corr_State": corr_state['state'],
                "AI_Insight": ai_insight,
                "Market_Context": market_narrative
            })
            
            print(f" → {signal} (Conv: {total_conviction:.1f}) | Itō: {ito_signal}")

        # 5. Export Report
        df_results = pd.DataFrame(self.analysis_results)
        df_results.to_csv(config.SIGNALS_FILE, index=False)
        print(f"\n✅ Informe de Inteligencia generado: {config.SIGNALS_FILE}")
        
        # 6. Portfolio Audit & Trading Status
        print("⚖️  Auditando riesgo de cartera...")
        
        from portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        risk_insight, _ = pm.analyze_risk()
        
        final_report_msg = ""
        # 7. Mobile Alert (Phase 3)
        from notifier import Notifier
        notifier = Notifier()
        
        # Filtrar Top Picks para el mensaje
        top_picks = [r for r in self.analysis_results if r['Confidence_Score'] >= 70]
        
        # Construir mensaje combinado
        if notifier.enabled:
            # We construct a custom message
            final_msg = notifier.format_morning_brief(market_narrative, top_picks, risk_insight)
            notifier.send_alert(final_msg)

if __name__ == "__main__":
    analyst = GlobalAnalyst()
    analyst.run_workflow()
