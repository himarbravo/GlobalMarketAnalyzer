import yfinance as yf
import config
import requests
import json
import re

class NewsAnalyst:
    def __init__(self):
        pass

    def get_news_sentiment(self, ticker_symbol):
        """
        Descarga noticias y calcula sentimiento usando AI.
        Retorna: {"Sentiment_Score": int, "News_Summary": str}
        """
        try:
            # 1. Fetch News
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news
            
            if not news:
                return {"Sentiment_Score": 0, "News_Summary": "Sin noticias recientes."}
            
            # Extract headlines (Last 3)
            headlines = []
            for n in news[:3]:
                # Intentar varios campos posibles por si yfinance cambia la API
                title = n.get('title') or n.get('content', {}).get('title')
                if title:
                    headlines.append(title)
            
            if not headlines:
                return {"Sentiment_Score": 0, "News_Summary": "No se encontraron titulares legibles."}

            headlines_text = "\n- ".join(headlines)
            
            # 2. AI Analysis
            result = self._analyze_with_ai(ticker_symbol, headlines_text)
            return result
            
        except Exception as e:
            print(f"⚠️ Error noticias {ticker_symbol}: {e}")
            return {"Sentiment_Score": 0, "News_Summary": "Error analizando noticias."}

    def _analyze_with_ai(self, ticker, headlines):
        """Envía los titulares al LLM para scoring."""
        provider = config.AI_CONFIG["PROVIDER"]
        
        # MOCK Fallback
        if provider == "MOCK":
            return {"Sentiment_Score": 10, "News_Summary": "Analisis simulado: Noticias mixtas."}

        # LOCAL / API
        try:
            system_prompt = "Eres un experto en sentimiento de mercado. Responde SOLO en JSON."
            user_prompt = config.PROMPTS["SENTIMENT_ANALYSIS"].format(ticker=ticker, headlines=headlines)
            
            # Construir payload para Ollama
            payload = {
                "model": config.AI_CONFIG["MODEL"],
                "prompt": f"System: {system_prompt}\nUser: {user_prompt}",
                "stream": False,
                "format": "json", # Ollama support parsing if model allows, forces structured output
                "options": {"temperature": 0.2}
            }

            if provider == "LOCAL":
                url = "http://localhost:11434/api/generate"    
            elif provider == "BRIDGE":
                url = "http://localhost:5050/api/generate"
                
            if provider in ["LOCAL", "BRIDGE"]:
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    json_str = response.json()['response']
                    # Intentar limpiar si el modelo es charlatán
                    data = self._clean_json(json_str)
                    return {
                        "Sentiment_Score": data.get("score", 0),
                        "News_Summary": data.get("summary", "N/A")
                    }
        except Exception as e:
            print(f"    (AI News Error: {e})")
            
        return {"Sentiment_Score": 0, "News_Summary": "Fallo en IA de Noticias"}

    def _clean_json(self, json_str):
        """Intenta extraer JSON válido de la respuesta de la IA."""
        try:
            return json.loads(json_str)
        except:
            # Simple regex fallback
            try:
                score = int(re.search(r'"score":\s*(-?\d+)', json_str).group(1))
                summary = re.search(r'"summary":\s*"(.*?)"', json_str).group(1)
                return {"score": score, "summary": summary}
            except:
                return {"score": 0, "summary": "Error parsing AI response"}
