# --- UNIVERSO DE ACTIVOS (COBERTURA INSTITUCIONAL: 150+ activos) ---
TICKERS = {
    "PORTFOLIO": ["NVDA", "BTC-USD"],
    "TECH_MEGA": ["GOOGL", "MSFT", "AAPL", "META", "AMZN"],
    "TECH_SEMIS": ["NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "MU", "QCOM"],
    "TECH_SOFTWARE": ["CRM", "ADBE", "NOW", "INTU", "WDAY", "TEAM"],
    "HEALTHCARE_PHARMA": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY"],
    "HEALTHCARE_BIOTECH": ["GILD", "AMGN", "REGN", "VRTX", "BIIB"],
    "HEALTHCARE_SERVICES": ["UNH", "CVS", "CI", "HUM"],
    "HEALTHCARE_DEVICES": ["TMO", "ABT", "DHR", "ISRG"],
    "FINANCIALS_BANKS": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "FINANCIALS_INSURANCE": ["BRK-B", "PGR", "ALL", "TRV"],
    "FINANCIALS_FINTECH": ["V", "MA", "PYPL", "SQ"],
    "ENERGY_OIL": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "ENERGY_RENEWABLE": ["NEE", "ENPH"],
    "CONSUMER_DISCRETIONARY": ["AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD"],
    "CONSUMER_STAPLES": ["WMT", "PG", "KO", "PEP", "COST", "CL"],
    "CONSUMER_ECOMMERCE": ["AMZN", "SHOP", "MELI"],
    "INDUSTRIALS_AERO": ["BA", "RTX", "LMT", "GD"],
    "INDUSTRIALS_MACHINERY": ["CAT", "DE", "CMI"],
    "INDUSTRIALS_TRANSPORT": ["UPS", "FDX", "DAL"],
    "COMMUNICATION": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
    "MATERIALS": ["LIN", "APD", "ECL", "NEM", "FCX"],
    "COMMODITIES_METALS": ["GLD", "SLV", "GDX"],
    "COMMODITIES_ENERGY": ["USO", "UNG"],
    "COMMODITIES_AGRI": ["DBA"],
    "REAL_ESTATE": ["PLD", "AMT", "EQIX", "PSA", "SPG"],
    "INTL_DEVELOPED": ["EWG", "EWU", "EWJ", "EWC", "EWA"],
    "INTL_EMERGING": ["FXI", "MCHI", "INDA", "EWZ", "EWW", "EWT", "EWY"],
    "INTL_STOCKS": ["BABA", "TSM", "ASML", "SAP", "NVO"],
    "CRYPTO": ["BTC-USD", "ETH-USD"],
    "BONDS_GOVT": ["TLT", "IEF", "SHY"],
    "BONDS_CORP": ["LQD", "HYG", "JNK"],
    "BONDS_INTL": ["BNDX"],
    "FACTORS": ["SPY", "QQQ", "IWM", "VTV", "VUG", "MTUM", "QUAL", "USMV"],
    "SECTOR_ETFS": ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE"],
    "MARKET_HEALTH": ["^VIX"]
}

# --- PARÁMETROS DE ESTRATEGIA ("LEJOS DEL RUIDO") ---
STRATEGY = {
    "TREND_WINDOW": 50,      # Días para definir la tendencia (SMA 50)
    "RSI_WINDOW": 14,        # Ventana estándar del RSI
    "RSI_OVERSOLD": 40,      # Comprar si está barato (y en tendencia) - Conservador (40 en vez de 30)
    "RSI_OVERBOUGHT": 75,    # Vender/Recortar si hay euforia
    "VOLATILITY_WINDOW": 21, # 1 mes de trading para medir "jaleo"
    "MAX_VIX": 25.0          # Si el VIX supera esto, NO COMPRAR NADA ("Stay Away")
}

# --- PARÁMETROS DE ANÁLISIS AVANZADO (NUEVO) ---
ANALYSIS_PARAMS = {
    "CONFIDENCE_WEIGHTS": {
        "TREND": 0.4,       # La tendencia es el rey
        "RSI": 0.3,         # El precio relativo importa
        "VOLATILITY": 0.3   # La seguridad es clave
    },
    "TREND_STRENGTH_THRESHOLD": 0.05 # 5% degap to consider strong trend
}

# --- INTEGRACIÓN IA (MODO FACTUAL ESTRICTO) ---
AI_CONFIG = {
    "PROVIDER": "BRIDGE",
    "API_KEY": None,
    "MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
    "TEMPERATURE": 0.1,   # Casi determinista para evitar alucinaciones
    "MAX_TOKENS": 200,     # Forzar concisión
    "TOP_P": 0.9
}

# --- PROMPTS (MODO FACTUAL - NO INTERPRETATIVO) ---
PROMPTS = {
    "SYSTEM_ROLE": """Eres un analista cuantitativo. SOLO reportas hechos numéricos verificables.
NO interpretes tendencias sin datos numéricos.
NO uses términos vagos como "broken", "frágil", "fuerte".
SOLO estados: alcista/bajista/lateral + porcentaje exacto.

Formato obligatorio:
- Dato numérico
- Comparación con referencia (ej: "5% sobre SMA50")
- Estado binario (alcista/bajista, sobre/bajo, alto/bajo)""",
    
    "MARKET_BRIEF": """Reporta SOLO hechos del mercado:

VIX: {VIX:.1f}
Régimen: {Regime} (confianza {RegimeConf:.0f}%)
Correlación promedio: {AvgCorr:.2f}
Estado: {CorrState}

En 2 frases:
1. Estado objetivo del VIX y régimen
2. Implicación para diversificación (correlación alta/baja)

NO uses metáforas. SOLO hechos numéricos.""",

    "ASSET_INSIGHT": """Reporta SOLO hechos para {Ticker}:

📊 Precio: ${Price:.2f}
├─ Señal: {Signal} (Convicción: {Confidence:.1f}%)
├─ Alpha: {Alpha:+.1f}% anual vs factores
├─ RSI: {RSI:.0f}{RSI_State}
├─ vs SMA50: {Price_vs_SMA}
├─ Régimen: {Regime}
└─ Target 1M: {Ito_Target}

En 1 frase: Resume SOLO el alpha y la señal. NO inventes interpretaciones.""",

    "SENTIMENT_ANALYSIS": """Analiza titulares para {ticker}:
{headlines}

Formato JSON obligatorio:
{{"score": -100 a +100, "summary": "evento clave en max 10 palabras"}}"""
}

# --- SITIOS DE ARCHIVOS ---
DATA_FILE = "datos_acciones.csv"
SIGNALS_FILE = "senales_diarias.csv"

# --- MATHEMATICA INTEGRATION ---
EXECUTE_MATHEMATICA = False
MATHEMATICA_PATH = "/Applications/Mathematica.app/Contents/MacOS/wolframscript"
NOTEBOOK_PATH = "Guasi_economica.nb"

# --- NOTIFICACIONES MÓVILES (Fase 3) ---
TELEGRAM_CONFIG = {
    "ENABLED": True,
    "BOT_TOKEN": "8404563487:AAFch-xukApTHXmVCO44kra5zhK10dbWnCI",
    "CHAT_ID": "7887900469"
}
