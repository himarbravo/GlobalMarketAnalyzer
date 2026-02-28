"""
CONFIG — GlobalMarketAnalyzer
================================
Configuración central. Credenciales desde .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ─── SUPABASE ────────────────────────────────────────────────────────────────
SUPABASE_URL              = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY         = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_DB_PASSWORD      = os.getenv("SUPABASE_DB_PASSWORD", "")

_project_id = SUPABASE_URL.replace("https://", "").replace(".supabase.co", "")
SUPABASE_DB_URL = (
    f"postgresql://postgres.{_project_id}:{SUPABASE_DB_PASSWORD}"
    f"@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"
)

# ─── FRED (Federal Reserve Economic Data) ─────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")



# ─── UNIVERSO DE ACTIVOS ──────────────────────────────────────────────────────
TICKERS = {
    "TECH_MEGA":     ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "TECH_SEMIS":    ["NVDA", "AMD", "TSM", "AVGO", "ASML", "MU", "QCOM"],
    "TECH_SOFTWARE": ["CRM", "ADBE", "NOW", "INTU", "WDAY"],
    "HEALTHCARE":    ["JNJ", "LLY", "ABBV", "MRK", "UNH", "PFE"],
    "BIOTECH":       ["GILD", "AMGN", "REGN", "VRTX"],
    "BANKS":         ["JPM", "BAC", "GS", "MS", "WFC"],
    "FINTECH":       ["V", "MA", "PYPL"],
    "ENERGY":        ["XOM", "CVX", "COP", "SLB"],
    "RENEWABLE":     ["NEE", "ENPH"],
    "CONSUMER_DISC": ["TSLA", "HD", "NKE", "MCD", "SBUX"],
    "CONSUMER_STA":  ["WMT", "PG", "KO", "PEP", "COST"],
    "INDUSTRIALS":   ["CAT", "DE", "BA", "RTX", "UPS"],
    "MATERIALS":     ["LIN", "FCX", "NEM"],
    "REAL_ESTATE":   ["PLD", "AMT", "EQIX"],
    "INTL_DEV":      ["EWG", "EWJ", "EWU", "EWC"],
    "INTL_EM":       ["FXI", "INDA", "EWZ", "EWT"],
    "INTL_STOCKS":   ["BABA", "SAP", "NVO", "LVMHF", "TTE", "SIE", "AZN",
                      "SONY", "TM"],
    "INTL_BANKS":    ["HSBC", "BNP.PA", "SAN", "ING",      # EUR banks
                      "MUFG", "SMFG",                      # JP banks
                      "ITUB", "HDB"],                       # EM banks
    "INTL_INDUSTRY": ["VALE", "PBR"],                       # EM industry
    "CRYPTO":        ["BTC-USD", "ETH-USD"],
    "BONDS_GOVT":    ["TLT", "IEF", "SHY"],
    "BONDS_CORP":    ["LQD", "HYG"],
    "FACTORS":       ["SPY", "QQQ", "IWM", "VTV", "VUG", "MTUM", "QUAL", "USMV"],
    "SECTORS":       ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE"],
    "COMMODITIES":   ["GLD", "SLV", "USO", "DBA"],
}

# Tickers de sector ETFs (para tabla sector_performance)
SECTOR_ETFS = {
    "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
    "XLE": "Energy", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
}

ASSET_TYPE_MAP = {
    "BTC-USD": "crypto", "ETH-USD": "crypto",
    "TLT": "bond", "IEF": "bond", "SHY": "bond", "LQD": "bond", "HYG": "bond",
    "GLD": "commodity", "SLV": "commodity", "USO": "commodity", "DBA": "commodity",
}

ETF_TICKERS = set(
    TICKERS["FACTORS"] + TICKERS["SECTORS"] + TICKERS["INTL_DEV"] +
    TICKERS["INTL_EM"] + TICKERS["BONDS_GOVT"] + TICKERS["BONDS_CORP"] +
    TICKERS["COMMODITIES"]
)

ASSET_TYPE_MAP.update({
    "LVMHF": "equity", "TTE": "equity", "SIE": "equity",
    "AZN": "equity", "SONY": "equity", "TM": "equity",
    "HSBC": "equity", "BNP.PA": "equity", "SAN": "equity", "ING": "equity",
    "MUFG": "equity", "SMFG": "equity",
    "ITUB": "equity", "HDB": "equity", "VALE": "equity", "PBR": "equity",
})


# ─── GRAFO JERÁRQUICO: ROLES + DIMENSIONES ──────────────────────────────────

# Nodos bancarios: crean dinero vía préstamos (f_bank = NIM × lending)
NODE_ROLES = {
    # US banks
    "JPM": "bank", "BAC": "bank", "GS": "bank", "MS": "bank", "WFC": "bank",
    # EUR banks
    "HSBC": "bank", "BNP.PA": "bank", "SAN": "bank", "ING": "bank",
    # JP banks
    "MUFG": "bank", "SMFG": "bank",
    # EM banks
    "ITUB": "bank", "HDB": "bank",
}
# Todo ticker NO listado aquí → role = "productive"

# País de cotización/exposición de cada ticker no-US
TICKER_COUNTRY = {
    # China
    "BABA": "CN", "FXI": "CN",
    # Europe
    "SAP": "DE", "EWG": "DE", "SIE": "DE",
    "NVO": "DK",
    "ASML": "NL", "ING": "NL",
    "EWU": "UK", "HSBC": "UK", "AZN": "UK",
    "BNP.PA": "FR", "LVMHF": "FR", "TTE": "FR",
    "SAN": "ES",
    # Asia
    "TSM": "TW", "EWT": "TW",
    "EWJ": "JP", "MUFG": "JP", "SMFG": "JP", "SONY": "JP", "TM": "JP",
    # Americas (non-US)
    "EWC": "CA",
    # Emerging Markets
    "EWZ": "BR", "ITUB": "BR", "VALE": "BR", "PBR": "BR",
    "INDA": "IN", "HDB": "IN",
}
# Todo ticker NO listado aquí → country = "US"

# ─── CAMPOS MONETARIOS POR DIVISA ────────────────────────────────────────────

# Zonas monetarias: cada zona tiene su propio Laplaciano y campo m
# Tickers se asignan a la zona de su país (via TICKER_COUNTRY)
COUNTRY_TO_ZONE = {
    "US": "USD", "CA": "USD",       # Norteamérica dolarizada
    "DE": "EUR", "NL": "EUR", "DK": "EUR", "UK": "EUR",
    "FR": "EUR", "ES": "EUR",       # Europa
    "JP": "ASIA", "CN": "ASIA", "TW": "ASIA",            # Asia
    "BR": "EM", "IN": "EM",                               # Emergentes
}
# Todo país no listado → "USD"

# Pares FX entre zonas: campo en macro_indicators + convención de signo
# sign: +1 si FX sube = moneda zona se fortalece, -1 si FX sube = se debilita
FX_PAIRS = {
    ("USD", "EUR"):  {"column": "eurusd",  "sign": -1},  # EURUSD sube → USD débil
    ("USD", "ASIA"): {"column": "usdjpy",  "sign": +1},  # USDJPY sube → USD fuerte vs Asia
    ("USD", "EM"):   {"column": "dxy",     "sign": +1},  # DXY sube → USD fuerte vs EM
}

# Dim 2: Deuda soberana / PIB (ratio, fuente: FMI 2024)
SOVEREIGN_DEBT_GDP = {
    "US": 1.20, "JP": 2.60, "DE": 0.65, "NL": 0.50,
    "CN": 0.80, "TW": 0.30, "DK": 0.35, "UK": 1.00,
    "BR": 0.75, "IN": 0.85, "CA": 0.65,
    "FR": 1.10, "ES": 1.05,
}

# Dim 3: Tipos de interés por banco central (FRED series)
CENTRAL_BANK_RATES = {
    "US": "FEDFUNDS",          # Federal Funds Rate
    "EU": "ECBDFR",            # ECB Deposit Facility Rate
    "JP": "IRSTCI01JPM156N",   # Japan Immediate Rate (call money)
    "UK": "IUDSOIA",           # UK Sterling Overnight Rate
}

# Indicadores macro internacionales (FRED series)
INTL_MACRO_SERIES = {
    # PMI manufacturero (>50 = expansión, <50 = contracción)
    "pmi_us":  "MANEMP",           # ISM Manufacturing Employment (proxy PMI)
    "pmi_eu":  "BSCICP03EZM460S",  # EU Business Confidence (proxy PMI)
    "pmi_jp":  "BSCICP03JPM460S",  # Japan Business Confidence
    "pmi_cn":  "BSCICP03CNM460S",  # China Business Confidence
    "pmi_uk":  "BSCICP03GBM460S",  # UK Business Confidence

    # PIB crecimiento real (trimestral)
    "gdp_us":  "A191RL1Q225SBEA",   # US Real GDP Growth (annualized)
    "gdp_eu":  "CLVMNACSCAB1GQEA19", # Euro Area Real GDP
    "gdp_jp":  "JPNRGDPEXP",        # Japan Real GDP
    "gdp_uk":  "NAEXKP01GBQ189S",   # UK Real GDP

    # Desempleo (mensual)
    "unemp_us": "UNRATE",           # US Unemployment Rate
    "unemp_eu": "LRHUTTTTEZM156S",  # Euro Area Unemployment
    "unemp_jp": "LRHUTTTTJPM156S",  # Japan Unemployment
}

# Mapeo: país → serie de tipo de interés de su banco central
COUNTRY_RATE_SERIES = {
    "US": "FEDFUNDS", "CA": "FEDFUNDS",     # USD zone
    "DE": "ECBDFR", "NL": "ECBDFR", "DK": "ECBDFR",
    "FR": "ECBDFR", "ES": "ECBDFR",          # EUR zone
    "JP": "IRSTCI01JPM156N",                 # Asia zone
    "UK": "IUDSOIA",                         # UK
    "CN": "FEDFUNDS", "TW": "FEDFUNDS",     # proxy: USD-linked
    "BR": "FEDFUNDS", "IN": "FEDFUNDS",     # proxy: USD-linked
}

# Legacy alias
FED_RATE_SERIES = "FEDFUNDS"

# Parámetros de correcciones dimensionales (calibrables)
DIM_PARAMS = {
    "beta_fx":      0.30,    # elasticidad flujo FX entre zonas
    "eta_debt":     0.02,    # peso drenaje deuda soberana
    "beta_r_bank":  -0.50,   # bancos ganan con subida tipos (negativo)
    "beta_r_prod":  0.30,    # empresas sufren (positivo, ×(1+leverage))
}

# Dim 1: Divisa por país → campo en macro_indicators (legacy, para graph_builder)
COUNTRY_CURRENCY = {
    "US": "dxy", "JP": "usdjpy", "DE": "eurusd", "NL": "eurusd",
    "DK": "eurusd", "UK": "gbpusd", "CN": "dxy", "TW": "dxy",
    "BR": "dxy", "IN": "dxy", "CA": "dxy",
    "FR": "eurusd", "ES": "eurusd",
}


def get_all_tickers() -> list:
    seen, result = set(), []
    for tickers in TICKERS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def get_asset_type(ticker: str) -> str:
    if ticker in ASSET_TYPE_MAP:
        return ASSET_TYPE_MAP[ticker]
    if ticker in ETF_TICKERS:
        return "etf"
    return "equity"


def get_sector(ticker: str) -> str:
    for sector, tickers in TICKERS.items():
        if ticker in tickers:
            return sector
    return "UNKNOWN"


def get_node_role(ticker: str) -> str:
    """Rol del nodo en el grafo jerárquico: 'bank' o 'productive'."""
    return NODE_ROLES.get(ticker, "productive")


def get_country(ticker: str) -> str:
    """País de exposición principal del ticker."""
    return TICKER_COUNTRY.get(ticker, "US")


def get_zone(ticker: str) -> str:
    """Zona monetaria del ticker: 'USD', 'EUR', 'ASIA', 'EM'."""
    country = get_country(ticker)
    return COUNTRY_TO_ZONE.get(country, "USD")


# ─── INDICADORES TÉCNICOS ─────────────────────────────────────────────────────
INDICATORS = {
    "SMA":         [5, 10, 20, 50, 100, 200],
    "EMA":         [9, 12, 21, 26, 50],
    "RSI":         [7, 14],
    "BB":          {"window": 20, "std": 2},
    "MACD":        {"fast": 12, "slow": 26, "signal": 9},
    "ATR":         [7, 14],
    "STOCHASTIC":  {"k_period": 14, "d_period": 3},
    "WILLIAMS_R":  14,
    "ADX":         14,
    "KELTNER":     {"ema": 20, "atr_mult": 2, "atr_period": 14},
    "ICHIMOKU":    {"tenkan": 9, "kijun": 26, "senkou_b": 52},
    "OBV":         True,
    "MFI":         14,
    "CMF":         20,
    "VOL_WINDOWS": [5, 10, 20, 60],
    "RETURNS":     [1, 2, 3, 5, 10, 20, 60, 120, 252],
    "SHARPE":      [20, 60],
    "SORTINO":     [20],
    "MAX_DD":      [20, 60],
}


# ─── MACRO (yfinance tickers → campo en DB) ───────────────────────────────────
MACRO_TICKERS = {
    "vix":        "^VIX",
    "sp500":      "^GSPC",
    "nasdaq":     "^IXIC",
    "dow_jones":  "^DJI",
    "russell_2000": "^RUT",
    "dxy":        "DX-Y.NYB",
    "gold":       "GC=F",
    "silver":     "SI=F",
    "oil_wti":    "CL=F",
    "oil_brent":  "BZ=F",
    "natural_gas": "NG=F",
    "copper":     "HG=F",
    "yield_2y":   "^IRX",
    "yield_10y":  "^TNX",
    "yield_30y":  "^TYX",
    "btc_usd":    "BTC-USD",
    "eth_usd":    "ETH-USD",
    "eurusd":     "EURUSD=X",
    "usdjpy":     "JPY=X",
    "gbpusd":     "GBPUSD=X",
}


# ─── INGESTA ──────────────────────────────────────────────────────────────────
INGESTION = {
    "DEFAULT_PERIOD":          "2y",
    "INCREMENTAL_BUFFER_DAYS": 7,
    "BATCH_SIZE":              15,
    "DELAY_BETWEEN_BATCHES":   1.2,
}
