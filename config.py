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
    "INTL_STOCKS":   ["BABA", "SAP", "NVO"],
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
