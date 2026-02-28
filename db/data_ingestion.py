"""
DATA INGESTION PIPELINE — GlobalMarketAnalyzer
================================================
Descarga datos de mercado desde yfinance y los sube a Supabase.
Calcula ~100 indicadores técnicos por cada fila de precios.

Modos:
  historical    Histórico completo (default: 2 años)
  incremental   Solo días nuevos
  macro         Solo indicadores macro
  fundamentals  Solo datos fundamentales por empresa
  sectors       Solo rotación sectorial

Uso:
  python data_ingestion.py                            # incremental
  python data_ingestion.py --mode historical
  python data_ingestion.py --mode historical --period 5y
  python data_ingestion.py --tickers NVDA AAPL MSFT   # específicos
"""

import argparse
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import config
from db.database_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

R = lambda x, d=4: round(x, d) if pd.notna(x) else None


# ═════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE INDICADORES TÉCNICOS (~100 columnas)
# ═════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos sobre DataFrame OHLCV."""
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)
    v  = df["volume"].astype(float)
    p  = config.INDICATORS

    # ══════════ MEDIAS MÓVILES ══════════
    for w in p["SMA"]:
        df[f"sma_{w}"] = c.rolling(w).mean()
    for w in p["EMA"]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # ══════════ RSI ══════════
    for period in p["RSI"]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # ══════════ MACD ══════════
    m = p["MACD"]
    ema_fast       = c.ewm(span=m["fast"], adjust=False).mean()
    ema_slow       = c.ewm(span=m["slow"], adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=m["signal"], adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ══════════ STOCHASTIC %K / %D ══════════
    st = p["STOCHASTIC"]
    low_n  = lo.rolling(st["k_period"]).min()
    high_n = h.rolling(st["k_period"]).max()
    df["stoch_k"] = (c - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    df["stoch_d"] = df["stoch_k"].rolling(st["d_period"]).mean()

    # ══════════ WILLIAMS %R ══════════
    wr_period = p["WILLIAMS_R"]
    wr_high = h.rolling(wr_period).max()
    wr_low  = lo.rolling(wr_period).min()
    df["williams_r"] = (wr_high - c) / (wr_high - wr_low).replace(0, np.nan) * -100

    # ══════════ ADX / +DI / -DI ══════════
    adx_p = p["ADX"]
    up_move   = h.diff()
    down_move = -lo.diff()
    plus_dm   = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm  = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    tr_series = pd.concat([
        h - lo,
        (h - c.shift()).abs(),
        (lo - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr_adx = tr_series.rolling(adx_p).mean()
    plus_di  = plus_dm.rolling(adx_p).mean() / atr_adx.replace(0, np.nan) * 100
    minus_di = minus_dm.rolling(adx_p).mean() / atr_adx.replace(0, np.nan) * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100

    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    df["adx"]      = dx.rolling(adx_p).mean()

    # ══════════ BOLLINGER BANDS ══════════
    bb = p["BB"]
    bb_mid = c.rolling(bb["window"]).mean()
    bb_std = c.rolling(bb["window"]).std()
    df["bb_middle"] = bb_mid
    df["bb_upper"]  = bb_mid + bb["std"] * bb_std
    df["bb_lower"]  = bb_mid - bb["std"] * bb_std
    bb_range = (df["bb_upper"] - df["bb_lower"])
    df["bb_width"]  = bb_range / bb_mid.replace(0, np.nan)
    df["bb_pct"]    = (c - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # ══════════ KELTNER CHANNELS ══════════
    kc = p["KELTNER"]
    kc_mid = c.ewm(span=kc["ema"], adjust=False).mean()
    kc_atr = tr_series.rolling(kc["atr_period"]).mean()
    df["keltner_upper"] = kc_mid + kc["atr_mult"] * kc_atr
    df["keltner_lower"] = kc_mid - kc["atr_mult"] * kc_atr

    # ══════════ ATR ══════════
    for atr_w in p["ATR"]:
        df[f"atr_{atr_w}"] = tr_series.rolling(atr_w).mean()

    # ══════════ VOLATILIDAD REALIZADA ══════════
    daily_ret = c.pct_change()
    for vw in p["VOL_WINDOWS"]:
        df[f"vol_{vw}d"] = daily_ret.rolling(vw).std() * np.sqrt(252) * 100

    # ══════════ ON-BALANCE VOLUME ══════════
    obv_dir = np.where(c > c.shift(), 1, np.where(c < c.shift(), -1, 0))
    df["obv"] = (v * obv_dir).cumsum()

    # ══════════ VWAP (media ponderada por volumen, aprox diaria) ══════════
    typical_price = (h + lo + c) / 3
    df["vwap"] = (typical_price * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # ══════════ VOLUMEN ══════════
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_ratio"]  = v / v.rolling(20).mean().replace(0, np.nan)

    # ══════════ MFI (Money Flow Index) ══════════
    mfi_period = p["MFI"]
    mf = typical_price * v
    pos_mf = pd.Series(np.where(typical_price > typical_price.shift(), mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(typical_price < typical_price.shift(), mf, 0), index=df.index)
    mf_ratio = pos_mf.rolling(mfi_period).sum() / neg_mf.rolling(mfi_period).sum().replace(0, np.nan)
    df["mfi"] = 100 - 100 / (1 + mf_ratio)

    # ══════════ CMF (Chaikin Money Flow) ══════════
    cmf_period = p["CMF"]
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    df["cmf"] = (clv * v).rolling(cmf_period).sum() / v.rolling(cmf_period).sum().replace(0, np.nan)

    # ══════════ ICHIMOKU CLOUD ══════════
    ichi = p["ICHIMOKU"]
    df["ichimoku_tenkan"]   = (h.rolling(ichi["tenkan"]).max() + lo.rolling(ichi["tenkan"]).min()) / 2
    df["ichimoku_kijun"]    = (h.rolling(ichi["kijun"]).max() + lo.rolling(ichi["kijun"]).min()) / 2
    df["ichimoku_senkou_a"] = (df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2
    df["ichimoku_senkou_b"] = (h.rolling(ichi["senkou_b"]).max() + lo.rolling(ichi["senkou_b"]).min()) / 2

    # ══════════ PIVOT POINTS ══════════
    df["pivot"]    = (h.shift() + lo.shift() + c.shift()) / 3
    df["pivot_r1"] = 2 * df["pivot"] - lo.shift()
    df["pivot_s1"] = 2 * df["pivot"] - h.shift()
    df["pivot_r2"] = df["pivot"] + (h.shift() - lo.shift())
    df["pivot_s2"] = df["pivot"] - (h.shift() - lo.shift())

    # ══════════ RETORNOS ══════════
    for days in p["RETURNS"]:
        df[f"returns_{days}d"] = c.pct_change(days) * 100

    # ══════════ SHARPE RATIO ══════════
    for sw in p["SHARPE"]:
        mean_ret = daily_ret.rolling(sw).mean()
        std_ret  = daily_ret.rolling(sw).std()
        df[f"sharpe_{sw}d"] = (mean_ret / std_ret.replace(0, np.nan)) * np.sqrt(252)

    # ══════════ SORTINO RATIO ══════════
    for sw in p["SORTINO"]:
        mean_ret = daily_ret.rolling(sw).mean()
        downside = daily_ret.clip(upper=0).rolling(sw).std()
        df[f"sortino_{sw}d"] = (mean_ret / downside.replace(0, np.nan)) * np.sqrt(252)

    # ══════════ MAX DRAWDOWN ══════════
    for dw in p["MAX_DD"]:
        roll_max = c.rolling(dw, min_periods=1).max()
        dd = (c - roll_max) / roll_max * 100
        df[f"max_drawdown_{dw}d"] = dd.rolling(dw, min_periods=1).min()

    # ══════════ PRECIO RELATIVO ══════════
    for w in [20, 50, 200]:
        sma_col = f"sma_{w}"
        if sma_col in df.columns:
            df[f"dist_sma_{w}"] = (c - df[sma_col]) / df[sma_col].replace(0, np.nan) * 100

    # 52-week high / low
    df["high_52w"]     = h.rolling(252, min_periods=1).max()
    df["low_52w"]      = lo.rolling(252, min_periods=1).min()
    df["dist_high_52w"] = (c - df["high_52w"]) / df["high_52w"].replace(0, np.nan) * 100
    df["dist_low_52w"]  = (c - df["low_52w"]) / df["low_52w"].replace(0, np.nan) * 100

    # ══════════ GAP ══════════
    df["gap_pct"] = (df["open"] - c.shift()) / c.shift().replace(0, np.nan) * 100

    return df


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA DE PRECIOS
# ═════════════════════════════════════════════════════════════════════════════

def _clean_record(record: dict) -> dict:
    """Limpia valores NaN/Inf para JSON de Supabase."""
    BIGINT_COLS = {"volume", "obv", "volume_sma_20"}
    clean = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = None
            elif k in BIGINT_COLS:
                clean[k] = int(v)
            else:
                clean[k] = round(float(v), 6)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        else:
            clean[k] = v
    return clean


def ingest_prices(db: DatabaseManager, tickers: list,
                  period: str = None, start_date: str = None) -> dict:
    """Descarga OHLCV, calcula indicadores y sube a Supabase."""
    batch_size = config.INGESTION["BATCH_SIZE"]
    delay      = config.INGESTION["DELAY_BETWEEN_BATCHES"]
    summary    = {}

    # Registrar activos en catálogo
    for ticker in tickers:
        db.upsert_asset(
            ticker=ticker,
            sector=config.get_sector(ticker),
            asset_type=config.get_asset_type(ticker),
        )

    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    for batch_idx, batch in enumerate(batches):
        logger.info(f"  Batch {batch_idx+1}/{len(batches)}: {batch}")

        try:
            raw = yf.download(
                batch,
                period=period,
                start=start_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"  ✗ Error en batch: {e}")
            time.sleep(delay * 2)
            continue

        if raw is None or raw.empty:
            logger.warning(f"  ⚠ Batch vacío")
            continue

        for ticker in batch:
            try:
                # yfinance v1.2+ siempre devuelve MultiIndex (Price, Ticker)
                if isinstance(raw.columns, pd.MultiIndex):
                    ticker_level = 'Ticker' if 'Ticker' in raw.columns.names else 1
                    tickers_in_data = raw.columns.get_level_values(ticker_level).unique()
                    if ticker not in tickers_in_data:
                        logger.warning(f"  ⚠ {ticker} no en respuesta")
                        summary[ticker] = 0
                        continue
                    df = raw.xs(ticker, axis=1, level=ticker_level).copy()
                else:
                    df = raw.copy()

                # Flatten column names (pueden venir como 'Close', 'Open', etc.)
                df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

                if "close" not in df.columns and "adj_close" in df.columns:
                    df["close"] = df["adj_close"]
                if "adj_close" not in df.columns:
                    df["adj_close"] = df.get("close")

                for col in ["open", "high", "low", "volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                df = df.dropna(subset=["close"])
                if df.empty:
                    summary[ticker] = 0
                    continue

                df.index = pd.to_datetime(df.index)

                # Calcular indicadores
                df = compute_indicators(df)

                # Preparar registros
                records = []
                for idx_date, row in df.iterrows():
                    record = {"ticker": ticker, "date": idx_date.strftime("%Y-%m-%d")}
                    for col in df.columns:
                        record[col] = row[col]
                    records.append(_clean_record(record))

                uploaded = db.upsert_prices_bulk(records)
                summary[ticker] = uploaded
                logger.info(f"    ✓ {ticker:<12} {uploaded:>5} registros")

            except Exception as e:
                logger.error(f"    ✗ {ticker}: {e}")
                summary[ticker] = 0

        if batch_idx < len(batches) - 1:
            time.sleep(delay)

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA INCREMENTAL
# ═════════════════════════════════════════════════════════════════════════════

def ingest_incremental(db: DatabaseManager, tickers: list) -> dict:
    """Descarga solo días nuevos desde la última fecha en Supabase."""
    buffer   = config.INGESTION["INCREMENTAL_BUFFER_DAYS"]
    fallback = (date.today() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    summary  = {}

    date_groups: dict = {}
    for ticker in tickers:
        latest = db.get_latest_date(ticker)
        if latest:
            start = (pd.Timestamp(latest) - timedelta(days=buffer)).strftime("%Y-%m-%d")
        else:
            start = fallback
        date_groups.setdefault(start, []).append(ticker)

    for start_date, group in date_groups.items():
        logger.info(f"\n  Desde {start_date}: {len(group)} tickers")
        result = ingest_prices(db, group, start_date=start_date)
        summary.update(result)

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA MACRO
# ═════════════════════════════════════════════════════════════════════════════

def ingest_macro(db: DatabaseManager, period: str = "2y") -> int:
    """Descarga indicadores macro y los sube a macro_indicators."""
    macro_map = config.MACRO_TICKERS
    logger.info(f"  Tickers macro: {list(macro_map.values())}")

    try:
        raw = yf.download(
            list(macro_map.values()),
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        logger.error(f"  ✗ Error macro: {e}")
        return 0

    if raw.empty:
        return 0

    records = []
    for dt in raw.index:
        row = {"date": dt.strftime("%Y-%m-%d")}
        for field, yf_ticker in macro_map.items():
            try:
                val = raw.xs(yf_ticker, axis=1, level=0).at[dt, "Close"]
                row[field] = None if pd.isna(val) else round(float(val), 4)
            except Exception:
                row[field] = None

        # Spreads calculados
        y10 = row.get("yield_10y")
        y2  = row.get("yield_2y")
        if y10 is not None and y2 is not None:
            row["yield_spread_10y_2y"] = round(y10 - y2, 4)

        records.append(row)

    inserted = db.upsert_macro_bulk(records)
    logger.info(f"  ✓ Macro: {inserted} días")
    return inserted


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA FRED (CPI, Credit Spreads)
# ═════════════════════════════════════════════════════════════════════════════

def ingest_fred(db: DatabaseManager) -> int:
    """Fetch FRED data (CPI, credit spreads) and merge into macro_indicators."""
    from db.fred_client import FREDClient

    if not config.FRED_API_KEY:
        logger.warning("  ⚠ FRED_API_KEY not set, skipping FRED ingestion")
        return 0

    try:
        fred = FREDClient(config.FRED_API_KEY)
    except Exception as e:
        logger.error(f"  ✗ FRED client error: {e}")
        return 0

    # Fields to fetch and their FRED series
    fred_fields = {
        "cpi_value":          "CPIAUCSL",
        "cpi_yoy":            "CPALTT01USM657N",
        "credit_spread_bbb":  "BAMLC0A4CBBB",
        "fed_rate":           "FEDFUNDS",        # Federal Funds Effective Rate
    }

    # Add international macro series from config
    try:
        import config as cfg
        # Central bank rates (ECB, BoJ, BoE)
        for zone, series_id in cfg.CENTRAL_BANK_RATES.items():
            field = f"rate_{zone.lower()}"
            if field not in fred_fields:
                fred_fields[field] = series_id
        # International indicators (PMI, GDP, unemployment)
        fred_fields.update(cfg.INTL_MACRO_SERIES)
    except Exception:
        pass  # config not available, use base fields only

    all_series = {}
    for field, series_id in fred_fields.items():
        try:
            df = fred.get_series(series_id)
            if not df.empty:
                all_series[field] = df
                logger.info(f"  ✓ FRED {field}: {len(df)} observations")
            else:
                logger.warning(f"  ⚠ FRED {field}: no data")
        except Exception as e:
            logger.warning(f"  ✗ FRED {field}: {e}")

    if not all_series:
        return 0

    # Build records: merge all series by date
    # Monthly series (CPI) need forward-fill to daily
    all_dates = set()
    for field, df in all_series.items():
        for d in df["date"]:
            all_dates.add(d.strftime("%Y-%m-%d"))

    # Also get existing macro dates to merge with
    existing = db.client.table("macro_indicators").select("date").order(
        "date", desc=True
    ).limit(1000).execute()
    if existing.data:
        for row in existing.data:
            all_dates.add(row["date"])

    records = []
    for date_str in sorted(all_dates):
        row = {"date": date_str}
        dt = pd.Timestamp(date_str)
        has_data = False
        for field, df in all_series.items():
            # Find most recent value on or before this date
            mask = df["date"] <= dt
            if mask.any():
                val = df.loc[mask, "value"].iloc[-1]
                row[field] = round(float(val), 4)
                has_data = True
        if has_data:
            records.append(row)

    if not records:
        return 0

    inserted = db.upsert_macro_bulk(records)
    logger.info(f"  ✓ FRED: {inserted} records merged into macro_indicators")
    return inserted


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA FUNDAMENTALS
# ═════════════════════════════════════════════════════════════════════════════

def ingest_fundamentals(db: DatabaseManager, tickers: list) -> dict:
    """Descarga datos fundamentales de yfinance → Supabase."""
    summary = {}
    today   = date.today().strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if not info or "regularMarketPrice" not in info:
                summary[ticker] = False
                continue

            data = {
                "market_cap":          info.get("marketCap"),
                "enterprise_value":    info.get("enterpriseValue"),
                "pe_ratio":            info.get("trailingPE"),
                "forward_pe":          info.get("forwardPE"),
                "peg_ratio":           info.get("pegRatio"),
                "pb_ratio":            info.get("priceToBook"),
                "ps_ratio":            info.get("priceToSalesTrailing12Months"),
                "ev_ebitda":           info.get("enterpriseToEbitda"),
                "ev_revenue":          info.get("enterpriseToRevenue"),
                "revenue":             info.get("totalRevenue"),
                "revenue_growth":      info.get("revenueGrowth"),
                "earnings":            info.get("netIncomeToCommon"),
                "earnings_growth":     info.get("earningsGrowth"),
                "ebitda":              info.get("ebitda"),
                "gross_margin":        info.get("grossMargins"),
                "operating_margin":    info.get("operatingMargins"),
                "net_margin":          info.get("profitMargins"),
                "eps":                 info.get("trailingEps"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "free_cash_flow":      info.get("freeCashflow"),
                "total_assets":        info.get("totalAssets"),
                "total_debt":          info.get("totalDebt"),
                "cash":                info.get("totalCash"),
                "debt_to_equity":      info.get("debtToEquity"),
                "current_ratio":       info.get("currentRatio"),
                "quick_ratio":         info.get("quickRatio"),
                "roe":                 info.get("returnOnEquity"),
                "roa":                 info.get("returnOnAssets"),
                "dividend_yield":      info.get("dividendYield"),
                "payout_ratio":        info.get("payoutRatio"),
                "beta":                info.get("beta"),
                "shares_outstanding":  info.get("sharesOutstanding"),
                "float_shares":        info.get("floatShares"),
                "insider_pct":         info.get("heldPercentInsiders"),
                "institutional_pct":   info.get("heldPercentInstitutions"),
                "target_mean_price":   info.get("targetMeanPrice"),
                "target_high_price":   info.get("targetHighPrice"),
                "target_low_price":    info.get("targetLowPrice"),
                "recommendation":      info.get("recommendationKey"),
                "num_analysts":        info.get("numberOfAnalystOpinions"),
                "fiscal_quarter":      str(info.get("mostRecentQuarter", "")),
            }

            # Net debt
            debt = data.get("total_debt")
            cash = data.get("cash")
            if debt is not None and cash is not None:
                data["net_debt"] = debt - cash

            # Actualizar metadata del activo
            db.upsert_asset(
                ticker=ticker,
                name=info.get("longName") or info.get("shortName"),
                sector=info.get("sector") or config.get_sector(ticker),
                industry=info.get("industry"),
                asset_type=config.get_asset_type(ticker),
                exchange=info.get("exchange"),
                country=info.get("country", "US"),
            )

            db.upsert_fundamentals(ticker, today, data)
            summary[ticker] = True
            logger.info(f"  ✓ {ticker:<12} fundamentals OK")
            time.sleep(0.4)

        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")
            summary[ticker] = False

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# INGESTA SECTOR ROTATION
# ═════════════════════════════════════════════════════════════════════════════

def ingest_sectors(db: DatabaseManager, period: str = "2y") -> int:
    """Descarga rendimiento de ETFs sectoriales y calcula rankings."""
    sector_tickers = list(config.SECTOR_ETFS.keys())
    spy_ticker = "SPY"

    logger.info(f"  Sectores: {sector_tickers}")

    try:
        all_tickers = sector_tickers + [spy_ticker]
        raw = yf.download(
            all_tickers, period=period,
            auto_adjust=True, progress=False, group_by="ticker"
        )
    except Exception as e:
        logger.error(f"  ✗ Error sectores: {e}")
        return 0

    if raw.empty:
        return 0

    # Extraer close y returns para cada sector
    records = []
    for dt in raw.index:
        # SPY close para relative strength
        try:
            spy_close = raw.xs(spy_ticker, axis=1, level=1).at[dt, "Close"]
        except Exception:
            spy_close = None

        daily_records = []
        for etf, name in config.SECTOR_ETFS.items():
            try:
                etf_data = raw.xs(etf, axis=1, level=1)
                close_val = etf_data.at[dt, "Close"]
                vol_val   = etf_data.at[dt, "Volume"]

                if pd.isna(close_val):
                    continue

                # Calcular retornos básicos
                close_series = etf_data["Close"].loc[:dt]
                r1  = close_series.pct_change(1).iloc[-1] * 100 if len(close_series) > 1 else None
                r5  = close_series.pct_change(5).iloc[-1] * 100 if len(close_series) > 5 else None
                r20 = close_series.pct_change(20).iloc[-1] * 100 if len(close_series) > 20 else None
                r60 = close_series.pct_change(60).iloc[-1] * 100 if len(close_series) > 60 else None

                # Relative strength vs SPY
                rs = None
                if spy_close and not pd.isna(spy_close) and spy_close != 0:
                    rs = round(float(close_val) / float(spy_close), 4)

                rec = {
                    "date":             dt.strftime("%Y-%m-%d"),
                    "sector_etf":       etf,
                    "sector_name":      name,
                    "close":            R(close_val),
                    "returns_1d":       R(r1),
                    "returns_5d":       R(r5),
                    "returns_20d":      R(r20),
                    "returns_60d":      R(r60),
                    "volume":           int(vol_val) if pd.notna(vol_val) else None,
                    "relative_strength": rs,
                }
                daily_records.append(rec)
            except Exception:
                continue

        # Calcular rankings
        if daily_records:
            sorted_1d  = sorted(daily_records, key=lambda x: x.get("returns_1d") or -999, reverse=True)
            sorted_20d = sorted(daily_records, key=lambda x: x.get("returns_20d") or -999, reverse=True)
            for rank, rec in enumerate(sorted_1d, 1):
                rec["rank_1d"] = rank
            for rank, rec in enumerate(sorted_20d, 1):
                for orig_rec in daily_records:
                    if orig_rec["sector_etf"] == rec["sector_etf"]:
                        orig_rec["rank_20d"] = rank
                        break

        records.extend(daily_records)

    # Subir en chunks
    if records:
        CHUNK = 400
        total = 0
        for i in range(0, len(records), CHUNK):
            chunk = records[i:i+CHUNK]
            db.client.table("sector_performance").upsert(
                chunk, on_conflict="sector_etf,date"
            ).execute()
            total += len(chunk)
        logger.info(f"  ✓ Sectores: {total} registros")
        return total

    return 0


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GlobalMarketAnalyzer — Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "incremental", "macro", "fred", "fundamentals", "sectors"],
        default="incremental",
    )
    parser.add_argument("--period", default=config.INGESTION["DEFAULT_PERIOD"])
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    db = DatabaseManager()

    if args.tickers:
        target_tickers = [t.upper() for t in args.tickers]
    else:
        target_tickers = config.get_all_tickers()

    price_tickers = [t for t in target_tickers if not t.startswith("^")]

    print(f"\n{'='*60}")
    print(f"  GlobalMarketAnalyzer — {args.mode.upper()}")
    print(f"  {len(price_tickers)} tickers")
    print(f"{'='*60}")

    t0 = time.time()

    if args.mode == "historical":
        logger.info(f"\n[1/5] Precios históricos ({args.period})...")
        ingest_prices(db, price_tickers, period=args.period)

        logger.info(f"\n[2/5] Macro ({args.period})...")
        ingest_macro(db, period=args.period)

        logger.info("\n[3/5] FRED (CPI, credit spreads)...")
        ingest_fred(db)

        logger.info("\n[4/5] Fundamentales...")
        equities = [t for t in price_tickers
                     if config.get_asset_type(t) in ("equity", "etf")]
        ingest_fundamentals(db, equities)

        logger.info("\n[5/5] Rotación sectorial...")
        ingest_sectors(db, period=args.period)

    elif args.mode == "incremental":
        logger.info("\n[1/2] Precios incrementales...")
        ingest_incremental(db, price_tickers)
        logger.info("\n[2/2] Macro incremental...")
        ingest_macro(db, period="5d")

    elif args.mode == "macro":
        ingest_macro(db, period=args.period)
        ingest_fred(db)

    elif args.mode == "fred":
        ingest_fred(db)

    elif args.mode == "fundamentals":
        equities = [t for t in price_tickers
                     if config.get_asset_type(t) in ("equity", "etf")]
        ingest_fundamentals(db, equities)

    elif args.mode == "sectors":
        ingest_sectors(db, period=args.period)

    elapsed = time.time() - t0

    stats = db.get_stats()
    print(f"\n{'='*60}")
    print(f"  ✓ Completado en {elapsed:.1f}s")
    print(f"{'='*60}")
    for table, count in stats.items():
        print(f"  {table:<25} {count:>10,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
