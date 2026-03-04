"""
DATABASE MANAGER — GlobalMarketAnalyzer (Supabase / PostgreSQL)
================================================================
Gestiona la conexión y operaciones sobre la base de datos en Supabase.

Usa el cliente oficial supabase-py para operaciones CRUD
y psycopg2 para inserciones masivas (upsert en bulk).
"""

import logging
import os
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
from supabase import create_client, Client

import config

logger = logging.getLogger(__name__)


def get_client(use_service_role: bool = True) -> Client:
    """
    Devuelve cliente Supabase.
    - service_role: permisos totales (INSERT, UPDATE, DELETE) — solo en backend
    - anon:         permisos de lectura pública
    """
    key = config.SUPABASE_SERVICE_ROLE_KEY if use_service_role else config.SUPABASE_ANON_KEY
    return create_client(config.SUPABASE_URL, key)


class DatabaseManager:
    """
    Interfaz de alto nivel para interactuar con Supabase.

    Uso:
        db = DatabaseManager()
        db.upsert_asset("NVDA", "NVIDIA", "TECH_SEMIS", "equity")
        df = db.get_prices("NVDA", start_date="2024-01-01")
    """

    def __init__(self):
        self.client: Client = get_client(use_service_role=True)
        logger.info(f"✓ Conectado a Supabase: {config.SUPABASE_URL}")

    def _execute_with_retry(self, query, max_attempts: int = 4):
        """
        Execute a Supabase query with exponential backoff and reconnect.

        Handles: connection resets, timeouts, transient network errors.
        On each failure we also recreate the client to get a fresh connection.
        """
        import time
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return query.execute()
            except Exception as e:
                last_exc = e
                err_msg = str(e).lower()
                is_transient = any(x in err_msg for x in [
                    'connection reset', 'connection refused', 'timed out',
                    'timeout', 'eof', 'broken pipe', 'connecterror',
                    'too many connections', 'reset by peer',
                ])
                if not is_transient or attempt >= max_attempts - 1:
                    raise
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                logger.warning(
                    f"Supabase transient error (attempt {attempt+1}/{max_attempts}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
                # Recreate client for a fresh connection
                try:
                    self.client = get_client(use_service_role=True)
                except Exception:
                    pass
                # Rebuild the query against the new client — not possible for
                # a built query, so caller must retry from scratch.
                # We raise after reconnect so the caller retries if wrapped properly.
                raise  # let the caller loop re-issue the query
        raise last_exc

    # ─── ASSETS ──────────────────────────────────────────────────────────────

    def upsert_asset(self, ticker: str, name: str = None, sector: str = None,
                     industry: str = None, asset_type: str = "equity",
                     currency: str = "USD", exchange: str = None,
                     country: str = "US") -> None:
        """Inserta o actualiza un activo en el catálogo maestro."""
        self.client.table("assets").upsert({
            "ticker":     ticker,
            "name":       name,
            "sector":     sector,
            "industry":   industry,
            "asset_type": asset_type,
            "currency":   currency,
            "exchange":   exchange,
            "country":    country,
        }, on_conflict="ticker").execute()

    def get_all_active_tickers(self) -> list[str]:
        """Devuelve todos los tickers activos del catálogo."""
        for attempt in range(4):
            try:
                res = (self.client.table("assets")
                       .select("ticker")
                       .eq("is_active", True)
                       .order("ticker")
                       .execute())
                return [r["ticker"] for r in res.data]
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 ** (attempt + 1))
                    try:
                        self.client = get_client(use_service_role=True)
                    except Exception:
                        pass
                else:
                    raise

    def get_assets(self, asset_type: str = None, sector: str = None) -> pd.DataFrame:
        """Devuelve catálogo de activos como DataFrame, con filtros opcionales."""
        for attempt in range(4):
            try:
                query = self.client.table("assets").select("*")
                if asset_type:
                    query = query.eq("asset_type", asset_type)
                if sector:
                    query = query.eq("sector", sector)
                res = query.order("ticker").execute()
                return pd.DataFrame(res.data)
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 ** (attempt + 1))
                    try:
                        self.client = get_client(use_service_role=True)
                    except Exception:
                        pass
                else:
                    raise

    # ─── PRICES ──────────────────────────────────────────────────────────────

    def upsert_prices_bulk(self, records: list[dict]) -> int:
        """
        Inserta/actualiza OHLCV + indicadores en masa.
        Devuelve número de registros procesados.
        """
        if not records:
            return 0

        # Supabase acepta hasta ~500 registros por llamada
        CHUNK = 400
        total = 0
        for i in range(0, len(records), CHUNK):
            chunk = records[i:i + CHUNK]
            self.client.table("prices").upsert(
                chunk, on_conflict="ticker,date"
            ).execute()
            total += len(chunk)

        return total

    def get_prices(self, ticker: str, start_date: str = None,
                   end_date: str = None, limit: int = None) -> pd.DataFrame:
        """
        Devuelve precios históricos de un ticker como DataFrame.
        Paginación automática para superar el límite de 1000 rows de Supabase.
        """
        all_data = []
        page_size = 1000
        offset = 0
        max_pages = 50  # seguridad: máx 50,000 rows por ticker

        for _ in range(max_pages):
            query = (self.client.table("prices")
                     .select("*")
                     .eq("ticker", ticker))

            if start_date:
                query = query.gte("date", start_date)
            if end_date:
                query = query.lte("date", end_date)

            query = query.order("date", desc=False)
            query = query.range(offset, offset + page_size - 1)

            # Retry with backoff + reconnect for connection issues
            res = None
            for attempt in range(4):
                try:
                    res = query.execute()
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    is_transient = any(x in err_msg for x in [
                        'connection reset', 'timed out', 'timeout',
                        'eof', 'broken pipe', 'connecterror', 'reset by peer'
                    ])
                    if attempt < 3 and is_transient:
                        import time
                        wait = 2 ** (attempt + 1)  # 2, 4, 8, 16s
                        logger.warning(f"get_prices retry {attempt+1}/3 in {wait}s: {e}")
                        time.sleep(wait)
                        try:
                            self.client = get_client(use_service_role=True)
                        except Exception:
                            pass
                    else:
                        raise
            if res is None or not res.data:
                break

            all_data.extend(res.data)
            offset += page_size

            # Si devolvió menos que page_size, hemos llegado al final
            if len(res.data) < page_size:
                break

            # Si se especificó limit y lo alcanzamos
            if limit and len(all_data) >= limit:
                all_data = all_data[:limit]
                break

        df = pd.DataFrame(all_data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
        return df

    def get_prices_multi(self, tickers: list[str], start_date: str = None,
                         end_date: str = None, column: str = "close") -> pd.DataFrame:
        """
        Devuelve una columna (por defecto 'close') de múltiples tickers
        como un DataFrame de columnas por ticker.

        Usa caché local en .cache/ para evitar queries repetidas a Supabase.
        """
        import hashlib, pickle
        from pathlib import Path

        # Build cache key from parameters
        cache_dir = Path(__file__).parent.parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        key = hashlib.md5(f"{sorted(tickers)}_{start_date}_{end_date}_{column}".encode()).hexdigest()[:12]
        cache_file = cache_dir / f"prices_{key}.pkl"

        # Try cache first
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass  # Corrupted cache, re-download

        # Download from Supabase
        frames = {}
        for ticker in tickers:
            df = self.get_prices(ticker, start_date=start_date, end_date=end_date)
            if not df.empty and column in df.columns:
                frames[ticker] = df[column]

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result.index = pd.to_datetime(result.index)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass  # Don't fail if cache write fails

        return result

    def get_latest_date(self, ticker: str) -> Optional[str]:
        """Devuelve la fecha más reciente disponible para un ticker."""
        res = (self.client.table("prices")
               .select("date")
               .eq("ticker", ticker)
               .order("date", desc=True)
               .limit(1)
               .execute())
        if res.data:
            return res.data[0]["date"]
        return None

    def get_latest_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Devuelve el último registro de precios para una lista de tickers."""
        frames = []
        for ticker in tickers:
            res = (self.client.table("prices")
                   .select("*")
                   .eq("ticker", ticker)
                   .order("date", desc=True)
                   .limit(1)
                   .execute())
            if res.data:
                frames.append(res.data[0])
        return pd.DataFrame(frames)

    # ─── FUNDAMENTALS ─────────────────────────────────────────────────────────

    def upsert_fundamentals(self, ticker: str, report_date: str, data: dict) -> None:
        """Inserta o actualiza fundamentales de un activo."""
        self.client.table("fundamentals").upsert(
            {"ticker": ticker, "report_date": report_date, **data},
            on_conflict="ticker,report_date"
        ).execute()

    def get_fundamentals(self, ticker: str = None) -> pd.DataFrame:
        """Devuelve fundamentales de uno o todos los activos."""
        for attempt in range(4):
            try:
                query = self.client.table("fundamentals").select("*")
                if ticker:
                    query = query.eq("ticker", ticker).order("report_date", desc=True)
                res = query.execute()
                return pd.DataFrame(res.data)
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 ** (attempt + 1))
                    try:
                        self.client = get_client(use_service_role=True)
                    except Exception:
                        pass
                else:
                    raise

    # ─── MACRO ────────────────────────────────────────────────────────────────

    def upsert_macro(self, date_str: str, data: dict) -> None:
        """Inserta o actualiza indicadores macro de una fecha."""
        self.client.table("macro_indicators").upsert(
            {"date": date_str, **data}, on_conflict="date"
        ).execute()

    def upsert_macro_bulk(self, records: list[dict]) -> int:
        """Inserta/actualiza indicadores macro en masa."""
        if not records:
            return 0
        CHUNK = 400
        total = 0
        for i in range(0, len(records), CHUNK):
            chunk = records[i:i + CHUNK]
            self.client.table("macro_indicators").upsert(
                chunk, on_conflict="date"
            ).execute()
            total += len(chunk)
        return total

    def get_macro(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Devuelve indicadores macro en un rango de fechas."""
        for attempt in range(4):
            try:
                query = self.client.table("macro_indicators").select("*")
                if start_date:
                    query = query.gte("date", start_date)
                if end_date:
                    query = query.lte("date", end_date)
                res = query.order("date", desc=False).execute()
                df = pd.DataFrame(res.data)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                return df
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 ** (attempt + 1))
                    try:
                        self.client = get_client(use_service_role=True)
                    except Exception:
                        pass
                else:
                    raise

    def get_macro_latest(self) -> Optional[dict]:
        """Devuelve el último registro macro disponible."""
        res = (self.client.table("macro_indicators")
               .select("*")
               .order("date", desc=True)
               .limit(1)
               .execute())
        return res.data[0] if res.data else None

    def get_macro_column(self, column: str) -> Optional[pd.Series]:
        """P2.2: Returns a single macro column as a pd.Series indexed by date."""
        for attempt in range(4):
            try:
                res = (self.client.table("macro_indicators")
                       .select(f"date,{column}")
                       .not_.is_(column, "null")
                       .order("date", desc=False)
                       .limit(2000)
                       .execute())
                if not res.data:
                    return None
                df = pd.DataFrame(res.data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                return df[column].astype(float)
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 ** (attempt + 1))
                    try:
                        self.client = get_client(use_service_role=True)
                    except Exception:
                        pass
                else:
                    return None  # macro column non-fatal — return None on failure

    # ─── KALMAN STATE (P3.2) ─────────────────────────────────────────────────

    def save_kalman_state(self, filter_name: str, state: dict) -> bool:
        """P3.2: Persist Kalman filter state to Supabase."""
        try:
            import json
            self.client.table("kalman_state").upsert({
                "filter_name": filter_name,
                "state_json": json.dumps(state),
            }, on_conflict="filter_name").execute()
            return True
        except Exception:
            return False  # table may not exist yet

    def load_kalman_state(self, filter_name: str) -> dict | None:
        """P3.2: Load Kalman filter state from Supabase."""
        try:
            import json
            r = (self.client.table("kalman_state")
                 .select("state_json")
                 .eq("filter_name", filter_name)
                 .limit(1)
                 .execute())
            if r.data and r.data[0].get("state_json"):
                return json.loads(r.data[0]["state_json"])
        except Exception:
            pass  # table may not exist yet
        return None

    # ─── SIGNALS ─────────────────────────────────────────────────────────────

    def log_signal(self, ticker: str, date_str: str, signal: str,
                   confidence: float, price: float, strategy: str = None,
                   regime: str = None, rationale: str = None,
                   technical_score: float = None, fundamental_score: float = None,
                   sentiment_score: float = None, macro_score: float = None) -> None:
        """Registra una señal de análisis."""
        self.client.table("signals").insert({
            "ticker":            ticker,
            "date":              date_str,
            "signal":            signal,
            "confidence":        confidence,
            "price":             price,
            "strategy":          strategy,
            "regime":            regime,
            "rationale":         rationale,
            "technical_score":   technical_score,
            "fundamental_score": fundamental_score,
            "sentiment_score":   sentiment_score,
            "macro_score":       macro_score,
        }).execute()

    def get_signals(self, ticker: str = None, signal_type: str = None,
                    limit: int = 100) -> pd.DataFrame:
        """Devuelve historial de señales."""
        query = self.client.table("signals").select("*")
        if ticker:
            query = query.eq("ticker", ticker)
        if signal_type:
            query = query.eq("signal", signal_type)
        res = query.order("created_at", desc=True).limit(limit).execute()
        return pd.DataFrame(res.data)

    # ─── NEWS SENTIMENT ───────────────────────────────────────────────────────

    def log_news(self, date_str: str, headline: str, sentiment_score: float,
                 ticker: str = None, source: str = None,
                 summary: str = None, category: str = None) -> None:
        """Registra un titular con score de sentimiento."""
        self.client.table("news_sentiment").insert({
            "ticker":          ticker,
            "date":            date_str,
            "headline":        headline,
            "source":          source,
            "sentiment_score": sentiment_score,
            "summary":         summary,
            "category":        category,
        }).execute()

    def get_sentiment(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Devuelve noticias recientes de un ticker."""
        from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        res = (self.client.table("news_sentiment")
               .select("*")
               .eq("ticker", ticker)
               .gte("date", from_date)
               .order("date", desc=True)
               .execute())
        return pd.DataFrame(res.data)

    # ─── UTILS ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Devuelve número de filas por tabla."""
        tables = ["assets", "prices", "fundamentals",
                  "macro_indicators", "signals", "news_sentiment"]
        stats = {}
        for table in tables:
            res = self.client.table(table).select("*", count="exact").limit(1).execute()
            stats[table] = res.count if res.count is not None else 0
        return stats

    def get_price_coverage(self) -> pd.DataFrame:
        """
        Devuelve resumen de cobertura de precios por ticker.
        (Número de días, fecha mínima y máxima)
        """
        # Supabase no soporta GROUP BY nativo desde el cliente,
        # usamos RPC o hacemos la agregación en Python con una muestra
        res = (self.client.table("prices")
               .select("ticker, date")
               .order("ticker")
               .execute())
        if not res.data:
            return pd.DataFrame()

        df = pd.DataFrame(res.data)
        return (df.groupby("ticker")["date"]
                .agg(trading_days="count", from_date="min", to_date="max")
                .reset_index()
                .sort_values("ticker"))
