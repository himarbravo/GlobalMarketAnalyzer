"""
FRED CLIENT — GlobalMarketAnalyzer
====================================
Lightweight client for the Federal Reserve Economic Data (FRED) API.

Used to fetch:
  - CPI (inflation) → replaces INFLATION_PROXY in capital_field.py
  - Treasury yields → dynamic WACC
  - Credit spreads → stress indicator
  - Fed funds rate → monetary policy

API docs: https://fred.stlouisfed.org/docs/api/fred/
Rate limit: 120 requests/minute (no throttling needed for our usage).
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FREDClient:
    """Client for FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Series we care about
    SERIES = {
        "cpi_value":          "CPIAUCSL",          # CPI Urban All Items (monthly)
        "cpi_yoy":            "CPALTT01USM657N",   # CPI YoY % change (monthly)
        "credit_spread_bbb":  "BAMLC0A4CBBB",      # BBB Option-Adjusted Spread (daily)
        "fed_rate":           "FEDFUNDS",           # Fed Funds Rate (monthly)
        "yield_3m_fred":      "DGS3MO",            # 3-Month Treasury (daily)
        "yield_2y_fred":      "DGS2",              # 2-Year Treasury (daily)
        "yield_10y_fred":     "DGS10",             # 10-Year Treasury (daily)
    }

    def __init__(self, api_key: str = None):
        if api_key is None:
            from config import FRED_API_KEY
            api_key = FRED_API_KEY
        if not api_key:
            raise ValueError(
                "FRED_API_KEY not set. Get one free at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.api_key = api_key

    def get_series(self, series_id: str,
                   start_date: str = None,
                   end_date: str = None) -> pd.DataFrame:
        """
        Fetch a FRED series as a DataFrame with columns ['date', 'value'].

        Args:
            series_id: FRED series ID (e.g., 'CPIAUCSL')
            start_date: 'YYYY-MM-DD' (default: 5 years ago)
            end_date: 'YYYY-MM-DD' (default: today)
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc",
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"FRED: no data for {series_id}")
            return pd.DataFrame(columns=["date", "value"])

        rows = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == "." or val is None:
                continue
            rows.append({
                "date": obs["date"],
                "value": float(val),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"FRED {series_id}: {len(df)} observations")
        return df

    def get_all(self, start_date: str = None) -> dict[str, pd.DataFrame]:
        """Fetch all configured series. Returns {field_name: DataFrame}."""
        result = {}
        for field, series_id in self.SERIES.items():
            try:
                df = self.get_series(series_id, start_date=start_date)
                result[field] = df
                logger.info(f"  ✓ {field} ({series_id}): {len(df)} rows")
            except Exception as e:
                logger.warning(f"  ✗ {field} ({series_id}): {e}")
                result[field] = pd.DataFrame(columns=["date", "value"])
        return result
