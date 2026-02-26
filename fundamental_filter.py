"""
FUNDAMENTAL FILTER — GlobalMarketAnalyzer
==========================================
Clasifica activos en "creadores de valor" vs "especulativos".
Genera el término fuente f(t) de la ecuación O-U.

Score Fundamental:
  F = w1·FCF_yield + w2·ROIC_excess + w3·growth_real + w4·quality
  f = γ · tanh(F / F₀)

  F >> 0  →  fuente de calor  (empresa genera capital)
  F ≈  0  →  neutro           (ETF, índice, carry)
  F << 0  →  sumidero         (especulativo, destruye)
"""

import numpy as np
import pandas as pd
from database_manager import DatabaseManager


# ── Pesos del score fundamental ──
W_FCF       = 0.35
W_ROIC      = 0.25
W_GROWTH    = 0.25
W_QUALITY   = 0.15

# ── Parámetros del source term ──
GAMMA       = 0.01    # tasa diaria máxima de source/sink
WACC_PROXY  = 0.08    # coste de capital estimado (8%)


def compute_fundamental_score(row: dict, inflation_annual: float = 0.03) -> float:
    """
    Calcula F para un activo dado su dict de fundamentales.
    Devuelve un float (puede ser negativo, cero, o positivo).
    """
    # FCF yield: cash libre que genera / capitalización
    mc  = row.get("market_cap")
    fcf = row.get("free_cash_flow")
    fcf_yield = (fcf / mc) if (mc and mc > 0 and fcf is not None) else 0.0

    # ROIC excess: retorna más de lo que cuesta el capital?
    roe = row.get("roe") or 0.0
    roa = row.get("roa") or 0.0
    roic = max(roe, roa)
    roic_excess = roic - WACC_PROXY

    # Crecimiento real: crece por encima de inflación?
    rev_growth = row.get("revenue_growth") or 0.0
    growth_real = rev_growth - inflation_annual

    # Quality: solvencia y estabilidad
    d2e = row.get("debt_to_equity")
    cr  = row.get("current_ratio")
    q_debt  = max(0.0, 1.0 - (d2e / 3.0)) if (d2e is not None) else 0.5
    q_liq   = 1.0 if (cr is not None and cr > 1.0) else 0.5
    quality = q_debt * q_liq

    F = (W_FCF * fcf_yield +
         W_ROIC * roic_excess +
         W_GROWTH * growth_real +
         W_QUALITY * quality)

    return float(F)


def compute_source_term(F: float, F0: float) -> float:
    """
    Convierte score fundamental F en fuente de calor f.
    f = γ · tanh(F / F₀)
    """
    if F0 == 0:
        return 0.0
    return GAMMA * np.tanh(F / F0)


class FundamentalFilter:
    """
    Carga fundamentales desde Supabase, calcula scores y source terms.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.scores: dict[str, float] = {}
        self.sources: dict[str, float] = {}
        self.classifications: dict[str, str] = {}

    def compute_all(self, inflation_annual: float = 0.03) -> pd.DataFrame:
        """
        Calcula scores para todos los activos con datos fundamentales.
        Returns DataFrame con columns: ticker, F, f, classification.
        """
        # Fetch all fundamentals
        resp = self.db.client.table("fundamentals").select("*").execute()
        if not resp.data:
            return pd.DataFrame()

        # Agrupar por ticker, tomar el más reciente
        df = pd.DataFrame(resp.data)
        df["report_date"] = pd.to_datetime(df["report_date"])
        latest = df.sort_values("report_date").groupby("ticker").last().reset_index()

        # Calcular score por activo
        results = []
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            F = compute_fundamental_score(row.to_dict(), inflation_annual)
            self.scores[ticker] = F
            results.append({"ticker": ticker, "F": F})

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        # F₀ = mediana de |F| (excluye ceros) para normalización
        nonzero = result_df["F"].abs()
        nonzero = nonzero[nonzero > 0]
        F0 = float(nonzero.median()) if len(nonzero) > 0 else 0.01

        # Source terms y clasificación
        for idx, row in result_df.iterrows():
            ticker = row["ticker"]
            F = row["F"]
            f = compute_source_term(F, F0)
            self.sources[ticker] = f

            if F > F0:
                cls = "value_creator"
            elif F > 0:
                cls = "neutral"
            elif F > -F0:
                cls = "speculative_mild"
            else:
                cls = "speculative"
            self.classifications[ticker] = cls

            result_df.at[idx, "f"] = f
            result_df.at[idx, "classification"] = cls

        # Activos sin fundamentales (ETFs, crypto) → neutral / speculative
        all_assets = self.db.client.table("assets").select("ticker,asset_type").execute()
        if all_assets.data:
            for asset in all_assets.data:
                t = asset["ticker"]
                if t not in self.scores:
                    atype = asset.get("asset_type", "equity")
                    if atype in ("crypto",):
                        self.scores[t] = -0.02
                        self.sources[t] = compute_source_term(-0.02, F0)
                        self.classifications[t] = "speculative"
                    elif atype in ("commodity", "bond"):
                        self.scores[t] = 0.0
                        self.sources[t] = 0.0
                        self.classifications[t] = "neutral"
                    else:
                        self.scores[t] = 0.0
                        self.sources[t] = 0.0
                        self.classifications[t] = "neutral"

        return result_df

    def get_source_vector(self, tickers: list[str]) -> np.ndarray:
        """Devuelve vector f ordenado según la lista de tickers."""
        return np.array([self.sources.get(t, 0.0) for t in tickers])

    def get_score_vector(self, tickers: list[str]) -> np.ndarray:
        """Devuelve vector F ordenado según la lista de tickers."""
        return np.array([self.scores.get(t, 0.0) for t in tickers])

    def summary(self) -> dict:
        """Resumen por clasificación."""
        from collections import Counter
        return dict(Counter(self.classifications.values()))
