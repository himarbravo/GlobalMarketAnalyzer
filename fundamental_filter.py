"""
FUNDAMENTAL FILTER — GlobalMarketAnalyzer
==========================================
Clasifica activos en "creadores de valor" vs "especulativos".
Genera el término fuente f(t) de la ecuación O-U.

Score Fundamental (7 componentes):
  F = w1·FCF_yield + w2·ROIC_excess + w3·growth_real
    + w4·quality + w5·valuation + w6·analyst + w7·momentum_quality
  f = γ · tanh(F / F₀)

  F >> 0  →  fuente de calor  (empresa genera capital)
  F ≈  0  →  neutro           (ETF, índice, carry)
  F << 0  →  sumidero         (especulativo, destruye)
"""

import numpy as np
import pandas as pd
from database_manager import DatabaseManager


# ── Pesos del score fundamental (7 componentes) ──
W_FCF       = 0.20    # Free cash flow yield
W_ROIC      = 0.20    # Return on invested capital vs WACC
W_GROWTH    = 0.15    # Revenue growth real
W_QUALITY   = 0.15    # Solvencia + margins
W_VALUATION = 0.15    # Forward PE, EV/EBITDA — ¿está barato?
W_ANALYST   = 0.10    # Consenso de analistas
W_MOMENTUM_Q = 0.05   # Momentum quality (beta-adjusted)

# ── Parámetros del source term ──
GAMMA       = 0.01    # tasa diaria máxima de source/sink
WACC_PROXY  = 0.08    # coste de capital estimado (8%)


def compute_fundamental_score(row: dict, inflation_annual: float = 0.03) -> float:
    """
    Calcula F para un activo usando 7 componentes.
    Usa todos los datos disponibles en la tabla fundamentals.
    """
    # ── 1. FCF yield ──
    mc  = row.get("market_cap")
    fcf = row.get("free_cash_flow")
    fcf_yield = (fcf / mc) if (mc and mc > 0 and fcf is not None) else 0.0

    # ── 2. ROIC excess ──
    # Usar ROIC real si disponible, sino max(ROE, ROA)
    roic = row.get("roic")
    if roic is None:
        roe = row.get("roe") or 0.0
        roa = row.get("roa") or 0.0
        roic = max(roe, roa)
    roic_excess = roic - WACC_PROXY

    # ── 3. Crecimiento real ──
    rev_growth = row.get("revenue_growth") or 0.0
    earn_growth = row.get("earnings_growth") or rev_growth  # fallback
    growth_real = (0.6 * rev_growth + 0.4 * earn_growth) - inflation_annual

    # ── 4. Quality: solvencia + margins ──
    d2e = row.get("debt_to_equity")
    cr  = row.get("current_ratio")
    gm  = row.get("gross_margin") or 0.5
    om  = row.get("operating_margin") or 0.1
    nm  = row.get("net_margin") or 0.05

    q_debt  = max(0.0, 1.0 - (d2e / 200.0)) if (d2e is not None) else 0.5
    q_liq   = min(1.0, cr / 2.0) if (cr is not None and cr > 0) else 0.5
    q_margin = min(1.0, gm * 1.5)  # gross margin 66%+ = score 1.0
    quality = 0.3 * q_debt + 0.2 * q_liq + 0.3 * q_margin + 0.2 * min(1.0, max(0, om * 5))

    # ── 5. Valuation: ¿está barato vs sector? ──
    fwd_pe = row.get("forward_pe")
    ev_ebitda = row.get("ev_ebitda")
    pb = row.get("pb_ratio")

    # Menor PE/EV_EBITDA = más barato = score más alto
    val_pe = max(0, 1.0 - (fwd_pe / 40.0)) if (fwd_pe and fwd_pe > 0) else 0.5
    val_ev = max(0, 1.0 - (ev_ebitda / 25.0)) if (ev_ebitda and ev_ebitda > 0) else 0.5
    val_pb = max(0, 1.0 - (pb / 10.0)) if (pb and pb > 0) else 0.5
    valuation = 0.4 * val_pe + 0.4 * val_ev + 0.2 * val_pb

    # ── 6. Analyst consensus ──
    target = row.get("target_mean_price")
    price = mc / (row.get("shares_outstanding") or 1e18) if mc else None
    rec = row.get("recommendation")
    n_analysts = row.get("num_analysts") or 0

    # Target upside
    if target and price and price > 0:
        upside = (target - price) / price
        analyst_score = np.clip(upside, -0.5, 0.5)  # cap at ±50%
    else:
        analyst_score = 0.0

    # Recommendation boost (1=strong buy, 5=strong sell)
    if rec:
        rec_map = {"strongBuy": 0.3, "buy": 0.15, "hold": 0.0,
                   "sell": -0.15, "strongSell": -0.3}
        if isinstance(rec, str):
            analyst_score += rec_map.get(rec, 0.0)

    # Weight by analyst coverage (more analysts = more reliable)
    analyst_weight = min(1.0, n_analysts / 20) if n_analysts else 0.3
    analyst_final = analyst_score * analyst_weight

    # ── 7. Momentum quality (beta-adjusted) ──
    beta = row.get("beta") or 1.0
    inst_pct = row.get("institutional_pct") or 0.5
    # High institutional + low beta = quality momentum
    momentum_q = inst_pct * (1.0 / max(beta, 0.5)) * 0.5

    # ── Score final ──
    F = (W_FCF       * fcf_yield +
         W_ROIC      * roic_excess +
         W_GROWTH    * growth_real +
         W_QUALITY   * quality +
         W_VALUATION * (valuation - 0.5) +  # centrar en 0
         W_ANALYST   * analyst_final +
         W_MOMENTUM_Q * (momentum_q - 0.25))  # centrar

    # Guard: si algún componente es NaN, devolver 0 (neutral)
    if np.isnan(F) or np.isinf(F):
        return 0.0

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
