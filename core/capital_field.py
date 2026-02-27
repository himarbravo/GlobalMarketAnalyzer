"""
CAPITAL FIELD c(t) — GlobalMarketAnalyzer
==========================================
Segundo campo del sistema dual: capital real creado por cada activo.

u(t) = precio/temperatura (lo que el mercado DICE) — se actualiza diario
c(t) = capital real (lo que la empresa CREA) — se actualiza trimestral, interpolado

La señal de mispricing es  δ(t) = u(t) - c(t):
  δ > 0 → precio > valor real → sobrevalorado (sell si trend confirma)
  δ < 0 → precio < valor real → infravalorado  (buy si trend confirma)

Medición profesional de c(t):
  Δc = (FCF_real - SBC_proxy) / Market_Cap   ← Free cash flow neto de dilución
     + max(0, ROIC - WACC) / 252             ← Creación de valor económico diario
     + buyback_yield / 252                    ← Capital devuelto al accionista
     - dilution_rate / 252                    ← Shares outstanding creciendo = destrucción
     - inflation / 252                        ← Ajuste real (nominal ≠ real)
"""

import numpy as np
import pandas as pd
from db.database_manager import DatabaseManager


# ── Parámetros ──
WACC_PROXY       = 0.08    # Coste de capital estimado (8% anual)
INFLATION_PROXY  = 0.03    # Inflación anual proxy (reemplazable por CPI real)
INTERP_CONFIDENCE = 0.02   # Reducción de confianza por día desde último earnings
MAX_CONFIDENCE   = 1.0     # Confianza máxima post-earnings
MIN_CONFIDENCE   = 0.3     # Confianza mínima cuando earnings muy viejos


class CapitalField:
    """
    Construye el campo c(t) = capital real acumulado por activo.

    Usa datos trimestrales de fundamentals y los interpola diariamente
    para compararlos con u(t) (precio diario).
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.c_daily: pd.DataFrame = pd.DataFrame()     # (T, N) capital acumulado
        self.delta: pd.DataFrame = pd.DataFrame()        # (T, N) mispricing δ=u-c
        self.confidence: pd.DataFrame = pd.DataFrame()   # (T, N) confianza del dato
        self.capital_rate: dict[str, float] = {}         # Δc trimestral por ticker
        self.capital_rate_daily: dict[str, float] = {}    # Δc diario = trimestral/252
        self.quality_flag: dict[str, str] = {}           # 'real', 'estimated', 'neutral'

    def build(self, tickers: list[str], price_dates: pd.DatetimeIndex,
              inflation_annual: float = INFLATION_PROXY) -> pd.DataFrame:
        """
        Construye c(t) para los tickers dados, alineado con price_dates.

        Returns:
            DataFrame (T, N) con capital acumulado real por activo.
        """
        # 0. Load real macro data: CPI and Treasury yields
        real_inflation, real_risk_free = self._load_macro_params()
        if real_inflation is not None:
            inflation_annual = real_inflation  # Override proxy with real CPI YoY

        # 1. Cargar TODOS los registros fundamentals (históricos, no solo último)
        resp = self.db.client.table("fundamentals").select(
            "ticker, report_date, market_cap, free_cash_flow, "
            "operating_cash_flow, capex, roic, roe, roa, "
            "revenue_growth, earnings_growth, buyback_yield, "
            "shares_outstanding, debt_to_equity, beta, total_debt"
        ).order("report_date").execute()

        if not resp.data:
            self._fill_neutral(tickers, price_dates)
            return self.c_daily

        fund_df = pd.DataFrame(resp.data)
        fund_df["report_date"] = pd.to_datetime(fund_df["report_date"])

        # 2. Para cada ticker: calcular Δc trimestral y acumular
        T = len(price_dates)
        N = len(tickers)
        c_matrix = np.zeros((T, N))
        conf_matrix = np.full((T, N), MIN_CONFIDENCE)

        for j, ticker in enumerate(tickers):
            tk_data = fund_df[fund_df["ticker"] == ticker].sort_values("report_date")

            if len(tk_data) == 0:
                # Sin fundamentals → neutral (ETFs, commodities, crypto)
                self.capital_rate[ticker] = 0.0
                self.quality_flag[ticker] = "neutral"
                continue

            # Dynamic WACC per ticker: risk_free + β × equity_premium
            last_row = tk_data.iloc[-1]
            beta_val = last_row.get("beta")
            if pd.notna(beta_val) and beta_val > 0:
                beta = float(beta_val)
            else:
                beta = 1.0  # market average

            risk_free = real_risk_free if real_risk_free is not None else 0.04
            equity_premium = 0.05  # Damodaran long-term avg
            wacc = risk_free + beta * equity_premium

            # Calcular Δc para cada trimestre
            quarterly_rates = []
            for _, row in tk_data.iterrows():
                dc = self._compute_quarterly_delta_c(row, inflation_annual, wacc)
                quarterly_rates.append({
                    "date": row["report_date"],
                    "delta_c": dc["delta_c"],
                    "confidence": dc["confidence"],
                    "components": dc
                })

            if not quarterly_rates:
                self.quality_flag[ticker] = "neutral"
                continue

            # Guardar último rate
            self.capital_rate[ticker] = quarterly_rates[-1]["delta_c"]
            self.capital_rate_daily[ticker] = quarterly_rates[-1]["delta_c"] / 252.0
            self.quality_flag[ticker] = "real" if len(quarterly_rates) >= 2 else "estimated"

            # 3. Interpolar diariamente: carry-forward con decay de confianza
            rate_idx = 0
            current_rate = 0.0
            current_conf = MIN_CONFIDENCE
            days_since_update = 0

            for t, date in enumerate(price_dates):
                # ¿Nuevo trimestre disponible?
                while (rate_idx < len(quarterly_rates) and
                       quarterly_rates[rate_idx]["date"] <= date):
                    current_rate = quarterly_rates[rate_idx]["delta_c"]
                    current_conf = quarterly_rates[rate_idx]["confidence"]
                    days_since_update = 0
                    rate_idx += 1

                # Acumular capital (diario = rate trimestral / ~63 trading days)
                c_matrix[t, j] = c_matrix[t - 1, j] + current_rate / 63.0 if t > 0 else 0.0

                # Decay de confianza: más días sin earnings → menos fiable
                days_since_update += 1
                conf_matrix[t, j] = max(
                    MIN_CONFIDENCE,
                    current_conf - INTERP_CONFIDENCE * (days_since_update / 63.0)
                )

        self.c_daily = pd.DataFrame(c_matrix, index=price_dates, columns=tickers)
        self.confidence = pd.DataFrame(conf_matrix, index=price_dates, columns=tickers)

        # Store params for reporting
        self._inflation_used = inflation_annual
        self._risk_free_used = risk_free if real_risk_free is not None else WACC_PROXY
        self._source = "FRED" if real_inflation is not None else "proxy"

        return self.c_daily

    def _load_macro_params(self):
        """Load real CPI and Treasury yields from macro_indicators."""
        inflation = None
        risk_free = None

        try:
            # CPI YoY — most recent value
            r = self.db.client.table("macro_indicators").select(
                "cpi_yoy"
            ).not_.is_("cpi_yoy", "null").order("date", desc=True).limit(1).execute()
            if r.data and r.data[0].get("cpi_yoy") is not None:
                inflation = r.data[0]["cpi_yoy"] / 100.0  # Convert % to decimal

            # Treasury 10Y yield — most recent
            r2 = self.db.client.table("macro_indicators").select(
                "yield_10y"
            ).not_.is_("yield_10y", "null").order("date", desc=True).limit(1).execute()
            if r2.data and r2.data[0].get("yield_10y") is not None:
                risk_free = r2.data[0]["yield_10y"] / 100.0  # Convert % to decimal
        except Exception:
            pass

        return inflation, risk_free

    def compute_mispricing(self, u: np.ndarray, tickers: list[str]) -> np.ndarray:
        """
        δ(t) = u(t) - c(t), normalizado por volatilidad.

        Returns:
            (T, N) array de mispricing z-scores.
        """
        if self.c_daily.empty:
            return np.zeros_like(u)

        c = self.c_daily.values
        T_u, N_u = u.shape
        T_c, N_c = c.shape

        # Alinear longitudes
        T = min(T_u, T_c)
        raw_delta = u[:T] - c[:T]

        # Normalizar por volatilidad rolling (20d)
        delta_z = np.zeros_like(raw_delta)
        for t in range(20, T):
            window = raw_delta[t-20:t]
            std = np.nanstd(window, axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            delta_z[t] = (raw_delta[t] - np.nanmean(window, axis=0)) / std

        self.delta = pd.DataFrame(delta_z[:T], columns=tickers[:N_c])

        # Ponderar por confianza
        conf = self.confidence.values[:T]
        delta_weighted = delta_z * conf

        return delta_weighted

    def _compute_quarterly_delta_c(self, row: pd.Series,
                                    inflation_annual: float,
                                    wacc: float = WACC_PROXY) -> dict:
        """
        Calcula la tasa trimestral de creación de capital real.

        Profesional-grade: 5 componentes netos con ajustes por calidad.
        Args:
            row: pandas Series with fundamental data
            inflation_annual: real CPI YoY or proxy
            wacc: dynamic WACC = risk_free + β × equity_premium
        """
        # Use pd.notna() throughout — Supabase NULL becomes pandas NaN,
        # and 'nan is not None' is True, which caused K=NaN for 34 tickers
        def _val(key):
            v = row.get(key)
            return v if pd.notna(v) else None

        mc = _val("market_cap")
        fcf = _val("free_cash_flow")
        ocf = _val("operating_cash_flow")
        capex = _val("capex")
        roic = _val("roic")
        roe = _val("roe")
        roa = _val("roa")
        rev_g = _val("revenue_growth")
        earn_g = _val("earnings_growth")
        buyback_y = _val("buyback_yield")
        shares = _val("shares_outstanding")
        d2e = _val("debt_to_equity")
        total_debt = _val("total_debt")

        confidence = MAX_CONFIDENCE
        components = {}

        # ── 1. FCF yield real ──
        # Proxy de SBC: si FCF < OCF - |capex|, la diferencia es SBC+otros
        if mc and mc > 0 and fcf is not None:
            if ocf is not None and capex is not None:
                # FCF ajustado = OCF - |capex| (sin SBC, que infla OCF)
                fcf_clean = ocf - abs(capex if capex else 0)
                sbc_proxy = max(0, fcf - fcf_clean)  # SBC estimado
                fcf_adjusted = fcf - sbc_proxy * 0.5  # penalizar ~50% del SBC
            else:
                fcf_adjusted = fcf
                confidence -= 0.1  # menos fiable sin OCF/capex

            fcf_yield = fcf_adjusted / mc  # anualizado
            components["fcf_yield"] = fcf_yield
        else:
            fcf_yield = 0.0
            confidence -= 0.2
            components["fcf_yield"] = 0.0

        # ── 2. Economic value added (ROIC - WACC) ──
        if roic is not None:
            eva = max(0, roic - wacc)  # solo creación, no destrucción aquí
        elif roe is not None:
            # Ajustar ROE por apalancamiento: ROE inflado por deuda no es capital real
            leverage_adj = min(1.0, 1.0 / (1 + (d2e/100 if d2e else 0)))
            eva = max(0, roe * leverage_adj - wacc)
            confidence -= 0.1
        elif roa is not None:
            eva = max(0, roa - wacc)
            confidence -= 0.15
        else:
            eva = 0.0
            confidence -= 0.2
        components["eva"] = eva

        # ── 3. Growth real (revenue growth - inflación) ──
        if rev_g is not None:
            growth_real = rev_g - inflation_annual
            # Penalizar crecimiento sin profitabilidad (quemar cash para crecer)
            if fcf_yield < 0:
                growth_real *= 0.3  # growth destruyendo capital = poco valor
        elif earn_g is not None:
            growth_real = earn_g - inflation_annual
            confidence -= 0.1
        else:
            growth_real = -inflation_annual  # no crece → pierde por inflación
            confidence -= 0.15
        components["growth_real"] = growth_real

        # ── 4. Shareholder return (buybacks + dividends) ──
        if buyback_y is not None and buyback_y > 0:
            shareholder_return = buyback_y  # buybacks = capital devuelto
        else:
            shareholder_return = 0.0
        components["shareholder_return"] = shareholder_return

        # ── 5. Dilution penalty (shares outstanding creciendo) ──
        # No podemos medir dilución sin historial de shares,
        # pero SBC proxy ya lo captura parcialmente
        dilution = 0.0  # TODO: comparar shares vs trimestre anterior
        components["dilution"] = dilution

        # ── 6. Delta debt (crédito nuevo = dinero creado) ──
        # Proxy: si total_debt subió, hubo inyección de crédito
        # Normalizado por market cap para hacerlo comparable
        delta_debt = 0.0
        if total_debt is not None and mc and mc > 0:
            # Positivo = empresa pidió más prestado (inyección)
            # Negativo = empresa pagó deuda (drenaje)
            # Usamos total_debt/mc como proxy del flujo de crédito
            debt_ratio = total_debt / mc
            # Escala: deuda al 50% del mc → delta_debt ≈ 0.10
            delta_debt = np.clip(debt_ratio * 0.20, -0.20, 0.20)
        components["delta_debt"] = delta_debt

        # ── Δc total (tasa anualizada) ──
        # Pesos: FCF (40%), EVA (25%), growth (20%), shareholder (15%)
        # delta_debt se suma aparte (no ponderado, es flujo de crédito)
        delta_c = (0.40 * fcf_yield +
                   0.25 * eva +
                   0.20 * growth_real +
                   0.15 * shareholder_return -
                   dilution -
                   inflation_annual +
                   0.10 * delta_debt)  # crédito nuevo = 10% peso en Δc

        # Sanity check
        delta_c = np.clip(delta_c, -0.50, 0.50)  # cap a ±50% anual
        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

        return {
            "delta_c": float(delta_c),
            "confidence": float(confidence),
            **components
        }

    def _fill_neutral(self, tickers, dates):
        """Rellena con ceros cuando no hay datos."""
        T = len(dates)
        N = len(tickers)
        self.c_daily = pd.DataFrame(np.zeros((T, N)), index=dates, columns=tickers)
        self.confidence = pd.DataFrame(
            np.full((T, N), MIN_CONFIDENCE), index=dates, columns=tickers
        )
        for t in tickers:
            self.capital_rate[t] = 0.0
            self.quality_flag[t] = "neutral"

    def summary(self) -> dict:
        """Resumen por calidad de datos."""
        from collections import Counter
        flags = Counter(self.quality_flag.values())
        rates = list(self.capital_rate.values())
        return {
            "quality_flags": dict(flags),
            "mean_delta_c": float(np.mean(rates)) if rates else 0,
            "median_delta_c": float(np.median(rates)) if rates else 0,
            "top_creators": sorted(
                self.capital_rate.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "top_destroyers": sorted(
                self.capital_rate.items(), key=lambda x: x[1]
            )[:10],
        }
