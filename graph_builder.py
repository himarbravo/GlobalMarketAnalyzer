"""
GRAPH BUILDER — GlobalMarketAnalyzer
======================================
Construye el grafo financiero multi-capa con:
  - Correlación cross-lag (quién lidera a quién)
  - 3 escalas temporales (20d, 60d, 120d)
  - Vecinos de 2º y 3er orden (W², W³)
  - Laplaciano fraccional con signo

Pipeline:
  Precios → Retornos reales → Cross-lag corr × 3 escalas →
  W_eff (con W²,W³) → L → Φ,λ → L^s
"""

import numpy as np
import pandas as pd
from database_manager import DatabaseManager


# ── Parámetros del grafo ──
CORR_THRESHOLD  = 0.25    # |ρ| mínima para crear arista (bajado de 0.30)
MAX_LAG         = 15      # lag máximo en días para cross-lag
SCALE_WINDOWS   = [20, 60, 120]  # ventanas multi-escala
SCALE_WEIGHTS   = [0.20, 0.50, 0.30]  # pesos base (adaptables)
W2_WEIGHT       = 0.15   # peso de vecinos de 2º orden
W3_WEIGHT       = 0.05   # peso de vecinos de 3er orden

# No-localidad
S_BASE          = 0.85
S_VIX_COEF      = 0.25
S_SPREAD_COEF   = 0.20
S_CREDIT_COEF   = 0.20
S_MIN           = 0.15
S_MAX           = 1.00

# Acoplamiento inter-moneda (USD es la base)
FX_COUPLING = {
    ("USD", "USD"): 1.0,
    ("USD", "EUR"): 0.6,  ("EUR", "USD"): 0.6,
    ("USD", "JPY"): 0.5,  ("JPY", "USD"): 0.5,
    ("USD", "GBP"): 0.7,  ("GBP", "USD"): 0.7,
    ("EUR", "EUR"): 1.0,
    ("EUR", "JPY"): 0.4,  ("JPY", "EUR"): 0.4,
    ("EUR", "GBP"): 0.8,  ("GBP", "EUR"): 0.8,
    ("JPY", "JPY"): 1.0,
    ("GBP", "GBP"): 1.0,
    ("JPY", "GBP"): 0.4,  ("GBP", "JPY"): 0.4,
}


class GraphBuilder:
    """
    Construye y mantiene el grafo de correlaciones financieras.

    Uso:
        gb = GraphBuilder(db)
        gb.load_data(start_date="2024-01-01")
        gb.build(reference_date="2026-02-25")
        L_s = gb.fractional_laplacian  # shape (N, N)
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

        # Datos
        self.tickers: list[str] = []
        self.N: int = 0
        self.returns: pd.DataFrame = pd.DataFrame()     # (T, N) log returns reales
        self.prices: pd.DataFrame = pd.DataFrame()       # (T, N) closes

        # Indicadores técnicos cargados
        self.vol_20d: pd.DataFrame = pd.DataFrame()      # (T, N) volatilidad realizada 20d
        self.volume: pd.DataFrame = pd.DataFrame()        # (T, N) volumen
        self.atr_14: pd.DataFrame = pd.DataFrame()        # (T, N) Average True Range

        # Macro
        self.vix: pd.Series = pd.Series(dtype=float)
        self.yield_spread: pd.Series = pd.Series(dtype=float)
        self.credit_spread: pd.Series = pd.Series(dtype=float)  # HYG vs TLT
        self.inflation_daily: pd.Series = pd.Series(dtype=float)

        # Grafo
        self.W: np.ndarray = np.array([])            # Adjacencia efectiva (N, N)
        self.W_direct: np.ndarray = np.array([])     # Adjacencia directa (1er vecino)
        self.W_lag: np.ndarray = np.array([])         # Lag óptimo por par (N, N)
        self.W_scales: dict = {}                      # {20: W_20d, 60: W_60d, 120: W_120d}
        self.L: np.ndarray = np.array([])             # Laplaciano (N, N)
        self.eigenvalues: np.ndarray = np.array([])   # (N,)
        self.eigenvectors: np.ndarray = np.array([])  # (N, N)
        self.s: float = S_BASE
        self.fractional_laplacian: np.ndarray = np.array([])  # L^s (N, N)
        self.scale_signals: dict = {}                 # señal por capa temporal

        # Temperaturas (capital real)
        self.u: np.ndarray = np.array([])             # (T, N)

    # ─────────────────────────────────────────────────────────────
    # PASO 0: Cargar datos de Supabase
    # ─────────────────────────────────────────────────────────────

    def load_data(self, start_date: str = None, end_date: str = None):
        """Carga precios y macro desde Supabase."""

        # --- Assets ---
        assets_resp = self.db.client.table("assets").select(
            "ticker, currency"
        ).eq("is_active", True).execute()

        if not assets_resp.data:
            raise ValueError("No hay activos en la DB")

        asset_info = {a["ticker"]: a.get("currency", "USD")
                      for a in assets_resp.data}
        self.tickers = sorted(asset_info.keys())
        self.N = len(self.tickers)

        # --- Precios (close) ---
        self.prices = self.db.get_prices_multi(
            self.tickers, start_date=start_date, end_date=end_date,
            column="close"
        )
        # Eliminar tickers sin datos
        self.prices = self.prices.dropna(axis=1, how="all")
        self.tickers = list(self.prices.columns)
        self.N = len(self.tickers)

        # --- Indicadores técnicos ---
        self.vol_20d = self.db.get_prices_multi(
            self.tickers, start_date=start_date, end_date=end_date,
            column="vol_20d"
        )
        self.volume = self.db.get_prices_multi(
            self.tickers, start_date=start_date, end_date=end_date,
            column="volume"
        )
        self.atr_14 = self.db.get_prices_multi(
            self.tickers, start_date=start_date, end_date=end_date,
            column="atr_14"
        )

        # --- Macro ---
        macro_resp = self.db.client.table("macro_indicators").select(
            "date, vix, yield_10y, yield_2y"
        ).order("date").execute()

        if macro_resp.data:
            macro_df = pd.DataFrame(macro_resp.data)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            macro_df = macro_df.set_index("date").sort_index()

            self.vix = macro_df["vix"].dropna()
            y10 = macro_df["yield_10y"].dropna()
            y2  = macro_df["yield_2y"].dropna()
            self.yield_spread = (y10 - y2).dropna()

            # Inflación diaria esperada (Fisher: breakeven ≈ yield_10y - TIPS)
            # Proxy: yield_2y / 252 (actualización diaria)
            self.inflation_daily = y2 / 252 / 100  # de % anual a decimal diario

        # Credit spread: proxy de tensión en mercado de deuda
        # Usamos los precios de HYG (high yield) vs TLT (treasuries)
        # Un spread creciente = tensión económica
        try:
            hyg = self.db.get_prices("HYG", column="close")
            tlt = self.db.get_prices("TLT", column="close")
            if not hyg.empty and not tlt.empty:
                # Ratio HYG/TLT: cae cuando hay tensión de crédito
                ratio = hyg["close"] / tlt["close"].reindex(hyg.index).ffill()
                # Retorno rolling 20d del ratio (negativo = tensión)
                self.credit_spread = -ratio.pct_change(20).dropna()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # PASO 0.5: Calcular retornos reales (deflactados)
    # ─────────────────────────────────────────────────────────────

    def _compute_real_returns(self):
        """Log retornos deflactados por inflación."""
        # Forzar float64 — Supabase puede devolver columnas como object
        prices_f = self.prices.apply(pd.to_numeric, errors="coerce").astype(np.float64)

        # NO usar .dropna() — eliminaría filas donde cualquier
        # ticker tiene NaN. pandas.corr() maneja NaN pairwise.
        log_ret = np.log(prices_f / prices_f.shift(1))
        log_ret = log_ret.iloc[1:]  # quitar primera fila (NaN por shift)

        # Alinear inflación con fechas de retornos
        pi = self.inflation_daily.reindex(log_ret.index).ffill().fillna(0)

        # Deflactar: r_real = r_nominal - π
        self.returns = log_ret.sub(pi, axis=0)

        # Temperaturas acumuladas (capital real)
        self.u = self.returns.fillna(0).cumsum().values.astype(np.float64)

    # ─────────────────────────────────────────────────────────────
    # PASO 1: Construir correlación → Grafo → Laplaciano
    # ─────────────────────────────────────────────────────────────

    def build(self, reference_date: str = None):
        """
        Construye el grafo multi-capa completo:
        1. Calcula retornos reales
        2. Para cada escala (20d, 60d, 120d): correlación cross-lag
        3. Combina escalas con pesos adaptativos
        4. Añade vecinos W², W³ para cadenas de contagio
        5. Calcula Laplaciano, eigendecomposition, L^s
        """
        self._compute_real_returns()

        ref = pd.Timestamp(reference_date) if reference_date else None

        # ─── Multi-escala: calcular W para cada ventana temporal ───
        self.W_scales = {}
        W_lag_best = np.zeros((self.N, self.N))  # lag óptimo (de la escala media)

        for win_size in SCALE_WINDOWS:
            if ref is not None:
                mask = self.returns.index <= ref
                window = self.returns[mask].tail(win_size)
            else:
                window = self.returns.tail(win_size)

            if len(window) < win_size // 3:
                # Insuficientes datos → usar lo que haya
                window = self.returns.tail(max(20, len(self.returns)))

            # Cross-lag correlation
            W_scale, W_lag_scale = self._compute_crosslag_corr(
                window.values.astype(np.float64)
            )
            self.W_scales[win_size] = W_scale

            # Guardar lag de la escala media (60d) como referencia
            if win_size == 60:
                W_lag_best = W_lag_scale

        self.W_lag = W_lag_best

        # ─── Combinar escalas con pesos adaptativos ───
        w_fast, w_mid, w_slow = self._adaptive_scale_weights()
        W_combined = np.zeros((self.N, self.N))
        for win_size, weight in zip(SCALE_WINDOWS, [w_fast, w_mid, w_slow]):
            W_combined += weight * self.W_scales[win_size]

        # ─── Volatility-adaptive threshold: alta vol → necesita más correlación ───
        if not self.vol_20d.empty:
            vol_data = self.vol_20d.tail(60).mean()
            vol_data = vol_data.reindex(self.tickers).fillna(30).values.astype(np.float64)
            vol_data = vol_data / 100  # de % a fracción
            # Threshold adaptativo: base + exceso de vol
            vol_pair = np.sqrt(np.outer(vol_data, vol_data))
            threshold_matrix = CORR_THRESHOLD + 0.05 * np.clip(vol_pair - 0.25, 0, 0.5)
            mask_below = np.abs(W_combined) < threshold_matrix
            W_combined[mask_below] = 0.0

        # ─── Volume weighting: aristas entre activos líquidos son más fiables ───
        # Se aplica DESPUÉS del threshold para no matar aristas con buena corr
        if not self.volume.empty:
            vol_avg = self.volume.tail(60).mean()
            vol_avg = vol_avg.reindex(self.tickers).fillna(0).values.astype(np.float64)
            vol_norm = vol_avg / (np.max(vol_avg) + 1e-10)  # [0,1]
            vol_weight = np.sqrt(np.outer(vol_norm, vol_norm))
            # Boost moderado: 0.85 base + 0.15 × vol (no reduce demasiado)
            vol_boost = 0.85 + 0.15 * vol_weight
            W_combined = W_combined * vol_boost

        self.W_direct = W_combined.copy()  # guardar antes de añadir W²,W³

        # ─── Vecinos de 2º y 3er orden (cadenas de contagio) ───
        # Usar W con signo: (-)(-)=(+), (-)(+)=(-) — propaga anti-corr correctamente
        W2 = W_combined @ W_combined
        np.fill_diagonal(W2, 0)
        W2_max = np.max(np.abs(W2)) + 1e-10
        W2_norm = W2 / W2_max  # mantiene signo propio

        W3 = W2 @ W_combined
        np.fill_diagonal(W3, 0)
        W3_max = np.max(np.abs(W3)) + 1e-10
        W3_norm = W3 / W3_max  # mantiene signo propio

        # Sumar: vecinos directos + indirectos (cada uno con su signo propio)
        self.W = W_combined + W2_WEIGHT * W2_norm + W3_WEIGHT * W3_norm

        # Garantizar simetría
        self.W = (self.W + self.W.T) / 2

        # ─── Laplaciano de grafo con signo ───
        D = np.diag(np.abs(self.W).sum(axis=1))
        self.L = D - self.W

        # ─── Eigendecomposition ───
        eigenvalues, eigenvectors = np.linalg.eigh(self.L)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        idx = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # ─── Señal multi-escala (es temporal o real?) ───
        self._compute_scale_signals()

        # ─── Calibrar s(t) y L^s ───
        self._calibrate_s(reference_date)
        self._compute_fractional_laplacian()

    # ─────────────────────────────────────────────────────────────
    # CROSS-LAG CORRELATION
    # ─────────────────────────────────────────────────────────────

    def _compute_crosslag_corr(self, data: np.ndarray) -> tuple:
        """
        Para cada par (i,j), encuentra el lag que maximiza |corr|.

        Args:
            data: (T, N) array de retornos, float64

        Returns:
            W: (N, N) pesos con signo (filtrados por umbral)
            W_lag: (N, N) lag óptimo por par (positivo = i lidera j)
        """
        T, N = data.shape
        W = np.zeros((N, N))
        W_lag = np.zeros((N, N), dtype=int)

        # Reemplazar NaN por 0 para correlación (pandas lo hace pairwise,
        # pero aquí lo hacemos manual por velocidad con lag)
        data_clean = np.nan_to_num(data, nan=0.0)

        for i in range(N):
            for j in range(i + 1, N):
                best_corr = 0.0
                best_lag = 0

                for lag in range(-MAX_LAG, MAX_LAG + 1):
                    if lag >= 0:
                        x = data_clean[:T - lag, i]
                        y = data_clean[lag:, j]
                    else:
                        x = data_clean[-lag:, i]
                        y = data_clean[:T + lag, j]

                    if len(x) < 15:
                        continue

                    # Correlación rápida (sin scipy)
                    x_m = x - x.mean()
                    y_m = y - y.mean()
                    denom = np.sqrt(np.sum(x_m**2) * np.sum(y_m**2))
                    if denom < 1e-12:
                        continue
                    c = np.sum(x_m * y_m) / denom

                    if abs(c) > abs(best_corr):
                        best_corr = c
                        best_lag = lag

                # Filtrar por umbral (mantiene signo)
                if best_corr > CORR_THRESHOLD:
                    w = best_corr - CORR_THRESHOLD
                elif best_corr < -CORR_THRESHOLD:
                    w = best_corr + CORR_THRESHOLD
                else:
                    w = 0.0

                W[i, j] = w
                W[j, i] = w
                W_lag[i, j] = best_lag
                W_lag[j, i] = -best_lag  # simétrico inverso

        return W, W_lag

    # ─────────────────────────────────────────────────────────────
    # PESOS ADAPTATIVOS POR ESCALA
    # ─────────────────────────────────────────────────────────────

    def _adaptive_scale_weights(self) -> tuple:
        """
        Pesos adaptativos según volatilidad:
        - Alta vol (VIX > 25) → más peso a escala rápida (20d)
        - Baja vol → más peso a escala lenta (120d)
        """
        vix_val = self.vix.iloc[-1] if len(self.vix) > 0 else 15.0
        vix_val = float(vix_val) if pd.notna(vix_val) else 15.0

        if vix_val > 30:  # Crisis
            return 0.50, 0.35, 0.15
        elif vix_val > 20:  # Estresado
            return 0.30, 0.45, 0.25
        else:  # Calma
            return SCALE_WEIGHTS[0], SCALE_WEIGHTS[1], SCALE_WEIGHTS[2]

    # ─────────────────────────────────────────────────────────────
    # SEÑAL MULTI-ESCALA
    # ─────────────────────────────────────────────────────────────

    def _compute_scale_signals(self):
        """
        Clasifica perturbaciones por capa temporal:
        - Solo 20d afectada → corrección temporal (oportunidad de compra)
        - 20d + 60d afectadas → rotación sectorial
        - Las 3 capas afectadas → cambio de tendencia real
        """
        self.scale_signals = {}

        for i, ticker in enumerate(self.tickers):
            # Grado (conectividad) del activo en cada escala
            degrees = {}
            for win_size in SCALE_WINDOWS:
                if win_size in self.W_scales:
                    W_s = self.W_scales[win_size]
                    degrees[win_size] = np.sum(np.abs(W_s[i, :]))
                else:
                    degrees[win_size] = 0.0

            d_fast = degrees.get(20, 0)
            d_mid = degrees.get(60, 0)
            d_slow = degrees.get(120, 0)

            # Clasificar
            if d_fast > 0.5 and d_mid < 0.2 and d_slow < 0.2:
                signal = "temporal"
            elif d_fast > 0.3 and d_mid > 0.3 and d_slow < 0.2:
                signal = "rotacion"
            elif d_fast > 0.2 and d_mid > 0.2 and d_slow > 0.2:
                signal = "tendencia"
            else:
                signal = "estable"

            self.scale_signals[ticker] = {
                "signal": signal,
                "d_fast": round(d_fast, 3),
                "d_mid": round(d_mid, 3),
                "d_slow": round(d_slow, 3),
            }

    # ─────────────────────────────────────────────────────────────
    # CALIBRAR s(t)
    # ─────────────────────────────────────────────────────────────

    def _calibrate_s(self, reference_date: str = None):
        """
        s(t) = clip(s_base - β₁·VIX_norm - β₂·spread_norm - β₃·credit_norm, s_min, s_max)
        """
        if reference_date and len(self.vix) > 0:
            ref = pd.Timestamp(reference_date)
            vix_val = self.vix.asof(ref)
            spread_val = self.yield_spread.asof(ref)
            credit_val = self.credit_spread.asof(ref) if len(self.credit_spread) > 0 else 0.0
        elif len(self.vix) > 0:
            vix_val = self.vix.iloc[-1]
            spread_val = self.yield_spread.iloc[-1]
            credit_val = self.credit_spread.iloc[-1] if len(self.credit_spread) > 0 else 0.0
        else:
            self.s = S_BASE
            return

        # Normalizar
        vix_norm = max(0.0, (vix_val - 15.0) / 25.0) if pd.notna(vix_val) else 0.0
        spread_norm = max(0.0, -spread_val) if pd.notna(spread_val) else 0.0
        credit_norm = max(0.0, credit_val * 10) if pd.notna(credit_val) else 0.0

        self.s = np.clip(
            S_BASE - S_VIX_COEF * vix_norm - S_SPREAD_COEF * spread_norm
            - S_CREDIT_COEF * credit_norm,
            S_MIN, S_MAX
        )

    # ─────────────────────────────────────────────────────────────
    # PASO 3: Laplaciano fraccional L^s
    # ─────────────────────────────────────────────────────────────

    def _compute_fractional_laplacian(self):
        """L^s = Φ · diag(λ₁ˢ, ..., λₙˢ) · Φᵀ"""
        lam_s = np.power(self.eigenvalues, self.s)
        # λ₁ = 0 → 0^s = 0 (conservación de capital)
        lam_s[0] = 0.0
        self.fractional_laplacian = (
            self.eigenvectors @ np.diag(lam_s) @ self.eigenvectors.T
        )

    # ─────────────────────────────────────────────────────────────
    # UTILIDADES DE ANÁLISIS
    # ─────────────────────────────────────────────────────────────

    def green_kernel(self, alpha: float, dt: float = 1.0) -> np.ndarray:
        """
        Kernel de Green: K = Φ · diag(e^{-α·λₖˢ·dt}) · Φᵀ
        K[i,j] = influencia de shock en j sobre i tras dt días.
        """
        lam_s = np.power(self.eigenvalues, self.s)
        lam_s[0] = 0.0
        decay = np.exp(-alpha * lam_s * dt)
        return self.eigenvectors @ np.diag(decay) @ self.eigenvectors.T

    def nonlocality_ratio(self, alpha: float, dt: float = 1.0) -> float:
        """
        NL = influencia no-local / influencia local.
        NL >> 1 → crisis sistémica, todo acoplado.
        Detecta aristas por W != 0 (incluye anti-correlaciones).
        """
        K = self.green_kernel(alpha, dt)
        has_edge = self.W != 0       # aristas positivas Y negativas
        np.fill_diagonal(has_edge, False)

        local_influence = np.abs(K[has_edge]).sum()
        no_edge = ~has_edge
        np.fill_diagonal(no_edge, False)
        nonlocal_influence = np.abs(K[no_edge]).sum()

        if local_influence == 0:
            return 0.0
        return float(nonlocal_influence / local_influence)

    def relaxation_times(self) -> np.ndarray:
        """τₖ = 1/(α·λₖˢ) — tiempos de relajación por modo (en días)."""
        lam_s = np.power(self.eigenvalues, self.s)
        lam_s[0] = np.nan  # modo constante no relaja
        with np.errstate(divide="ignore"):
            return 1.0 / lam_s

    def mode_interpretation(self, k: int) -> dict:
        """Devuelve los tickers dominantes en el modo k."""
        phi_k = self.eigenvectors[:, k]
        idx_pos = np.argsort(phi_k)[::-1][:5]
        idx_neg = np.argsort(phi_k)[:5]
        return {
            "eigenvalue": float(self.eigenvalues[k]),
            "positive": [(self.tickers[i], round(float(phi_k[i]), 3)) for i in idx_pos],
            "negative": [(self.tickers[i], round(float(phi_k[i]), 3)) for i in idx_neg],
        }

    def get_temperatures(self) -> pd.DataFrame:
        """Devuelve DataFrame (T, N) de temperaturas u[i,t]."""
        dates = self.returns.index
        return pd.DataFrame(
            self.u, index=dates, columns=self.tickers
        )
