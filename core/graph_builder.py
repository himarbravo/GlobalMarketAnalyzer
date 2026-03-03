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
from db.database_manager import DatabaseManager
from core.ukf import UKF_S
import config


# ── Parámetros del grafo ──
CORR_THRESHOLD  = 0.25    # |ρ| mínima para crear arista (bajado de 0.30)
MAX_LAG         = 15      # lag máximo en días para cross-lag
SCALE_WINDOWS   = [20, 60, 120]  # ventanas multi-escala
SCALE_WEIGHTS   = [0.20, 0.50, 0.30]  # pesos base (adaptables)
W2_WEIGHT       = 0.15   # peso de vecinos de 2º orden
W3_WEIGHT       = 0.05   # peso de vecinos de 3er orden

# No-localidad
S_BASE          = 0.85
S_VIX_COEF      = 0.20   # VIX: miedo general
S_DXY_COEF      = 0.15   # DXY: flight to USD = contagio global
S_SPREAD_COEF   = 0.15   # Yield curve: inversión = recesión
S_CREDIT_COEF   = 0.15   # Credit spread: tensión deuda (level)
S_CREDIT_DELTA  = 0.20   # P2.2: Credit spread delta: widening speed (early warning)
S_RATE_MOM      = 0.15   # P2.1: Central bank rate momentum (hiking = stress)
S_COPPER_COEF   = 0.10   # Copper: salud industrial (caída = estrés)
S_OIL_COEF      = 0.05   # Oil: shock energético
S_MIN           = 0.15
S_MAX           = 1.00




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
        self.credit_spread: pd.Series = pd.Series(dtype=float)
        self.credit_spread_delta: pd.Series = pd.Series(dtype=float)  # P2.2
        self.rate_momentum: pd.Series = pd.Series(dtype=float)  # P2.1

        # P3.1: UKF for s — tracks regime transitions
        self.ukf_s = UKF_S(s_init=S_BASE)
        # P3.2: Try to restore saved UKF state
        saved = db.load_kalman_state('ukf_s')
        if saved:
            self.ukf_s = UKF_S.from_dict(saved)
        self.inflation_daily: pd.Series = pd.Series(dtype=float)
        self.dxy: pd.Series = pd.Series(dtype=float)              # Dollar index
        self.copper: pd.Series = pd.Series(dtype=float)            # Industrial health
        self.oil: pd.Series = pd.Series(dtype=float)               # Energy stress

        # ── Grafo jerárquico ──
        self.node_roles: list[str] = []    # 'bank' | 'productive' por nodo
        self.node_countries: list[str] = []  # 'US', 'JP', etc. por nodo
        self.node_zones: list[str] = []      # 'USD', 'EUR', 'ASIA', 'EM'
        self.bank_indices: list[int] = []    # índices de nodos bancarios
        self.prod_indices: list[int] = []    # índices de nodos productivos
        self.currency_zones: dict = {}       # zone_id → [indices]

        # Dimensiones (campos que modulan todos los nodos)
        self.dimensions: dict = {
            'currency': {},        # país → pd.Series de divisa c(t)
            'fed_rate': pd.Series(dtype=float),  # r(t) tipo FED
            'sovereign_debt': {},  # país → ratio deuda/PIB
            'fx_returns': {},      # (zone_a, zone_b) → pd.Series de retorno FX
        }

        # Per-zone Laplacians (campos monetarios separados)
        self.zone_laplacians: dict = {}     # zone → {L, eigenvalues, eigenvectors, L_s}

        # Sectores dinámicos (cargados de assets table)
        self.sector_map: dict = {}   # ticker → sector
        self.sectors: dict = {}      # sector → [ticker list]

        # Grafo
        self.W: np.ndarray = np.array([])            # Adjacencia efectiva (N, N)
        self.W_direct: np.ndarray = np.array([])     # Adjacencia directa (1er vecino)
        self.W_directed: np.ndarray = np.array([])   # Aristas dirigidas bank↔company (N, N)
        self.W_lag: np.ndarray = np.array([])         # Lag óptimo por par (N, N)
        self.W_scales: dict = {}                      # {20: W_20d, 60: W_60d, 120: W_120d}
        self.L: np.ndarray = np.array([])             # Laplaciano (N, N)
        self.eigenvalues: np.ndarray = np.array([])   # (N,)
        self.eigenvectors: np.ndarray = np.array([])  # (N, N)
        self.s: float = S_BASE
        self.s_prev: float = S_BASE        # P5: previous s for ds/dt
        self.ds_dt: float = 0.0            # P5: rate of change of s
        self.fractional_laplacian: np.ndarray = np.array([])  # L^s (N, N)
        self.scale_signals: dict = {}                 # señal por capa temporal

        # Temperaturas (capital real)
        self.u: np.ndarray = np.array([])             # (T, N)

    # ─────────────────────────────────────────────────────────────
    # PASO 0: Cargar datos de Supabase
    # ─────────────────────────────────────────────────────────────

    def load_data(self, start_date: str = None, end_date: str = None):
        """Carga precios y macro desde Supabase."""

        # Default: 2 years of data (more than enough for 120d windows)
        if start_date is None:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

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

        # ── Clasificar nodos por rol, país y zona monetaria ──
        # MUST be after ticker filtering so indices match self.tickers
        self.node_roles = [config.get_node_role(t) for t in self.tickers]
        self.node_countries = [config.get_country(t) for t in self.tickers]
        self.node_zones = [config.get_zone(t) for t in self.tickers]
        self.bank_indices = [i for i, r in enumerate(self.node_roles) if r == 'bank']
        self.prod_indices = [i for i, r in enumerate(self.node_roles) if r == 'productive']

        # Agrupar índices por zona monetaria
        self.currency_zones = {}
        for i, z in enumerate(self.node_zones):
            self.currency_zones.setdefault(z, []).append(i)

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

        # --- Macro (with pagination) ---
        macro_data = []
        offset = 0
        while True:
            q = self.db.client.table("macro_indicators").select(
                "date, vix, yield_10y, yield_2y, dxy, copper, oil_wti"
            ).order("date")
            if start_date:
                q = q.gte("date", start_date)
            q = q.range(offset, offset + 999)
            resp = q.execute()
            if not resp.data:
                break
            macro_data.extend(resp.data)
            if len(resp.data) < 1000:
                break
            offset += 1000
        macro_resp_data = macro_data

        if macro_resp_data:
            macro_df = pd.DataFrame(macro_resp_data)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            macro_df = macro_df.set_index("date").sort_index()

            self.vix = macro_df["vix"].dropna()
            y10 = macro_df["yield_10y"].dropna()
            y2  = macro_df["yield_2y"].dropna()
            self.yield_spread = (y10 - y2).dropna()

            # DXY, copper, oil
            self.dxy = macro_df["dxy"].dropna() if "dxy" in macro_df.columns else pd.Series(dtype=float)
            self.copper = macro_df["copper"].dropna() if "copper" in macro_df.columns else pd.Series(dtype=float)
            self.oil = macro_df["oil_wti"].dropna() if "oil_wti" in macro_df.columns else pd.Series(dtype=float)

            # Inflación diaria esperada (proxy: yield_2y / 252)
            self.inflation_daily = y2 / 252 / 100

            # ── Dim 1: Divisas ──
            for country, fx_col in config.COUNTRY_CURRENCY.items():
                if fx_col in macro_df.columns:
                    self.dimensions['currency'][country] = macro_df[fx_col].dropna()

            # ── Dim 3: Fed Rate (si disponible en macro_indicators) ──
            if 'fed_rate' in macro_df.columns:
                fed = macro_df['fed_rate'].dropna()
                self.dimensions['fed_rate'] = fed
                # P2.1: Rate momentum — 3-month change and acceleration
                if len(fed) > 60:
                    # 60-day rate change (proxy for 3-month change)
                    rate_change = fed.diff(60).dropna()
                    self.rate_momentum = rate_change
                elif len(fed) > 20:
                    rate_change = fed.diff(20).dropna()
                    self.rate_momentum = rate_change

        # ── Dim 2: Deuda soberana (estático, de config) ──
        self.dimensions['sovereign_debt'] = config.SOVEREIGN_DEBT_GDP.copy()

        # --- Sectores dinámicos desde assets table ---
        sector_resp = self.db.client.table("assets").select(
            "ticker, sector"
        ).eq("is_active", True).execute()
        if sector_resp.data:
            for row in sector_resp.data:
                t, s = row["ticker"], row.get("sector", "UNKNOWN")
                self.sector_map[t] = s
                self.sectors.setdefault(s, []).append(t)

        # Credit spread: real FRED IG/HY if available, else HYG/TLT proxy
        try:
            fred_spread = self.db.get_macro_column('credit_spread_hy')
            if fred_spread is not None and len(fred_spread) > 20:
                self.credit_spread = fred_spread.dropna()
                # P2.2: delta = 5-day rate of change (widening speed)
                self.credit_spread_delta = fred_spread.pct_change(5).dropna()
            else:
                # Fallback: HYG/TLT price ratio proxy
                hyg = self.db.get_prices("HYG", column="close")
                tlt = self.db.get_prices("TLT", column="close")
                if not hyg.empty and not tlt.empty:
                    ratio = hyg["close"] / tlt["close"].reindex(hyg.index).ffill()
                    self.credit_spread = -ratio.pct_change(20).dropna()
                    self.credit_spread_delta = self.credit_spread.pct_change(5).dropna()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # ARISTAS DIRIGIDAS: BANKS ↔ COMPANIES
    # ─────────────────────────────────────────────────────────────

    def _build_directed_edges(self):
        """
        Construye aristas dirigidas entre bancos y empresas productivas.

        Descendente (bank → company): préstamos = inyección de dinero
          w_down = bank_exposure × same_country_factor

        Ascendente (company → bank): pago de intereses = drenaje
          w_up = debt_ratio_company × cost_of_debt
        """
        self.W_directed = np.zeros((self.N, self.N))

        if not self.bank_indices or not self.prod_indices:
            return

        # --- Proxy de tamaño/capacidad de cada banco ---
        # Usamos market cap relativo como proxy de lending capacity
        bank_weights = np.zeros(len(self.bank_indices))
        for idx, bi in enumerate(self.bank_indices):
            # Último precio disponible como proxy de tamaño relativo
            if not self.prices.empty:
                ticker = self.tickers[bi]
                if ticker in self.prices.columns:
                    last_price = self.prices[ticker].dropna()
                    bank_weights[idx] = float(last_price.iloc[-1]) if len(last_price) > 0 else 1.0

        # Normalizar: peso relativo entre bancos
        bw_sum = bank_weights.sum()
        if bw_sum > 0:
            bank_weights /= bw_sum
        else:
            bank_weights = np.ones(len(self.bank_indices)) / len(self.bank_indices)

        # --- Construir aristas descendentes: bank → company ---
        LENDING_BASE = 0.02  # peso base de arista de préstamo

        for bi_idx, bi in enumerate(self.bank_indices):
            bank_country = self.node_countries[bi]
            for pi in self.prod_indices:
                prod_country = self.node_countries[pi]
                # Bancos prestan más a empresas del mismo país
                country_factor = 1.0 if bank_country == prod_country else 0.2
                self.W_directed[bi, pi] = LENDING_BASE * bank_weights[bi_idx] * country_factor

        # --- Construir aristas ascendentes: company → bank ---
        # Proxy: empresas con más deuda pagan más intereses a los bancos
        # Cargamos debt_to_equity de fundamentals
        try:
            fund_resp = self.db.client.table("fundamentals").select(
                "ticker, debt_to_equity"
            ).order("report_date", desc=True).execute()

            debt_map = {}
            if fund_resp.data:
                for row in fund_resp.data:
                    t = row["ticker"]
                    if t not in debt_map and row.get("debt_to_equity") is not None:
                        debt_map[t] = min(float(row["debt_to_equity"]) / 200.0, 1.0)
        except Exception:
            debt_map = {}

        INTEREST_BASE = 0.01  # peso base de arista de intereses

        for pi in self.prod_indices:
            ticker = self.tickers[pi]
            debt_ratio = debt_map.get(ticker, 0.3)  # default moderado
            prod_country = self.node_countries[pi]

            for bi_idx, bi in enumerate(self.bank_indices):
                bank_country = self.node_countries[bi]
                country_factor = 1.0 if bank_country == prod_country else 0.2
                self.W_directed[pi, bi] = INTEREST_BASE * debt_ratio * country_factor

    # ─────────────────────────────────────────────────────────────
    # PASO 0.5: Calcular retornos reales (deflactados)
    # ─────────────────────────────────────────────────────────────

    def _compute_real_returns(self):
        """
        Log retornos en MONEDA LOCAL, deflactados por inflación.

        Para cada ticker en zona no-USD:
          r_local = r_usd - r_fx
        Esto elimina el ruido FX de las correlaciones.
        """
        # Forzar float64 — Supabase puede devolver columnas como object
        prices_f = self.prices.apply(pd.to_numeric, errors="coerce").astype(np.float64)

        # Log retornos nominales (en USD, como vienen de yfinance)
        log_ret = np.log(prices_f / prices_f.shift(1))
        log_ret = log_ret.iloc[1:]  # quitar primera fila (NaN por shift)

        # Alinear inflación con fechas de retornos
        pi = self.inflation_daily.reindex(log_ret.index).ffill().fillna(0)

        # Deflactar: r_real = r_nominal - π
        self.returns = log_ret.sub(pi, axis=0)

        # ── Convertir a retornos en moneda local ──
        # Para tickers no-USD: restar el retorno FX para aislar
        # el movimiento del activo en su propia moneda
        self.returns_local = self.returns.copy()

        for zone_pair, fx_info in config.FX_PAIRS.items():
            col = fx_info['column']
            sign = fx_info['sign']
            zone_a, zone_b = zone_pair  # e.g. ("USD", "EUR")

            # Buscar serie FX en dimensions
            fx_series = self.dimensions['currency'].get(
                next((c for c, cc in config.COUNTRY_CURRENCY.items()
                      if cc == col), None), None
            )
            if fx_series is None and col in ['dxy', 'eurusd', 'usdjpy']:
                # Intentar directamente desde atributos macro
                if col == 'dxy' and len(self.dxy) > 0:
                    fx_series = self.dxy
                elif col == 'eurusd' and 'eurusd' in self.dimensions.get('currency', {}):
                    fx_series = self.dimensions['currency'].get('DE')
                elif col == 'usdjpy' and 'usdjpy' in self.dimensions.get('currency', {}):
                    fx_series = self.dimensions['currency'].get('JP')

            if fx_series is not None and len(fx_series) > 1:
                # Log retorno del FX
                r_fx = np.log(fx_series / fx_series.shift(1)).dropna()
                r_fx = r_fx.reindex(self.returns.index).ffill().fillna(0)

                # Almacenar para acoplamiento FX en heat_engine
                self.dimensions['fx_returns'][zone_pair] = r_fx

                # Ajustar retornos de tickers de la zona no-USD
                target_zone = zone_b
                for i, ticker in enumerate(self.tickers):
                    if self.node_zones[i] == target_zone and ticker in self.returns_local.columns:
                        # r_local = r_usd - sign * r_fx
                        self.returns_local[ticker] -= sign * r_fx

        # Temperaturas acumuladas (capital real, en moneda local)
        self.u = self.returns_local.fillna(0).cumsum().values.astype(np.float64)

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

        # Garantizar simetría (intra-nivel)
        self.W = (self.W + self.W.T) / 2

        # ─── Aristas dirigidas inter-nivel (banks ↔ companies) ───
        self._build_directed_edges()

        # ─── Laplaciano de grafo con signo ───
        # Usamos W simétrico para el Laplaciano (difusión intra-nivel)
        # Las aristas dirigidas se aplican por separado en heat_engine.solve()
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

        # ─── Per-zone Laplacians (campos monetarios separados) ───
        self._build_zone_laplacians()

    def _build_zone_laplacians(self):
        """
        Construye Laplacianos independientes por zona monetaria.

        Cada zona usa un subgrafo extraído de W (solo nodos de esa zona).
        El heat_engine resuelve la ecuación POR ZONA y las acopla via FX.
        """
        self.zone_laplacians = {}

        for zone_id, indices in self.currency_zones.items():
            n_z = len(indices)
            if n_z < 2:
                # Zona con 1 nodo: no hay Laplaciano, trivial
                self.zone_laplacians[zone_id] = {
                    'indices': indices,
                    'tickers': [self.tickers[i] for i in indices],
                    'L': np.zeros((1, 1)),
                    'eigenvalues': np.array([0.0]),
                    'eigenvectors': np.array([[1.0]]),
                    'L_s': np.zeros((1, 1)),
                }
                continue

            # Extraer submatriz W para esta zona
            idx = np.array(indices)
            W_zone = self.W[np.ix_(idx, idx)]

            # Laplaciano de la zona
            D_z = np.diag(np.abs(W_zone).sum(axis=1))
            L_z = D_z - W_zone

            # Eigendecomposition
            evals, evecs = np.linalg.eigh(L_z)
            evals = np.maximum(evals, 0.0)
            sort_idx = np.argsort(evals)
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]

            # Laplaciano fraccional L_z^s
            lam_s = np.power(evals, self.s)
            lam_s[0] = 0.0  # conservación de capital
            L_s_z = evecs @ np.diag(lam_s) @ evecs.T

            self.zone_laplacians[zone_id] = {
                'indices': indices,
                'tickers': [self.tickers[i] for i in indices],
                'L': L_z,
                'eigenvalues': evals,
                'eigenvectors': evecs,
                'L_s': L_s_z,
            }

    # ─────────────────────────────────────────────────────────────
    # CROSS-LAG CORRELATION
    # ─────────────────────────────────────────────────────────────

    def _compute_crosslag_corr(self, data: np.ndarray) -> tuple:
        """
        Para cada par (i,j), encuentra el lag que maximiza |corr|.

        DECONTAMINATION: Antes de correlacionar, elimina el factor
        común de mercado (retorno medio cross-sectional) para que W
        capture relaciones reales (sustitución, supply chain) y no
        co-movimiento espurio por exposición al mercado.

        Args:
            data: (T, N) array de retornos, float64

        Returns:
            W: (N, N) pesos con signo (filtrados por umbral)
            W_lag: (N, N) lag óptimo por par (positivo = i lidera j)
        """
        T, N = data.shape
        W = np.zeros((N, N))
        W_lag = np.zeros((N, N), dtype=int)

        # Reemplazar NaN por 0
        data_clean = np.nan_to_num(data, nan=0.0)

        # ─── DECONTAMINATION: quitar factor de mercado ───
        # El factor de mercado es el retorno medio cross-sectional
        # β_i = cov(ret_i, ret_mkt) / var(ret_mkt)
        # ret_residual_i = ret_i - β_i · ret_mkt
        mkt_ret = np.nanmean(data_clean, axis=1)  # (T,) retorno medio del mercado
        mkt_var = np.var(mkt_ret)
        if mkt_var > 1e-12:
            betas = np.array([
                np.cov(data_clean[:, i], mkt_ret)[0, 1] / mkt_var
                for i in range(N)
            ])
            data_resid = data_clean - np.outer(mkt_ret, betas)  # (T, N)
        else:
            data_resid = data_clean

        for i in range(N):
            for j in range(i + 1, N):
                best_corr = 0.0
                best_lag = 0

                for lag in range(-MAX_LAG, MAX_LAG + 1):
                    if lag >= 0:
                        x = data_resid[:T - lag, i]
                        y = data_resid[lag:, j]
                    else:
                        x = data_resid[-lag:, i]
                        y = data_resid[:T + lag, j]

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
        s(t) = clip(s_base - Σ βᵢ·xᵢ_norm, s_min, s_max)

        6 indicadores de estrés:
          1. VIX: miedo general (normalizado vs 15-40 rango)
          2. DXY: flight-to-USD (normalizado vs 95-110 rango, sube=stress)
          3. Yield spread: inversión = recesión (spread negativo=stress)
          4. Credit spread: tensión de deuda
          5. Copper: caída = estrés industrial (normalizado como % cambio 60d)
          6. Oil: shock energético (volatilidad del oil)
        """
        def _get_val(series, ref=None):
            """Extrae valor de la serie en ref o último disponible."""
            if len(series) == 0:
                return None
            if ref:
                v = series.asof(pd.Timestamp(ref))
            else:
                v = series.iloc[-1]
            return v if pd.notna(v) else None

        # P5: track previous s for ds/dt computation
        self.s_prev = self.s

        if len(self.vix) == 0:
            self.s = S_BASE
            self.ds_dt = 0.0
            return

        vix_val    = _get_val(self.vix, reference_date)
        spread_val = _get_val(self.yield_spread, reference_date)
        credit_val = _get_val(self.credit_spread, reference_date)
        dxy_val    = _get_val(self.dxy, reference_date)
        copper_val = _get_val(self.copper, reference_date)
        oil_val    = _get_val(self.oil, reference_date)

        # Z-normalizar cada indicador por SU PROPIA historia
        # stress = max(0, z-score) → solo contribuye si está por encima de su media
        def _z_stress(val, series, invert=False):
            """Z-score normalizado a [0, ~2]. invert=True para indicadores donde baja=stress."""
            if val is None or len(series) < 30:
                return 0.0
            mu = series.mean()
            sigma = series.std()
            if sigma < 1e-10:
                return 0.0
            z = (val - mu) / sigma
            if invert:
                z = -z  # invertir: caída = stress positivo
            return max(0.0, z)  # Solo stress positivo

        # VIX sube = stress, DXY sube = stress (flight to USD)
        # spread baja = stress (inversión), copper baja = stress, oil volatilidad
        vix_norm    = _z_stress(vix_val, self.vix)
        dxy_norm    = _z_stress(dxy_val, self.dxy)
        spread_norm = _z_stress(spread_val, self.yield_spread, invert=True)
        credit_norm = _z_stress(credit_val, self.credit_spread) if len(self.credit_spread) > 0 else 0.0

        # P2.2: Credit spread DELTA — widening speed as early warning
        credit_delta_val = _get_val(self.credit_spread_delta, reference_date)
        credit_delta_norm = _z_stress(credit_delta_val, self.credit_spread_delta) if len(self.credit_spread_delta) > 0 else 0.0

        # P2.1: Rate momentum — hiking cycle = contractionary stress
        rate_mom_val = _get_val(self.rate_momentum, reference_date)
        rate_mom_norm = _z_stress(rate_mom_val, self.rate_momentum) if len(self.rate_momentum) > 0 else 0.0

        copper_norm = _z_stress(copper_val, self.copper, invert=True)

        # Oil: usar volatilidad reciente como stress
        if len(self.oil) > 20:
            oil_vol = self.oil.tail(20).pct_change().std()
            oil_vol_hist = self.oil.pct_change().std()
            oil_norm = max(0.0, (oil_vol - oil_vol_hist) / max(oil_vol_hist, 1e-6))
        else:
            oil_norm = 0.0

        # Heuristic s (used as UKF prior attractor)
        s_heuristic = np.clip(
            S_BASE
            - S_VIX_COEF    * vix_norm
            - S_DXY_COEF    * dxy_norm
            - S_SPREAD_COEF * spread_norm
            - S_CREDIT_COEF * credit_norm
            - S_CREDIT_DELTA * credit_delta_norm   # P2.2: early warning
            - S_RATE_MOM    * rate_mom_norm           # P2.1: CB rate momentum
            - S_COPPER_COEF * copper_norm
            - S_OIL_COEF    * oil_norm,
            S_MIN, S_MAX
        )

        # P3.1: UKF refines s using heuristic as prior
        self.ukf_s.predict(prior_s=s_heuristic)

        # If we have recent prediction errors, update UKF
        if hasattr(self, '_last_prediction_errors') and len(self._last_prediction_errors) > 0:
            self.ukf_s.update(self._last_prediction_errors)
            self.s = self.ukf_s.get_s()
        else:
            # No prediction errors yet → use heuristic
            self.s = s_heuristic

        # P5: compute ds/dt (rate of change)
        self.ds_dt = self.s - self.s_prev

        # P3.2: Persist UKF state
        self.db.save_kalman_state('ukf_s', self.ukf_s.to_dict())

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
