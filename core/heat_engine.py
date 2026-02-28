"""
HEAT ENGINE — GlobalMarketAnalyzer
====================================
Resuelve advección-difusión fraccional CON INERCIA en el grafo:

  γ·d²m/dt² + dm/dt = -α · L^s · m + v(t) + f(t)

  inercia:    γ·d²m/dt²       → el dinero tiene masa (momentum de mercado)
  fricción:   dm/dt           → amortiguamiento natural
  difusión:  -α·L^s·m         → equilibrar (calor fluye de caliente a frío)
  advección:  v(t)            → expectativa (dónde se MOVERÁ el dinero)
  fuente:     f(t)            → fundamentales (quién genera/destruye valor)

  γ=1: O-U puro (sin inercia, comportamiento original)
  γ>1: trends persisten (el mercado tiene memoria)
  γ→∞: momentum puro (trend following)

Calcula:
  - u_pred(t): predicción del modelo
  - ε(t) = u_real - u_pred: residuos
  - z(t) = ε / σ_ε: residuos normalizados
  - v(t): velocidad macro (dónde fluye el capital)
  - refuge_signal: cuándo ir a refugio y cuándo salir
"""

import numpy as np
import pandas as pd


# ── Parámetros del solver ──
ALPHA_DEFAULT   = 0.05    # coeficiente de difusión base
ALPHA_RANGE     = (0.005, 0.50)  # rango ampliado (antes 0.01-0.20)
SIGMA_WINDOW    = 20      # ventana para normalizar residuos
RETURN_HORIZON  = 5       # días para probabilidad de reversión
LAMBDA_TREND_TH = 0.10    # modos con λ < esto son TENDENCIA
TRAIN_FRAC      = 0.70    # fracción para calibración (rest = test)
ALPHA_REG       = 0.10    # regularización: penalizar α bajos
MOMENTUM_WIN    = 20      # ventana para estimar momentum de modos lentos
GAMMA_DEFAULT   = 5.0     # inercia: γ=1 → O-U puro, γ>1 → momentum
GAMMA_RANGE     = (1.0, 50.0)  # rango para calibración de γ


class HeatEngine:
    """
    Solver O-U en espacio espectral.

    Uso:
        engine = HeatEngine(graph_builder, fundamental_filter)
        engine.solve()
        residuals = engine.residuals      # (T, N)
        z_scores  = engine.z_scores       # (T, N)
        spectral  = engine.spectral_res   # (T, N) en modos del grafo
    """

    def __init__(self, graph_builder, fundamental_filter):
        self.gb = graph_builder
        self.ff = fundamental_filter

        self.alpha: float = ALPHA_DEFAULT
        self.gamma: float = GAMMA_DEFAULT
        self.tickers = graph_builder.tickers
        self.N = graph_builder.N

        # Resultados
        self.u_real: np.ndarray = np.array([])       # (T, N)
        self.u_pred: np.ndarray = np.array([])       # (T, N)
        self.residuals: np.ndarray = np.array([])    # (T, N) ε
        self.z_scores: np.ndarray = np.array([])     # (T, N) z
        self.spectral_res: np.ndarray = np.array([]) # (T, N) ε_k
        self.alpha_per_mode: np.ndarray = np.array([])  # (N,) α_k
        self.sigma_residual: np.ndarray = np.array([])  # (N,) σ por activo

        # Advección: velocidad macro y flujos de capital
        self.macro_velocity: np.ndarray = np.array([])  # (T, N) v(t)
        self.capital_flow: np.ndarray = np.array([])     # (T,) net capital in system
        self.refuge_signal: float = 0.0                  # [-1, +1] cuándo ir a refugio

        # Campo dual: mispricing δ(t) = u(t) - c(t)
        self.mispricing: np.ndarray = np.array([])       # (T, N) δ weighted by confidence
        self.capital_field = None                         # CapitalField instance

        # Inicializar alpha_per_mode con defaults
        self.alpha_per_mode = np.full(self.N, ALPHA_DEFAULT)

    def calibrate_alpha(self):
        """
        Calibra α con train/test split + regularización.
        α_k = argmin_α  [err_test(α) + λ_reg · (α_min/α)²]
        """
        from scipy.optimize import minimize_scalar

        u = self.gb.u.astype(np.float64)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        T_len = len(u)

        if T_len < 60:
            self.alpha = ALPHA_DEFAULT
            self.alpha_per_mode = np.full(self.N, ALPHA_DEFAULT)
            return

        Phi = self.gb.eigenvectors
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)
        lam_s[0] = 0.0
        f_vec = np.nan_to_num(self._compute_dynamic_f())
        f_k = Phi.T @ f_vec
        u_k = u @ Phi  # (T, N)

        # Train/test split temporal
        t_split = int(T_len * TRAIN_FRAC)

        self.alpha_per_mode = np.full(self.N, ALPHA_DEFAULT)
        self.mode_type = np.array(['revert'] * self.N, dtype=object)  # 'trend' o 'revert'

        # Clasificar modos: trend vs revert
        for k in range(self.N):
            if lam_s[k] < LAMBDA_TREND_TH:
                self.mode_type[k] = 'trend'

        # Calibrar α global con OUT-OF-SAMPLE + regularización
        def global_error_oos(alpha):
            mu = alpha * lam_s
            decay = np.where(mu > 1e-10, np.exp(-mu), 1.0)
            eq_term = np.where(mu > 1e-10, f_k / np.maximum(mu, 1e-10) * (1.0 - decay), f_k)
            # Predecir solo en TEST set (t >= t_split)
            # Usando datos de train como base
            u_pred_k = u_k[t_split:-1] * decay[np.newaxis, :] + eq_term[np.newaxis, :]
            err = float(np.sum((u_k[t_split+1:] - u_pred_k) ** 2))
            # Regularización: penalizar α muy bajo (overfitting)
            reg = ALPHA_REG * (ALPHA_RANGE[0] / max(alpha, 1e-6)) ** 2
            return err + reg * err  # relative regularization

        result = minimize_scalar(global_error_oos, bounds=ALPHA_RANGE, method='bounded')
        self.alpha = float(result.x)

        # Calibrar α por modo (solo modos revert, primeros 30)
        for k in range(min(self.N, 30)):
            if self.mode_type[k] == 'trend' or lam_s[k] < 1e-10:
                self.alpha_per_mode[k] = self.alpha
                continue

            def mode_error_oos(a, _k=k):
                mk = a * lam_s[_k]
                dk = np.exp(-mk) if mk > 1e-10 else 1.0
                eqk = (f_k[_k] / mk * (1 - dk)) if mk > 1e-10 else f_k[_k]
                pred = u_k[t_split:-1, _k] * dk + eqk
                err = float(np.sum((u_k[t_split+1:, _k] - pred) ** 2))
                reg = ALPHA_REG * (ALPHA_RANGE[0] / max(a, 1e-6)) ** 2
                return err + reg * err

            res = minimize_scalar(mode_error_oos, bounds=ALPHA_RANGE, method='bounded')
            self.alpha_per_mode[k] = float(res.x)

    def calibrate_gamma(self):
        """
        Calibra γ (inercia) via grid search con train/test split.
        γ=1 → O-U puro (sin inercia), γ>1 → el dinero tiene momentum.

        Ecuación de 2do orden: γ·m'' + m' = -μ·(m - m_eq)
        Discretizada: m[t+1] = m[t] + (1-1/γ)·v[t] - (μ/γ)·(m[t] - m_eq)
        """
        u = self.gb.u.astype(np.float64)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        T_len = len(u)

        if T_len < 60:
            self.gamma = GAMMA_DEFAULT
            return

        Phi = self.gb.eigenvectors
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)
        lam_s[0] = 0.0
        f_vec = np.nan_to_num(self._compute_dynamic_f())
        f_k = Phi.T @ f_vec
        m_k = u @ Phi

        alpha_vec = np.full(self.N, self.alpha)
        alpha_vec[:min(self.N, 30)] = self.alpha_per_mode[:min(self.N, 30)]
        mu = alpha_vec * lam_s
        m_eq = np.where(mu > 1e-10, f_k / mu, 0.0)

        t_split = int(T_len * TRAIN_FRAC)

        # Spectral velocity: v_k[t] = m_k[t] - m_k[t-1]
        vel = np.zeros_like(m_k)
        vel[1:] = np.diff(m_k, axis=0)

        best_gamma = 1.0  # default = sin inercia (comportamiento original)
        best_err = np.inf

        for gamma_c in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0]:
            mw = 1.0 - 1.0 / gamma_c       # momentum weight
            rw = mu / gamma_c               # revert weight

            pred = (m_k[t_split:-1]
                    + mw * vel[t_split:-1]
                    - rw * (m_k[t_split:-1] - m_eq))
            err = float(np.sum((m_k[t_split+1:] - pred) ** 2))

            if err < best_err:
                best_err = err
                best_gamma = gamma_c

        self.gamma = best_gamma

    def _compute_dynamic_f(self):
        """
        Calcula f dinámico por rol de nodo.

        bank:       f = yield_spread × lending_weight (crean dinero)
        productive: f = (dK/dt + crédito_in - intereses_out) × S(t)
        """
        f = np.zeros(self.N)

        # Yield spread (proxy del margen bancario)
        yield_spread = 0.02  # default 200bps
        if hasattr(self.gb, 'yield_spread') and len(self.gb.yield_spread) > 0:
            last_spread = self.gb.yield_spread.iloc[-1]
            if not np.isnan(last_spread):
                yield_spread = max(0.005, float(last_spread) / 100.0)

        # Composite sentiment: S_fund × S_macro(PMI) × S_fear(VIX) × S_earnings
        sentiment = self.ff.get_composite_sentiment_vector(self.tickers)

        # Capital creation rate dK/dt (diario)
        capital_rates = np.zeros(self.N)
        if self.capital_field is not None:
            for i, t in enumerate(self.tickers):
                capital_rates[i] = self.capital_field.capital_rate_daily.get(t, 0.0)

        # Directed edges (from graph_builder)
        W_dir = self.gb.W_directed if hasattr(self.gb, 'W_directed') and \
                self.gb.W_directed.size > 0 else np.zeros((self.N, self.N))

        # Node roles (padded to match self.N)
        roles = self.gb.node_roles if hasattr(self.gb, 'node_roles') else []
        if len(roles) < self.N:
            roles = list(roles) + ['productive'] * (self.N - len(roles))

        for i in range(self.N):
            if roles[i] == 'bank':
                # Bancos: crean dinero. f = margen × capacidad de préstamo
                lending_capacity = W_dir[i, :].sum()  # aristas descendentes
                f[i] = yield_spread * lending_capacity * sentiment[i]
            else:
                # Productive: f = (dK/dt + crédito recibido - intereses pagados) × S
                dk_dt = capital_rates[i]
                credit_in = W_dir[:, i].sum()   # préstamos recibidos de bancos
                interest_out = W_dir[i, :].sum()  # intereses pagados a bancos
                f[i] = (dk_dt + credit_in - interest_out) * sentiment[i]

        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_dim_corrections(self, m_current: np.ndarray) -> np.ndarray:
        """
        Computes dimensional corrections Ω(t) + FX coupling Φ(t).

        Ω_i = -η · (D/GDP) · m_i/252          (sovereign debt drag)
              - β_r · (dr_FED/dt) · m_i        (rate effect, role-dependent)

        Φ_i = β_fx · r_fx · m̄_other_zone      (FX coupling between zones)
        """
        import config as cfg

        omega = np.zeros(self.N)
        params = cfg.DIM_PARAMS

        # --- Debt drag: constant slow drain proportional to D/GDP ---
        countries = self.gb.node_countries if hasattr(self.gb, 'node_countries') else []
        if len(countries) < self.N:
            countries = list(countries) + ['US'] * (self.N - len(countries))
        for i in range(self.N):
            d_gdp = cfg.SOVEREIGN_DEBT_GDP.get(countries[i], 1.0)
            omega[i] -= params['eta_debt'] * d_gdp * m_current[i] / 252.0

        # --- Interest rate effect: per-country, role-dependent ---
        # Each node feels its OWN central bank's rate changes
        roles = self.gb.node_roles if hasattr(self.gb, 'node_roles') else []
        if len(roles) < self.N:
            roles = list(roles) + ['productive'] * (self.N - len(roles))

        # Pre-compute dr/dt per rate series (avoid recomputing for each node)
        rate_changes = {}  # series_id → dr/dt
        fed_rate = self.gb.dimensions.get('fed_rate', pd.Series(dtype=float))
        if isinstance(fed_rate, pd.Series) and len(fed_rate) >= 2:
            r_diff = fed_rate.diff().tail(20).mean()
            if pd.notna(r_diff):
                rate_changes['FEDFUNDS'] = float(r_diff) / 100.0

        # Try to load other central bank rates from macro_indicators
        try:
            for zone, series_id in cfg.CENTRAL_BANK_RATES.items():
                if series_id in rate_changes:
                    continue
                field = f"rate_{zone.lower()}"
                r = self.ff.db.client.table("macro_indicators").select(
                    f"date, {field}"
                ).not_.is_(field, "null").order("date", desc=True).limit(20).execute()
                if r.data and len(r.data) >= 2:
                    vals = [float(row[field]) for row in r.data if row.get(field) is not None]
                    if len(vals) >= 2:
                        dr = (vals[0] - vals[-1]) / len(vals) / 100.0
                        rate_changes[series_id] = dr
        except Exception:
            pass

        # Apply rate effect per node using its country's central bank
        for i in range(self.N):
            country = countries[i]
            series_id = cfg.COUNTRY_RATE_SERIES.get(country, 'FEDFUNDS')
            dr_dt = rate_changes.get(series_id, 0.0)

            if abs(dr_dt) < 1e-8:
                continue

            if roles[i] == 'bank':
                # Banks gain from rate hikes (NIM widens)
                omega[i] += params['beta_r_bank'] * dr_dt * m_current[i]
            else:
                # Companies suffer, more if leveraged
                lev = 1.0
                if self.capital_field is not None:
                    d2e = self.capital_field.capital_rate.get(self.tickers[i], 0)
                    lev = 1.0 + max(0, -d2e)
                omega[i] -= params['beta_r_prod'] * lev * dr_dt * m_current[i]

        # --- FX coupling: flow between zones ---
        fx_returns = self.gb.dimensions.get('fx_returns', {})
        zones = self.gb.node_zones if hasattr(self.gb, 'node_zones') else ['USD'] * self.N
        zone_indices = self.gb.currency_zones if hasattr(self.gb, 'currency_zones') else {}

        for (zone_a, zone_b), r_fx_series in fx_returns.items():
            if len(r_fx_series) == 0:
                continue
            # Last FX return
            r_fx_last = float(r_fx_series.iloc[-1]) if pd.notna(r_fx_series.iloc[-1]) else 0.0

            if abs(r_fx_last) < 1e-8:
                continue

            # Mean m in zone_a (capital available to flow)
            idx_a = zone_indices.get(zone_a, [])
            m_bar_a = np.mean(m_current[idx_a]) if idx_a else 0.0

            # Mean m in zone_b
            idx_b = zone_indices.get(zone_b, [])
            m_bar_b = np.mean(m_current[idx_b]) if idx_b else 0.0

            # FX_PAIRS sign convention: positive sign = zone_a strengthens
            fx_info = cfg.FX_PAIRS.get((zone_a, zone_b), {})
            sign = fx_info.get('sign', 1)

            # Flow: when zone_a's currency strengthens, money flows FROM zone_b TO zone_a
            flow = params['beta_fx'] * sign * r_fx_last

            # Apply: zone_a gains, zone_b loses
            for i in idx_a:
                omega[i] += flow * m_bar_b / max(len(idx_a), 1)
            for i in idx_b:
                omega[i] -= flow * m_bar_a / max(len(idx_b), 1)

        return np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)

    def solve(self, calibrate: bool = True):
        """
        Solve the water-landscape model with INERTIA (2nd order):

        γ·d²m/dt² + dm/dt = -α·L^s·m + v(t) + f(t)

        When γ=1: reduces to original O-U (no inertia)
        When γ>1: money has momentum, trends persist
        When γ→∞: pure trend following

        Spectral decomposition per mode k:
          m_k[t+1] = m_k[t] + (1-1/γ)·v_k[t] - (μ_k/γ)·(m_k[t] - m_eq_k)
          + advection v(t) pushes money per macro flows
        """
        # m(t) = money field ≡ u(t) (cumulative real returns = money assigned)
        m = self.gb.u.astype(np.float64)  # (T, N)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        T_len = len(m)
        if T_len < 2:
            raise ValueError("Datos insuficientes para resolver")

        if calibrate:
            self.calibrate_alpha()
            self.calibrate_gamma()
            print(f"    α={self.alpha:.4f} | γ={self.gamma:.1f} "
                  f"(momentum_weight={1-1/self.gamma:.2f})")

        Phi = self.gb.eigenvectors
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)
        lam_s[0] = 0.0

        # Build capital field early (needed for dynamic f)
        try:
            from core.capital_field import CapitalField
            cf = CapitalField(self.ff.db)
            dates = self.gb.returns.index
            cf.build(self.tickers, dates)
            self.capital_field = cf
        except Exception as e:
            self.capital_field = None
            print(f"    ⚠ Capital field para f dinámico no disponible: {e}")

        # Dynamic f: por rol (bank vs productive)
        f_vec = self._compute_dynamic_f()
        f_k = Phi.T @ f_vec

        # Macro velocity v(t) — direction of money flows
        self._compute_macro_velocity(T_len)

        # Project money to spectral space
        m_k_real = m @ Phi  # (T, N)

        # ═══ SECOND-ORDER SOLVER: Damped oscillator with inertia ═══
        # γ·m'' + m' = -μ·(m - m_eq) + advection
        # Discrete: m[t+1] = m[t] + (1-1/γ)·v[t] - (μ/γ)·(m[t]-m_eq) + adv
        #
        # γ=1  → pure O-U (backward compatible with previous version)
        # γ>1  → money has inertia, trends persist longer
        # γ→∞  → pure momentum (random walk with trend following)
        m_k_pred = np.zeros_like(m_k_real)
        m_k_pred[0] = m_k_real[0]

        alpha_vec = np.full(self.N, self.alpha)
        alpha_vec[:min(self.N, 30)] = self.alpha_per_mode[:min(self.N, 30)]
        mu = alpha_vec * lam_s  # restoring rate per mode

        # Equilibrium per mode: m_eq_k = f_k / μ_k
        m_eq = np.where(mu > 1e-10, f_k / mu, 0.0)

        # Dimensional corrections Ω(t): debt drag + Fed rate + FX coupling
        # Computed in physical space, then projected to spectral
        omega_phys = self._compute_dim_corrections(m[-1])  # use last known m
        omega_k = Phi.T @ omega_phys
        # Ω shifts the equilibrium: m_eq_k += Ω_k / μ_k (safe division)
        m_eq += np.where(mu > 1e-10, omega_k / mu, 0.0)

        # Inertia coefficients (derived from γ)
        gamma = self.gamma
        momentum_weight = 1.0 - 1.0 / gamma   # 0 when γ=1, →1 when γ→∞
        revert_weight = mu / gamma             # strong when γ=1, weak when γ→∞

        # Spectral velocity: v_k[t] = m_k[t] - m_k[t-1]
        spectral_velocity = np.zeros_like(m_k_real)
        spectral_velocity[1:] = np.diff(m_k_real, axis=0)

        # Project v(t) to spectral space for advection
        v_spectral = self.macro_velocity @ Phi if len(self.macro_velocity) > 0 \
            else np.zeros_like(m_k_real)

        ADV_WEIGHT = 0.05

        for t in range(T_len - 1):
            # Second-order prediction: position + inertia + restoring force
            m_k_pred[t + 1] = (
                m_k_real[t]                                          # current position
                + momentum_weight * spectral_velocity[t]             # inertia term
                - revert_weight * (m_k_real[t] - m_eq)              # restoring force
            )

            # Advection: macro velocity pushes money
            if t < len(v_spectral):
                m_k_pred[t + 1] += ADV_WEIGHT * v_spectral[t]

        # Spectral residuals
        self.spectral_res = m_k_real - m_k_pred

        # Store spectral velocity for diagnostics
        self.spectral_velocity = spectral_velocity

        # Mode classification (informational, preserved for compatibility)
        self.is_trend_mode = np.array([lam_s[k] < LAMBDA_TREND_TH for k in range(self.N)])
        self.n_trend_modes = int(np.sum(self.is_trend_mode))
        self.n_revert_modes = self.N - self.n_trend_modes

        # Reconstruct in real space
        self.u_real = m  # m(t) = money = price (they're the same thing)
        self.u_pred = m_k_pred @ Phi.T
        self.residuals = self.u_real - self.u_pred

        # Sigma per asset
        self.sigma_residual = np.nanstd(self.residuals, axis=0)

        # Z-scores
        self._compute_z_scores()

        # Refuge signal
        self._compute_refuge_signal()

        # === WATER-LANDSCAPE: build K(t) terrain and compute λ = m/K ===
        self._compute_landscape()

    def _compute_macro_velocity(self, T_len: int):
        """
        v(t) = β · ΔM(t) + injection(t)

        ΔM(t): tasa de cambio z-normalizada de [VIX, DXY, yield_curve, copper, oil]
        β:     sensibilidad de cada activo a cada macro (rolling 120d regression)
        injection(t): capital neto entrando/saliendo del sistema

        El capital NO se conserva: QE inyecta, QT extrae, defaults destruyen.
        injection(t) = Σ_i ret_i(t) · cap_i  (si el mercado sube en agregado,
        hay capital neto entrando — vía earnings, buybacks, inversión real)
        """
        gb = self.gb
        returns = gb.returns.apply(pd.to_numeric, errors='coerce').fillna(0).values
        T, N = returns.shape

        # Macro series como arrays
        macro_series = {}
        dates = gb.returns.index

        for name, series in [("vix", gb.vix), ("dxy", gb.dxy),
                              ("yield", gb.yield_spread),
                              ("copper", gb.copper), ("oil", gb.oil)]:
            if len(series) > 0:
                macro_series[name] = series.reindex(dates).ffill().bfill().values
            else:
                macro_series[name] = np.zeros(T)

        n_macro = len(macro_series)
        macro_names = list(macro_series.keys())
        macro_data = np.column_stack([macro_series[n] for n in macro_names])  # (T, 5)

        # ΔM(t): z-normalized rate of change (20d window)
        WIN = 20
        delta_macro = np.zeros_like(macro_data)
        for j in range(n_macro):
            for t in range(WIN, T):
                change = macro_data[t, j] - macro_data[t - WIN, j]
                std = np.std(macro_data[max(0, t-120):t, j])
                delta_macro[t, j] = change / max(std, 1e-8)

        # β(t): sensibilidad rolling (120d) de cada activo a cada macro
        BETA_WIN = 120
        betas = np.zeros((T, N, n_macro))  # (T, N, 5)

        for t in range(BETA_WIN, T):
            ret_window = returns[t-BETA_WIN:t, :]       # (120, N)
            dm_window = delta_macro[t-BETA_WIN:t, :]    # (120, 5)

            for j in range(n_macro):
                dm_j = dm_window[:, j]
                dm_var = np.var(dm_j)
                if dm_var < 1e-12:
                    continue
                for i in range(N):
                    cov_ij = np.cov(ret_window[:, i], dm_j)[0, 1]
                    betas[t, i, j] = cov_ij / dm_var

        # v(t) = Σ_j β_ij · ΔM_j(t)  → (T, N)
        self.macro_velocity = np.zeros((T, N))
        for t in range(BETA_WIN, T):
            self.macro_velocity[t] = betas[t] @ delta_macro[t]  # (N,5)·(5,) = (N,)

        # Capital flow: net capital entering/leaving the system
        # Si Σ returns > 0 → capital neto entrando (earnings, buybacks, QE)
        # Si Σ returns < 0 → capital saliendo (QT, defaults, panic selling)
        self.capital_flow = np.nanmean(returns, axis=1)  # (T,) retorno medio del mercado

        # Suavizar capital flow (20d EMA)
        cap_ema = np.zeros(T)
        alpha_ema = 2.0 / (WIN + 1)
        cap_ema[0] = self.capital_flow[0]
        for t in range(1, T):
            cap_ema[t] = alpha_ema * self.capital_flow[t] + (1 - alpha_ema) * cap_ema[t-1]
        self.capital_flow_smooth = cap_ema

        # Añadir injection al velocity: cuando capital entra al sistema,
        # empuja a TODOS los activos proporcionalmente a su beta de mercado
        for t in range(BETA_WIN, T):
            injection = cap_ema[t] * 10  # scale factor
            self.macro_velocity[t] += injection  # uniform push

    def _compute_refuge_signal(self):
        """
        Refuge signal ∈ [-1, +1]:
          +1 = ir a refugio (bonds/gold) — capital sale de equity
          -1 = ir a equity — capital entra a risk-on
           0 = neutral

        Basado en:
          1. v(t) de activos refuge vs equity
          2. capital_flow_smooth dirección
          3. s(t) nivel de stress
        """
        gb = self.gb
        tickers = self.tickers
        sectors = gb.sectors

        # Indices de activos refuge vs equity
        refuge_idx = []
        equity_idx = []
        for s in ["BONDS_GOVT", "BONDS_CORP", "COMMODITIES"]:
            refuge_idx.extend([tickers.index(t) for t in sectors.get(s, []) if t in tickers])
        for s in ["TECH_MEGA", "TECH_SEMIS", "BANKS", "ENERGY", "CONSUMER_DISC"]:
            equity_idx.extend([tickers.index(t) for t in sectors.get(s, []) if t in tickers])

        if not refuge_idx or not equity_idx or len(self.macro_velocity) == 0:
            self.refuge_signal = 0.0
            return

        # v medio de refugio vs equity (último valor)
        v_refuge = np.mean(self.macro_velocity[-1, refuge_idx])
        v_equity = np.mean(self.macro_velocity[-1, equity_idx])

        # Capital flow direction
        cap_dir = self.capital_flow_smooth[-1] if len(self.capital_flow_smooth) > 0 else 0

        # Refuge signal:
        # v_refuge > v_equity → capital fluye a refugio → +1
        # v_equity > v_refuge → capital fluye a equity → -1
        raw = v_refuge - v_equity
        # Normalizar a [-1, +1]
        self.refuge_signal = float(np.clip(raw * 20, -1, 1))

        # Modular por stress: si s bajo (crisis), signal es más creíble
        self.refuge_signal *= (1 + (1 - gb.s))

    def _compute_landscape(self):
        """
        Water-landscape model:
          K(t) = terrain (capital real, from capital_field)
          λ = m/K = valuation multiple (derived)
          δ = λ - λ_eq(regime) = mispricing signal

        Also computes L_K: capital-weighted Laplacian for λ contagion.
        Big-K companies drag neighbors' λ toward theirs.
        """
        import json
        from pathlib import Path

        T, N = self.u_real.shape

        try:
            if self.capital_field is not None:
                cf = self.capital_field
            else:
                from core.capital_field import CapitalField
                cf = CapitalField(self.ff.db)
                dates = self.gb.returns.index
                cf.build(self.tickers, dates)
                self.capital_field = cf

            # K(t) = terrain — accumulated real capital per asset
            K = cf.c_daily.values if hasattr(cf, 'c_daily') else None
        except Exception as e:
            K = None
            self.capital_field = None
            print(f"    ⚠ Capital field K(t) no disponible: {e}")

        # --- Load regime-calibrated λ_eq ---
        regime_params = {}
        cal_path = Path(__file__).parent / "calibration_results.json"
        if cal_path.exists():
            try:
                with open(cal_path) as f:
                    cal_data = json.load(f)
                regime_params = cal_data.get("regime_params", {})
            except Exception:
                pass

        # --- Classify current regime from s(t) ---
        s = self.gb.s
        if s < 0.40:
            current_regime = "crisis"
        elif s < 0.65:
            current_regime = "stress"
        elif s < 0.85:
            current_regime = "normal"
        else:
            current_regime = "hype"

        # λ_eq for current regime (fallback: 18 = typical PE)
        lambda_eq = regime_params.get(current_regime, {}).get("lambda_eq", 18.0)

        # --- Compute λ = m/K and δ = λ - λ_eq ---
        if K is not None and K.shape == self.u_real.shape:
            # Avoid division by zero: where K ≈ 0, set λ to neutral
            K_safe = np.where(np.abs(K) > 1e-6, K, np.nan)
            lambda_field = self.u_real / K_safe  # (T, N)
            lambda_field = np.nan_to_num(lambda_field, nan=lambda_eq)

            # δ(t) = λ(t) - λ_eq(regime)
            # Positive δ → overvalued for this regime
            # Negative δ → undervalued
            delta = lambda_field - lambda_eq

            # Z-normalize δ per asset for signal compatibility
            delta_std = np.nanstd(delta, axis=0)
            delta_std = np.where(delta_std > 1e-8, delta_std, 1.0)
            self.mispricing = delta / delta_std

            # --- L_K: capital-weighted Laplacian for contagion ---
            # [L_K·λ]_i = (1/K_i) * Σ_j L_ij * K_j * λ_j
            # Big-K firms drag neighbors' λ toward theirs
            K_last = K[-1]  # use latest K for weighting
            K_abs = np.abs(K_last) + 1e-8
            L = np.diag(np.sum(self.gb.W, axis=1)) - self.gb.W
            # Guard: cap K ratios to prevent Inf from extreme K differences
            K_ratio = K_abs[np.newaxis, :] / K_abs[:, np.newaxis]
            K_ratio = np.clip(K_ratio, 0.01, 100.0)
            L_K = L * K_ratio
            L_K = np.nan_to_num(L_K, nan=0.0, posinf=0.0, neginf=0.0)
            self.L_K = L_K

            # λ contagion signal: how much λ_i differs from K-weighted neighbors
            lambda_last = lambda_field[-1] if len(lambda_field) > 0 else np.full(N, lambda_eq)
            self.lambda_contagion = L_K @ (lambda_last - lambda_eq)

            # Store for signal_generator
            self.lambda_field = lambda_field
            self.lambda_eq = lambda_eq
            self.current_regime = current_regime

            quality_flag = "real"
        else:
            # Fallback: no K data → mispricing = 0 (neutral)
            self.mispricing = np.zeros((T, N))
            self.lambda_field = np.full((T, N), lambda_eq)
            self.lambda_contagion = np.zeros(N)
            self.lambda_eq = lambda_eq
            self.current_regime = current_regime
            self.L_K = np.eye(N)
            quality_flag = "neutral"

        # Store quality for signal weighting
        if not hasattr(self, 'capital_field') or self.capital_field is None:
            self._landscape_quality = "neutral"
        else:
            self._landscape_quality = quality_flag

        print(f"    Régimen: {current_regime} | λ_eq = {lambda_eq:.1f} | "
              f"Landscape: {quality_flag}")

    def _compute_z_scores(self):
        """z[i,t] = ε[i,t] / σ_ε[i, rolling]"""
        T_len, N = self.residuals.shape
        self.z_scores = np.zeros_like(self.residuals)

        for i in range(N):
            eps_series = pd.Series(self.residuals[:, i])
            sigma = eps_series.rolling(SIGMA_WINDOW, min_periods=5).std()
            sigma = sigma.replace(0, np.nan).fillna(eps_series.std())
            self.z_scores[:, i] = (eps_series / sigma).fillna(0).values

    def compute_probability(self, ticker_idx: int, horizon: int = None) -> dict:
        """
        Probabilidad analítica O-U de que el activo i revierta.

        En O-U, la distribución del proceso en t+Δt dado u(t) es:
          u(t+Δt) ~ N(μ_eq + (u(t) - μ_eq)·e^{-α·Δt},  σ²·(1-e^{-2α·Δt})/(2α))

        Devuelve:
          P(reversion): prob de que el residuo se reduzca (vuelva a 0)
          P(continue):  prob de que siga en la misma dirección
          expected_return: retorno esperado por la reversión O-U
        """
        if horizon is None:
            horizon = RETURN_HORIZON

        i = ticker_idx
        if len(self.residuals) == 0 or i >= self.N:
            return {"p_reversion": 0.5, "p_continue": 0.5, "expected_return": 0.0}

        eps_now = self.residuals[-1, i]
        sigma = self.sigma_residual[i] if self.sigma_residual[i] > 1e-10 else 0.01

        # Tasa efectiva de reversión para este activo
        # (promedio ponderado de μ_k de los modos donde participa)
        Phi = self.gb.eigenvectors
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)
        alpha_vec = np.full(self.N, self.alpha)
        alpha_vec[:min(self.N, 20)] = self.alpha_per_mode[:min(self.N, 20)]
        mu_vec = alpha_vec * lam_s

        # Tasa efectiva = Σ_k φ_k(i)² · μ_k  (ponderado por participación)
        participation = Phi[i, :] ** 2
        mu_eff = np.sum(participation * mu_vec)
        mu_eff = max(mu_eff, 1e-6)

        # O-U: valor esperado en t+horizon
        decay_h = np.exp(-mu_eff * horizon)
        eps_expected = eps_now * decay_h  # revierte hacia 0

        # Varianza condicional
        var_h = (sigma ** 2) * (1 - np.exp(-2 * mu_eff * horizon)) / (2 * mu_eff)
        std_h = np.sqrt(max(var_h, 1e-12))

        # P(reversión) = P(|ε(t+h)| < |ε(t)|) para procesos O-U
        # = P(ε se acerca a 0)
        from scipy import stats
        if eps_now > 0:
            # Activo caliente → reversión = baja
            p_reversion = stats.norm.cdf(0, loc=eps_expected, scale=std_h)
        elif eps_now < 0:
            # Activo frío → reversión = sube
            p_reversion = 1 - stats.norm.cdf(0, loc=eps_expected, scale=std_h)
        else:
            p_reversion = 0.5

        # Retorno esperado por reversión (en %)
        expected_return = -(eps_now - eps_expected) * 100

        return {
            "p_reversion": round(float(np.clip(p_reversion, 0, 1)), 3),
            "p_continue": round(float(np.clip(1 - p_reversion, 0, 1)), 3),
            "expected_return": round(float(expected_return), 3),
            "mu_effective": round(float(mu_eff), 4),
            "half_life_days": round(float(np.log(2) / mu_eff), 1),
        }

    def get_residuals_df(self) -> pd.DataFrame:
        """Residuos como DataFrame indexado por fecha."""
        dates = self.gb.returns.index
        T_len = min(len(dates), len(self.residuals))
        return pd.DataFrame(
            self.residuals[:T_len],
            index=dates[:T_len],
            columns=self.tickers
        )

    def get_z_scores_df(self) -> pd.DataFrame:
        """Z-scores como DataFrame."""
        dates = self.gb.returns.index
        T_len = min(len(dates), len(self.z_scores))
        return pd.DataFrame(
            self.z_scores[:T_len],
            index=dates[:T_len],
            columns=self.tickers
        )

    def get_spectral_df(self) -> pd.DataFrame:
        """Residuos espectrales como DataFrame (columnas = modo k)."""
        dates = self.gb.returns.index
        T_len = min(len(dates), len(self.spectral_res))
        return pd.DataFrame(
            self.spectral_res[:T_len],
            index=dates[:T_len],
            columns=[f"mode_{k}" for k in range(self.N)]
        )

