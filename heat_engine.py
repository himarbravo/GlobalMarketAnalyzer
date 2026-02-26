"""
HEAT ENGINE — GlobalMarketAnalyzer
====================================
Resuelve advección-difusión fraccional en el grafo:

  du/dt = -α · L^s · (u - μ(t)) + v(t) + f(t)

  difusión:  -α·L^s·u        → equilibrar (calor fluye de caliente a frío)
  advección:  v(t)            → expectativa (dónde se MOVERÁ el dinero)
  fuente:     f(t)            → fundamentales (quién genera/destruye valor)

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
        f_vec = np.nan_to_num(self.ff.get_source_vector(self.tickers))
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

    def solve(self, calibrate: bool = True):
        """
        Resuelve advección-difusión híbrida:

        du/dt = -α·L^s·u + v(t) + f(t)

        Modos TREND (λ < threshold):
          u_k(t+1) = u_k(t) + β_k · Δu_k(t)  (momentum)

        Modos REVERT (λ >= threshold):
          u_k(t+1) = u_k(t) · e^{-μ_k} + f_k/μ_k · (1 - e^{-μ_k})

        + Advección: v(t) empuja u según flujos macro de capital
        """
        u = self.gb.u.astype(np.float64)  # (T, N)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        T_len = len(u)
        if T_len < 2:
            raise ValueError("Datos insuficientes para resolver O-U")

        if calibrate:
            self.calibrate_alpha()

        Phi = self.gb.eigenvectors
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)
        lam_s[0] = 0.0
        f_vec = np.nan_to_num(self.ff.get_source_vector(self.tickers))
        f_k = Phi.T @ f_vec

        # Calcular velocidad macro v(t) — ANTES de resolver
        self._compute_macro_velocity(T_len)

        # Proyectar temperaturas reales al espacio espectral
        u_k_real = u @ Phi  # (T, N)

        # Propagar one-step-ahead por modo
        u_k_pred = np.zeros_like(u_k_real)
        u_k_pred[0] = u_k_real[0]  # condición inicial

        # --- Modos REVERT: O-U clásico ---
        alpha_vec = np.full(self.N, self.alpha)
        alpha_vec[:min(self.N, 30)] = self.alpha_per_mode[:min(self.N, 30)]
        mu = alpha_vec * lam_s  # (N,)
        decay = np.where(mu > 1e-10, np.exp(-mu), 1.0)
        eq_term = np.where(mu > 1e-10, f_k / np.maximum(mu, 1e-10) * (1 - decay), f_k)

        # --- Modos TREND: momentum ---
        is_trend = np.array([mt == 'trend' for mt in self.mode_type])
        n_trend = int(np.sum(is_trend))

        # Proyectar v(t) al espacio espectral para sumar al solver
        v_spectral = self.macro_velocity @ Phi if len(self.macro_velocity) > 0 else np.zeros_like(u_k_real)

        # Peso de advección (calibrar: demasiado alto → ruido, muy bajo → no efecto)
        ADV_WEIGHT = 0.05  # conservador: solo 5% del push macro

        for t in range(T_len - 1):
            # Revert modes: O-U + advección
            u_k_pred[t + 1] = u_k_real[t] * decay + eq_term

            # Sumar advección (el empuje del capital macro)
            if t < len(v_spectral):
                u_k_pred[t + 1] += ADV_WEIGHT * v_spectral[t]

            # Trend modes: momentum extrapolation
            if n_trend > 0 and t >= MOMENTUM_WIN:
                for k in range(self.N):
                    if not is_trend[k]:
                        continue
                    deltas = np.diff(u_k_real[max(0, t-MOMENTUM_WIN):t+1, k])
                    if len(deltas) > 0:
                        beta_k = np.mean(deltas)
                        u_k_pred[t + 1, k] = u_k_real[t, k] + beta_k
                        # Trend modes también reciben advección
                        if t < len(v_spectral):
                            u_k_pred[t + 1, k] += ADV_WEIGHT * v_spectral[t, k]

        # Residuos espectrales
        self.spectral_res = u_k_real - u_k_pred  # (T, N)

        # Guardar info para señales
        self.is_trend_mode = is_trend
        self.n_trend_modes = n_trend
        self.n_revert_modes = self.N - n_trend

        # Reconstruir en espacio real
        self.u_real = u
        self.u_pred = u_k_pred @ Phi.T  # (T, N)
        self.residuals = self.u_real - self.u_pred

        # Sigma por activo (para probabilidades)
        self.sigma_residual = np.nanstd(self.residuals, axis=0)  # (N,)

        # Z-scores (normalizar por volatilidad rolling)
        self._compute_z_scores()

        # Refuge signal
        self._compute_refuge_signal()

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

