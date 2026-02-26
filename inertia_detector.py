"""
INERTIA DETECTOR — GlobalMarketAnalyzer
=========================================
Marco de inercia riguroso con 5 componentes:

  1. Espacio de fases (u, du/dt): trayectorias espiral → convergencia/divergencia
  2. Masa efectiva: cuánto cuesta mover cada activo
  3. Momento angular espectral: rotación sectorial
  4. Flujo de energía entre modos: qué "temas" ganan/pierden capital
  5. Histéresis: asimetría entre subir y bajar

Señales de salida:
  - asset_scores: score compuesto por activo
  - phase_state: estado dinámico (convergent/divergent/cyclic/stable)
  - rotation_info: dirección y velocidad de rotación sectorial
"""

import numpy as np
import pandas as pd


# ── Parámetros ──
PHASE_WINDOW    = 20     # días para calcular velocidad
Z_ANOMALY       = 2.0    # umbral z-score para anomalía
ANOMALY_DAYS    = 5      # días consecutivos
ENERGY_WINDOW   = 10     # días para calcular flujo de energía


class InertiaDetector:
    """
    Detecta inercias de mercado con 5 métodos complementarios.

    Uso:
        detector = InertiaDetector(heat_engine)
        detector.analyze()
        detector.get_phase_report()
    """

    def __init__(self, heat_engine):
        self.engine = heat_engine
        self.N = heat_engine.N
        self.tickers = heat_engine.tickers

        # Resultados
        self.asset_scores: dict[str, float] = {}
        self.phase_states: dict[str, dict] = {}     # espacio de fases
        self.effective_mass: np.ndarray = np.array([])  # (N,)
        self.angular_momentum: np.ndarray = np.array([])  # (N_modes,)
        self.energy_flow: np.ndarray = np.array([])  # (N_modes,)
        self.hysteresis: np.ndarray = np.array([])   # (N,)

        # Alertas
        self.mode_alerts: list[dict] = []
        self.asset_alerts: list[dict] = []

    def analyze(self):
        """Pipeline completo de detección de inercias."""
        self._compute_phase_space()
        self._compute_effective_mass()
        self._compute_angular_momentum()
        self._compute_energy_flow()
        self._compute_hysteresis()
        self._detect_mode_alerts()
        self._compute_asset_scores()

    # ═══════════════════════════════════════════════════════════
    # 1. ESPACIO DE FASES (u, du/dt)
    # ═══════════════════════════════════════════════════════════

    def _compute_phase_space(self):
        """
        Para cada activo, calcula la trayectoria en (posición, velocidad).
        Clasifica el estado dinámico según la forma de la trayectoria.
        """
        u = self.engine.u_real  # (T, N) temperaturas reales
        T_len = u.shape[0]

        if T_len < PHASE_WINDOW + 5:
            self.phase_states = {t: {"state": "insufficient_data"}
                                for t in self.tickers}
            return

        self.phase_states = {}

        for i, ticker in enumerate(self.tickers):
            u_i = u[:, i].astype(np.float64)

            # Velocidad = derivada discreta suavizada (media móvil 3d)
            v_raw = np.diff(u_i)
            v = pd.Series(v_raw).rolling(3, min_periods=1).mean().values

            # Últimos PHASE_WINDOW días
            u_window = u_i[-PHASE_WINDOW:]
            v_window = v[-PHASE_WINDOW:]

            # Radio de la espiral: r(t) = √(u² + v²)
            r = np.sqrt(u_window**2 + v_window**2)
            r = np.maximum(r, 1e-10)  # evitar div/0

            # Derivada del radio (dr/dt > 0 → diverge, < 0 → converge)
            dr = np.diff(r)
            dr_dt_mean = np.mean(dr[-5:]) if len(dr) > 5 else 0.0

            # Curvatura de la trayectoria (para detectar ciclos)
            # κ = |u'·v'' - v'·u''| / (u'² + v'²)^{3/2}
            u_dot = np.diff(u_window)
            v_dot = np.diff(v_window)
            if len(u_dot) > 2 and len(v_dot) > 2:
                u_ddot = np.diff(u_dot)
                v_ddot = np.diff(v_dot)
                min_len = min(len(u_dot) - 1, len(v_dot) - 1)
                numer = np.abs(u_dot[:min_len] * v_ddot[:min_len]
                             - v_dot[:min_len] * u_ddot[:min_len])
                denom = (u_dot[:min_len]**2 + v_dot[:min_len]**2)**1.5
                kappa = numer / np.maximum(denom, 1e-12)
                mean_kappa = np.nanmean(kappa)
            else:
                mean_kappa = 0.0

            # Clasificar estado
            r_last = r[-1]
            if dr_dt_mean > 0.002 and r_last > 0.05:
                state = "divergent"   # burbuja o tendencia acelerante
            elif dr_dt_mean < -0.002:
                state = "convergent"  # reverting al equilibrio
            elif mean_kappa > 0.5 and r_last > 0.02:
                state = "cyclic"      # ciclo límite / oscilación
            else:
                state = "stable"      # cerca del equilibrio

            self.phase_states[ticker] = {
                "state": state,
                "radius": round(float(r_last), 4),
                "dr_dt": round(float(dr_dt_mean), 5),
                "position": round(float(u_window[-1]), 4),
                "velocity": round(float(v_window[-1]), 5),
                "curvature": round(float(mean_kappa), 4),
            }

    # ═══════════════════════════════════════════════════════════
    # 2. MASA EFECTIVA
    # ═══════════════════════════════════════════════════════════

    def _compute_effective_mass(self):
        """
        Masa = cuánto cuesta mover el activo.
        M_eff[i] = (1/σ_vol[i]) · D[i] · (1 + centrality[i])

        Alta masa → señal más significativa (AAPL z=2 > ETH z=2)
        """
        # Volatilidad inversa
        returns = self.engine.gb.returns.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).astype(np.float64)
        vol = returns.std().values
        inv_vol = 1.0 / np.maximum(vol, 1e-6)

        # Grado (conectividad en el grafo)
        degree = np.abs(self.engine.gb.W).sum(axis=1)

        # Centralidad eigenvector (importancia sistémica)
        # = componente del primer eigenvector no trivial (Fiedler)
        if len(self.engine.gb.eigenvectors) > 0:
            fiedler = np.abs(self.engine.gb.eigenvectors[:, 1])
        else:
            fiedler = np.ones(self.N)

        # Masa efectiva (normalizada 0-1)
        mass_raw = inv_vol * degree * (1 + fiedler)
        mass_max = np.max(mass_raw) if np.max(mass_raw) > 0 else 1.0
        self.effective_mass = mass_raw / mass_max

    # ═══════════════════════════════════════════════════════════
    # 3. MOMENTO ANGULAR ESPECTRAL (rotación sectorial)
    # ═══════════════════════════════════════════════════════════

    def _compute_angular_momentum(self):
        """
        L_k(t) = u_k(t) · v_k(t)  (producto en plano u-v del modo k)

        L_k > 0 → capital rotando en dirección positiva
        L_k < 0 → rotando en dirección opuesta
        |dL_k/dt| → torque (aceleración de la rotación)
        """
        spectral = self.engine.spectral_res  # (T, N)

        if len(spectral) < 5:
            self.angular_momentum = np.zeros(self.N)
            return

        # Posición espectral (últimos datos)
        u_k = spectral[-1, :]  # posición actual

        # Velocidad espectral
        v_k = spectral[-1, :] - spectral[-2, :]  # derivada discreta

        # Momento angular L = u × v (en 2D es el "pseudo-vector")
        self.angular_momentum = u_k * v_k

    # ═══════════════════════════════════════════════════════════
    # 4. FLUJO DE ENERGÍA ENTRE MODOS
    # ═══════════════════════════════════════════════════════════

    def _compute_energy_flow(self):
        """
        E_k = ½ · (u_k² + v_k²/ω_k²)

        dE_k/dt > 0 → el modo k gana capital
        dE_k/dt < 0 → el modo k pierde capital
        """
        spectral = self.engine.spectral_res
        T_len = len(spectral)

        if T_len < ENERGY_WINDOW + 2:
            self.energy_flow = np.zeros(self.N)
            return

        lam_s = np.power(self.engine.gb.eigenvalues, self.engine.gb.s)
        lam_s = np.maximum(lam_s, 1e-6)

        # Energía por modo en dos instantes
        t1 = T_len - ENERGY_WINDOW
        t2 = T_len - 1

        u_k1 = spectral[t1, :]
        u_k2 = spectral[t2, :]
        v_k1 = spectral[t1, :] - spectral[max(t1 - 1, 0), :]
        v_k2 = spectral[t2, :] - spectral[t2 - 1, :]

        E1 = 0.5 * (u_k1**2 + v_k1**2 / lam_s)
        E2 = 0.5 * (u_k2**2 + v_k2**2 / lam_s)

        # Flujo = cambio de energía / ventana
        self.energy_flow = (E2 - E1) / ENERGY_WINDOW

    # ═══════════════════════════════════════════════════════════
    # 5. HISTÉRESIS (asimetría up/down)
    # ═══════════════════════════════════════════════════════════

    def _compute_hysteresis(self):
        """
        H[i] = ε_up / ε_down

        H > 1 → más fácil subir que bajar → tendencia alcista tiene inercia
        H < 1 → más fácil bajar que subir → tendencia bajista tiene inercia
        H ≈ 1 → simétrico
        """
        residuals = self.engine.residuals  # (T, N)
        returns = self.engine.gb.returns.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).values.astype(np.float64)

        T_len = min(len(residuals), len(returns))
        self.hysteresis = np.ones(self.N)

        for i in range(self.N):
            r_i = returns[:T_len, i]
            eps_i = residuals[:T_len, i]

            up_mask = r_i > 0
            down_mask = r_i < 0

            eps_up = np.mean(np.abs(eps_i[up_mask])) if up_mask.sum() > 10 else 1e-6
            eps_down = np.mean(np.abs(eps_i[down_mask])) if down_mask.sum() > 10 else 1e-6

            self.hysteresis[i] = eps_up / max(eps_down, 1e-10)

    # ═══════════════════════════════════════════════════════════
    # MODE ALERTS (compatible con versión anterior)
    # ═══════════════════════════════════════════════════════════

    def _detect_mode_alerts(self):
        """Detecta anomalías por modo espectral."""
        self.mode_alerts = []
        spectral = self.engine.spectral_res

        for k in range(min(self.N, 20)):
            alert = {
                "mode": k,
                "eigenvalue": float(self.engine.gb.eigenvalues[k]),
                "reversal": False,
                "inflection": False,
                "persistent_anomaly": False,
                "energy_flow": float(self.energy_flow[k]),
                "angular_momentum": float(self.angular_momentum[k]),
            }

            # Anomalía persistente
            if len(spectral) >= ANOMALY_DAYS:
                eps_k = np.nan_to_num(
                    spectral[-ANOMALY_DAYS:, k].astype(np.float64)
                )
                sigma_k = np.nanstd(spectral[:, k].astype(np.float64))
                if sigma_k > 1e-10:
                    z_k = np.abs(eps_k) / sigma_k
                    if np.all(z_k > Z_ANOMALY):
                        alert["persistent_anomaly"] = True

            # Reversión: energía decreciendo + momento angular cambiando signo
            if self.energy_flow[k] < -0.001 and abs(self.angular_momentum[k]) > 0.001:
                alert["reversal"] = True

            # Inflexión: flujo cambiando de signo
            if len(spectral) > ENERGY_WINDOW + 5:
                t_mid = len(spectral) - ENERGY_WINDOW
                flow_before = np.mean(spectral[t_mid-5:t_mid, k]**2)
                flow_after = np.mean(spectral[-5:, k]**2)
                if np.sign(flow_after - flow_before) != np.sign(self.energy_flow[k]):
                    alert["inflection"] = True

            self.mode_alerts.append(alert)

    # ═══════════════════════════════════════════════════════════
    # ASSET SCORES (combinando los 5 componentes)
    # ═══════════════════════════════════════════════════════════

    def _compute_asset_scores(self):
        """
        Score compuesto por activo:
          S[i] = w₁·z_phase + w₂·mass_adj + w₃·rotation + w₄·energy + w₅·hysteresis
        """
        Phi = self.engine.gb.eigenvectors
        lam_s = np.power(self.engine.gb.eigenvalues, self.engine.gb.s)
        z_real = self.engine.z_scores

        self.asset_scores = {}
        self.asset_alerts = []

        for i, ticker in enumerate(self.tickers):
            # ── Componente 1: Espacio de fases ──
            phase = self.phase_states.get(ticker, {})
            phase_score = 0.0
            state = phase.get("state", "stable")

            if state == "divergent":
                # Divergente → señal en dirección del momentum
                phase_score = phase.get("velocity", 0) * 10
            elif state == "convergent":
                # Convergente → señal opuesta al desplazamiento
                phase_score = -phase.get("position", 0) * 5
            elif state == "cyclic":
                # Cíclico → señal depende de la posición en el ciclo
                phase_score = -phase.get("velocity", 0) * 3

            # ── Componente 2: Masa como ponderador ──
            mass = self.effective_mass[i] if len(self.effective_mass) > i else 0.5
            # La masa amplifica la señal (activos pesados → señales más fiables)
            mass_factor = 0.5 + mass  # rango [0.5, 1.5]

            # ── Componente 3: Rotación espectral ──
            rotation_score = 0.0
            for k in range(min(self.N, 20)):
                if self.engine.gb.eigenvalues[k] < 1e-10:
                    continue
                w_k = 1.0 / max(lam_s[k], 1e-6)
                participation = Phi[i, k] ** 2
                rotation_score += w_k * participation * self.angular_momentum[k]

            # Normalizar rotación
            rot_norm = sum(
                (1.0 / max(lam_s[k], 1e-6)) * Phi[i, k]**2
                for k in range(min(self.N, 20))
                if self.engine.gb.eigenvalues[k] > 1e-10
            )
            if rot_norm > 0:
                rotation_score /= rot_norm

            # ── Componente 4: Flujo de energía ──
            energy_score = 0.0
            for k in range(min(self.N, 20)):
                if self.engine.gb.eigenvalues[k] < 1e-10:
                    continue
                participation = Phi[i, k] ** 2
                energy_score += participation * self.energy_flow[k]

            # ── Componente 5: Histéresis ──
            h = self.hysteresis[i] if len(self.hysteresis) > i else 1.0
            hyst_bias = np.log(max(h, 0.01))  # >0 si alcista, <0 si bajista

            # ── Score compuesto ──
            z_i = float(z_real[-1, i]) if len(z_real) > 0 else 0.0

            score = mass_factor * (
                0.30 * phase_score +                    # espacio de fases
                0.25 * z_i +                            # z-score real
                0.20 * rotation_score * 100 +           # rotación espectral
                0.15 * energy_score * 1000 +            # flujo de energía
                0.10 * hyst_bias                        # histéresis
            )

            self.asset_scores[ticker] = float(score)

            if abs(score) > 0.5:
                self.asset_alerts.append({
                    "ticker": ticker,
                    "score": round(float(score), 3),
                    "direction": "cold" if score < 0 else "hot",
                    "state": state,
                    "mass": round(float(mass), 3),
                    "hysteresis": round(float(h), 3),
                })

    # ═══════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════

    def get_alerts(self) -> list[dict]:
        """Alertas de activos ordenadas por |score|."""
        return sorted(self.asset_alerts,
                      key=lambda x: abs(x["score"]), reverse=True)

    def get_mode_summary(self) -> pd.DataFrame:
        """Resumen de los primeros 10 modos con energía y momento."""
        rows = []
        for alert in self.mode_alerts[:10]:
            rows.append({
                "mode":             alert["mode"],
                "eigenvalue":       round(alert["eigenvalue"], 4),
                "energy_flow":      round(alert["energy_flow"], 6),
                "angular_momentum": round(alert["angular_momentum"], 6),
                "reversal":         alert["reversal"],
                "inflection":       alert["inflection"],
                "anomaly":          alert["persistent_anomaly"],
            })
        return pd.DataFrame(rows)

    def get_phase_report(self) -> str:
        """Reporte legible del estado de fases de todos los activos."""
        lines = [
            f"{'='*70}",
            f"  PHASE SPACE REPORT",
            f"{'='*70}",
            "",
            f"  {'Ticker':<8} {'Estado':<12} {'Radio':>8} {'dr/dt':>8} "
            f"{'Pos':>8} {'Vel':>8} {'Mass':>6} {'H':>6}",
            f"  {'-'*68}",
        ]

        # Sort by radius (most active first)
        sorted_tickers = sorted(
            self.phase_states.items(),
            key=lambda x: x[1].get("radius", 0),
            reverse=True,
        )

        for ticker, phase in sorted_tickers[:20]:
            i = self.tickers.index(ticker) if ticker in self.tickers else -1
            mass = self.effective_mass[i] if i >= 0 and i < len(self.effective_mass) else 0
            h = self.hysteresis[i] if i >= 0 and i < len(self.hysteresis) else 1.0

            state_emoji = {
                "divergent": "🔺",
                "convergent": "🔻",
                "cyclic": "🔄",
                "stable": "⚖️",
            }.get(phase.get("state", ""), "❓")

            lines.append(
                f"  {ticker:<8} {state_emoji} {phase.get('state', '?'):<10} "
                f"{phase.get('radius', 0):>8.4f} {phase.get('dr_dt', 0):>8.5f} "
                f"{phase.get('position', 0):>8.4f} {phase.get('velocity', 0):>8.5f} "
                f"{mass:>6.3f} {h:>6.2f}"
            )

        # Summary
        states = [p.get("state", "?") for p in self.phase_states.values()]
        n_div = states.count("divergent")
        n_conv = states.count("convergent")
        n_cyc = states.count("cyclic")
        n_stab = states.count("stable")

        lines.append(f"\n  Resumen: {n_div} divergentes, {n_conv} convergentes, "
                    f"{n_cyc} cíclicos, {n_stab} estables")
        lines.append(f"{'='*70}")
        return "\n".join(lines)
