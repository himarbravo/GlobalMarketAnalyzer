"""
PERTURBATION SIMULATOR — GlobalMarketAnalyzer
================================================
Simula cómo una perturbación en un activo se propaga por el grafo O-U.

Uso:
    sim = PerturbationSimulator(graph_builder, heat_engine)
    impact = sim.simulate("FCX", delta=-0.05, horizon=20)
    # → muestra impacto en cada activo durante 20 días

    sim.scenario([("FCX", -0.05), ("USO", -0.10)])
    # → perturbaciones múltiples simultáneas
"""

import numpy as np
import pandas as pd


class PerturbationSimulator:
    """
    Propaga perturbaciones a través del grafo financiero usando
    el kernel de Green del proceso O-U fraccional.

    La respuesta del activo j a una perturbación δ₀ en el activo i es:

        response_j(t) = Σ_k  φ_k(i) · φ_k(j) · e^{-α_k·λ_k^s·t} · δ₀

    donde φ_k son los eigenvectores del Laplaciano, λ_k los eigenvalores,
    y α_k la tasa de reversión por modo.
    """

    def __init__(self, graph_builder, heat_engine):
        self.gb = graph_builder
        self.engine = heat_engine
        self.N = graph_builder.N
        self.tickers = graph_builder.tickers

    def simulate(self, ticker: str, delta: float = -0.05,
                 horizon: int = 20) -> pd.DataFrame:
        """
        Simula la propagación de una perturbación δ en un activo.

        Args:
            ticker: ticker del activo perturbado
            delta: magnitud de la perturbación (e.g. -0.05 = caída del 5%)
            horizon: días de simulación

        Returns:
            DataFrame (horizon, N) con el impacto acumulado en cada activo
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} no encontrado en el grafo")

        i = self.tickers.index(ticker)

        # Construir perturbación inicial en espacio real
        delta_0 = np.zeros(self.N)
        delta_0[i] = delta

        # Propagar usando kernel de Green
        return self._propagate(delta_0, horizon, label=f"{ticker} {delta:+.1%}")

    def scenario(self, events: list[tuple], horizon: int = 20) -> pd.DataFrame:
        """
        Simula múltiples perturbaciones simultáneas.

        Args:
            events: lista de (ticker, delta), e.g.
                    [("FCX", -0.05), ("USO", -0.10), ("TLT", +0.03)]
            horizon: días de simulación
        """
        delta_0 = np.zeros(self.N)
        label_parts = []

        for ticker, delta in events:
            if ticker in self.tickers:
                i = self.tickers.index(ticker)
                delta_0[i] = delta
                label_parts.append(f"{ticker}={delta:+.1%}")

        label = " + ".join(label_parts)
        return self._propagate(delta_0, horizon, label=label)

    def _propagate(self, delta_0: np.ndarray, horizon: int,
                   label: str = "") -> pd.DataFrame:
        """
        Propaga un impulso δ₀ a través del grafo O-U.

        response(t) = Φ · diag(e^{-μ_k · t}) · Φ^T · δ₀
        """
        Phi = self.gb.eigenvectors  # (N, N)
        lam_s = np.power(self.gb.eigenvalues, self.gb.s)

        # Tasa por modo
        alpha_vec = np.full(self.N, self.engine.alpha)
        n_per_mode = min(self.N, 20)
        alpha_vec[:n_per_mode] = self.engine.alpha_per_mode[:n_per_mode]
        mu = alpha_vec * lam_s  # (N,)

        # Proyectar impulso a espacio espectral
        delta_k = Phi.T @ delta_0  # (N,)

        # Propagar t = 0, 1, ..., horizon-1
        response = np.zeros((horizon, self.N))
        for t in range(horizon):
            # Respuesta espectral: cada modo decae exponencialmente
            decay = np.exp(-mu * t)
            response_k = delta_k * decay
            # Proyectar de vuelta a espacio real
            response[t] = Phi @ response_k

        # DataFrame
        df = pd.DataFrame(
            response,
            index=[f"t+{t}" for t in range(horizon)],
            columns=self.tickers,
        )
        df.attrs["label"] = label
        return df

    def impact_report(self, ticker: str, delta: float = -0.05,
                      horizon: int = 20, top_n: int = 15) -> str:
        """
        Genera un reporte legible del impacto de una perturbación.
        """
        df = self.simulate(ticker, delta, horizon)

        # Impacto acumulado (integral de la respuesta)
        cumulative = df.sum(axis=0)
        # Impacto máximo (pico)
        peak = df.abs().max(axis=0)
        # Día del pico
        peak_day = df.abs().idxmax(axis=0)

        lines = [
            f"{'='*70}",
            f"  PERTURBATION ANALYSIS: {ticker} {delta:+.1%}",
            f"  Horizonte: {horizon} días | {self.N} activos",
            f"{'='*70}",
            "",
        ]

        # Top impactados (excluyendo el propio activo)
        others = cumulative.drop(ticker) if ticker in cumulative.index else cumulative
        top_negative = others.nsmallest(top_n)
        top_positive = others.nlargest(top_n)

        lines.append("  🔴 Más afectados negativamente:")
        for t, v in top_negative.items():
            if abs(v) > 1e-6:
                lag = self.gb.W_lag[self.tickers.index(ticker),
                                    self.tickers.index(t)]
                lag_str = f"lag={lag:+d}d" if lag != 0 else "instantáneo"
                scale = self.gb.scale_signals.get(t, {}).get("signal", "?")
                lines.append(f"    {t:<8} impacto={v:+.4f}  "
                           f"pico={peak[t]:.4f} en {peak_day[t]}  "
                           f"{lag_str}  [{scale}]")

        lines.append(f"\n  🟢 Beneficiados (anti-correlados):")
        for t, v in top_positive.items():
            if abs(v) > 1e-6:
                lag = self.gb.W_lag[self.tickers.index(ticker),
                                    self.tickers.index(t)]
                lag_str = f"lag={lag:+d}d" if lag != 0 else "instantáneo"
                lines.append(f"    {t:<8} impacto={v:+.4f}  "
                           f"pico={peak[t]:.4f} en {peak_day[t]}  "
                           f"{lag_str}")

        # Impacto por sector
        lines.append(f"\n  📊 Impacto por sector:")
        from model_diagnostic import SECTORS
        for sector, stickers in SECTORS.items():
            sector_impact = [cumulative[t] for t in stickers
                           if t in cumulative.index]
            if sector_impact:
                avg_impact = np.mean(sector_impact)
                if abs(avg_impact) > 1e-5:
                    tag = "🔴" if avg_impact < 0 else "🟢"
                    lines.append(f"    {tag} {sector:<14} {avg_impact:+.4f}")

        # Propagación temporal
        lines.append(f"\n  ⏳ Propagación temporal (media de |impacto| por día):")
        for t in [0, 1, 3, 5, 10, min(19, horizon-1)]:
            if t < horizon:
                avg_abs = np.mean(np.abs(df.iloc[t].values))
                lines.append(f"    t+{t:<3d}: {avg_abs:.4f}")

        lines.append(f"{'='*70}")
        return "\n".join(lines)

    def impact_matrix(self, delta: float = -0.05, horizon: int = 10) -> pd.DataFrame:
        """
        Matriz NxN: impacto acumulado de perturbar cada activo en cada otro.
        Lento (N simulaciones) pero muestra la estructura completa.
        """
        impact = np.zeros((self.N, self.N))
        for i in range(self.N):
            delta_0 = np.zeros(self.N)
            delta_0[i] = delta
            df = self._propagate(delta_0, horizon)
            impact[i, :] = df.sum(axis=0).values

        return pd.DataFrame(
            impact,
            index=self.tickers,
            columns=self.tickers,
        )
