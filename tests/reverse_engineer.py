"""
REVERSE ENGINEER — GlobalMarketAnalyzer
==========================================
Ingeniería inversa de perturbaciones de mercado.

Como un sismógrafo: dado un movimiento anómalo del mercado,
¿dónde fue el "epicentro" y cómo se propagó?

Funcionalidades:
  1. Epicenter detection: ¿qué activo/sector fue la fuente?
  2. Propagation path: ¿cómo se difundió por el grafo?
  3. Event classification: ¿qué tipo de evento fue?
  4. Historical matching: ¿se parece a algún evento previo?

Uso:
    re = ReverseEngineer(graph_builder, heat_engine)
    event = re.analyze_date("2025-04-09")    # día del crash
    print(event.report())

    # ¿A qué se parece este movimiento?
    re.find_similar_events("2025-04-09")
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# Taxonomía de eventos
EVENT_TYPES = {
    "systemic_shock":     "Shock sistémico — afecta a todo el mercado (>40% activos)",
    "flight_to_safety":   "Flight to safety — capital fluye a refugios (bonos, oro)",
    "credit_stress":      "Tensión de crédito — diferenciales de deuda se amplían",
    "rate_shock":         "Shock de tipos — bonos reaccionan a cambio de política monetaria",
    "commodity_shock":    "Shock de commodities — materias primas mueven todo",
    "currency_shock":     "Shock de divisa — DXY mueve exportadores e importadores",
    "earnings_surprise":  "Sorpresa de earnings — una empresa se mueve >10% sola",
    "liquidity_crisis":   "Crisis de liquidez — VIX dispara, correlaciones → 1",
    "contagion":          "Contagio — un sector arrastra a otros con retardo",
    "momentum_unwind":    "Reversión de momentum — los ganadores caen, perdedores suben",
    "tech_rotation":      "Rotación tech↔valor — capital migra entre growth y value",
    "sector_rotation":    "Rotación sectorial genérica — un sector sube, otro baja",
    "idiosyncratic":      "Evento idiosincrático — afecta solo a 1-2 activos",
    "unknown":            "No clasificado",
}


@dataclass
class MarketEvent:
    """Resultado del análisis de ingeniería inversa."""
    date: str
    epicenter: str                          # ticker más probable
    epicenter_return: float                 # retorno del epicenter
    event_type: str                         # clave de EVENT_TYPES
    affected_sectors: dict = field(default_factory=dict)
    propagation_path: list = field(default_factory=list)
    spectral_signature: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: float = 0.0

    def report(self) -> str:
        lines = [
            f"{'='*70}",
            f"  🔍 REVERSE ENGINEERING — {self.date}",
            f"{'='*70}",
            f"",
            f"  Epicentro:    {self.epicenter} ({self.epicenter_return:+.2%})",
            f"  Tipo:         {EVENT_TYPES.get(self.event_type, self.event_type)}",
            f"  Confianza:    {self.confidence:.0%}",
            f"",
        ]

        if self.propagation_path:
            lines.append("  Cadena de propagación:")
            for step in self.propagation_path[:8]:
                lines.append(f"    {step['ticker']:<8} t+{step['delay']:>2d}d  "
                           f"{step['return']:>+.2%}  {step['role']}")

        if self.affected_sectors:
            lines.append(f"\n  Sectores afectados:")
            for sector, impact in sorted(self.affected_sectors.items(),
                                        key=lambda x: abs(x[1]), reverse=True):
                if abs(impact) > 0.002:
                    tag = "🔴" if impact < -0.01 else ("🟢" if impact > 0.01 else "🟡")
                    lines.append(f"    {tag} {sector:<14} {impact:>+.2%}")

        lines.append(f"{'='*70}")
        return "\n".join(lines)


class ReverseEngineer:
    """Sismología financiera: triangula el epicentro de perturbaciones."""

    # Mapeo de sectores DB → categorías de clasificación
    SECTOR_GROUPS = {
        "Tech":       ["TECH_MEGA", "TECH_SEMIS", "TECH_SOFTWARE"],
        "Finanzas":   ["BANKS", "FINTECH"],
        "Salud":      ["HEALTHCARE", "BIOTECH"],
        "Energía":    ["ENERGY", "RENEWABLE"],
        "Industrial": ["INDUSTRIALS", "MATERIALS"],
        "Consumo":    ["CONSUMER_DISC", "CONSUMER_STA"],
        "Bonds":      ["BONDS_GOVT", "BONDS_CORP"],
        "Commodities":["COMMODITIES"],
        "Crypto":     ["CRYPTO"],
        "Intl":       ["INTL_DEV", "INTL_EM", "INTL_STOCKS"],
        "Factors":    ["FACTORS"],
        "Sectors":    ["SECTORS"],
        "Real_Estate":["REAL_ESTATE"],
    }

    def __init__(self, graph_builder, heat_engine):
        self.gb = graph_builder
        self.engine = heat_engine
        self.N = graph_builder.N
        self.tickers = graph_builder.tickers

        # Construir sectores dinámicamente desde graph_builder.sectors
        self.SECTORS = self._build_sectors()

    def _build_sectors(self) -> dict:
        """Construye dict {categoría: [tickers]} desde graph_builder.sectors."""
        sectors = {}
        gb_sectors = getattr(self.gb, 'sectors', {})

        if gb_sectors:
            # Usar sectores reales de la DB
            for category, db_names in self.SECTOR_GROUPS.items():
                tickers = []
                for db_name in db_names:
                    tickers.extend(gb_sectors.get(db_name, []))
                # Solo incluir tickers que estén en el universo actual
                tickers = [t for t in tickers if t in self.tickers]
                if tickers:
                    sectors[category] = tickers
        else:
            # Fallback: hardcoded mínimo para funcionar sin DB
            sectors = {
                "Tech":       [t for t in ["AAPL","MSFT","GOOGL","NVDA","META","AMD"] if t in self.tickers],
                "Finanzas":   [t for t in ["JPM","BAC","GS","V"] if t in self.tickers],
                "Bonds":      [t for t in ["TLT","IEF","SHY","HYG"] if t in self.tickers],
                "Commodities":[t for t in ["GLD","SLV","USO","DBA"] if t in self.tickers],
            }

        return sectors

    def analyze_date(self, date_str: str) -> MarketEvent:
        """
        Analiza un día específico: ¿qué pasó y por qué?

        1. Identifica el epicentro (activo con mayor z-score anómalo)
        2. Traza la cadena de propagación usando lags del grafo
        3. Clasifica el tipo de evento por la firma espectral
        """
        returns = self.gb.returns.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).astype(np.float64)

        target = pd.Timestamp(date_str)
        if target not in returns.index:
            # Buscar fecha más cercana
            diffs = (returns.index - target).total_seconds().abs()
            target = returns.index[np.argmin(diffs)]

        d_idx = returns.index.get_loc(target)

        # ── 1. Encontrar epicentro ──
        day_returns = returns.iloc[d_idx].values
        vol = returns.iloc[max(0, d_idx-60):d_idx].std().values
        vol = np.maximum(vol, 1e-6)
        z_day = day_returns / vol

        # Epicentro = activo con mayor |z| ese día
        epicenter_idx = int(np.argmax(np.abs(z_day)))
        epicenter = self.tickers[epicenter_idx]
        epicenter_ret = float(day_returns[epicenter_idx])

        # ── 2. Cadena de propagación ──
        propagation = self._trace_propagation(returns, d_idx, epicenter_idx)

        # ── 3. Firma espectral ──
        Phi = self.gb.eigenvectors
        spectral_sig = Phi.T @ day_returns  # proyectar retornos al espacio espectral

        # ── 4. Clasificar tipo de evento ──
        event_type, confidence = self._classify_event(
            day_returns, z_day, spectral_sig, epicenter_idx
        )

        # ── 5. Impacto por sector ──
        sector_impact = self._compute_sector_impact(day_returns, d_idx, returns)

        return MarketEvent(
            date=str(target.date()),
            epicenter=epicenter,
            epicenter_return=epicenter_ret,
            event_type=event_type,
            affected_sectors=sector_impact,
            propagation_path=propagation,
            spectral_signature=spectral_sig,
            confidence=confidence,
        )

    def _trace_propagation(self, returns: pd.DataFrame, d_idx: int,
                           epicenter_idx: int) -> list:
        """
        Traza cómo se propagó el shock desde el epicentro.
        Usa los lags del grafo para ordenar temporalmente.
        """
        path = []
        W_lag = self.gb.W_lag
        W = self.gb.W

        # Reconstruir la cadena: epicentro → vecinos directos → 2º orden
        # Ordenado por lag (quién reacciona primero)
        for j in range(self.N):
            if j == epicenter_idx:
                continue

            lag = int(W_lag[epicenter_idx, j])
            weight = float(W[epicenter_idx, j])

            if abs(weight) < 0.01:
                continue

            # Retorno real del activo j en t+lag (si existe)
            t_response = d_idx + lag
            if 0 <= t_response < len(returns):
                real_ret = float(returns.iloc[t_response, j])
            else:
                real_ret = 0.0

            role = "co-move" if weight > 0 else "anti-corr"
            path.append({
                "ticker": self.tickers[j],
                "delay": lag,
                "return": real_ret,
                "weight": weight,
                "role": role,
            })

        # Ordenar por delay (propagación temporal)
        path.sort(key=lambda x: (x["delay"], -abs(x["weight"])))
        return path

    def _classify_event(self, day_returns, z_day, spectral_sig,
                        epicenter_idx) -> tuple:
        """
        Clasifica el evento por su firma espectral y patrones de mercado.
        Orden de reglas: de más específico a más genérico.
        """
        n_affected = int(np.sum(np.abs(z_day) > 1.5))
        n_extreme = int(np.sum(np.abs(z_day) > 3.0))

        def group_ret(tickers):
            indices = [self.tickers.index(t) for t in tickers if t in self.tickers]
            return np.mean(day_returns[indices]) if indices else 0.0

        r_tech    = group_ret(self.SECTORS["Tech"])
        r_bonds   = group_ret(self.SECTORS["Bonds"])
        r_commod  = group_ret(self.SECTORS["Commodities"])
        r_energy  = group_ret(self.SECTORS["Energía"])
        r_finance = group_ret(self.SECTORS["Finanzas"])
        r_consumo = group_ret(self.SECTORS["Consumo"])
        r_salud   = group_ret(self.SECTORS["Salud"])
        r_indust  = group_ret(self.SECTORS["Industrial"])

        # Entropía espectral
        energy = spectral_sig ** 2
        energy_norm = energy / (np.sum(energy) + 1e-10)
        entropy = -np.sum(energy_norm * np.log(energy_norm + 1e-10))
        rel_entropy = entropy / np.log(self.N)

        # ── Regla 1: Idiosincrático (solo 1-2 activos extremos) ──
        if n_extreme <= 2 and n_affected <= 5:
            epicenter_z = abs(z_day[epicenter_idx])
            if epicenter_z > 8:  # Earnings surprise: movimiento brutal en 1 activo
                return "earnings_surprise", 0.80
            return "idiosyncratic", 0.75

        # ── Regla 2: Sistémico (>40% activos, entropía alta) ──
        if n_affected > self.N * 0.4 and rel_entropy > 0.5:
            # ¿Es crisis de liquidez? (VIX implícito alto)
            r_spy = group_ret(["SPY"])
            if abs(r_spy) > 0.03:  # SPY mueve >3%
                return "liquidity_crisis", 0.80
            return "systemic_shock", 0.80

        # ── Regla 3: Flight to safety ──
        if r_bonds > 0.003 and r_commod > 0.003 and r_tech < -0.003:
            return "flight_to_safety", 0.75

        # ── Regla 4: Credit stress ──
        r_hyg = group_ret(["HYG"])
        r_tlt = group_ret(["TLT"])
        if r_hyg < -0.005 and r_tlt > 0.003:
            return "credit_stress", 0.70

        # ── Regla 5: Rate shock (bonos mueven mucho) ──
        if abs(r_bonds) > 0.01 and abs(r_finance) > 0.005:
            return "rate_shock", 0.70

        # ── Regla 6: Commodity shock ──
        if abs(r_commod) > 0.015 or abs(r_energy) > 0.02:
            return "commodity_shock", 0.65

        # ── Regla 7: Currency shock ──
        # Exportadores EUR (ASML,SAP,NVO) vs USD nativos se mueven opuesto
        r_eur = group_ret(["ASML", "SAP", "NVO", "EWG", "EWU"])
        r_usd = group_ret(["AAPL", "MSFT", "JPM", "WMT"])
        if abs(r_eur - r_usd) > 0.015:
            return "currency_shock", 0.60

        # ── Regla 8: Momentum unwind ──
        # Los ganadores recientes caen, los perdedores suben
        r_momentum = group_ret(["MTUM"])
        r_value = group_ret(["VTV"])
        if r_momentum < -0.01 and r_value > 0.005:
            return "momentum_unwind", 0.60

        # ── Regla 9: Tech rotation (tech vs value) ──
        if abs(r_tech - r_finance) > 0.015 or abs(r_tech - r_consumo) > 0.015:
            if r_tech > 0 and (r_finance < 0 or r_consumo < 0 or r_salud < 0):
                return "tech_rotation", 0.55
            elif r_tech < 0 and (r_finance > 0 or r_consumo > 0 or r_salud > 0):
                return "tech_rotation", 0.55

        # ── Regla 10: Contagio (pocos afectados pero vecinos del epicentro) ──
        if 3 < n_extreme <= 10:
            epicenter_neighbors = np.sum(np.abs(self.gb.W[epicenter_idx, :]) > 0.05)
            extreme_as_neighbors = 0
            extreme_indices = np.where(np.abs(z_day) > 3.0)[0]
            for j in extreme_indices:
                if abs(self.gb.W[epicenter_idx, j]) > 0.03:
                    extreme_as_neighbors += 1
            if extreme_as_neighbors > n_extreme * 0.3:
                return "contagion", 0.60

        # ── Regla 11: Rotación sectorial genérica ──
        sector_rets = {s: group_ret(t) for s, t in self.SECTORS.items()}
        max_sect = max(sector_rets, key=lambda s: sector_rets[s])
        min_sect = min(sector_rets, key=lambda s: sector_rets[s])
        if sector_rets[max_sect] > 0.005 and sector_rets[min_sect] < -0.005:
            return "sector_rotation", 0.50

        return "unknown", 0.3

    def _compute_sector_impact(self, day_returns, d_idx, returns) -> dict:
        """Impacto medio por sector (retorno acumulado 5d)."""
        impact = {}
        for sector, tickers_list in self.SECTORS.items():
            indices = [self.tickers.index(t) for t in tickers_list
                      if t in self.tickers]
            if not indices:
                continue
            # Retorno acumulado 5 días desde el evento
            end = min(d_idx + 6, len(returns))
            cum_ret = returns.iloc[d_idx:end, indices].sum().mean()
            impact[sector] = float(cum_ret)
        return impact

    def find_similar_events(self, date_str: str, top_n: int = 5) -> list:
        """
        Encuentra días con firma espectral similar al evento dado.
        Útil para: "esto se parece a lo que pasó el día X".
        """
        target_event = self.analyze_date(date_str)
        target_sig = target_event.spectral_signature

        returns = self.gb.returns.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).astype(np.float64)

        Phi = self.gb.eigenvectors
        similarities = []

        for d_idx in range(30, len(returns) - 5):
            d = returns.index[d_idx]
            if str(d.date()) == date_str:
                continue

            day_ret = returns.iloc[d_idx].values
            sig = Phi.T @ day_ret
            # Cosine similarity
            cos_sim = np.dot(target_sig, sig) / (
                np.linalg.norm(target_sig) * np.linalg.norm(sig) + 1e-10
            )
            similarities.append((str(d.date()), cos_sim, d_idx))

        # Top N
        similarities.sort(key=lambda x: abs(x[1]), reverse=True)
        results = []
        for date_, sim, d_idx in similarities[:top_n]:
            event = self.analyze_date(date_)
            results.append({
                "date": date_,
                "similarity": round(sim, 3),
                "epicenter": event.epicenter,
                "type": event.event_type,
                "same_sign": sim > 0,  # True = mismo patrón, False = opuesto
            })
        return results

    def scan_all_events(self, sigma_threshold: float = 2.5) -> list:
        """
        Escanea toda la historia y clasifica todos los eventos significativos.
        """
        returns = self.gb.returns.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).astype(np.float64)

        vol = returns.std().values
        events = []

        for d_idx in range(30, len(returns) - 5):
            day_ret = returns.iloc[d_idx].values
            z = day_ret / np.maximum(vol, 1e-6)
            max_z = np.max(np.abs(z))

            if max_z > sigma_threshold:
                event = self.analyze_date(str(returns.index[d_idx].date()))
                events.append(event)

        return events

    def event_history_report(self, sigma_threshold: float = 3.0) -> str:
        """Genera un reporte de todos los eventos detectados."""
        events = self.scan_all_events(sigma_threshold)

        # Conteo por tipo
        type_counts = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        lines = [
            f"{'='*70}",
            f"  📋 EVENT HISTORY — {len(events)} eventos detectados (|z| > {sigma_threshold}σ)",
            f"{'='*70}",
            "",
            "  Distribución por tipo:",
        ]
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {c:>3}× {EVENT_TYPES.get(t, t)}")

        lines.append(f"\n  {'Fecha':>12} {'Epicentro':>10} {'Retorno':>8} "
                    f"{'Tipo':>20} {'Conf':>5}")
        lines.append(f"  {'-'*60}")

        for e in events[:30]:
            lines.append(f"  {e.date:>12} {e.epicenter:>10} "
                        f"{e.epicenter_return:>+8.2%} "
                        f"{e.event_type:>20} {e.confidence:>5.0%}")

        lines.append(f"{'='*70}")
        return "\n".join(lines)
