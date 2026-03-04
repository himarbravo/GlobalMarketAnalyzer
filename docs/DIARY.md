# Development Diary — GlobalMarketAnalyzer

## 2026-03-04

### Session: Project Status Assessment — Lo que funciona y lo que bloquea

**Objetivo**: Evaluar honestamente el estado del proyecto tras P0–P6 para priorizar los próximos pasos.

#### ✅ Lo que el proyecto hace bien

| Fortaleza | Evidencia |
|---|---|
| Modelo físico sólido | R² > 0.97 (OOS), drift parámetros mínimo (CV α = 0.16) |
| Infraestructura completa | 148 tickers, 4 zonas, Supabase, cache local, estado Kalman persistente |
| Validación rigurosa y honesta | 3 crisis walk-forward + 3-fold cross-val + paper trading con scoring |
| Iteración medible P4→P6 | MaxDD crisis: -49% → -14% (s-gate) → -3.5% best case (VIX+Combo) |
| Anticipación demostrada | UKF detectó Volmageddon 132d antes, Fed 2022 103d antes |
| Documentación excepcional | MATHEMATICS.md 703 líneas, analogía agua-paisaje, diagrams Mermaid |

#### 🔴 Problemas que bloquean el avance

**1. Hit Rate ≈ 50% (problema central)**
Los z-scores miden *cuánto* se desvió un activo pero no si el equilibrio sigue existiendo. P7 (Modal Overlap + Entropía Von Neumann) planteó la solución teórica pero **no está implementado**. Sin resolver esto, no hay edge real.

**2. Retornos negativos incluso en bull markets**
P6 Combo: Bull 2019 = -8.7%, AI Rally = -15.1%. El problema no es solo el gate de crisis — la generación de alpha por z-score MR no es suficiente por sí sola.

**3. Ruido dimensional**
102 activos / 300 días → ratio obs/params ≈ 0.06 para la matriz de correlación. Muchos modos del Laplaciano son ruido. Clustering jerárquico (Sec 13.7 de MATHEMATICS.md) propuesto pero **no implementado**.

**4. Bug conocido en W²/W³**
`graph_builder.py` usa `|W|·|W|` en vez de `W·W` para vecinos de 2º/3er orden — pierde signo de anti-correlación. (Nota: DIARY 2026-02-26 dice que se corrigió, pero MATHEMATICS.md línea 79 sigue marcándolo como pendiente — verificar estado real.)

**5. Parámetros fijos sin calibración empírica**
Solo α, γ, s se calibran. β_fx=0.30, η=0.02, β_r_bank=-0.50, β_r_prod=+0.30, β₂=0.15, β₃=0.05 son ad hoc.

**6. Sizing y timing primitivos**
Posiciones iguales cada 20 días, z-score fijo 0.8, sin Kelly sizing, sin uso de P(rev) para dimensionar. Issue #14 lo documenta.

#### 📊 Mapa de progreso

```
Modelo Físico          ██████████ 95%
Infraestructura        █████████░ 90%
Validación             ████████░░ 80%
Regime Gate            ███████░░░ 70%
Reversibilidad (P7)    ██░░░░░░░░ 20% (solo teoría)
Sizing / Execution     █░░░░░░░░░ 10%
Clustering jerárquico  ░░░░░░░░░░  0%
```

**Diagnóstico**: Plataforma de investigación excelente. Para operar con capital real → resolver (1) reversibilidad/hit rate y (2) sizing inteligente.

**Riesgo principal**: Dirección — mucha investigación abierta (P7, clustering, Granger, eigenvalores complejos) pero poca convergencia hacia implementación y prueba. Issues #13 y #14 definen el camino.

---

## 2026-03-02

### Session: P0-P3 Implementation Sprint (6 PRs merged)

**P0 — Graph expansion (PR #3)**
- 80 → 148 tickers across 10 countries (US, FR, DE, UK, JP, KR, BR, IN, MX, SA)
- 4 zones: USD (~50), EUR (~25), ASIA (~22), EM (~20)
- 13 banks (HSBC, BNP.PA, SAN, ING, MUFG, SMFG, ITUB, HDB, etc.)
- Sub-zones for FX coupling: Nordics, Latam, MENA

**P1 — Trading strategy calibration (PR #5)**
- Adaptive Z_ENTRY: VIX-based (lower in calm, higher in stress)
- 3-tier cost model: 5bps US, 15bps intl, 25bps EM
- Hard stop at -10% per position
- Walk-forward composite weight optimization
- **Result**: Sharpe 0.89 (down from 1.16 due to realistic costs), 3/3 trials positive

**P2 — Information edge (PRs #7, #8, #9)**
- P2.2: Credit spread delta — early warning signal (rate of widening > level)
- P2.3: Asymmetric earnings surprise (-0.3 miss / +0.2 beat) + analyst target gap
- P2.1: Central bank rate momentum (60-day FEDFUNDS change)
- All defensive signals — ready for crisis, don't hurt calm performance

**P3 — Extended Bayesian adaptation (PR #12)**
- P3.1: UKF (Unscented Kalman Filter) for s parameter
  - Merwe scaled sigma points, 1D state
  - Heuristic s as prior attractor, prediction errors as measurement
  - Feedback loop: heat_engine → spectral residuals → graph_builder → UKF update
- P3.2: Kalman state persistence in Supabase (JSONB)
  - save/load on each calibration cycle
  - Eliminates 20-day warmup on restart

**Issues closed**: #4 (P1), #6 (P2), #11 (P3)
**Next priority**: P4 — Crisis validation (COVID, 2022 rate hikes, Volmageddon)

---

## 2026-02-26

### Session: Model improvements + Audit + Phase 1 fixes

**Mejoras implementadas:**
- Cross-lag correlation [-15,+15] en graph_builder
- Multi-escala 20d/60d/120d con pesos adaptativos
- Vecinos W²/W³ para chains de contagio
- Perturbation simulator (nuevo módulo)
- Inertia detector reescrito (5 componentes: phase space, masa, momento angular, energía, histéresis)
- Reverse engineer (sismología financiera, 13 tipos de eventos)
- Model diagnostic v2 (non-locality, event backtesting, crisis simuladas)

**Bugs críticos corregidos:**
- `data_ingestion.py`: `xs(level=1)` → `xs(level=0)` — macro data estaba toda NULL
- `graph_builder.py`: W²/W³ usaban `|W|@|W|` (perdía signo) → ahora `W@W`
- `heat_engine.py`: brute-force α → `scipy.optimize.minimize_scalar`
- `signal_generator.py`: F≥0 para BUY demasiado restrictivo → F≥-0.05

**Resultados diagnóstico v2:**
- Non-locality W²/W³: 20/20 = 100% validada
- Event backtesting: avg corr = -0.105 (single-stock exógeno, pero escenarios sistémicos coherentes)
- Phase space: 39 divergentes, 21 convergentes, 40 cíclicos
- s(t) = 0.821 (ahora calibra con VIX real)

**Documentación:**
- MATHEMATICS.md: fundamento matemático completo
- README.md: mapa de arquitectura

**Próximo:** Fase 3 — Macro completo (DXY, copper, divisas, fed_rate)

---

### Phase 2: Indicadores técnicos integrados

**Re-ingested 51,664 records** — todos los 57 indicadores técnicos ahora poblados:
- SMAs (5,10,20,50,100,200), EMAs, RSI (14,7), MACD, Stochastic, Williams%R
- ADX, Bollinger (upper/lower/width/pct), Keltner, ATR (14,7)
- Vol realizeda (5d,10d,20d,60d), OBV, VWAP, Volume ratio, MFI, CMF
- Ichimoku (tenkan/kijun/senkou), Pivot points
- Returns (1d-252d), Sharpe, Sortino, Max Drawdown
- Dist to SMAs, 52w high/low, Gap%

**Integración en graph_builder:**
- Volume weighting: boost = 0.85 + 0.15 × √(vol_i × vol_j) → liquid pairs more reliable
- Adaptive threshold: CORR_THRESHOLD + 0.05 × max(vol_pair - 0.25, 0) → high-vol needs more corr
- Result: 2632+ / 682- edges (was 3014+/2137- without vol weighting)
- Order matters: threshold first, then vol boost (prevents low-vol dampening from killing good corr)

---

### Phase 3: Macro completo + eliminación de hardcoded

**Análisis de debilidades encontradas:**
1. `s(t)` solo usaba VIX+yield_spread — ignoraba DXY, copper, oil (3 indicadores de stress global)
2. `credit_spread` siempre vacío — HYG/TLT ratio fallaba silenciosamente
3. `reverse_engineer.py` tenía ~80 tickers hardcoded en 9 sectores
4. `FX_COUPLING` dict definido (13 líneas) pero nunca usado
5. Los sectores de `assets` table no se usaban en ningún módulo

**Fixes implementados:**
- `_calibrate_s()` ahora usa 6 indicadores: VIX (0.20), DXY (0.15), yield_spread (0.15), credit_spread (0.15), copper (0.10), oil (0.05)
- Sectores cargados dinámicamente desde `assets` table → `gb.sectors` dict
- `reverse_engineer.py` usa `SECTOR_GROUPS` mapping: DB names → classification categories
- Eliminado `FX_COUPLING` (dead code)
- Cargados DXY (505 pts), copper (505 pts), oil_wti (505 pts) nuevos en `graph_builder`

**Resultados:**
- s(t) = 0.826 (antes 0.821 — DXY=97.7 por debajo de stress threshold 98, no añade stress)
- 13 categorías de sector (vs 9 hardcoded): +Intl, +Factors, +Sectors, +Real_Estate
- Event 2025-04-09: 13 sectores afectados (antes solo 9), incluyendo Real_Estate +5.5% e Intl +7.3%

---

### Phase 4: Fundamentales completos (7 componentes)

**Score enriquecido de 4 a 7 componentes:**
| Component | Weight | Columns Used |
|---|---|---|
| FCF yield | 0.20 | free_cash_flow, market_cap |
| ROIC excess | 0.20 | roic (real), roe, roa |
| Growth real | 0.15 | revenue_growth, earnings_growth |
| Quality | 0.15 | debt_to_equity, current_ratio, gross_margin, operating_margin |
| Valuation | 0.15 | forward_pe, ev_ebitda, pb_ratio |
| Analyst | 0.10 | target_mean_price, recommendation, num_analysts, shares_outstanding |
| Momentum Q | 0.05 | beta, institutional_pct |

**Resultados:**
- Top: MA(+0.53), NVDA(+0.42), AMGN(+0.39), AAPL(+0.38), LLY(+0.37)
- 25 value_creators, 1 speculative_mild, 42 speculative (ETFs sin fundamentals)
- NaN guard añadido para tickers con datos incompletos
- Uso total de columnas: 14/52 fundamentals + ~15 macro + vol/volume = **~35 columnas activas** (antes 9)
