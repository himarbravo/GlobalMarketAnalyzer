# Development Diary — GlobalMarketAnalyzer

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
