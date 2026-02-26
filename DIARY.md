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
