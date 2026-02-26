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

**Próximo:** Fase 2 — calcular indicadores técnicos NULL + integrar vol/RSI/BB al modelo
