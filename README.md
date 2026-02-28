# GlobalMarketAnalyzer

Water-landscape model on a fractional graph with inertia, hierarchical monetary transmission, and dimensional fields.

## The Model

Three fundamental quantities, two equations, everything derived:

```
MONEY (m) ── the water that flows between investors on the graph
  γ·d²m/dt² + dm/dt = -α·L^s·m + v(t) + f(t) + Ω(t)

CAPITAL (K) ── the terrain, fixed at each company, changes slowly
  dK/dt = g(t)  (CAPEX, R&D, depreciation, earnings, debt)

PRICE (u) = λ·K = m ── the observable
  λ = m/K ── valuation multiple (derived)
```

### Inertia (γ)
The equation is **2nd order**: money has mass. γ=1 → classic O-U (no momentum), γ>1 → trends persist. γ is auto-calibrated via out-of-sample grid search.

### Hierarchical Graph

```
╔══════════════════════════════════════════════════════╗
║  DIMENSIONS (fields, modulate all nodes of a country) ║
║                                                       ║
║  Dim 1: Currency c(t)     DXY, EURUSD, USDJPY        ║
║  Dim 2: Sovereign Debt    Debt/GDP ratio per country  ║
║  Dim 3: Fed Rate r(t)     Central bank interest rate  ║
║                                                       ║
║  → Ω(t) = dc/dt·m - D/GDP·η·m - βr·dr/dt·m          ║
╠══════════════════════════════════════════════════════╣
║  DIRECTED GRAPH (2 levels, money creation channel)    ║
║                                                       ║
║  Banks →(lending)→ Companies                          ║
║  Companies →(interest)→ Banks                         ║
║                                                       ║
║  Intra-level: ←→ correlation (undirected, as before)  ║
╚══════════════════════════════════════════════════════╝
```

### Source Term f(t) — per role

| Role | f(t) | Meaning |
|---|---|---|
| `bank` | NIM × lending volume | Banks create money via loans |
| `productive` | (dK/dt + credit - debt·r) × S(t) | Capital creation + loans - interest, × sentiment |

**Signal**: δ = λ - λ_eq(regime). Overvalued if δ > 0, undervalued if δ < 0.

## Architecture

```
┌──────────────────────── PIPELINE ──────────────────────────────┐
│                                                                 │
│  config.py              Constants, node roles, dimensions       │
│  db/database_manager.py Supabase connection and queries         │
│  db/data_ingestion.py   Price, macro, fundamental data loading  │
│  db/fred_client.py      FRED API (CPI, yields, credit spread)  │
│  db/schema.sql          Supabase table definitions              │
│                                                                 │
│  ┌──── CORE MODEL ─────────────────────────────────────────┐   │
│  │  graph_builder.py       Multi-layer graph + directed     │   │
│  │                         edges (banks↔companies)          │   │
│  │                         + dimension loading              │   │
│  │  fundamental_filter.py  Fundamental scores → S(t)        │   │
│  │  capital_field.py       K(t) terrain from earnings       │   │
│  │  heat_engine.py         2nd-order money equation solver  │   │
│  │                         with inertia (γ) + dimensions    │   │
│  │  inertia_detector.py    Phase space, mass, rotation      │   │
│  │  perturbation_simulator.py  Shock propagation            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  core/signal_generator.py   data → graph → solve → signals     │
│  core/regime_calibrator.py  Historical calibration              │
│                                                                 │
│  ┌──── DIAGNOSTICS ────────────────────────────────────────┐   │
│  │  tests/model_diagnostic.py    Tests 1-6                  │   │
│  │  tests/model_diagnostic_v2.py Tests 7-11                 │   │
│  │  tests/historical_tests.py    Walk-forward backtests     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Execution

```bash
# Daily signals
python core/signal_generator.py

# Basic diagnostics
python tests/model_diagnostic.py

# Historical calibration by market regime
python core/regime_calibrator.py

# Full walk-forward tests
python tests/historical_tests.py
```

## Documentation

- [MATHEMATICS.md](docs/MATHEMATICS.md) — Complete mathematical formulation
- [DIARY.md](docs/DIARY.md) — Development log

## Roadmap

### ✅ Implemented
- **Inertia (γ)**: 2nd-order equation, auto-calibrated
- **Hierarchical graph**: 2 roles (bank/productive), directed edges
- **Dynamic f(t)**: role-dependent source term with S(t) sentiment
- **Multi-currency fields**: 4 zones (USD, EUR, ASIA, EM) with local-currency returns
- **Dimensional corrections Ω(t)**: sovereign debt drag, role-dependent Fed rate, FX coupling

### Next
- **Bayesian adaptation**: Kalman filter to correct f(t) based on prediction errors
- **Parameter optimization**: grid search over α, γ, η, β_fx with Sharpe ratio loss
- **Backtest**: multi-currency hierarchical vs single-USD flat graph
