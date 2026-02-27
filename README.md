# GlobalMarketAnalyzer

Water-landscape model on a fractional graph for market analysis.

## The Model

Three fundamental quantities, two equations, everything else derived:

```
MONEY (m) ── the water that flows between investors on the graph
  dm/dt = -α·L^s·m + v(t) + QE

CAPITAL (K) ── the terrain, fixed at each company, changes slowly
  dK/dt = g(t)  (CAPEX, R&D, depreciation, earnings)

PRICE (u) = λ·K = m ── the observable
  λ = m/K ── valuation multiple (derived, not independent)

L^s ── fractional Laplacian: s=1 local diffusion, s→0 global panic
L_K ── capital-weighted Laplacian: big companies drag neighbors' λ
v(t) ── macro velocity: where money flows (QE, rate hikes, rotation)
```

**Signal**: δ = λ - λ_eq(regime). Overvalued if δ > 0, undervalued if δ < 0.
Strong buy: δ < 0 AND v > 0 (undervalued + money arriving).

## Architecture

```
┌─────────────────────────── PIPELINE ──────────────────────────────┐
│                                                                    │
│  config.py             Constants and global parameters             │
│  database_manager.py   Supabase connection and queries             │
│  data_ingestion.py     Price, macro, fundamental data loading      │
│  schema.sql            Supabase table definitions                  │
│                                                                    │
│  ┌──── CORE MODEL ────────────────────────────────────────────┐   │
│  │  graph_builder.py       Multi-layer graph with cross-lag   │   │
│  │                         3 scales (20/60/120d) + W²,W³      │   │
│  │  fundamental_filter.py  Fundamental scores (F)             │   │
│  │  capital_field.py       K(t) terrain from earnings data     │   │
│  │  heat_engine.py         Money equation solver + landscape   │   │
│  │  inertia_detector.py    Phase space, mass, rotation         │   │
│  │  perturbation_simulator.py  Shock propagation               │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  signal_generator.py   Pipeline: data → graph → solve → signals   │
│  regime_calibrator.py  Historical calibration (7 market periods)   │
│                                                                    │
│  ┌──── DIAGNOSTICS ───────────────────────────────────────────┐   │
│  │  model_diagnostic.py    Tests 1-6 (tracking, prediction)    │   │
│  │  model_diagnostic_v2.py Tests 7-11 (non-locality, events)   │   │
│  │  historical_tests.py    Walk-forward backtests              │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

## Execution

```bash
# Daily signals
python signal_generator.py

# Basic diagnostics
python model_diagnostic.py

# Historical calibration by market regime
python regime_calibrator.py

# Full walk-forward tests
python historical_tests.py
```

## Documentation

- [MATHEMATICS.md](MATHEMATICS.md) — Complete mathematical formulation
- [DIARY.md](DIARY.md) — Development log
