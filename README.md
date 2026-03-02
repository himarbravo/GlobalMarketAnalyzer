# GlobalMarketAnalyzer

Water-landscape model on a hierarchical fractional graph with multi-currency monetary fields.

## The Equation

$$\gamma \cdot m'' + m' = -\alpha \cdot L_z^s \cdot m + f_i(t) + \Omega_i(t) + v(t)$$

| Term | What it does | Updates |
|---|---|---|
| γ·m'' | **Inertia** — money has mass, trends persist | Auto-calibrated (grid search) |
| -α·L_z^s·m | **Diffusion** — money flows between connected assets within zone z | α auto-calibrated, s adapts to regime |
| f_i(t) | **Source** — injection/drain per node role | Recalculated each step |
| Ω_i(t) | **Dimensions** — debt drag + interest rate + FX coupling | Per-country, per-role |
| v(t) | **Macro velocity** — direction of global money flow | From macro indicators |

## Graph Structure

```
╔═══════════════════════════════════════════════════════════════════════╗
║  DIMENSIONS (external fields, affect all nodes)                      ║
║                                                                      ║
║  Ω_debt  = -η · D/GDP · m/252            (sovereign debt drain)     ║
║  Ω_rate  = -β_r · dr/dt · m              (per central bank, ±role) ║
║  Φ_FX    = β_fx · r_fx · m̄_other_zone   (FX coupling)             ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ZONA USD (~50 nodos)       ║  ZONA EUR (~25 nodos)                 ║
║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐ ║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐           ║
║  │JPM BAC│  │AAPL NVDA   │ ║  │HSBC   │  │SAP  ASML   │           ║
║  │GS  MS │  │TSLA TLT    │ ║  │BNP.PA │  │NVO  SIE    │           ║
║  │WFC    │→→│GLD  BTC    │ ║  │SAN ING│→→│LVMHF TTE   │           ║
║  └───────┘←←└────────────┘ ║  │BBVA   │←←│AZN  NESN   │           ║
║  L_USD (retornos en USD)    ║  └───────┘  └────────────┘            ║
║         ↕ EURUSD                    ↕ USDJPY                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║  ZONA ASIA (~22 nodos)      ║  ZONA EM (~20 nodos)                  ║
║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐ ║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐           ║
║  │MUFG   │  │TSM  SONY   │ ║  │ITUB   │  │VALE EWZ    │           ║
║  │SMFG   │→→│TM   BABA   │ ║  │HDB    │→→│PBR  INDA   │           ║
║  │005930 │←←│EWJ  BHP    │ ║  │TLKM   │←←│AMX  2222   │           ║
║  L_ASIA (moneda local)      ║  L_EM (moneda local)                  ║
╚═══════════════════════════════════════════════════════════════════════╝
  →→ = lending (bank creates money)    ←← = interest (company pays back)
  ←→ = correlation (undirected)        ↕  = FX coupling between zones
```

## Step-by-Step: How the Solver Works

### Step 1 — Load Data
- **Trigger**: `signal_generator.py` daily run
- **Input**: Supabase tables (prices, fundamentals, macro_indicators)
- **Output**: Prices, volumes, macro series, fundamental scores

### Step 2 — Build Graph (`graph_builder.py`)
- Compute **local-currency returns** (SAP in EUR, TSM in TWD proxy)
- Calculate **cross-lag correlation** at 3 scales (20d, 60d, 120d)
- Build **adjacency matrix W** with volume weighting + 2nd/3rd order neighbors
- Assign **node roles** (bank/productive), **countries**, **zones**
- Build **directed edges** (bank → company lending, company → bank interest)
- Compute **Laplacian L per zone** and eigendecomposition
- Calibrate **s(t)** from VIX, credit spreads, copper, oil, DXY

### Step 3 — Compute Capital Field (`capital_field.py`)
- Load quarterly fundamentals (FCF, CAPEX, ROIC, debt, growth)
- Calculate **dK/dt** = change in capital per quarter
- Include **delta_debt** (net new borrowing = money creation proxy)
- Interpolate to daily: `capital_rate_daily = dK/dt / 252`

### Step 4 — Compute Sentiment (`fundamental_filter.py`)
- **S_fund** ∈ [0.5, 2.0]: from 7-component fundamental score (quarterly)
- **S_macro** ∈ [0.7, 1.3]: from PMI of ticker's country (monthly)
- **S_fear** ∈ [0.6, 1.2]: from VIX (daily)
- **S_earnings** ∈ [0.7, 1.2]: from eps_surprise (asymmetric: -0.3 miss/+0.2 beat) + analyst target gap (P2.3)
- **S_composite = S_fund × S_macro × S_fear × S_earnings**

### Step 5 — Compute Dynamic f(t) (`heat_engine.py`)
- **Banks**: `f = yield_spread × lending_capacity × S`
- **Productive**: `f = (dK/dt + credit_in - interest_out) × S`

### Step 6 — Compute Dimensional Corrections Ω(t) (`heat_engine.py`)
- **Sovereign debt**: `Ω_debt = -η × D/GDP × m / 252` (slow constant drain)
- **Interest rate** (per country's central bank):
  - Banks: `Ω = +β_r_bank × dr/dt × m` (gain from rate hikes)
  - Companies: `Ω = -β_r_prod × (1+leverage) × dr/dt × m` (suffer)
- **FX coupling**: when EUR/USD rises → capital flows from USD zone to EUR zone

### Step 7 — Solve 2nd-Order Equation (`heat_engine.py`)
- Project everything to **spectral space** (eigenvectors of L_z)
- Shift equilibrium by Ω: `m_eq += Ω_k / μ_k`
- For each timestep: `m[t+1] = m[t] + momentum × v[t] - restoring × (m[t] - m_eq)`
- Back to physical space: `m_pred = m_k_pred @ Φ^T`

### Step 8 — Generate Signals
- **δ = m_pred - m_real**: positive → overvalued (SELL), negative → undervalued (BUY)
- Combine with technical indicators, regime classification
- Output: BUY/SELL/HOLD with confidence score

## Regime Adaptation

The equation is the same in all regimes. What changes are the **parameters**:

| Parameter | Bull (VIX~15) | Stress (VIX~30) | Crisis (VIX~45) |
|---|---|---|---|
| s (diffusion reach) | ~0.9 (local) | ~0.5 (regional) | ~0.2 (global panic) |
| α (diffusion speed) | ~0.02 (slow) | ~0.04 (medium) | ~0.06 (fast) |
| γ (inertia) | ~10 (trends) | ~3 (reduced) | ~1 (no momentum) |
| S_fear (via VIX) | ~1.1 (confident) | ~0.9 (cautious) | ~0.7 (scared) |

## Parameters

| Parameter | Value | Calibrated? | Meaning |
|---|---|---|---|
| α | ~0.02 | ✅ OOS (EKF) | Diffusion speed |
| γ | ~5 | ✅ Grid search (EKF) | Inertia / momentum |
| s | ~0.8 | ✅ UKF (P3.1) | Fractional exponent — tracked through regime transitions |
| β_fx | 0.30 | ❌ Fixed | FX flow elasticity |
| η | 0.02 | ❌ Fixed | Sovereign debt weight |
| β_r_bank | -0.50 | ❌ Fixed | Bank rate sensitivity |
| β_r_prod | +0.30 | ❌ Fixed | Company rate sensitivity |
| S_CREDIT_DELTA | 0.20 | ❌ Fixed (P2.2) | Credit spread widening speed |
| S_RATE_MOM | 0.15 | ❌ Fixed (P2.1) | Central bank rate momentum |

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
- Inertia (γ), hierarchical graph, directed edges
- Dynamic f(t) per role, composite sentiment S(t)
- Multi-currency fields (4 zones, local returns, per-zone Laplacians)
- Dimensional corrections Ω(t) (debt + rate + FX coupling)
- International macro indicators (ECB, BoJ, PMI, GDP)
- **P0**: Expanded universe (**148 tickers**, 10 countries, 4 zones) — PR #3
- **Bayesian adaptation**: 3 parallel EKF filters for f_k, α, γ
- **P1**: Trading strategy — adaptive Z_ENTRY, 3-tier costs (5/15/25 bps), hard stop -10% — PR #5
- **P2.2**: Credit spread delta (early warning before VIX) — PR #7
- **P2.3**: Earnings whisper + analyst target gap — PR #8
- **P2.1**: Central bank rate momentum — PR #9
- **P3.1**: UKF for s (regime transition tracking via sigma points) — PR #12
- **P3.2**: Kalman state persistence (Supabase JSONB) — PR #12

### Backtest Results (with realistic costs)

```
MR (z-score) Strategy:
  Sharpe:     0.89
  Return:     +17.2%
  α vs SPY:   +0.6%
  α vs Rand:  +49.5%
  MaxDD:      -12.4%
  Win trials: 3/3 ✅
  Cost model: 5bps US / 15bps intl / 25bps EM
```

### Diagnostic: Where Alpha Comes From

| Source | Contribution | Evidence |
|---|---|---|
| **Multi-zone graph** | ~73% | Trial 1 (intl) Sharpe=2.37 vs Trial 3 (US only) Sharpe=-0.23 |
| **EKF adaptation** | ~22% | Sharpe 0.20→0.47 from Bayesian filters |
| **γ calibration** | ~5% | Marginal improvement when momentum detected |
| **Equation itself** | ~0% | EMH: market already prices equilibrium dynamics |

Alpha lives in **cross-zone latency** (days-weeks to arbitrage between jurisdictions), not in intra-zone dynamics (milliseconds, already arbed by HFT).

### 🟡 Current Limitations

1. **No crisis validation** — only tested on 2025-2026 (bull market)
2. **Composite strategy loses** — weights 0.4z/0.3F/0.3δ need better optimization
3. **Defensive signals untested** — credit spread delta, rate momentum ready but no stress events in test window
4. **UKF lacks stress data** — regime transition detection can't be validated without crisis periods

### Next: P4 — Validation (Critical)

**P4 — Crisis validation and paper trading**
- [ ] Crisis backtest: COVID Mar 2020, Fed rate hikes 2022, Volmageddon 2018
- [ ] Cross-validation: train 2020-2023, test 2024-2026 and vice versa
- [ ] Paper trading: 6 months of live signals without execution
- [ ] Stress test: verify credit spread delta catches COVID-style drawdown
- [ ] Verify UKF s tracking: does s drop before VIX spikes?
