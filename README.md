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
║  ZONA USD (~50 nodos)       ║  ZONA EUR (~13 nodos)                 ║
║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐ ║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐           ║
║  │JPM BAC│  │AAPL NVDA   │ ║  │HSBC   │  │SAP  ASML   │           ║
║  │GS  MS │  │TSLA TLT    │ ║  │BNP.PA │  │NVO  SIE    │           ║
║  │WFC    │→→│GLD  BTC    │ ║  │SAN ING│→→│LVMHF TTE   │           ║
║  └───────┘←←└────────────┘ ║  └───────┘←←│AZN  EWG    │           ║
║  L_USD (retornos en USD)    ║  L_EUR (retornos en EUR)              ║
║         ↕ EURUSD                    ↕ USDJPY                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║  ZONA ASIA (~10 nodos)      ║  ZONA EM (~8 nodos)                  ║
║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐ ║  ┌─BANKS─┐  ┌─PRODUCTIVE─┐           ║
║  │MUFG   │  │TSM  SONY   │ ║  │ITUB   │  │VALE EWZ    │           ║
║  │SMFG   │→→│TM   BABA   │ ║  │HDB    │→→│PBR  INDA   │           ║
║  └───────┘←←│EWJ  FXI    │ ║  └───────┘←←│EWT         │           ║
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
- **S_earnings** ∈ [0.8, 1.2]: from eps_surprise, decays over 20 days
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
| α | ~0.02 | ✅ OOS | Diffusion speed |
| γ | ~5 | ✅ Grid search | Inertia / momentum |
| s | ~0.8 | ✅ Daily (VIX) | Fractional exponent |
| β_fx | 0.30 | ❌ Fixed | FX flow elasticity |
| η | 0.02 | ❌ Fixed | Sovereign debt weight |
| β_r_bank | -0.50 | ❌ Fixed | Bank rate sensitivity |
| β_r_prod | +0.30 | ❌ Fixed | Company rate sensitivity |

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
- Expanded universe (13 banks, ~80 tickers across 4 zones)
- **Bayesian adaptation**: 3 parallel filters in solve():
  - KF for f_k (source, linear)
  - EKF for α_k (diffusion, per-mode Jacobian)
  - EKF for γ (inertia, global Jacobian)

### Diagnostic: Where Alpha Comes From

| Source | Contribution | Evidence |
|---|---|---|
| **Multi-zone graph** | ~73% | Trial 1 (intl) Sharpe=2.37 vs Trial 3 (US only) Sharpe=-0.23 |
| **EKF adaptation** | ~22% | Sharpe 0.20→0.47 from Bayesian filters |
| **γ calibration** | ~5% | Marginal improvement when momentum detected |
| **Equation itself** | ~0% | EMH: market already prices equilibrium dynamics |

Alpha lives in **cross-zone latency** (days-weeks to arbitrage between jurisdictions), not in intra-zone dynamics (milliseconds, already arbed by HFT).

### 🔴 Current Weaknesses

1. **1 of 3 trials works** — model only generates alpha with multi-zone diversification
2. **9 trades in 289 days** — Z_ENTRY=1.5 too restrictive, need ~200 trades/year
3. **No crisis data** — only 2025-2026 (bull market), no bear market validation
4. **Composite strategy loses money** — weights 0.4z/0.3F/0.3δ are arbitrary
5. **10bps cost model** — real-world slippage, spread, market impact not included

### Next: Alpha Increase Roadmap (Priority Order)

**P0 — Graph expansion (biggest alpha driver)**
- [ ] 20-25 tickers per zone (currently EUR=13, ASIA=10, EM=8)
- [ ] Sub-zones (Nordics, MENA, Latam) for finer cross-zone gradients
- [ ] Cross-zone explicit edges (BoJ → TLT, sovereign wealth fund flows)
- [ ] Non-US yields (Bund 10Y, JGB 10Y) as per-zone signals

**P1 — Trading strategy calibration**
- [ ] Adaptive Z_ENTRY (lower in bull, higher in crisis) → more trades
- [ ] Optimize composite weights via walk-forward Sharpe maximization
- [ ] Position sizing (Kelly criterion) and stop losses
- [ ] Realistic cost model with spread + slippage

**P2 — Information edge in f(t)**
- [ ] NLP sentiment from central bank speeches (predict rate decisions)
- [ ] Alternative data: satellite (ports, factories), web traffic
- [ ] Real-time credit spreads (IG/HY) for faster regime detection than VIX
- [ ] Earnings whisper consensus vs surprise for S_earnings

**P3 — Extended Bayesian adaptation**
- [ ] UKF for s (fractional exponent, highly nonlinear λ^s)
- [ ] Joint EKF state vector [f, α, γ, s] with full covariance
- [ ] Online learning: persist Kalman state across daily runs

**P4 — Validation**
- [ ] Paper trading: 6 months of live signals without execution
- [ ] Crisis backtest: COVID Mar 2020, Fed 2022, Volmageddon 2018
- [ ] Cross-validation: train 2020-2023, test 2024-2026 and vice versa

