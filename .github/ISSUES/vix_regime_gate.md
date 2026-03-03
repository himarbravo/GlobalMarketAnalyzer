# P6: VIX-Based Regime Gate

## Problem
The current regime gate uses `s(t)` thresholds (0.40 for REFUGE, 0.70 for DEFENSIVE) to switch between trading modes. However, `s(t)` measures the **non-locality of Laplacian diffusion** (physics parameter), not market regime. Evidence:

| Period | s value | Is Crisis? |
|---|---|---|
| COVID Crash | 0.790 | YES |
| Bull 2019 | 0.777 | NO |
| AI Rally 2023-24 | 0.540 | NO |
| Fed Rate Hikes | 0.701 | YES |

s doesn't distinguish crisis from bull — values overlap completely. No fixed threshold can work.

## Solution
Separate the two roles of `s(t)`:
1. **s → Laplacian physics** (R² 0.97, keep as is)
2. **VIX → regime gate** (forward-looking, options-based)

### VIX Gate Thresholds
- `VIX > 35` → **REFUGE**: close all positions, long refuge ETFs (TLT, GLD, SHY)
- `VIX > 25` → **DEFENSIVE**: pairs at 25% size, no directional MR
- `VIX ≤ 25` → **ALPHA**: full pairs trading (dollar-neutral long-short)

### Why VIX
- Volmageddon: VIX > 30 ✅ → should be DEFENSIVE
- COVID: VIX > 80 ✅ → should be REFUGE
- Fed 2022: VIX > 30 ✅ → should be DEFENSIVE
- Bull 2019: VIX ~12-15 ✅ → should be ALPHA
- AI Rally: VIX ~13-18 ✅ → should be ALPHA

## Parent Issue
Part of Issue #13 (Regime Gate Investigation)

## Acceptance Criteria
- [ ] Gate uses VIX instead of s for regime detection
- [ ] Combo strategy uses VIX gate + pairs trading
- [ ] Backtest shows Combo outperforms both pure MR and pure Gate
- [ ] s remains unchanged for Laplacian physics
