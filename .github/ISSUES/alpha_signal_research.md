# P8: Alpha Signal Research — Beyond Z-Score Levels

## Background

The P6 VIX gate correctly identifies market regimes (ALPHA/DEFENSIVE/REFUGE), and the Combo (Pairs + Gate) strategy outperforms all other strategies. However, **all strategies still lose money** even in full ALPHA mode in bull markets (-8.7% Bull 2019, -15.1% AI Rally).

The root cause: **z-score levels don't predict future returns**. A z-score of -2 tells us an asset is below its graph equilibrium, but doesn't tell us whether it will revert (temporary dislocation) or continue (structural change). Hit rate is ~50% — essentially a coin flip.

P7 investigated eigenvector-based modal overlap as a "reversibility chivato" but it proved **too noisy** (O_avg 0.07-0.17 even in stable markets). This issue proposes 5 alternative directions.

## Context from P7 (dead end)
- Eigenvectors of the 102-asset Laplacian rotate constantly due to noise (ratio observaciones/parámetros ~0.06)
- Shrinkage (Ledoit-Wolf) helps in bull (+117%) but not in crisis
- Entropy S(G) ≈ 4.55 uniformly — doesn't distinguish regimes
- Spectral clustering degenerate: 92-99/102 assets fall in one cluster

---

## Proposed Research Lines

### 1. 🟢 dz/dt — Z-Score Velocity (smallest effort, highest potential)

**Hypothesis**: the *speed* of change in z-score is more predictive than the *level*. A fast drop (dz/dt << 0) suggests a shock (likely reversible), while a slow drift suggests structural change (irreversible).

**Implementation**:
```python
dz = z[t] - z[t - REFIT_DAYS]
# Trade only when |dz/dt| > threshold (fast moves = reversible)
# Skip slow drifts (structural)
```

**Effort**: ~10 lines in `crisis_backtest.py`
**Test**: compare hit rate for fast-z vs slow-z trades

---

### 2. 🟡 APC Gate — Average Pairwise Correlation

**Hypothesis**: when average pairwise correlation rises, the market is in contagion mode and individual z-scores are unreliable. When APC is low, assets move independently and z-scores are meaningful.

**Evidence**: previous experiment showed APC-based gate reduced MaxDD from -49% to -14% (see KI: `regime_stress_indicators.md`).

**Implementation**: compute `APC = mean(|corr_ij|)` per window, use as gate threshold.

**Effort**: ~30 lines in `graph_builder.py` + backtest integration

---

### 3. 🟢 Shrinkage (Ledoit-Wolf)

**Hypothesis**: shrinkage reduces noise in the correlation matrix, producing more stable z-scores that better predict mean reversion.

**Evidence**: prototype exists in deleted `spectral_diagnostic.py`. In bull markets, shrunk eigenvectors are 2× more stable.

**Implementation**: apply shrinkage to W in `graph_builder.build()` before computing Laplacian.

**Effort**: ~20 lines in `graph_builder.py`

---

### 4. 🟢 Aggregate Spectral Metrics

**Hypothesis**: while individual eigenvectors are noisy, aggregate properties (trace, spectral radius, λ₁/λ_N ratio) are stable and may serve as regime indicators.

**Candidates**:
- `Tr(L) / N` — average connectivity strength
- `λ₁ / λ_N` — spectral gap ratio (cluster structure vs noise)
- `Σ λ_k² / (Σ λ_k)²` — eigenvalue concentration (Herfindahl index)

**Effort**: ~15 lines, eigenvalues already computed

---

### 5. 🔴 Fundamental-Based Alpha (δ = λ - λ*(R))

**Hypothesis**: z-scores alone lack fundamental information. Using the capital-based valuation signal δ from MATHEMATICS.md Section 10 could provide genuine alpha.

**Implementation**: requires full calibration of λ*(R) by regime and asset, earnings data, and ROIC/WACC estimates.

**Effort**: ~200+ lines, new calibration pipeline

---

## Recommended Execution Order

```
1. dz/dt velocity filter     → Quick test, could immediately improve hit rate
2. Shrinkage                  → Cleaner correlations → better z-scores
3. Aggregate spectral metrics → New regime indicators  
4. APC gate                   → Proven to work, medium effort
5. Fundamental α              → Full pipeline, biggest potential payoff
```

## Acceptance Criteria
- [ ] At least one approach improves hit rate from ~50% to >55%
- [ ] At least one approach produces positive returns in Bull 2019 or AI Rally
- [ ] Combined approach (best gate + best signal) achieves Sharpe > 0 in at least 3/5 periods

## Parent Issue
- Successor to P7 (reversibility_detection.md — RESOLVED, dead end)
- Builds on P6 (VIX gate — working, but alpha engine is the bottleneck)
