# Enriquecer datos macro y fundamentales para modelo Water-Landscape

## Contexto

El modelo Water-Landscape usa 3 campos:
- **m(t)** — dinero/precio (actualizado diario desde prices)
- **K(t)** — capital real (actualizado trimestral desde fundamentals)
- **λ = m/K** — múltiplo de valoración (derivado)

Actualmente K usa **proxies fijos** para inflación (3% anual) y WACC (8%), lo que limita la precisión de λ y δ = λ - λ_eq.

## Mejoras propuestas

### Tier 1 — Alto impacto, APIs gratuitas

| Dato | Impacto en el modelo | Fuente | Implementación |
|---|---|---|---|
| **CPI mensual** | Inflación real reemplaza INFLATION_PROXY=3% en capital_field.py | FRED API (series CPIAUCSL) | Añadir a macro_indicators, interpolar diario |
| **Treasury yields 2Y/10Y** | WACC real reemplaza WACC_PROXY=8% | FRED API (DGS2, DGS10) | WACC = risk_free + β·equity_premium |
| **Earnings dates** | Saber cuándo se actualiza K → confidence más preciso | yfinance calendar | Añadir campo earnings_date a fundamentals |
| **Shares outstanding histórico** | Medir dilución real (SBC real vs proxy) | SEC EDGAR | Guardar trimestral, calcular Δshares |

### Tier 2 — Alto impacto, más trabajo

| Dato | Impacto | Fuente |
|---|---|---|
| **CAPEX vs R&D separados** | R&D es inversión futura, no gasto corriente → K más preciso | 10-K filings / FMP API |
| **Forward P/E (consensus)** | λ forward es más predictivo que trailing | yfinance / FMP |
| **Credit spreads BBB-AAA** | WACC varía con condiciones crediticias | FRED (BAA10Y) |
| **Sector WACC diferenciado** | Tech WACC ≠ Utilities WACC ≠ Banks | Damodaran dataset |

### Tier 3 — Diferenciadores

| Dato | Impacto | Fuente |
|---|---|---|
| **Insider transactions** | Señal fuerte sobre K real | SEC Form 4 |
| **Options put/call OI** | Smart money expectations de λ | CBOE |
| **ETF flows** | Flujos pasivos distorsionan m sin afectar K | ETF.com API |

## Impacto esperado

- **K más preciso** → λ_eq mejor → δ (mispricing) más fiable
- **WACC dinámico** → K negativo de bancos se calibra (su WACC es distinto)
- **Forward λ** → señales predictivas mejoran (C2/C3 hit rates)
- **Cross-validation**: R²=0.977±0.007 (no overfitting), MR P&L=+8.8% media

## Prioridad de implementación

1. CPI + Treasury yields (FRED) — 1 día
2. Forward P/E + earnings dates (yfinance) — 1 día
3. Credit spreads (FRED) — medio día
4. Sector WACC (Damodaran CSV anual) — medio día

## Estado actual del modelo (27 Feb 2026)

- Tests: 37/38 PASS (Math 16/16, Data 12/12, Capacity 9/10)
- N=102 activos, 4156 aristas
- Cross-val 7 periodos: no overfitting
- Corr(s, VIX) global = -0.606
