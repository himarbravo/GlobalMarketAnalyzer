# P7: Reversibility Detection — Modal Overlap & Graph Entropy

## 🧠 The Core Problem

Our z-scores tell us *how far* an asset is from equilibrium, but not *whether the equilibrium still exists*. This is why the hit rate is ~50% — we can't distinguish between:

### Case A — Temporary Dislocation (Reversible) ✅

```
AAPL cae -5% por pánico general del mercado.
Sus fundamentales no cambiaron. Sus vecinos (MSFT, GOOGL) se mantienen.
Las correlaciones siguen iguales.

→ El equilibrio sigue siendo válido.
→ El z-score negativo predice correctamente un rebote.
→ El trade funciona.
```

**Ejemplos reales:**
- **Flash Crash agosto 2015**: China devalúa yuan → S&P -11% en 5 días → todo rebota en 3 semanas. Correlaciones tech intactas. Las "tuberías" del grafo no cambiaron.
- **Q4 2018 sell-off**: Fed sube tipos → mercado -20% → V-recovery en enero 2019. Los modos del grafo estables.
- **COVID crash industriales**: Boeing -70% pero la relación Boeing↔Airbus↔Honeywell se mantuvo. Cuando el sector rebotó, Boeing rebotó.

### Case B — Structural Change (Irreversible) ❌

```
AAPL cae -5% porque perdió un juicio de patentes.
Su posición competitiva cambió. La relación con sus vecinos está mutando.

→ El viejo equilibrio ya no existe. Hay un nuevo equilibrio.
→ El z-score negativo NO predice un rebote (la "piscina se achicó").
→ El trade falla.
```

**Ejemplos reales:**
- **META febrero 2022**: -26% en un día por Reality Labs. GOOGL y MSFT no cayeron. La relación META↔GOOGL cambió: META dejó de ser "big tech ads" y pasó a ser "metaverse bet". Z-score decía "comprar META" pero cayó otro -40%.
- **Intel 2020-2024**: perdió cuota frente a AMD/TSMC gradualmente. Su correlación con el sector semicon fue cayendo. Z-score decía "INTC barata" durante años, pero la piscina se encogía permanentemente ($65 → $20).
- **Energy vs Tech 2022**: la correlación energy↔tech se invirtió con los tipos altos. XOM subía mientras QQQ bajaba. Los modos del grafo rotaron completamente.

## 🔬 Diagnóstico Matemático

El problema se puede formular rigurosamente:

### ¿Por qué R² = 0.97 no implica poder predictivo?

El R² mide qué tan bien el equilibrio del Laplaciano **describe** la distribución actual de capital. Pero describir el presente ≠ predecir el futuro.

Analogía: un mapa altamente preciso (R² 0.97) no predice dónde habrá tráfico mañana.

### ¿Qué sería un "chivato" de reversibilidad?

Necesitamos detectar si el **grafo en sí** está cambiando. No los precios sobre el grafo (eso son los z-scores), sino la **estructura** del grafo. Para esto proponemos dos herramientas del álgebra espectral:

---

## 📐 Propuesta: Modal Overlap (Correlación entre Modos)

El Laplaciano L tiene eigenvectores φ₁, φ₂, ... φ_N (los "modos de vibración"). Cada modo define un patrón colectivo de activos.

**Chivato**: comparar los eigenvectores de dos ventanas temporales:

```
Overlap del modo k = |⟨φₖ(t-Δt) | φₖ(t)⟩|²
```

- Overlap ≈ 1 → el modo no cambió → estructura intacta → z-scores válidos → TRADEAR
- Overlap ≈ 0 → el modo rotó → estructura mutó → viejo equilibrio inválido → NO TRADEAR

### Ejemplo: META 2022
Los modos del cluster "big tech ads" rotaron cuando META pivoteó a metaverse. El overlap habría caído bruscamente → detector de que el z-score de META no era fiable.

### Ejemplo: Flash Crash 2015
Los modos se mantuvieron estables pese al crash → overlap ≈ 1 → z-scores seguían siendo válidos → tradear fue correcto.

---

## 📐 Propuesta: Von Neumann Entropy del Grafo

La entropía del grafo mide cuánta "estructura" hay vs "aleatoriedad":

```
S = -Tr(ρ · log(ρ))    donde  ρ = L̃ / Tr(L̃)
```

- S baja → el grafo tiene estructura clara (pocos modos dominan)
- S alta → el grafo es más aleatorio (todos los modos contribuyen igual)

**Chivato**: el cambio de entropía

```
ΔS = S(t) - S(t-Δt)
```

- |ΔS| ≈ 0 → no se produjo entropía → proceso reversible → equilibrio válido → TRADEAR
- |ΔS| grande → entropía producida → proceso irreversible → nuevo equilibrio → NO TRADEAR

---

## 🎯 Combinación: Filtro de Reversibilidad

```
Para cada refit day:
  1. Calcular ΔS (entropía global del grafo)
     → Si |ΔS| > umbral_global: DETENER todo trading (el mercado está en transición)

  2. Para cada activo con |z| > threshold:
     a. Identificar en qué modo φₖ participa más
     b. Calcular overlap de ese modo con la ventana anterior
     c. Si overlap > umbral_modal: TRADEAR (reversible)
        Si overlap < umbral_modal: SKIP (irreversible, nuevo equilibrio)
```

## Implementation Plan

### Phase 1: Diagnostic
- [ ] Compute modal overlap for each crisis period (do modes actually rotate during structural changes?)
- [ ] Compute Von Neumann entropy timeline (does entropy spike when z-scores fail?)
- [ ] Correlate hit rate with modal overlap (does high overlap → higher hit rate?)

### Phase 2: Integration  
- [ ] Add `modal_overlap()` and `graph_entropy()` to `GraphBuilder`
- [ ] Add reversibility filter to Pairs + Gate strategy
- [ ] Re-run crisis backtest with filtered vs unfiltered

### Phase 3: Validation
- [ ] Walk-forward test: does filtering improve Sharpe ratio?
- [ ] Out-of-sample test on new period
- [ ] Compare hit rate with and without reversibility filter

## Acceptance Criteria
- [ ] Modal overlap distinguishes Case A from Case B in at least 3/5 test periods
- [ ] Filtered Pairs strategy has hit rate > 55% (vs current 48-54%)
- [ ] |ΔS| correlates with subsequent MR failure

## References
- Von Neumann entropy of graphs: Braunstein et al. (2006)
- Graph spectral analysis for financial networks: Marti et al. (2021)
- Mean reversion vs structural breaks: Hendricks & Wilcox (2014)
