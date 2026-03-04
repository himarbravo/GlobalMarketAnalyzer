# Arquitectura del Proyecto — GlobalMarketAnalyzer

## Estructura de Carpetas

```
GlobalMarketAnalyzer/
├── core/                  ← Motor del modelo O-U sobre grafo fraccional
├── db/                    ← Conexión a Supabase y gestión de datos
├── ml/                    ← El Matemático — predicciones ML
├── tests/                 ← Backtests, validación, y tests unitarios
├── docs/                  ← Documentación técnica
├── viz/                   ← Visualizaciones 3D
├── output/                ← Resultados guardados (JSON/HTML)
├── logs/                  ← Logs de ejecución
├── .cache/                ← Cache local de precios (evita queries repetidas)
├── config.py              ← Configuración global (tickers, zonas, parámetros)
├── signal_generator.py    ← Genera señales BUY/SELL/WATCH/HOLD
├── paper_trader.py        ← Paper trading diario + scoring
└── README.md              ← Documentación general del proyecto
```

---

## `core/` — El Motor Matemático

El corazón del sistema. Implementa la ecuación O-U, el grafo, y los módulos de análisis.

| Archivo | Tamaño | Qué hace |
|---|---|---|
| **graph_builder.py** | 43KB | Construye el grafo multi-capa: cross-lag correlations (20/60/120d), correcciones FX/tipos/deuda, W²/W³ contagio, Laplaciano L, eigendecomposition, L^s fraccional, calibración de s(t) |
| **heat_engine.py** | 41KB | Resuelve la ecuación O-U en espacio espectral: solver 2º orden, z-scores, probabilidades analíticas P(rev), half-life, advección macro, flujos de dislocation |
| **fundamental_filter.py** | 16KB | Score fundamental de 7 componentes (FCF yield, ROIC, growth, quality, valuation, analyst, momentum). Clasifica activos como value_creator / speculative |
| **capital_field.py** | 16KB | Modelo dinero/capital/precio (Sec 10 de MATHEMATICS.md): ecuación del dinero sobre grafo, ecuación del capital, múltiplos λ |
| **inertia_detector.py** | 20KB | 5 componentes de inercia: espacio de fases (u, ů), masa efectiva, momento angular espectral, flujo de energía, histéresis |
| **regime_calibrator.py** | 14KB | Calibra α, γ por régimen (Calm/Normal/Stress/Crisis). Walk-forward con parámetros congelados |
| **reversibility.py** | 11KB | P7: filtro de reversibilidad sector-correlación. Compara correlación con peers entre ventanas para detectar cambios estructurales |
| **ukf.py** | 8KB | Unscented Kalman Filter para tracking de s(t). Sigma points de Merwe, feedback con errores de predicción |
| **perturbation_simulator.py** | 8KB | Simula shocks: "¿qué pasa si oil sube 20%?" Propaga perturbación por los eigenvectors |
| **__init__.py** | 1KB | Exports del paquete |

---

## `db/` — Datos y Persistencia

Gestión de la conexión a Supabase y la ingesta de datos.

| Archivo | Tamaño | Qué hace |
|---|---|---|
| **database_manager.py** | 23KB | API principal: get_prices, get_macro, get_fundamentals. Paginación, cache local .pkl, exponential backoff |
| **data_ingestion.py** | 33KB | Ingesta masiva desde APIs: precios (yfinance), fundamentales (quarterly), indicadores técnicos (51 columnas), macro (FRED) |
| **fred_client.py** | 4KB | Cliente para FRED API (Federal Reserve): VIX, yields, credit spreads, DXY, commodities |
| **schema.sql** | 20KB | Esquema Supabase: tables prices, assets, macro_indicators, fundamentals, signals, kalman_state |
| **__init__.py** | 0.3KB | Exports |

---

## `ml/` — El Matemático (ML Predictions)

Scripts de machine learning para predecir cantidades objetivas del mercado.

| Archivo | Tamaño | Qué hace |
|---|---|---|
| **regime_model.py** | 17KB | Baseline: clasifica regímenes (risk-on/transition/risk-off) con LightGBM. 38 features macro, temporal split. Resultado: 50.8% accuracy |
| **multi_target.py** | 15KB | 8 targets: volatilidad (5d/20d), tail risk, VIX dir, credit spread, yield curve, stock-bond corr, SPY dir. Resultado: yield (AUC 0.65) y tail risk (AUC 0.60) son predecibles |
| **graph_vs_macro.py** | 13KB | Comparación v1: macro features vs macro + eigenvalues crudos (numpy). Resultado: eigen crudos mejoran tail risk +0.048 |
| **graph_vs_macro_v2.py** | 8KB | Comparación v2: macro vs macro + GraphBuilder completo (72 walk-forward builds). Resultado: GraphBuilder mejora tail risk +0.020, menos que crudos |

---

## `tests/` — Validación y Backtesting

Tests unitarios, backtests, y diagnósticos del modelo.

| Archivo | Tamaño | Qué hace |
|---|---|---|
| **crisis_backtest.py** | 27KB | Walk-forward en 3 crisis: Volmageddon, COVID, Fed 2022. Con/sin regime gate |
| **model_diagnostic.py** | 22KB | Diagnóstico v1: non-locality, phase space, event backtesting |
| **model_diagnostic_v2.py** | 15KB | Diagnóstico v2: mejorado con event types y crisis simuladas |
| **reverse_engineer.py** | 18KB | Sismología financiera: epicentro, firma espectral, similitud entre eventos |
| **historical_tests.py** | 18KB | Tests históricos walk-forward |
| **backtest.py** | 19KB | Backtest de la estrategia MR completa |
| **crossval_train_test.py** | 11KB | 3-fold cross-validation temporal |
| **crossval_full.py** | 7KB | Cross-validation completa con todas las variantes |
| **crossval_s_vix.py** | 5KB | Cross-val específica para comparar s-gate vs VIX-gate |
| **test_reversibility.py** | 10KB | 13 unit tests para el módulo de reversibilidad (P7) |
| **__init__.py** | — | |

---

## `docs/` — Documentación

| Archivo | Tamaño | Contenido |
|---|---|---|
| **MATHEMATICS.md** | 32KB | Fundamento teórico completo (706 líneas): ecuación O-U, grafo multi-capa, Laplaciano fraccional, probabilidades, inercia, analogía agua/paisaje, reversibilidad, clustering |
| **DIARY.md** | 12KB | Diario cronológico del desarrollo: cada sesión con objetivos, resultados, y hallazgos |
| **STRATEGY_DIAGNOSTIC.md** | 11KB | Diagnóstico de las estrategias (MR, Pairs, Gate, Combo) con resultados |

---

## Archivos raíz

| Archivo | Qué hace |
|---|---|
| **config.py** (16KB) | Tickers por zona (US, EUR, ASIA, EM), parámetros del modelo (α, s, γ), umbrales, claves API |
| **signal_generator.py** (19KB) | Genera señales BUY/SELL/WATCH/HOLD combinando z-scores, P(rev), fundamentales, regime gate, reversibility filter |
| **paper_trader.py** (11KB) | Ejecuta paper trading diario: carga señales, simula posiciones, scoring con `--review` |
| **README.md** (11KB) | Documentación general, arquitectura, instalación |

---

## Flujo de ejecución

```
                    Supabase (datos)
                         │
                    db/database_manager.py
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         core/       core/       core/
    graph_builder  fundamental  regime_calibrator
         │         _filter.py        │
         │              │            │
         ▼              ▼            ▼
    core/heat_engine.py ◄────────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
  z-scores  P(rev)  inercia
    │         │        │
    └────┬────┘────────┘
         ▼
  signal_generator.py  ◄── core/reversibility.py (P7 filter)
         │
         ▼
   paper_trader.py → Supabase (signals)
         
   ml/ (El Matemático) → predice yield curve, tail risk, VIX
```
