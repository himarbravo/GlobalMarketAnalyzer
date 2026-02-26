# GlobalMarketAnalyzer

Modelo Ornstein-Uhlenbeck en grafo fraccional para análisis de mercado.

## Arquitectura

```
┌─────────────────────────────── PIPELINE ───────────────────────────────┐
│                                                                        │
│  config.py             Constantes y parámetros globales                │
│  database_manager.py   Conexión y queries a Supabase                   │
│  data_ingestion.py     Carga de precios, macro, fundamentales          │
│  schema.sql            Definición de tablas en Supabase                 │
│                                                                        │
│  ┌──── CORE MODEL ─────────────────────────────────────────────────┐   │
│  │  graph_builder.py       Grafo multi-capa con cross-lag          │   │
│  │                         3 escalas (20/60/120d) + W²,W³          │   │
│  │  fundamental_filter.py  Scores fundamentales (PE, ROE, etc.)    │   │
│  │  heat_engine.py         Solver O-U fraccional                   │   │
│  │  inertia_detector.py    Inercia: phase space, masa, rotación    │   │
│  │  perturbation_simulator.py  Propagación de shocks               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  signal_generator.py   Pipeline: datos → graph → O-U → señales        │
│                                                                        │
│  ┌──── DIAGNOSTICS ────────────────────────────────────────────────┐   │
│  │  model_diagnostic.py    Tests 1-6 (tracking, predicción, etc.)  │   │
│  │  model_diagnostic_v2.py Tests 7-11 (non-locality, eventos)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

## Ejecución

```bash
# Señales del día
python signal_generator.py

# Diagnóstico básico
python model_diagnostic.py

# Tests avanzados (non-locality, eventos, perturbaciones)
python model_diagnostic_v2.py
```
