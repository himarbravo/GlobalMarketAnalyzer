# Backtest Manual con Gemini (Prompts por Fecha)

Objetivo: construir una cartera dirigida por LLM y luego backtestearla.

Instrucciones rápidas:
1. Para cada bloque de fecha, copia el prompt completo en Gemini.
2. Pega la respuesta de Gemini justo debajo, en la sección `RESPUESTA GEMINI`.
3. Mantén el formato JSON exacto para que luego yo lo pueda parsear y backtestear.

Reglas del experimento (fijas para todo el documento):
- Universo permitido: `SPY, QQQ, IWM, TLT, SHY, GLD, UUP`
- Suma de pesos: `100`
- Sin cortos (no negativos)
- Máximo por activo: `40`
- Frecuencia de rebalanceo: mensual
- Horizonte de cada decisión: 1 mes

Formato de salida obligatorio (JSON):
```json
{
  "date": "YYYY-MM-DD",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0,
  "weights": {
    "SPY": 0,
    "QQQ": 0,
    "IWM": 0,
    "TLT": 0,
    "SHY": 0,
    "GLD": 0,
    "UUP": 0
  },
  "thesis": [
    "3-5 bullets cortos con rationale"
  ],
  "risk_triggers": [
    "3 señales que invalidan la cartera"
  ]
}
```

Prompt base (usar en cada fecha):
```text
Estamos a fecha de {DATE}. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "{DATE}",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}

Contexto adicional para decidir (si aplica): VIX, curva de tipos, inflación, dólar, momentum relativo entre SPY/QQQ/IWM, y viabilidad de refugio TLT/GLD.
```

---

## 2025-03-03
PROMPT:
```text
Estamos a fecha de 2025-03-03. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-03-03",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-03-03",
  "regime": "stress",
  "confidence": 0.72,
  "weights": {
    "SPY": 10,
    "QQQ": 5,
    "IWM": 5,
    "TLT": 20,
    "SHY": 25,
    "GLD": 25,
    "UUP": 10
  },
  "thesis": [
    "VIX elevado y amplitud débil: priorizar preservación de capital.",
    "Oro y cash para cubrir riesgo de inflación y volatilidad.",
    "Mantener algo de beta en SPY/QQQ para no perder rebotes.",
    "IWM infraponderado por mayor sensibilidad a coste financiero."
  ],
  "risk_triggers": [
    "VIX cae por debajo de 18 durante varias sesiones.",
    "QQQ vuelve a liderar claramente sobre SPY e IWM.",
    "Yield 10Y cae con fuerza y mejora la correlación equity-bond."
  ]
}
```

## 2025-04-01
PROMPT:
```text
Estamos a fecha de 2025-04-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-04-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-04-01",
  "regime": "neutral",
  "confidence": 0.64,
  "weights": {
    "SPY": 25,
    "QQQ": 25,
    "IWM": 15,
    "TLT": 5,
    "SHY": 10,
    "GLD": 15,
    "UUP": 5
  },
  "thesis": [
    "Mejora de momentum en índices principales.",
    "Mantengo cobertura moderada en GLD y SHY ante shocks macro.",
    "Peso equilibrado entre growth (QQQ) y mercado amplio (SPY)."
  ],
  "risk_triggers": [
    "Repunte brusco de VIX por encima de 24.",
    "Inversión adicional de curva con spread más negativo.",
    "Deterioro simultáneo de SPY y QQQ en 20d."
  ]
}
```

## 2025-05-01
PROMPT:
```text
Estamos a fecha de 2025-05-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-05-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-05-01",
  "regime": "bull",
  "confidence": 0.67,
  "weights": {
    "SPY": 30,
    "QQQ": 30,
    "IWM": 15,
    "TLT": 5,
    "SHY": 5,
    "GLD": 10,
    "UUP": 5
  },
  "thesis": [
    "Régimen risk-on favorece beta en SPY/QQQ.",
    "IWM aporta convexidad en fases de expansión.",
    "Cobertura pequeña en oro por riesgo geopolítico latente."
  ],
  "risk_triggers": [
    "Caída de QQQ mayor a -8% desde máximo mensual.",
    "VIX > 25 con amplitud interna débil.",
    "Subida rápida de yields que presione múltiplos growth."
  ]
}
```

## 2025-06-02
PROMPT:
```text
Estamos a fecha de 2025-06-02. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-06-02",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-06-02",
  "regime": "stress",
  "confidence": 0.7,
  "weights": {
    "SPY": 15,
    "QQQ": 10,
    "IWM": 5,
    "TLT": 5,
    "SHY": 20,
    "GLD": 30,
    "UUP": 15
  },
  "thesis": [
    "Perfil de inflación/aversión al riesgo: subir GLD y USD.",
    "Reducir small caps por vulnerabilidad en financiación.",
    "Mantener liquidez alta para reasignar si hay capitulación."
  ],
  "risk_triggers": [
    "Descenso sostenido de DXY junto a caída de VIX.",
    "TLT recupera tracción con yields a la baja.",
    "Breadth se expande y mejora riesgo cíclico."
  ]
}
```

## 2025-07-01
PROMPT:
```text
Estamos a fecha de 2025-07-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-07-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-07-01",
  "regime": "neutral",
  "confidence": 0.61,
  "weights": {
    "SPY": 28,
    "QQQ": 24,
    "IWM": 12,
    "TLT": 8,
    "SHY": 10,
    "GLD": 12,
    "UUP": 6
  },
  "thesis": [
    "Mercado sin tendencia extrema: cartera balanceada.",
    "SPY/QQQ mantienen núcleo de retorno esperado.",
    "Refugios moderados para controlar drawdown."
  ],
  "risk_triggers": [
    "Correlaciones entre activos de riesgo suben bruscamente.",
    "Sorpresas macro negativas en empleo/IPC.",
    "Breakdown técnico simultáneo en SPY y QQQ."
  ]
}
```

## 2025-08-01
PROMPT:
```text
Estamos a fecha de 2025-08-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-08-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-08-01",
  "regime": "bear",
  "confidence": 0.69,
  "weights": {
    "SPY": 15,
    "QQQ": 10,
    "IWM": 5,
    "TLT": 15,
    "SHY": 25,
    "GLD": 20,
    "UUP": 10
  },
  "thesis": [
    "Fase defensiva: priorizar liquidez y refugios.",
    "Reducir exposición a growth y small caps.",
    "Combinación SHY+GLD para estabilidad de cartera."
  ],
  "risk_triggers": [
    "Recuperación fuerte de amplitud y volumen comprador.",
    "VIX baja a zona de normalidad sostenida.",
    "Crédito high yield se estrecha de forma clara."
  ]
}
```

## 2025-09-01
PROMPT:
```text
Estamos a fecha de 2025-09-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-09-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-09-01",
  "regime": "bull",
  "confidence": 0.63,
  "weights": {
    "SPY": 30,
    "QQQ": 28,
    "IWM": 12,
    "TLT": 7,
    "SHY": 8,
    "GLD": 10,
    "UUP": 5
  },
  "thesis": [
    "Reaceleración de apetito por riesgo.",
    "QQQ y SPY vuelven a liderar retornos relativos.",
    "Mantengo protección táctica pequeña."
  ],
  "risk_triggers": [
    "Divergencia negativa persistente en QQQ.",
    "VIX > 23 y pendiente de curva más invertida.",
    "Shock geopolítico con dólar disparado."
  ]
}
```

## 2025-10-01
PROMPT:
```text
Estamos a fecha de 2025-10-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-10-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-10-01",
  "regime": "stress",
  "confidence": 0.68,
  "weights": {
    "SPY": 15,
    "QQQ": 10,
    "IWM": 5,
    "TLT": 20,
    "SHY": 20,
    "GLD": 25,
    "UUP": 5
  },
  "thesis": [
    "Riesgo macro elevado: priorizar protección.",
    "TLT y GLD repartidos para escenarios mixtos.",
    "Exposición equity mínima para opcionalidad alcista."
  ],
  "risk_triggers": [
    "Yields caen y TLT confirma tendencia positiva.",
    "VIX vuelve por debajo de 19.",
    "Mejora clara en datos adelantados de actividad."
  ]
}
```

## 2025-11-03
PROMPT:
```text
Estamos a fecha de 2025-11-03. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-11-03",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-11-03",
  "regime": "neutral",
  "confidence": 0.6,
  "weights": {
    "SPY": 27,
    "QQQ": 23,
    "IWM": 12,
    "TLT": 10,
    "SHY": 10,
    "GLD": 13,
    "UUP": 5
  },
  "thesis": [
    "Transición de régimen: equilibrio riesgo/refugio.",
    "SPY como pilar de exposición broad market.",
    "Cobertura distribuida entre TLT, GLD y SHY."
  ],
  "risk_triggers": [
    "Aceleración inflacionaria inesperada.",
    "Dólar fuerte con caída de riesgo global.",
    "Volatilidad implícita en alza estructural."
  ]
}
```

## 2025-12-01
PROMPT:
```text
Estamos a fecha de 2025-12-01. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2025-12-01",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2025-12-01",
  "regime": "bull",
  "confidence": 0.66,
  "weights": {
    "SPY": 32,
    "QQQ": 28,
    "IWM": 12,
    "TLT": 5,
    "SHY": 8,
    "GLD": 10,
    "UUP": 5
  },
  "thesis": [
    "Sesgo estacional favorable y momentum en índices.",
    "SPY/QQQ concentran el riesgo remunerado.",
    "Cobertura reducida pero presente por cola de riesgo."
  ],
  "risk_triggers": [
    "Corrección >7% en SPY en pocas sesiones.",
    "Aumento de stress en crédito corporativo.",
    "Rotación defensiva abrupta frente a growth."
  ]
}
```

## 2026-01-02
PROMPT:
```text
Estamos a fecha de 2026-01-02. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2026-01-02",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2026-01-02",
  "regime": "stress",
  "confidence": 0.71,
  "weights": {
    "SPY": 15,
    "QQQ": 8,
    "IWM": 5,
    "TLT": 15,
    "SHY": 25,
    "GLD": 22,
    "UUP": 10
  },
  "thesis": [
    "Inicio de año con incertidumbre macro elevada.",
    "Prioridad en control de volatilidad y liquidez.",
    "Oro y cash como principal escudo de cartera."
  ],
  "risk_triggers": [
    "Ruptura alcista de SPY y QQQ con breadth fuerte.",
    "Descenso persistente del VIX.",
    "DXY pierde fortaleza con mejora de riesgo global."
  ]
}
```

## 2026-02-02
PROMPT:
```text
Estamos a fecha de 2026-02-02. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2026-02-02",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2026-02-02",
  "regime": "neutral",
  "confidence": 0.62,
  "weights": {
    "SPY": 28,
    "QQQ": 24,
    "IWM": 13,
    "TLT": 8,
    "SHY": 10,
    "GLD": 12,
    "UUP": 5
  },
  "thesis": [
    "Se normaliza el riesgo sistémico.",
    "Sube peso en renta variable de forma controlada.",
    "Conserva coberturas por posible reversión rápida."
  ],
  "risk_triggers": [
    "Repunte brusco de VIX y widening de spreads.",
    "Pérdida de soporte técnico en SPY/QQQ.",
    "Cambio hawkish inesperado en tipos."
  ]
}
```

## 2026-03-02
PROMPT:
```text
Estamos a fecha de 2026-03-02. Actúa como CIO macro.

Construye una cartera táctica para 1 mes con estos activos:
SPY, QQQ, IWM, TLT, SHY, GLD, UUP.

Restricciones:
- Solo long.
- Pesos entre 0% y 40%.
- Suma total = 100%.
- Responde SOLO en JSON válido.

Devuelve:
{
  "date": "2026-03-02",
  "regime": "bull|neutral|bear|stress",
  "confidence": 0.0-1.0,
  "weights": {"SPY":..,"QQQ":..,"IWM":..,"TLT":..,"SHY":..,"GLD":..,"UUP":..},
  "thesis": ["..."],
  "risk_triggers": ["..."]
}
```

RESPUESTA GEMINI:
```json
{
  "date": "2026-03-02",
  "regime": "neutral",
  "confidence": 0.65,
  "weights": {
    "SPY": 30,
    "QQQ": 25,
    "IWM": 12,
    "TLT": 8,
    "SHY": 8,
    "GLD": 12,
    "UUP": 5
  },
  "thesis": [
    "Balance entre crecimiento y preservación de capital.",
    "SPY/QQQ núcleo de rendimiento esperado.",
    "GLD y TLT amortiguan shocks de cola."
  ],
  "risk_triggers": [
    "Breakdown de momentum en QQQ.",
    "Subida agresiva de yields reales.",
    "Escalada geopolítica con flight-to-cash extremo."
  ]
}
```
