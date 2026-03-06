# 📊 Dashboard — Guía de Uso e Intuiciones

## Quick Start

```bash
PYTHONPATH=. python dashboard/api.py
# Abrir http://localhost:8050
```

## Flujo de Trabajo Diario

1. **Abrir el dashboard** → revisar las alertas de salud (Health) primero
2. **Pulsar "📋 Copiar Prompt"** → se copia el prompt completo con todos los datos
3. **Pegar en Gemini/ChatGPT** → obtener el análisis del LLM
4. **Comparar** las recomendaciones del LLM con los datos del dashboard

## Secciones del Prompt (15 bloques)

El prompt que se genera contiene estos bloques de datos, en este orden:

| # | Sección | Qué contiene | Cómo interpretarlo |
|---|---------|-------------|-------------------|
| 1 | **CONTEXT** | Descripción del sistema | Le dice al LLM qué herramientas tienes |
| 2 | **MERCADO** | SPY, QQQ, IWM, TLT, GLD, VIX | Panorama general: ¿risk-on o risk-off? |
| 3 | **CURVA DE TIPOS** | Yields + spreads 10Y-2Y, 10Y-3M | Spread > 0 = curva normal. Si se invierte → recesión ~12 meses |
| 4 | **FRED MACRO** | 17 indicadores Supabase | IPC, desempleo, PMI, HY spread, petróleo, oro, cobre |
| 5 | **TITULARES** | 10 headlines de Google News | Contexto cualitativo para el LLM |
| 6 | **SENTIMIENTO** | Fear & Greed + ETF flows | F&G < 25 = miedo extremo (oportunidad contrarian) |
| 7 | **RÉGIMEN HMM** | Bull/Neutral/Bear + transiciones | Probabilidad de cambio a 1 semana |
| 8 | **O-U** | (Desactivado) | Solo aparece si `include_system=True` |
| 9 | **STOCKS** | Top 5 por sector GICS | Ranking fundamental (EPS, revenue, ROIC) |
| 10 | **MARKOWITZ** | Pesos óptimos max-Sharpe | Los pesos que minimizan riesgo para el mismo retorno |
| 11 | **RIESGO** | CVaR, VaR, correlaciones | El peor día probable y alertas de concentración |
| 12 | **FACTOR TIMING** | Reranking por régimen | En bear: calidad > momentum. En bull: momentum > todo |
| 13 | **PREGUNTAS** | 11 preguntas dirigidas | Fuerzan al LLM a cubrir todos los ángulos |

## Intuiciones Clave

### 🔴 Régimen HMM
- Si dice **BEAR con >90%**, el mercado está en modo de riesgo. El modelo NO está diseñado para predecir cuándo cambia, sino para decirte **dónde estás ahora**.
- La **transición a 1 semana** es la clave: si P(bear→bull) sube de 10% a 25%, puede ser señal temprana de giro.
- **Duración media** del bear (~23 días) te da perspectiva: si llevas 20 días en bear, la probabilidad estadística de salir aumenta.

### 📊 Markowitz
- Los pesos de Markowitz son **matemáticamente óptimos pero no perfectos**. Si pone 40% en un solo stock, es porque históricamente tuvo el mejor ratio retorno/riesgo, pero no sabe si mañana pierde un contrato.
- **Sharpe > 1.5** es excelente. **> 2.0** es raro y probablemente sobreajustado.
- Compara los pesos de Markowitz con los que recomienda el LLM. Si difieren mucho, pregunta por qué.

### ⚠️ Correlaciones
- **High correlations (>0.7)**: si dos acciones están muy correladas, tener ambas es como apostar doble. Markowitz ya intenta reducir esto, pero si ves AAPL-NVDA a 0.85, el riesgo real es mayor de lo que parece.
- **Correlation changes**: es la alerta más importante. Si AAPL-V pasa de 0.08 a 0.42 en 30 días, significa que **todo se está moviendo junto** → las correlaciones se rompen en crisis. Es una señal de estrés sistémico.

### 🔄 Factor Timing
- En **BEAR**: el sistema baja el peso del momentum (las acciones que más subieron caen más fuerte) y sube calidad + valor.
- **Trampa de valor**: un stock barato con revenue negativo NO es value, es una trampa. El sistema filtra esto (rev > 0 para sumar puntos de value).
- En **BULL**: confía en el momentum. Las acciones que suben tienden a seguir subiendo (efecto Jegadeesh-Titman).

### 📉 CVaR (la métrica más importante)
- **CVaR 95% = -2.66%** significa: "En el peor 5% de días, espera perder AL MENOS un 2.66% en promedio". Con un portfolio de €10.000, eso es -€266 en un mal día.
- Si el CVaR pasa de -2% a -4%, algo cambió — revisa correlaciones y régimen.
- **Max Drawdown -14.6%** es la peor caída pico-a-valle del último año. Es tu escenario "pesadilla realista".

## Reglas de Oro

1. **Nunca ignores el CVaR**. Es tu seguro de vida.
2. **El HMM no predice, clasifica**. No dice cuándo cambia el mercado, dice dónde estás.
3. **Si todas las correlaciones suben, reduce exposición**. Es la señal de contagio más fiable.
4. **El LLM es tu segundo cerebro**, no tu jefe. Si dice "compra BA" pero el sistema marca -287% EPS y factor_timing la pone última, confía en los datos.
5. **Revisa el prompt diariamente**. Las condiciones cambian y el briefing se actualiza con datos frescos.
