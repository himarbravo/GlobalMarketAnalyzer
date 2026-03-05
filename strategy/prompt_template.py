"""
PROMPT TEMPLATE — Modular LLM prompt builder
===============================================
Generates a structured prompt for Gemini/ChatGPT with all available
system data. Each section can be toggled on/off.

Usage:
    from strategy.prompt_template import build_prompt
    prompt = build_prompt(snapshot)
    # Copy prompt to Gemini
"""

from strategy.glossary import CRISIS_TYPES


# ── SYSTEM DESCRIPTION ──
SYSTEM_CONTEXT = """Eres un analista de mercados experto y consejero de inversión.
Estoy usando un sistema cuantitativo propio que combina:

1. **Momentum fundamental sector-neutral**: ranking trimestral por sector GICS
   (revenue QoQ, EPS growth, ROIC trend). Top 5 acciones por sector.
2. **Macro y curva de tipos**: yields reales, spread 10Y-2Y/3M, datos FRED
   (IPC, desempleo, PMI, HY spread, petróleo, oro, cobre).
3. **Valoración macro**: CAPE Shiller y Indicador Buffett como alertas.
4. **Sentimiento**: Fear & Greed Index + flujos ETF (SPY/QQQ/TLT/GLD/IWM).
5. **Titulares recientes**: headlines de mercado para contexto cualitativo.
6. **Clasificación de régimen**: basada en VIX, spreads y pendiente de curva.

Necesito tu CRITERIO HUMANO — no reglas mecánicas.
Quiero que interpretes estos datos como lo haría un gestor experimentado
que entiende las sutilezas que un modelo cuantitativo no captura.

IMPORTANTE: Mi cartera está diversificada por sector (top 5 por cada sector GICS).
Analiza riesgos y oportunidades tanto a nivel individual como sectorial."""


def _section_market(data):
    """Generate market data section."""
    lines = []
    lines.append("═══ MERCADO ═══")
    for tk in ['SPY', 'QQQ', 'IWM', 'TLT', 'SHY', 'GLD', 'UUP']:
        if tk in data.get('market', {}):
            m = data['market'][tk]
            lines.append(f"  {tk} ({m['desc']}): ${m['price']:.2f}, "
                        f"1d:{m['chg_1d']:+.1%}, 5d:{m['chg_5d']:+.1%}, "
                        f"20d:{m['chg_20d']:+.1%}, vs MA20:{m['price']/m['ma20']-1:+.1%}")

    vix = data.get('market', {}).get('^VIX')
    if vix:
        lines.append(f"  VIX: {vix['price']:.1f} (MA20:{vix['ma20']:.1f}, "
                    f"MA50:{vix['ma50']:.1f}, 1d:{vix['chg_1d']:+.1%}, "
                    f"5d:{vix['chg_5d']:+.1%})")
    return '\n'.join(lines)


def _section_yields(data):
    """Generate yield curve section."""
    lines = ["═══ CURVA DE TIPOS ═══"]
    yields = data.get('yields', {})
    order = {'3M': 0, '5Y': 1, '10Y': 2, '30Y': 3}
    for label, ydata in sorted(yields.items(), key=lambda x: order.get(x[0], 4)):
        lines.append(f"  {label}: {ydata['yield']:.2f}% "
                    f"(1w:{ydata['chg_1w']:+.2f}%, 1m:{ydata['chg_1m']:+.2f}%)")
    if '10Y' in yields and '3M' in yields:
        spread = yields['10Y']['yield'] - yields['3M']['yield']
        lines.append(f"  Spread 10Y-3M: {spread:+.2f}%")
    return '\n'.join(lines)


def _section_fred(data):
    """Generate FRED macro section."""
    lines = ["═══ INDICADORES MACRO (FRED) ═══"]
    for code, fdata in data.get('fred', {}).items():
        chg = fdata['value'] - fdata['prev_value']
        arrow = '↑' if chg > 0 else ('↓' if chg < 0 else '→')
        lines.append(f"  {fdata['name']}: {fdata['value']:.2f} {arrow} "
                    f"(prev: {fdata['prev_value']:.2f}, {fdata['prev_date']})")
    return '\n'.join(lines)


def _section_system(data):
    """Generate O-U system analytics section (if available)."""
    sa = data.get('system_analytics', {})
    if not sa or 'z_scores' not in sa:
        return ""  # O-U system disabled or empty, skip section

    lines = ["═══ ANÁLISIS DEL SISTEMA O-U (experimental) ═══"]
    lines.append("NOTA: Estos datos son experimentales, interpretar con cautela.")
    lines.append("")

    # Z-scores
    zs = sa.get('z_scores', {})
    if zs:
        sorted_z = sorted(zs.items(), key=lambda x: x[1])
        lines.append("Z-scores (desviación vs equilibrio teórico):")
        lines.append("  Más infravalorados:")
        for tk, z in sorted_z[:5]:
            lines.append(f"    {tk}: z={z:+.3f}")
        lines.append("  Más sobrevalorados:")
        for tk, z in sorted_z[-5:]:
            lines.append(f"    {tk}: z={z:+.3f}")

    # Refuge signal
    if 'refuge_signal' in sa:
        rs = sa['refuge_signal']
        lines.append(f"Señal de refugio: {rs:+.3f} (negativo=equity, positivo=refugio)")

    # Calibrated params
    if 's' in sa:
        lines.append(f"Exponente fraccional (s): {sa['s']:.3f}")
    if 'gamma' in sa:
        lines.append(f"Inercia (γ): {sa['gamma']:.1f}")

    return '\n'.join(lines)


def _section_stocks(data):
    """Generate sector-grouped momentum stocks section."""
    import pandas as pd
    from collections import defaultdict

    lines = ["═══ MI CARTERA: TOP 5 POR SECTOR (sector-neutral) ═══"]
    stocks = data.get('stocks', [])
    sa = data.get('system_analytics', {})
    z_scores = sa.get('z_scores', {})

    # Group by sector
    by_sector = defaultdict(list)
    for s in stocks:
        by_sector[s.get('sector', 'Unknown')].append(s)

    for sector in sorted(by_sector.keys()):
        sector_stocks = by_sector[sector]
        lines.append(f"\n── {sector} ({len(sector_stocks)}) ──")

        for i, s in enumerate(sector_stocks[:5]):
            line = f"  {i+1}. {s['ticker']} (score:{s.get('score',0):.0f}/5"
            if 'rev_qoq' in s and s['rev_qoq']:
                line += f", rev:{s['rev_qoq']:+.0%}"
            if 'eps_growth' in s and s['eps_growth'] is not None and pd.notna(s['eps_growth']):
                line += f", EPS:{s['eps_growth']:+.0%}"
            line += ")"

            if 'price' in s:
                line += f" → ${s['price']:.1f}"
            if 'chg_20d' in s and s['chg_20d'] is not None:
                line += f", 20d:{s['chg_20d']:+.1%}"
            if 'vol_20d' in s and s['vol_20d'] is not None:
                line += f", vol:{s['vol_20d']:.0f}%"

            if s['ticker'] in z_scores:
                line += f", z={z_scores[s['ticker']]:+.2f}"

            lines.append(line)

    return '\n'.join(lines)


def _section_questions():
    """Generate targeted questions."""
    return """═══ PREGUNTAS ═══
1. ¿Mi diagnóstico automático es correcto? ¿Qué matices le faltan?
2. Dado los yields, la curva, y la inflación: ¿TLT es buen refugio HOY
   o debería usar GLD, cash, o mantenerme en equity?
3. ¿Qué precedente histórico se parece más a la combinación actual de señales?
4. De mi cartera, ¿alguna empresa es especialmente VULNERABLE al régimen actual?
   (dependencia de China, deuda alta, márgenes en riesgo, etc.)
5. Los z-scores muestran activos sobre/infravalorados según mi modelo O-U.
   ¿Son oportunidades reales o trampas de valor?
6. ¿Cuál es el escenario MÁS PELIGROSO que podría ocurrir esta semana?
7. Si tuvieras que elegir SOLO 5 acciones de mi cartera para las próximas
   4 semanas, ¿cuáles elegirías y por qué?
8. ¿Hay algún indicador que contradiga al resto? ¿Dónde está la señal más débil?
9. ¿Hay algún riesgo GEOPOLÍTICO actual (conflictos armados, sanciones,
   rutas comerciales, embargo, tensiones entre potencias) que pueda impactar
   mi cartera y que mis datos cuantitativos NO capturen?
10. ¿Qué sectores de mi cartera son más vulnerables a los titulares recientes?
11. Para tus TOP 5 acciones elegidas, dame: peso recomendado (% de cartera),
    target de precio a 4 semanas, y stop-loss sugerido."""


def _section_headlines(data):
    """Generate recent market headlines section."""
    headlines = data.get('headlines', [])
    if not headlines:
        return ""
    lines = ["═══ TITULARES RECIENTES ═══"]
    for h in headlines[:10]:
        src = f" ({h['source']})" if h.get('source') else ""
        lines.append(f"  • [{h.get('date', '')}] {h['title']}{src}")
    return '\n'.join(lines)


def _section_sentiment(data):
    """Generate sentiment & flows section."""
    lines = ["═══ SENTIMIENTO Y FLUJOS ═══"]

    # Fear & Greed
    fg = data.get('fear_greed')
    if fg:
        val = fg['value']
        clf = fg['classification']
        lines.append(f"  Fear & Greed Index: {val}/100 ({clf})")
        if val <= 25:
            lines.append("  → Miedo extremo: históricamente buen momento para comprar.")
        elif val >= 75:
            lines.append("  → Codicia extrema: históricamente momento de cautela.")

    # ETF Flows
    flows = data.get('etf_flows', {})
    if flows:
        lines.append("  Flujos ETF (vol 5d vs 20d + dirección precio):")
        for tk in ['SPY', 'QQQ', 'TLT', 'GLD', 'IWM']:
            f = flows.get(tk)
            if f:
                emoji = '🟢' if f['signal'] == 'INFLOW' else ('🔴' if f['signal'] == 'OUTFLOW' else '⚪')
                lines.append(
                    f"    {emoji} {tk} ({f['name']}): {f['signal']} "
                    f"(vol:{f['vol_ratio']:.2f}x, precio:{f['price_5d']:+.1%})")

    return '\n'.join(lines)


def build_prompt(snapshot, sections=None):
    """
    Build the complete LLM prompt from a snapshot dict.

    Args:
        snapshot: dict with keys 'market', 'yields', 'fred',
                  'headlines', 'system_analytics', 'stocks'
        sections: list of sections to include, or None for all.
                  Options: 'context', 'market', 'yields', 'fred',
                  'headlines', 'system', 'stocks', 'questions'

    Returns:
        str: complete prompt ready to paste into Gemini
    """
    if sections is None:
        sections = ['context', 'market', 'yields', 'fred',
                    'headlines', 'sentiment', 'system', 'stocks', 'questions']

    parts = []

    if 'context' in sections:
        parts.append(SYSTEM_CONTEXT)

    if 'market' in sections:
        parts.append(_section_market(snapshot))

    if 'yields' in sections:
        parts.append(_section_yields(snapshot))

    if 'fred' in sections:
        parts.append(_section_fred(snapshot))

    if 'headlines' in sections:
        hl = _section_headlines(snapshot)
        if hl:
            parts.append(hl)

    if 'sentiment' in sections:
        parts.append(_section_sentiment(snapshot))

    if 'system' in sections:
        parts.append(_section_system(snapshot))

    if 'stocks' in sections:
        parts.append(_section_stocks(snapshot))

    if 'questions' in sections:
        parts.append(_section_questions())

    return '\n\n'.join(parts)


def build_calendar_prompt(snapshot):
    """
    Build a focused prompt asking the LLM for upcoming key dates.
    Includes the user's tickers so the LLM can give earnings dates.
    """
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')

    # Extract tickers from portfolio
    tickers = [s['ticker'] for s in snapshot.get('stocks', [])]
    ticker_str = ', '.join(tickers[:20]) if tickers else 'AAPL, NVDA, AMZN, MSFT, META'

    return f"""Eres un analista de mercados. HOY es {today}.

Necesito un CALENDARIO de eventos que pueden mover el mercado
en las próximas 4 semanas. Para cada evento, dame:
- Fecha exacta (o estimada si no se ha confirmado)
- Qué es
- Impacto esperado en mi cartera
- Qué debería vigilar o hacer ANTES del evento

══ MI CARTERA ACTUAL ══
{ticker_str}

══ EVENTOS QUE NECESITO ══
1. REUNIONES DE LA FED (FOMC): ¿cuándo es la próxima? ¿Se espera
   cambio de tipos? ¿Qué implica para TLT y el mercado?

2. DATOS DE INFLACIÓN: ¿cuándo sale el próximo CPI/PCE?
   ¿El consenso espera subida o bajada?

3. DATOS DE EMPLEO: ¿cuándo sale el próximo Non-Farm Payrolls?
   ¿Cómo afecta a mi modelo si sorprende?

4. EARNINGS DE MIS EMPRESAS: De mi cartera ({ticker_str}),
   ¿cuáles reportan resultados en las próximas 4 semanas?
   Dame fecha, consenso EPS, y si es riesgo o oportunidad.

5. VENCIMIENTOS IMPORTANTES: ¿Hay vencimiento de opciones (OpEx),
   rebalanceo de índices, o "triple witching" próximamente?

6. OTROS EVENTOS MACRO: GDP, PMI, reuniones de otros bancos centrales
   (BCE, BoJ), eventos geopolíticos programados.

══ FORMATO DE RESPUESTA ══
Dame una tabla ordenada por fecha con columnas:
| Fecha | Evento | Impacto | Acción sugerida |

Después de la tabla, dime:
- ¿Cuál es el evento MÁS PELIGROSO para mi cartera?
- ¿Debería ajustar posiciones ANTES de algún evento concreto?
- ¿Hay alguna "ventana tranquila" donde es seguro mantener posiciones?"""
