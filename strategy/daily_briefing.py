"""
DAILY BRIEFING — Market Context Report
=========================================
Generates a comprehensive market report with all available indicators,
TOP 20 momentum stocks, and contextual analysis. Designed to be read
by a human or fed to an LLM for deeper interpretation.

Usage:
    python strategy/daily_briefing.py
    python strategy/daily_briefing.py --output report.md
    python strategy/daily_briefing.py --telegram
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── CONFIGURATION ──
TOP_N = 20
VIX_GATE = 19
VIX_MA_WINDOW = 20
REFUGE_ALLOC = {'TLT': 0.5, 'GLD': 0.5}


def fetch_market_data():
    """Fetch all market data from yfinance."""
    print("  Fetching market data...", flush=True)

    # Core instruments
    tickers = {
        'SPY':   'S&P 500',
        'QQQ':   'Nasdaq 100',
        'IWM':   'Russell 2000 (small cap)',
        'TLT':   'Treasuries 20+ años',
        'SHY':   'Treasuries 1-3 años (cash proxy)',
        'GLD':   'Gold',
        'UUP':   'US Dollar Index',
        '^VIX':  'VIX (volatilidad)',
        '^TNX':  'Yield 10Y',
        '^IRX':  'Yield 3 meses (T-Bill)',
        'DX-Y.NYB': 'DXY Dollar Index',
    }

    data = {}
    for tk, desc in tickers.items():
        try:
            hist = yf.Ticker(tk).history(period="6mo")
            if not hist.empty:
                last = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last
                ma20 = float(hist['Close'].tail(20).mean())
                ma50 = float(hist['Close'].tail(50).mean()) if len(hist) >= 50 else ma20
                chg_1d = (last / prev - 1) if prev != 0 else 0
                chg_5d = (last / float(hist['Close'].iloc[-6]) - 1) if len(hist) > 5 else 0
                chg_20d = (last / float(hist['Close'].iloc[-21]) - 1) if len(hist) > 20 else 0
                data[tk] = {
                    'desc': desc, 'price': last, 'prev': prev,
                    'ma20': ma20, 'ma50': ma50,
                    'chg_1d': chg_1d, 'chg_5d': chg_5d, 'chg_20d': chg_20d,
                }
        except Exception:
            pass

    return data


def fetch_yield_curve():
    """Fetch yield curve data."""
    print("  Fetching yield curve...", flush=True)
    curve = {}
    tickers = {'^IRX': '3M', '^FVX': '5Y', '^TNX': '10Y', '^TYX': '30Y'}
    for tk, label in tickers.items():
        try:
            h = yf.Ticker(tk).history(period="3mo")
            if not h.empty:
                curve[label] = {
                    'yield': float(h['Close'].iloc[-1]),
                    'chg_1w': float(h['Close'].iloc[-1] - h['Close'].iloc[-6]) if len(h) > 5 else 0,
                    'chg_1m': float(h['Close'].iloc[-1] - h['Close'].iloc[-22]) if len(h) > 21 else 0,
                }
        except Exception:
            pass
    return curve


def fetch_fred_macro():
    """Fetch key macro indicators from FRED via Supabase."""
    print("  Fetching FRED macro data...", flush=True)
    macro = {}
    try:
        from db.database_manager import DatabaseManager
        db = DatabaseManager()
        indicators = {
            'CPIAUCSL': ('CPI (Inflación)', 'Índice de precios al consumo. Si sube >3%YoY = inflación. TLT PELIGROSO en inflación alta.'),
            'UNRATE': ('Desempleo', 'Tasa de desempleo USA. <4% = economía fuerte. >5% = posible recesión. Fed baja tipos cuando sube.'),
            'FEDFUNDS': ('Fed Funds Rate', 'Tipo de interés de referencia. Si sube = Fed endureciendo = malo para growth. Si baja = Fed estimulando.'),
            'T10Y2Y': ('Spread 10Y-2Y', 'Diferencial yield curve. Negativo = inversión = recesión inminente. 100% track record desde 1970.'),
            'BAMLH0A0HYM2': ('High Yield Spread', 'Spread de bonos basura. <4% = calma. >6% = estrés crédito. >8% = pánico crédito (como 2008).'),
            'DTWEXBGS': ('Trade Weighted USD', 'Índice dólar ponderado por comercio. Fuerte = flight-to-safety. Débil = apetito por riesgo global.'),
            'DGS10': ('Yield 10Y (FRED)', 'Rendimiento bonos 10 años desde FRED. Referencia para hipotecas y valoraciones.'),
        }
        for ind_code, (name, explanation) in indicators.items():
            try:
                resp = db.client.table('macro_data').select('date,value').eq('indicator', ind_code).order('date', desc=True).limit(5).execute()
                if resp.data and len(resp.data) > 0:
                    latest = resp.data[0]
                    prev = resp.data[-1] if len(resp.data) > 1 else latest
                    macro[ind_code] = {
                        'name': name,
                        'explanation': explanation,
                        'value': float(latest['value']),
                        'date': latest['date'],
                        'prev_value': float(prev['value']),
                        'prev_date': prev['date'],
                    }
            except Exception:
                pass
    except Exception as e:
        print(f"    ⚠️ FRED not available: {e}", flush=True)
    return macro


def fetch_momentum_stocks():
    """Compute momentum scores for universe of stocks."""
    print("  Computing momentum scores...", flush=True)
    try:
        from ml.fundamental_momentum import load_quarterly_data, compute_momentum_features
        qdf, _ = load_quarterly_data()
        mom = compute_momentum_features(qdf)
        return mom.nlargest(TOP_N, 'momentum_score')
    except Exception as e:
        print(f"    ⚠️ Could not load from Supabase: {e}", flush=True)
        return None


def fetch_stock_prices(tickers):
    """Fetch recent price action for momentum stocks."""
    print("  Fetching stock prices...", flush=True)
    stock_data = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="3mo")
            if not h.empty and len(h) > 20:
                last = float(h['Close'].iloc[-1])
                stock_data[t] = {
                    'price': last,
                    'chg_1d': float(h['Close'].pct_change().iloc[-1]),
                    'chg_5d': float(last / h['Close'].iloc[-6] - 1) if len(h) > 5 else 0,
                    'chg_20d': float(last / h['Close'].iloc[-21] - 1) if len(h) > 20 else 0,
                    'from_high': float(last / h['Close'].max() - 1),
                    'vol_20d': float(h['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100),
                }
        except Exception:
            pass
    return stock_data


def classify_regime(market):
    """Classify current market regime based on indicators."""
    signals = {}

    # VIX analysis
    if '^VIX' in market:
        vix = market['^VIX']
        vix_level = vix['price']
        vix_vs_ma = vix['price'] / vix['ma20'] - 1

        if vix_level < 15:
            signals['vix'] = ('🟢 CALMA', f'VIX={vix_level:.1f}, mercado tranquilo. Buen momento para momentum.')
        elif vix_level < 20:
            signals['vix'] = ('🟡 ALERTA', f'VIX={vix_level:.1f}, elevándose. Vigilar pero no actuar.')
        elif vix_level < 30:
            signals['vix'] = ('🟠 ESTRÉS', f'VIX={vix_level:.1f}, mercado nervioso. Considerar refugio parcial.')
        else:
            signals['vix'] = ('🔴 PÁNICO', f'VIX={vix_level:.1f}, miedo extremo. Refugio total recomendado.')

        if vix_vs_ma > 0.15:
            signals['vix_spike'] = ('⚠️ SPIKE', f'VIX +{vix_vs_ma:.0%} sobre MA20. Pico de miedo — históricamente es momento de compra contrarian en 1-2 semanas.')
        elif vix_vs_ma < -0.15:
            signals['vix_complacency'] = ('⚠️ COMPLACENCIA', f'VIX -{abs(vix_vs_ma):.0%} bajo MA20. Mucha calma — cuidado con shock repentino.')

    # Yield curve analysis
    if '^TNX' in market and '^IRX' in market:
        y10 = market['^TNX']['price']
        y3m = market['^IRX']['price']
        spread = y10 - y3m

        if spread < -0.5:
            signals['curve'] = ('🔴 INVERTIDA', f'Spread 10Y-3M = {spread:+.2f}%. Curva invertida → señal clásica de recesión en 6-18 meses. Históricamente precede todas las recesiones desde 1970.')
        elif spread < 0:
            signals['curve'] = ('🟠 PLANA', f'Spread 10Y-3M = {spread:+.2f}%. Curva plana/ligeramente invertida. El mercado de bonos no espera crecimiento.')
        elif spread > 1.5:
            signals['curve'] = ('🟢 EMPINADA', f'Spread 10Y-3M = {spread:+.2f}%. Curva empinada → economía acelerando. Buen momento para riesgo.')
        else:
            signals['curve'] = ('🟡 NORMAL', f'Spread 10Y-3M = {spread:+.2f}%. Curva normal.')

    # Yields direction (inflation signal)
    if '^TNX' in market:
        y_chg = market['^TNX']['chg_20d']
        y_level = market['^TNX']['price']
        if y_chg > 0.3:
            signals['yields'] = ('⚠️ YIELDS ↑↑', f'Yield 10Y = {y_level:.2f}%, subió {y_chg:+.2f}% en 20d. Tipos SUBIENDO → TLT va a CAER. NO usar TLT como refugio. Precedente: 2022 (TLT -31%).')
        elif y_chg < -0.3:
            signals['yields'] = ('✅ YIELDS ↓↓', f'Yield 10Y = {y_level:.2f}%, bajó {y_chg:+.2f}% en 20d. Tipos BAJANDO → TLT SUBE. TLT es buen refugio ahora. Flight-to-safety activo.')
        else:
            signals['yields'] = ('🟡 YIELDS →', f'Yield 10Y = {y_level:.2f}%, estable ({y_chg:+.2f}% 20d). Sin dirección clara en tipos.')

    # Dollar (DXY)
    if 'UUP' in market:
        dxy = market['UUP']
        dxy_chg = dxy['chg_20d']
        if dxy_chg > 0.02:
            signals['dollar'] = ('💵 USD FUERTE', f'Dólar +{dxy_chg:.1%} en 20d. Flight-to-safety → inversores buscan seguridad en USD. Malo para mercados emergentes y commodities.')
        elif dxy_chg < -0.02:
            signals['dollar'] = ('💵 USD DÉBIL', f'Dólar {dxy_chg:.1%} en 20d. Debilidad del dólar → bueno para mercados emergentes, GLD, y exportadores USA.')
        else:
            signals['dollar'] = ('💵 USD NEUTRO', f'Dólar estable ({dxy_chg:.1%} 20d). Sin flujo claro de capital.')

    # Gold
    if 'GLD' in market:
        gld = market['GLD']
        gld_chg = gld['chg_20d']
        if gld_chg > 0.05:
            signals['gold'] = ('🥇 ORO ↑↑', f'GLD +{gld_chg:.1%} en 20d. Demanda de refugio ALTA. Inversores comprando protección. Señal de incertidumbre global.')
        elif gld_chg < -0.03:
            signals['gold'] = ('🥇 ORO ↓', f'GLD {gld_chg:.1%} en 20d. Sin demanda de refugio. Confianza en el mercado.')

    # Small vs Large cap
    if 'SPY' in market and 'IWM' in market:
        spy_chg = market['SPY']['chg_20d']
        iwm_chg = market['IWM']['chg_20d']
        rotation = iwm_chg - spy_chg
        if rotation > 0.03:
            signals['rotation'] = ('🔄 ROTACIÓN → SMALL', f'Small caps (+{iwm_chg:.1%}) baten large caps (+{spy_chg:.1%}). Apetito por riesgo, dinero fluyendo a empresas pequeñas. Señal bull.')
        elif rotation < -0.03:
            signals['rotation'] = ('🔄 ROTACIÓN → LARGE', f'Large caps (+{spy_chg:.1%}) baten small caps (+{iwm_chg:.1%}). "Fly to quality" — inversores prefieren empresas grandes/seguras. Señal de cautela.')

    # Bonds / TLT safety
    if 'TLT' in market:
        tlt = market['TLT']
        tlt_chg = tlt['chg_20d']
        if tlt_chg > 0.03:
            signals['bonds'] = ('🛡️ BONOS ↑', f'TLT +{tlt_chg:.1%} en 20d. Bonos subiendo → mercado buscando seguridad. TLT SÍ funciona como refugio ahora.')
        elif tlt_chg < -0.03:
            signals['bonds'] = ('⚠️ BONOS ↓', f'TLT {tlt_chg:.1%} en 20d. Bonos CAYENDO → NO USAR TLT como refugio. El "refugio seguro" está perdiendo dinero. Usar GLD o cash en su lugar.')

    return signals


def detect_crisis_type(signals):
    """Determine what TYPE of crisis we're in, if any."""
    crisis_text = ""

    vix_stressed = any('ESTRÉS' in s[0] or 'PÁNICO' in s[0] for s in signals.values())
    yields_up = any('YIELDS ↑' in s[0] for s in signals.values())
    yields_down = any('YIELDS ↓' in s[0] for s in signals.values())
    dollar_strong = any('USD FUERTE' in s[0] for s in signals.values())
    gold_up = any('ORO ↑' in s[0] for s in signals.values())
    bonds_down = any('BONOS ↓' in s[0] for s in signals.values())

    if not vix_stressed:
        crisis_text = "Sin estrés detectado. Régimen normal de mercado."
    elif vix_stressed and yields_down and dollar_strong:
        crisis_text = ("🚨 CRISIS DE PÁNICO (tipo COVID/GFC)\n"
                      "  Inversores huyendo a seguridad (USD, bonos)\n"
                      "  → TLT funciona como refugio ✅\n"
                      "  → GLD funciona como refugio ✅\n"
                      "  → Precedentes: marzo 2020, sept 2008")
    elif vix_stressed and yields_up and bonds_down:
        crisis_text = ("🚨 CRISIS DE INFLACIÓN (tipo 2022)\n"
                      "  Fed endureciendo, tipos subiendo, bonos caen\n"
                      "  → TLT es PELIGROSO como refugio ❌\n"
                      "  → GLD mixto (depende de tipos reales)\n"
                      "  → Mejor refugio: CASH (SHY, money market)\n"
                      "  → Precedentes: ene-oct 2022")
    elif vix_stressed and gold_up and not yields_down:
        crisis_text = ("🚨 CRISIS GEOPOLÍTICA (tipo tariffs/guerra)\n"
                      "  Incertidumbre geopolítica, no macro pura\n"
                      "  → GLD funciona como refugio ✅\n"
                      "  → TLT depende de hacia dónde van tipos\n"
                      "  → Precedentes: tariffs 2025, guerra Ucrania")
    elif vix_stressed:
        crisis_text = ("⚠️ ESTRÉS SIN CLASIFICAR\n"
                      "  Señales mixtas, no encaja en patrón claro\n"
                      "  → Cautela, reducir exposición\n"
                      "  → No ir 100% a un solo refugio")

    return crisis_text


def generate_briefing(market, yields, momentum_df, stock_prices, fred=None, output_file=None):
    """Generate the full briefing report."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = []

    def w(line="", end='\n'):
        if end == '':
            if lines:
                lines[-1] += line
            else:
                lines.append(line)
        else:
            lines.append(line)

    # ═══ HEADER ═══
    w(f"# 📊 DAILY MARKET BRIEFING — {now}")
    w()

    # ═══ VIX GATE STATUS ═══
    w("## 1. VIX GATE — Estado actual")
    w()
    if '^VIX' in market:
        vix = market['^VIX']
        gate_status = "🔴 REFUGIO" if vix['price'] >= VIX_GATE else "🟢 MOMENTUM"
        ma_status = "🔴 POR ENCIMA" if vix['price'] > vix['ma20'] else "🟢 POR DEBAJO"
        w(f"  VIX actual:     {vix['price']:.1f}")
        w(f"  VIX MA{VIX_MA_WINDOW}:       {vix['ma20']:.1f}")
        w(f"  VIX MA50:       {vix['ma50']:.1f}")
        w(f"  Cambio 1d:      {vix['chg_1d']:+.1%}")
        w(f"  Cambio 5d:      {vix['chg_5d']:+.1%}")
        w(f"  Cambio 20d:     {vix['chg_20d']:+.1%}")
        w(f"  Gate fijo (>{VIX_GATE}): {gate_status}")
        w(f"  Gate MA{VIX_MA_WINDOW}:      {ma_status}")
        w()
        w(f"  INTERPRETACIÓN: VIX mide el miedo del mercado.")
        w(f"  <15 = calma total, 15-20 = normal, 20-30 = nerviosismo, >30 = pánico.")
        w(f"  Si VIX sube RÁPIDO (>15% sobre MA20), suele ser momento de pánico")
        w(f"  pero históricamente también es oportunidad contrarian en 1-2 semanas.")
    w()

    # ═══ MACRO DASHBOARD ═══
    w("## 2. INDICADORES MACRO — Qué dicen los mercados")
    w()

    # Yields
    w("### Curva de tipos (yields)")
    if yields:
        for label, ydata in sorted(yields.items(), key=lambda x: {'3M':0,'5Y':1,'10Y':2,'30Y':3}.get(x[0],4)):
            direction = "↑" if ydata['chg_1w'] > 0.05 else ("↓" if ydata['chg_1w'] < -0.05 else "→")
            w(f"  {label}: {ydata['yield']:.2f}% {direction} (1w: {ydata['chg_1w']:+.2f}%, 1m: {ydata['chg_1m']:+.2f}%)")
        if '10Y' in yields and '3M' in yields:
            spread = yields['10Y']['yield'] - yields['3M']['yield']
            w(f"  Spread 10Y-3M: {spread:+.2f}%")
        w()
        w("  INTERPRETACIÓN:")
        w("  - Yields SUBIENDO = Fed endureciendo o inflación → malo para bonos y growth stocks")
        w("  - Yields BAJANDO = mercado espera recesión o recortes → bueno para bonos, malo para economía")
        w("  - Curva INVERTIDA (10Y < 3M) = señal de recesión más fiable que existe (100% track record)")
        w("  - Curva MUY EMPINADA = economía acelerando, buen momento para riesgo")
    w()

    # FRED macro indicators
    if fred:
        w("### Indicadores macro (FRED)")
        for code, fdata in fred.items():
            chg = fdata['value'] - fdata['prev_value']
            direction = '↑' if chg > 0 else ('↓' if chg < 0 else '→')
            w(f"  {fdata['name']}: {fdata['value']:.2f} {direction} (prev: {fdata['prev_value']:.2f}, {fdata['prev_date']})")
            w(f"    → {fdata['explanation']}")

            # Specific implications for refuge
            if code == 'CPIAUCSL' and fdata['value'] > 280:  # roughly >3% YoY
                w(f"    ⚠️ Inflación ALTA → bonos largos (TLT) son MALA idea como refugio")
            if code == 'BAMLH0A0HYM2' and fdata['value'] > 5:
                w(f"    🚨 Spread HY >{fdata['value']:.1f}% → estrés crediticio, posible contagio")
            if code == 'UNRATE' and fdata['value'] > 5:
                w(f"    ⚠️ Desempleo alto → Fed probablemente baja tipos → TLT SUBE (buen refugio)")
            if code == 'FEDFUNDS':
                if chg > 0:
                    w(f"    ⚠️ Fed SUBIENDO tipos → presión sobre growth stocks y TLT")
                elif chg < 0:
                    w(f"    ✅ Fed BAJANDO tipos → bueno para TLT y growth stocks")
            w()
        w()

    # Market instruments
    w("### Instrumentos clave")
    instrument_comments = {
        'SPY':  'Índice amplio USA. Si cae con VIX bajo → corrección sana. Si cae con VIX alto → posible crash.',
        'QQQ':  'Tech/growth. Más sensible a tipos que SPY. Si QQQ cae más que SPY → rotación fuera de tech.',
        'IWM':  'Small caps. Indicador de apetito por riesgo. Si IWM sube más que SPY → dinero fluyendo a riesgo.',
        'TLT':  'Bonos largo plazo. SOLO funciona como refugio cuando yields bajan. En crisis de inflación, TLT CAEA.',
        'SHY':  'Bonos corto plazo (proxy cash). Refugio universal, funciona SIEMPRE pero rinde poco.',
        'GLD':  'Oro. Refugio en geopolítica y devaluación. No tan bueno en tipos reales altos.',
        'UUP':  'Dólar. USD fuerte = flight-to-safety. USD débil = apetito por riesgo global.',
    }
    for tk in ['SPY', 'QQQ', 'IWM', 'TLT', 'SHY', 'GLD', 'UUP']:
        if tk in market:
            m = market[tk]
            w(f"  {tk} ({m['desc']}): ${m['price']:.2f}")
            w(f"    1d: {m['chg_1d']:+.1%}  5d: {m['chg_5d']:+.1%}  20d: {m['chg_20d']:+.1%}  vs MA20: {m['price']/m['ma20']-1:+.1%}")
            if tk in instrument_comments:
                w(f"    → {instrument_comments[tk]}")
            w()

    # ═══ REGIME CLASSIFICATION ═══
    w("## 3. DIAGNÓSTICO — Tipo de régimen actual")
    w()

    signals = classify_regime(market)
    for key, (label, explanation) in signals.items():
        w(f"  {label}")
        w(f"    {explanation}")
        w()

    crisis = detect_crisis_type(signals)
    w(f"### Clasificación de crisis:")
    w(f"  {crisis}")
    w()

    # ═══ REFUGE RECOMMENDATION ═══
    w("## 4. REFUGIO RECOMENDADO")
    w()

    # Determine best refuge based on signals
    yields_up = any('YIELDS ↑' in s[0] for s in signals.values())
    bonds_down = any('BONOS ↓' in s[0] for s in signals.values())
    gold_up = any('ORO ↑' in s[0] for s in signals.values())
    vix_stressed = any('ESTRÉS' in s[0] or 'PÁNICO' in s[0] for s in signals.values())

    if not vix_stressed:
        w("  Sin estrés → no hace falta refugio. Mantener momentum.")
    elif yields_up or bonds_down:
        w("  ⚠️ YIELDS SUBIENDO → TLT ES PELIGROSO")
        w("  Refugio sugerido: 70% GLD + 30% SHY (cash)")
        w("  NO usar TLT hasta que yields empiecen a bajar.")
    elif gold_up:
        w("  Refugio sugerido: 60% GLD + 40% TLT")
    else:
        w("  Refugio estándar: 50% TLT + 50% GLD")
    w()

    # ═══ MOMENTUM PORTFOLIO ═══
    w("## 5. TOP 20 MOMENTUM — Ranking de stocks")
    w()

    if momentum_df is not None and len(momentum_df) > 0:
        # Available columns vary, use what we have
        cols_available = momentum_df.columns.tolist()

        w(f"  {'#':<3} {'Ticker':<8} {'Score':>6} ", end='')
        extra_cols = [c for c in ['rev_last_qoq', 'eps_growth_total', 'margin_delta', 'roic_trend'] if c in cols_available]
        for c in extra_cols:
            short = {'rev_last_qoq': 'RevQoQ', 'eps_growth_total': 'EPS Gr', 'margin_delta': 'MargΔ', 'roic_trend': 'ROIC'}
            w(f"{short.get(c, c):>8} ", end='')
        if stock_prices:
            w(f"{'Price':>8} {'1d':>7} {'5d':>7} {'20d':>7} {'Hi%':>7} {'Vol%':>6}", end='')
        w()

        w(f"  {'─'*3} {'─'*8} {'─'*6} ", end='')
        for c in extra_cols:
            w(f"{'─'*8} ", end='')
        if stock_prices:
            w(f"{'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6}", end='')
        w()

        for i, (_, row) in enumerate(momentum_df.iterrows()):
            t = row['ticker']
            s = row['momentum_score']
            line = f"  {i+1:<3} {t:<8} {s:>5.0f} "

            for c in extra_cols:
                val = row.get(c, 0)
                if 'qoq' in c or 'growth' in c:
                    line += f"  {val:>+6.1%} "
                else:
                    line += f"  {val:>+6.3f} "

            if t in stock_prices:
                sp = stock_prices[t]
                line += f"  ${sp['price']:>6.1f} {sp['chg_1d']:>+6.1%} {sp['chg_5d']:>+6.1%} {sp['chg_20d']:>+6.1%} {sp['from_high']:>+6.1%} {sp['vol_20d']:>5.0f}%"

            w(line)

        w()
        w("  INTERPRETACIÓN:")
        w("  - Score 4-5: fundamentales mejorando fuerte (revenue + EPS + márgenes + FCF + deuda)")
        w("  - Score 2-3: mejora moderada, vigilar tendencia")
        w("  - Score 0-1: fundamentales estancados o deteriorándose")
        w("  - RevQoQ: crecimiento de ingresos trimestre a trimestre")
        w("  - EPS Gr: crecimiento de beneficios por acción (total sobre el periodo)")
        w("  - MargΔ: cambio en margen operativo (positivo = empresa más eficiente)")
        w("  - ROIC: tendencia del retorno sobre capital invertido")
        w("  - Vol%: volatilidad anualizada — >40% = muy volátil, <20% = estable")
        w("  - Hi%: distancia al máximo de 3 meses — cerca de 0% = en máximos")
    else:
        w("  ⚠️ No se pudieron cargar scores de momentum (Supabase no disponible)")
    w()

    # ═══ DECISION MATRIX ═══
    w("## 6. ACCIÓN SUGERIDA")
    w()
    w("  Según las reglas mecánicas:")
    if '^VIX' in market:
        vix = market['^VIX']
        if vix['price'] < VIX_GATE:
            w(f"  → 🟢 MOMENTUM: VIX={vix['price']:.1f} < {VIX_GATE}. Mantener cartera de momentum.")
        else:
            w(f"  → 🔴 REFUGIO: VIX={vix['price']:.1f} ≥ {VIX_GATE}. Mover a activos defensivos.")
            if yields_up or bonds_down:
                w(f"  → ⚠️ PERO yields subiendo, NO TLT. Usar GLD + cash.")
    w()
    w("  PERO recuerda: estas reglas son simplistas.")
    w("  Usa el contexto de las secciones 2-4 para decidir con juicio.")
    w()

    # ═══ LLM PROMPT ═══
    w("## 7. PROMPT PARA LLM")
    w("```")
    w("Soy un inversor retail con la siguiente cartera y datos de mercado.")
    w("Necesito tu análisis experto con sentido común, no reglas mecánicas.")
    w()
    w("DATOS DE HOY:")
    if '^VIX' in market:
        w(f"  VIX: {market['^VIX']['price']:.1f} (MA20: {market['^VIX']['ma20']:.1f})")
    if '^TNX' in market:
        w(f"  Yield 10Y: {market['^TNX']['price']:.2f}% ({market['^TNX']['chg_20d']:+.2f}% 20d)")
    if 'TLT' in market:
        w(f"  TLT (bonos): {market['TLT']['chg_20d']:+.1%} 20d")
    if 'GLD' in market:
        w(f"  GLD (oro): {market['GLD']['chg_20d']:+.1%} 20d")
    if 'SPY' in market:
        w(f"  SPY: {market['SPY']['chg_20d']:+.1%} 20d")
    w()
    w(f"MI DIAGNÓSTICO: {crisis}")
    w()
    if momentum_df is not None:
        top5 = momentum_df.head(5)['ticker'].tolist()
        w(f"MI CARTERA: {top5} (top 5 momentum por fundamentales)")
    w()
    w("PREGUNTAS:")
    w("1. ¿Mi diagnóstico es correcto? ¿Qué riesgos no estoy viendo?")
    w("2. ¿Es mejor refugiarme en TLT+GLD, solo GLD, cash, o mantener?")
    w("3. ¿Qué precedente histórico se parece más a la situación actual?")
    w("4. ¿Hay alguna empresa de mi cartera que sea especialmente vulnerable?")
    w("5. ¿Cuál es el escenario más peligroso que podría ocurrir esta semana?")
    w("```")
    w()

    report = '\n'.join(lines)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n  Report saved to {output_file}", flush=True)

    return report


def main():
    parser = argparse.ArgumentParser(description='Daily Market Briefing')
    parser.add_argument('--output', '-o', help='Save report to file')
    parser.add_argument('--telegram', action='store_true', help='Send via Telegram')
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════", flush=True)
    print("  GENERATING DAILY BRIEFING", flush=True)
    print("═══════════════════════════════════════════════════", flush=True)

    market = fetch_market_data()
    yields = fetch_yield_curve()
    fred = fetch_fred_macro()
    momentum = fetch_momentum_stocks()

    stock_tickers = momentum['ticker'].tolist() if momentum is not None else []
    stock_prices = fetch_stock_prices(stock_tickers) if stock_tickers else {}

    report = generate_briefing(market, yields, momentum, stock_prices, fred, args.output)

    print(report)

    if args.telegram:
        from strategy.daily_signal import send_telegram
        # Telegram has 4096 char limit, send summary only
        summary = report[:4000] + "\n\n[Truncado — ver informe completo]"
        if send_telegram(summary):
            print("\n✅ Sent to Telegram")


if __name__ == '__main__':
    main()
