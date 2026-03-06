"""
DATA PIPELINE — Fetches and structures all data for the dashboard
===================================================================
Central class that fetches market data, macro indicators, system
analytics, and momentum rankings into a single JSON-serializable snapshot.

Usage:
    from dashboard.data_pipeline import DashboardPipeline
    pipe = DashboardPipeline()
    snapshot = pipe.build_snapshot()
    prompt = pipe.build_llm_prompt()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ── CACHE ──
_cache = {}
_cache_ttl = 300  # 5 minutes


def _cached(key, ttl=_cache_ttl):
    """Simple in-memory cache decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            now = time.time()
            if key in _cache and now - _cache[key]['ts'] < ttl:
                return _cache[key]['data']
            result = func(*args, **kwargs)
            _cache[key] = {'data': result, 'ts': now}
            return result
        return wrapper
    return decorator


class DashboardPipeline:
    """Fetches all data sources and builds a unified snapshot."""

    MARKET_TICKERS = {
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
    }

    YIELD_TICKERS = {
        '^IRX': '3M', '^FVX': '5Y', '^TNX': '10Y', '^TYX': '30Y',
    }

    FRED_INDICATORS = {
        # column_name → display name
        'cpi_yoy':              'CPI (Inflación YoY)',
        'credit_spread_bbb':    'High Yield Spread (BBB)',
        'yield_spread_10y_2y':  'Spread 10Y-2Y',
        'oil_wti':              'Petróleo WTI',
        'dxy':                  'Dollar Index (DXY)',
        'gold':                 'Oro (Futures)',
        'copper':               'Cobre',
        'natural_gas':          'Gas Natural',
    }

    TOP_N = 20

    def __init__(self):
        self._snapshot = None

    # ─── Data Fetchers ───

    def fetch_market(self):
        """Fetch market prices from yfinance."""
        data = {}
        for tk, desc in self.MARKET_TICKERS.items():
            try:
                hist = yf.Ticker(tk).history(period="1y")
                if hist.empty or len(hist) < 2:
                    continue
                close = hist['Close']
                last = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                data[tk] = {
                    'desc': desc,
                    'price': last,
                    'prev': prev,
                    'ma20': float(close.tail(20).mean()),
                    'ma50': float(close.tail(50).mean()) if len(close) >= 50 else float(close.tail(20).mean()),
                    'chg_1d': (last / prev - 1) if prev else 0,
                    'chg_5d': (last / float(close.iloc[-6]) - 1) if len(close) > 5 else 0,
                    'chg_20d': (last / float(close.iloc[-21]) - 1) if len(close) > 20 else 0,
                    'history': [float(x) for x in close.tail(250).values],
                    'dates': [d.strftime('%Y-%m-%d') for d in close.tail(250).index],
                }
            except Exception:
                pass
        return data

    def fetch_yields(self):
        """Fetch yield curve data."""
        curve = {}
        for tk, label in self.YIELD_TICKERS.items():
            try:
                h = yf.Ticker(tk).history(period="3mo")
                if h.empty or len(h) < 2:
                    continue
                close = h['Close']
                curve[label] = {
                    'yield': float(close.iloc[-1]),
                    'chg_1w': float(close.iloc[-1] - close.iloc[-6]) if len(close) > 5 else 0,
                    'chg_1m': float(close.iloc[-1] - close.iloc[-22]) if len(close) > 21 else 0,
                    'history': [float(x) for x in close.tail(60).values],
                    'dates': [d.strftime('%Y-%m-%d') for d in close.tail(60).index],
                }
            except Exception:
                pass
        return curve

    def fetch_fred(self):
        """Fetch macro indicators from Supabase macro_indicators table."""
        macro = {}
        try:
            from db.database_manager import DatabaseManager
            db = DatabaseManager()

            # Get 2 most recent rows for comparison
            cols = ','.join(['date'] + list(self.FRED_INDICATORS.keys()))
            resp = (db.client.table('macro_indicators')
                    .select(cols)
                    .order('date', desc=True)
                    .limit(10)
                    .execute())

            if resp.data and len(resp.data) >= 2:
                # Find latest row with actual data for each indicator
                for col, name in self.FRED_INDICATORS.items():
                    latest_val = None
                    latest_date = None
                    prev_val = None
                    prev_date = None

                    # Walk rows to find latest non-null, then previous non-null
                    for row in resp.data:
                        val = row.get(col)
                        if val is not None:
                            if latest_val is None:
                                latest_val = float(val)
                                latest_date = row['date']
                            elif prev_val is None:
                                prev_val = float(val)
                                prev_date = row['date']
                                break

                    if latest_val is not None:
                        macro[col] = {
                            'name': name,
                            'value': latest_val,
                            'date': latest_date,
                            'prev_value': prev_val if prev_val is not None else latest_val,
                            'prev_date': prev_date if prev_date else latest_date,
                        }
        except Exception:
            pass
        return macro

    def fetch_headlines(self, n=10):
        """Fetch latest market headlines from Google News RSS."""
        headlines = []
        try:
            import feedparser
            feed = feedparser.parse(
                'https://news.google.com/rss/search?q=stock+market+S%26P+500+Fed+economy&hl=en-US&gl=US&ceid=US:en'
            )
            for entry in feed.entries[:n]:
                title = entry.title
                source = ''
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0].strip()
                    source = parts[1].strip()
                headlines.append({
                    'title': title,
                    'source': source,
                    'date': entry.get('published', '')[:16],
                })
        except Exception:
            pass
        return headlines

    def fetch_fear_greed(self):
        """Fetch Fear & Greed Index from alternative.me API."""
        try:
            import requests
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=8)
            if r.status_code == 200:
                d = r.json()
                entry = d.get('data', [{}])[0]
                return {
                    'value': int(entry.get('value', 0)),
                    'classification': entry.get('value_classification', 'Unknown'),
                }
        except Exception:
            pass
        return None

    def fetch_etf_flows(self):
        """Estimate ETF flows using volume ratio + price direction."""
        etfs = {
            'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'TLT': 'Bonos 20Y+',
            'GLD': 'Oro', 'IWM': 'Small Caps',
        }
        flows = {}
        for tk, name in etfs.items():
            try:
                h = yf.Ticker(tk).history(period='3mo')
                if len(h) > 20:
                    vol_20d = float(h['Volume'].tail(20).mean())
                    vol_5d = float(h['Volume'].tail(5).mean())
                    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
                    price_5d = float(h['Close'].iloc[-1] / h['Close'].iloc[-6] - 1)

                    if price_5d > 0 and vol_ratio > 1.1:
                        signal = 'INFLOW'
                    elif price_5d < 0 and vol_ratio > 1.1:
                        signal = 'OUTFLOW'
                    else:
                        signal = 'NEUTRAL'

                    flows[tk] = {
                        'name': name,
                        'vol_ratio': round(vol_ratio, 2),
                        'price_5d': round(price_5d, 4),
                        'signal': signal,
                    }
            except Exception:
                pass
        return flows

    def fetch_momentum_ranking(self):
        """Fetch momentum stocks with sector classification.
        Returns (stocks_flat, sectors_grouped):
        - stocks_flat: top 5 per sector = sector-neutral portfolio
        - sectors_grouped: dict of sector → list of stocks (up to TOP_N)
        """
        import json

        # Load sector map
        sector_map = {}
        sector_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sector_map.json')
        try:
            with open(sector_path) as f:
                sector_map = json.load(f)
        except Exception:
            pass

        all_stocks = []
        try:
            from ml.fundamental_momentum import load_quarterly_data, compute_momentum_features
            qdf, _ = load_quarterly_data()
            mom = compute_momentum_features(qdf)

            # Filter to classified stocks only (exclude ETFs)
            classified = mom[mom['ticker'].isin(sector_map.keys())].copy()
            classified = classified.sort_values('momentum_score', ascending=False)

            # Fetch prices for top stocks per sector
            seen_sectors = {}
            for _, row in classified.iterrows():
                ticker = row['ticker']
                sec_info = sector_map.get(ticker, {})
                sector = sec_info.get('sector', 'Unknown')

                # Track how many per sector
                seen_sectors[sector] = seen_sectors.get(sector, 0) + 1
                if seen_sectors[sector] > self.TOP_N:  # max 20 per sector
                    continue

                stock = {
                    'ticker': ticker,
                    'score': float(row['momentum_score']),
                    'rev_qoq': float(row.get('rev_last_qoq', 0)),
                    'eps_growth': float(row.get('eps_growth_total', 0)) if pd.notna(row.get('eps_growth_total')) else None,
                    'roic_trend': float(row.get('roic_trend', 0)),
                    'sector': sector,
                    'industry': sec_info.get('industry', ''),
                }
                try:
                    h = yf.Ticker(ticker).history(period="3mo")
                    if not h.empty and len(h) > 20:
                        last = float(h['Close'].iloc[-1])
                        stock.update({
                            'price': last,
                            'chg_1d': float(h['Close'].pct_change().iloc[-1]),
                            'chg_5d': float(last / h['Close'].iloc[-6] - 1) if len(h) > 5 else 0,
                            'chg_20d': float(last / h['Close'].iloc[-21] - 1) if len(h) > 20 else 0,
                            'from_high': float(last / h['Close'].max() - 1),
                            'vol_20d': float(h['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100),
                        })
                except Exception:
                    pass
                all_stocks.append(stock)
        except Exception:
            pass

        # Group by sector
        sectors_grouped = {}
        for s in all_stocks:
            sec = s['sector']
            if sec not in sectors_grouped:
                sectors_grouped[sec] = []
            sectors_grouped[sec].append(s)

        # Sort each sector by score
        for sec in sectors_grouped:
            sectors_grouped[sec].sort(key=lambda x: x['score'], reverse=True)

        # Flat list: top 5 per sector
        stocks_flat = []
        for sec, stocks in sorted(sectors_grouped.items()):
            stocks_flat.extend(stocks[:5])

        return stocks_flat, sectors_grouped

    def fetch_system_analytics(self):
        """Run core system analytics (O-U, graph, entropy)."""
        analytics = {}
        try:
            from db.database_manager import DatabaseManager
            from core.graph_builder import GraphBuilder
            from core.fundamental_filter import FundamentalFilter
            from core.heat_engine import HeatEngine

            db = DatabaseManager()
            gb = GraphBuilder(db)
            gb.load_data()
            gb.build()

            ff = FundamentalFilter(db)
            ff.compute_all()

            engine = HeatEngine(gb, ff)
            engine.solve(calibrate=True)

            # Z-scores
            if engine.z_scores is not None and len(engine.z_scores) > 0:
                latest_z = engine.z_scores[-1]
                analytics['z_scores'] = {
                    gb.tickers[i]: float(latest_z[i])
                    for i in range(min(len(gb.tickers), len(latest_z)))
                }

            # Refuge signal
            if hasattr(engine, 'refuge_signal') and engine.refuge_signal is not None:
                rs = engine.refuge_signal
                if isinstance(rs, (int, float, np.floating)):
                    analytics['refuge_signal'] = float(rs)
                elif hasattr(rs, '__len__') and len(rs) > 0:
                    analytics['refuge_signal'] = float(rs[-1])

            # Calibrated params
            analytics['s'] = float(gb.s) if hasattr(gb, 's') else 0.5
            analytics['gamma'] = float(engine.gamma) if hasattr(engine, 'gamma') else 1.0

            # Von Neumann entropy
            try:
                from core.reversibility import compute_von_neumann_entropy
                returns = gb.returns if hasattr(gb, 'returns') else None
                if returns is not None and len(returns) > 60:
                    corr = np.corrcoef(returns[-60:].T)
                    eigvals = np.linalg.eigvalsh(corr)
                    eigvals = eigvals[eigvals > 1e-10]
                    analytics['entropy'] = float(compute_von_neumann_entropy(eigvals))
            except Exception:
                pass

        except Exception as e:
            analytics['error'] = str(e)[:200]

        return analytics

    def classify_regime(self, market):
        """Classify current regime and crisis type."""
        regime = {'signals': {}, 'crisis_type': 'normal'}

        vix = market.get('^VIX', {})
        vix_level = vix.get('price', 0)

        # VIX level
        if vix_level < 15:
            regime['vix_state'] = 'calm'
        elif vix_level < 20:
            regime['vix_state'] = 'normal'
        elif vix_level < 30:
            regime['vix_state'] = 'stress'
        else:
            regime['vix_state'] = 'panic'

        # Refuge viability
        tlt = market.get('TLT', {})
        gld = market.get('GLD', {})
        tnx = market.get('^TNX', {})

        tlt_ok = tlt.get('chg_20d', 0) > -0.02
        gld_ok = gld.get('chg_20d', 0) > -0.02
        yields_up = tnx.get('chg_20d', 0) > 0.003

        regime['refuge'] = {
            'tlt_viable': tlt_ok and not yields_up,
            'gld_viable': gld_ok,
            'cash_only': yields_up and not gld_ok,
        }

        # Crisis type
        stressed = vix_level >= 20
        if not stressed:
            regime['crisis_type'] = 'none'
        elif yields_up and tlt.get('chg_20d', 0) < -0.02:
            regime['crisis_type'] = 'inflation'
        elif gld.get('chg_20d', 0) > 0.03 and not (tnx.get('chg_20d', 0) < -0.003):
            regime['crisis_type'] = 'geopolitical'
        elif tnx.get('chg_20d', 0) < -0.003:
            regime['crisis_type'] = 'panic'
        else:
            regime['crisis_type'] = 'unclassified'

        # Recommended refuge
        if regime['crisis_type'] == 'none':
            regime['recommended_refuge'] = 'Mantener momentum'
        elif regime['crisis_type'] == 'inflation':
            regime['recommended_refuge'] = '70% GLD + 30% SHY (cash). NO TLT.'
        elif regime['crisis_type'] == 'panic':
            regime['recommended_refuge'] = '50% TLT + 50% GLD'
        elif regime['crisis_type'] == 'geopolitical':
            regime['recommended_refuge'] = '60% GLD + 40% TLT'
        else:
            regime['recommended_refuge'] = '40% GLD + 30% TLT + 30% SHY'

        return regime

    def compute_model_health(self, market):
        """Compute model stability witnesses — overfitting/regime change alerts."""
        alerts = []
        vix_data = market.get('^VIX', {})
        vix_hist = vix_data.get('history', [])
        vix_now = vix_data.get('price', 0)
        spy_data = market.get('SPY', {})
        tlt_data = market.get('TLT', {})
        gld_data = market.get('GLD', {})

        # ── 1. VIX Distribution Shift (multi-window) ──
        # Historical VIX benchmarks by period:
        HIST_PERIODS = {
            '2004-07 (bull)':    {'mean': 14.0, 'std': 3.0},
            '2008-09 (crisis)':  {'mean': 32.0, 'std': 12.0},
            '2010-19 (expansión)': {'mean': 16.5, 'std': 5.0},
            '2020 (COVID)':      {'mean': 29.0, 'std': 10.0},
            '2022 (inflación)':  {'mean': 25.5, 'std': 5.0},
            '1990-2024 (global)': {'mean': 19.0, 'std': 7.0},
        }
        windows = [('20d', 20), ('60d', 60), ('120d', 120)]
        if len(vix_hist) >= 20:
            window_stats = []
            for wname, wlen in windows:
                if len(vix_hist) >= wlen:
                    subset = vix_hist[-wlen:]
                    wmean = float(np.mean(subset))
                    wstd = float(np.std(subset))
                    window_stats.append((wname, wlen, wmean, wstd))

            # Find closest historical period
            current_mean = window_stats[0][2]  # 20d mean
            best_match = min(HIST_PERIODS.items(),
                             key=lambda x: abs(x[1]['mean'] - current_mean))

            # Multi-window detail
            detail_lines = []
            for wname, wlen, wmean, wstd in window_stats:
                z = abs(wmean - 19.0) / 7.0
                detail_lines.append(f'{wname}: media={wmean:.1f}, std={wstd:.1f}, z={z:.1f}σ')
            detail_str = ' | '.join(detail_lines)

            # Overall severity based on longest available window
            longest = window_stats[-1]
            z_shift = abs(longest[2] - 19.0) / 7.0

            # Period similarity analysis
            period_match_str = f'Periodo más similar: {best_match[0]} (media={best_match[1]["mean"]})'
            crisis_periods = ['2008-09 (crisis)', '2020 (COVID)', '2022 (inflación)']

            if z_shift > 2.0:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'alert',
                    'title': f'🔴 VIX fuera de rango: {longest[2]:.1f} ({z_shift:.1f}σ)',
                    'detail': f'{detail_str}\n{period_match_str}',
                    'implication': f'Los umbrales VIX<15/20/30 pueden NO ser válidos. '
                                   f'{"Se parece a un periodo de crisis histórica." if best_match[0] in crisis_periods else "Régimen inusual sin precedente claro."}',
                    'methodology': f'Media VIX en {len(windows)} ventanas vs media histórica (19±7). '
                                   f'Periodos de referencia: bull (14), expansión (16.5), global (19), '
                                   f'inflación (25.5), COVID (29), crisis (32).',
                })
            elif z_shift > 1.0:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'warning',
                    'title': f'🟡 VIX elevado: {longest[2]:.1f} ({z_shift:.1f}σ)',
                    'detail': f'{detail_str}\n{period_match_str}',
                    'implication': 'Umbrales aún válidos pero en zona límite. Vigilar tendencia.',
                    'methodology': f'Comparación multi-ventana. Warning si >1σ.',
                })
            else:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'ok',
                    'title': f'🟢 Distribución VIX normal: {longest[2]:.1f}',
                    'detail': f'{detail_str}\n{period_match_str}',
                    'implication': 'Umbrales del backtest siguen siendo válidos.',
                    'methodology': 'Media VIX multi-ventana dentro de ±1σ de la media histórica.',
                })

        # ── 2. VIX Gate Frequency (multi-window) ──
        # In backtest, VIX>MA20 triggers ~37% of the time.
        if len(vix_hist) >= 20:
            ma20_vals = [np.mean(vix_hist[max(0,i-19):i+1]) for i in range(19, len(vix_hist))]
            vix_subset = vix_hist[19:]

            # Compute gate freq at multiple windows
            gate_windows = []
            for gname, glen in [('20d', 20), ('60d', 60), ('120d', 120), ('todo', len(vix_subset))]:
                if len(vix_subset) >= glen:
                    sub_v = vix_subset[-glen:]
                    sub_m = ma20_vals[-glen:]
                    ga = sum(1 for v, m in zip(sub_v, sub_m) if v > m)
                    gf = ga / glen
                    gate_windows.append((gname, glen, gf))

            freq_detail = ' | '.join(f'{n}: {f*100:.0f}%' for n, _, f in gate_windows)
            # Use longest window for severity
            freq = gate_windows[-1][2] if gate_windows else 0
            expected = 0.37

            # Trend: is frequency increasing? (short > long = worsening)
            trend_note = ''
            if len(gate_windows) >= 2:
                short_f = gate_windows[0][2]
                long_f = gate_windows[-1][2]
                if short_f > long_f + 0.15:
                    trend_note = ' ⬆️ TENDENCIA: frecuencia AUMENTANDO (corto plazo > largo plazo).'
                elif short_f < long_f - 0.15:
                    trend_note = ' ⬇️ MEJORANDO: frecuencia bajando (corto plazo < largo plazo).'

            if abs(freq - expected) > 0.20:
                alerts.append({
                    'id': 'gate_frequency',
                    'severity': 'alert',
                    'title': f'🔴 Gate VIX>MA20 anómalo: {freq*100:.0f}% (esperado ~37%)',
                    'detail': f'Frecuencia por ventana: {freq_detail}.{trend_note}',
                    'implication': f'{"Gate SIEMPRE activo → refugio permanente = sin alpha. El modelo asume que esto ocurre ~37%% del tiempo." if freq > 0.57 else "Gate RARA VEZ activo → sin protección. El mercado está demasiado tranquilo para que el gate funcione."}',
                    'methodology': f'Gate = VIX > MA20. Frecuencia medida en {len(gate_windows)} ventanas. '
                                   'Backtest walk-forward (2004-2026, 19 periodos): media 37%, rango 25-50%. '
                                   'Alerta si la ventana más larga difiere >20pp de 37%.',
                })
            elif abs(freq - expected) > 0.10:
                alerts.append({
                    'id': 'gate_frequency',
                    'severity': 'warning',
                    'title': f'🟡 Gate VIX>MA20 algo desviado: {freq*100:.0f}%',
                    'detail': f'Frecuencia por ventana: {freq_detail}.{trend_note}',
                    'implication': 'Dentro del rango histórico pero en el extremo. Vigilar.',
                    'methodology': 'Rango backtest: 25-50%. Warning si difiere 10-20pp de 37%.',
                })
            else:
                alerts.append({
                    'id': 'gate_frequency',
                    'severity': 'ok',
                    'title': f'🟢 Gate VIX>MA20: {freq*100:.0f}% (esperado ~37%)',
                    'detail': f'Frecuencia por ventana: {freq_detail}.{trend_note}',
                    'implication': 'El modelo opera en régimen similar al de entrenamiento.',
                    'methodology': f'Calculado en {len(gate_windows)} ventanas.',
                })

        # ── 3. SPY-TLT Correlation ──
        # Our strategy ASSUMES negative correlation. If positive, refuge is broken.
        spy_hist = spy_data.get('history', [])
        tlt_hist = tlt_data.get('history', [])
        if len(spy_hist) > 30 and len(tlt_hist) > 30:
            min_len = min(len(spy_hist), len(tlt_hist), 40)
            spy_ret = np.diff(np.log(spy_hist[-min_len:]))
            tlt_ret = np.diff(np.log(tlt_hist[-min_len:]))
            if len(spy_ret) == len(tlt_ret) and len(spy_ret) > 5:
                corr = float(np.corrcoef(spy_ret, tlt_ret)[0, 1])
                if corr > 0.3:
                    alerts.append({
                        'id': 'spy_tlt_corr',
                        'severity': 'alert',
                        'title': f'🔴 SPY-TLT correlación POSITIVA: {corr:.2f}',
                        'detail': f'SPY y TLT se mueven en la MISMA dirección (corr={corr:.2f}). '
                                  'Esto rompe la premisa de que TLT es refugio cuando SPY cae.',
                        'implication': 'CRÍTICO: La estrategia de refugio en TLT NO FUNCIONA en este régimen. '
                                       'Si SPY cae, TLT también caerá. Usar GLD o SHY en su lugar. '
                                       'Esto ocurrió en 2022 (inflación alta + Fed subiendo tipos).',
                        'methodology': f'Correlación log-returns de SPY y TLT en {min_len} días. '
                                       'Normal: -0.3 a +0.1. Alerta si >+0.30.',
                    })
                elif corr > 0.1:
                    alerts.append({
                        'id': 'spy_tlt_corr',
                        'severity': 'warning',
                        'title': f'🟡 SPY-TLT correlación débil: {corr:.2f}',
                        'detail': f'La correlación SPY-TLT ({corr:.2f}) está en zona ambigua. '
                                  'No es negativa fuerte como se espera.',
                        'implication': 'TLT puede funcionar como refugio parcial, pero con menos eficacia.',
                        'methodology': f'Correlación de log-returns en {min_len} días.',
                    })
                else:
                    alerts.append({
                        'id': 'spy_tlt_corr',
                        'severity': 'ok',
                        'title': f'🟢 SPY-TLT correlación negativa: {corr:.2f}',
                        'detail': f'SPY y TLT se mueven en direcciones opuestas (corr={corr:.2f}). '
                                  'TLT funciona como refugio.',
                        'implication': 'La premisa de refugio del modelo es válida.',
                        'methodology': f'Correlación log-returns en {min_len} días. Esperado: <+0.10.',
                    })

        # ── 4. Entropy Change (directional) ──
        # Computing entropy requires returns matrix, use SPY+TLT+GLD+QQQ+IWM as proxy
        proxy_hists = {k: market.get(k, {}).get('history', [])
                       for k in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'UUP']}
        min_hist = min((len(v) for v in proxy_hists.values() if len(v) > 0), default=0)
        if min_hist > 40:
            try:
                from core.reversibility import compute_von_neumann_entropy
                returns_mat = np.column_stack([
                    np.diff(np.log(v[-min_hist:])) for v in proxy_hists.values() if len(v) >= min_hist
                ])
                # Recent vs older entropy
                mid = len(returns_mat) // 2
                corr_old = np.corrcoef(returns_mat[:mid].T)
                corr_new = np.corrcoef(returns_mat[mid:].T)

                eig_old = np.linalg.eigvalsh(corr_old)
                eig_new = np.linalg.eigvalsh(corr_new)
                eig_old = eig_old[eig_old > 1e-10]
                eig_new = eig_new[eig_new > 1e-10]

                ent_old = float(compute_von_neumann_entropy(eig_old))
                ent_new = float(compute_von_neumann_entropy(eig_new))
                ent_change = ent_new - ent_old

                if ent_change < -0.3:
                    alerts.append({
                        'id': 'entropy',
                        'severity': 'alert',
                        'title': f'🔴 Entropía CAYENDO fuerte: {ent_new:.2f} (antes: {ent_old:.2f})',
                        'detail': f'Cambio de entropía: {ent_change:.2f}. Los activos se están '
                                  'moviendo JUNTOS (contagio). La diversificación pierde eficacia.',
                        'implication': 'Cuando la entropía cae drásticamente, TODOS los activos se correlacionan. '
                                       'Nuestro modelo O-U asume estructura de correlación estable. Si el grafo de '
                                       'correlaciones muta, las z-scores y señales de refugio pueden fallar.',
                        'methodology': f'Entropía de Von Neumann calculada sobre la matriz de correlaciones '
                                       f'de 6 ETFs proxy (SPY,QQQ,IWM,TLT,GLD,UUP). '
                                       f'Se comparan dos ventanas de {mid} días. Alerta si cae >0.30.',
                    })
                elif ent_change < -0.15:
                    alerts.append({
                        'id': 'entropy',
                        'severity': 'warning',
                        'title': f'🟡 Entropía bajando: {ent_new:.2f} (antes: {ent_old:.2f})',
                        'detail': f'Cambio: {ent_change:.2f}. Tendencia a mayor correlación entre activos.',
                        'implication': 'La diversificación funciona peor de lo habitual. Vigilar.',
                        'methodology': f'Entropía VN sobre 6 ETFs. Warning si cae >0.15.',
                    })
                else:
                    alerts.append({
                        'id': 'entropy',
                        'severity': 'ok',
                        'title': f'🟢 Entropía estable: {ent_new:.2f}',
                        'detail': f'Cambio: {ent_change:+.2f}. La estructura de correlaciones es estable.',
                        'implication': 'El grafo de correlaciones no ha mutado. Modelo O-U fiable.',
                        'methodology': 'Entropía VN sobre 6 ETFs proxy.',
                    })
            except Exception:
                pass

        # ── 5. Refuge Effectiveness ──
        # If VIX>20 AND recommended refuge (TLT/GLD) also losing money → model broken
        if vix_now >= 20:
            tlt_chg = tlt_data.get('chg_20d', 0)
            gld_chg = gld_data.get('chg_20d', 0)
            if tlt_chg < -0.03 and gld_chg < -0.03:
                alerts.append({
                    'id': 'refuge_effectiveness',
                    'severity': 'alert',
                    'title': '🔴 Refugio NO FUNCIONA: TLT y GLD caen en estrés',
                    'detail': f'VIX={vix_now:.0f} (estrés), pero TLT {tlt_chg*100:+.1f}% y GLD {gld_chg*100:+.1f}% en 20d. '
                              'AMBOS refugios están fallando.',
                    'implication': 'CRÍTICO: El modelo dice "refugio" pero el refugio TAMBIÉN pierde dinero. '
                                   'Esto ocurre en crisis de liquidez extrema (todo se vende = solo cash). '
                                   'Mover a SHY (cash) inmediatamente.',
                    'methodology': 'Si VIX≥20 y tanto TLT como GLD pierden >3% en 20d, los refugios no protegen.',
                })
            elif tlt_chg < -0.03 or gld_chg < -0.03:
                failing = 'TLT' if tlt_chg < -0.03 else 'GLD'
                ok_one = 'GLD' if tlt_chg < -0.03 else 'TLT'
                alerts.append({
                    'id': 'refuge_effectiveness',
                    'severity': 'warning',
                    'title': f'🟡 Refugio parcial: {failing} falla pero {ok_one} funciona',
                    'detail': f'VIX={vix_now:.0f}, {failing} pierde dinero pero {ok_one} aguanta.',
                    'implication': f'Concentrar refugio en {ok_one} y SHY. No usar {failing}.',
                    'methodology': 'Se verifica que los refugios suban (o al menos no caigan >3%) cuando VIX≥20.',
                })
            else:
                alerts.append({
                    'id': 'refuge_effectiveness',
                    'severity': 'ok',
                    'title': '🟢 Refugios funcionan correctamente en estrés',
                    'detail': f'VIX={vix_now:.0f} (estrés) y los refugios aguantan: '
                              f'TLT {tlt_chg*100:+.1f}%, GLD {gld_chg*100:+.1f}%.',
                    'implication': 'El modelo y la realidad coinciden. Refugio válido.',
                    'methodology': 'TLT y GLD no pierden >3% en 20d durante estrés.',
                })

        # ── 6. VIX Spike Without Recovery ──
        # Backtest assumes VIX spikes revert in 1-2 weeks
        if len(vix_hist) >= 20:
            vix_5d = vix_hist[-5:]
            vix_20d = vix_hist[-20:]
            if all(v > 25 for v in vix_5d) and all(v > 22 for v in vix_20d):
                alerts.append({
                    'id': 'vix_persistent',
                    'severity': 'warning',
                    'title': '🟡 VIX persistente >25 durante 20 días',
                    'detail': f'VIX lleva 20+ días por encima de 22 sin revertir. '
                              'Nuestro backtest asume que los spikes revierten en 1-2 semanas.',
                    'implication': 'Si el VIX permanece elevado, el gate estará activo mucho tiempo → '
                                   'perdemos oportunidades de alpha. Puede indicar un cambio de régimen estructural.',
                    'methodology': 'Se verifica si VIX>25 durante 5d consecutivos Y >22 durante 20d.',
                })

        # ── 7. CAPE Ratio (Shiller PE) ──
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get('https://www.multpl.com/shiller-pe',
                             headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            soup = BeautifulSoup(r.text, 'html.parser')
            cape_el = soup.select_one('#current')
            if cape_el:
                cape_text = cape_el.get_text(strip=True)
                # Extract numeric value
                import re
                cape_match = re.search(r'([\d.]+)', cape_text)
                if cape_match:
                    cape = float(cape_match.group(1))
                    hist_mean = 17.3
                    # Historical: mean 17.3, dot-com peak 44.2, 2007 peak 27.5
                    if cape > 35:
                        alerts.append({
                            'id': 'cape_ratio',
                            'severity': 'alert',
                            'title': f'🔴 CAPE Shiller: {cape:.1f} (media histórica: 17.3)',
                            'detail': f'El mercado cotiza a {cape/hist_mean:.1f}x su valoración media. '
                                      f'Solo se ha visto algo similar antes del crash de las .com (44.2) '
                                      f'y en 2021 (38.6).',
                            'implication': 'Territorio de burbuja. El momentum funciona, pero el riesgo de '
                                           'corrección del 30-40% es históricamente alto. '
                                           'Considerar reducir exposición a equity y aumentar cash.',
                            'methodology': f'CAPE = Precio S&P500 / Media de beneficios reales de 10 años. '
                                           f'Fuente: multpl.com. Media histórica (1871-2024): 17.3. '
                                           f'Alerta si >35, warning si >25.',
                        })
                    elif cape > 25:
                        alerts.append({
                            'id': 'cape_ratio',
                            'severity': 'warning',
                            'title': f'🟡 CAPE Shiller elevado: {cape:.1f}',
                            'detail': f'El mercado cotiza a {cape/hist_mean:.1f}x su valoración media. '
                                      f'Caro pero no en extremo histórico.',
                            'implication': 'Los retornos esperados a 10 años son menores que la media. '
                                           'El momentum sigue funcionando pero con un "techo" más bajo.',
                            'methodology': f'CAPE = P/E ajustado por ciclo (10 años). Media: 17.3.',
                        })
                    else:
                        alerts.append({
                            'id': 'cape_ratio',
                            'severity': 'ok',
                            'title': f'🟢 CAPE Shiller normal: {cape:.1f}',
                            'detail': f'Mercado en valoración razonable ({cape/hist_mean:.1f}x media).',
                            'implication': 'Buen momento para equity. Valoraciones no son obstáculo.',
                            'methodology': 'CAPE < 25 = rango normal-bajo.',
                        })
        except Exception:
            pass

        # ── 8. Buffett Indicator (Market Cap / GDP) ──
        try:
            r2 = requests.get('https://www.currentmarketvaluation.com/models/buffett-indicator.php',
                              headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            soup2 = BeautifulSoup(r2.text, 'html.parser')
            # Find percentage value
            buffett_val = None
            for t in soup2.find_all(string=lambda x: x and '%' in str(x)):
                txt = t.strip()
                match = re.search(r'(\d+)%', txt)
                if match and 50 < int(match.group(1)) < 500:
                    buffett_val = int(match.group(1))
                    break

            if buffett_val:
                if buffett_val > 200:
                    alerts.append({
                        'id': 'buffett_indicator',
                        'severity': 'alert',
                        'title': f'🔴 Indicador Buffett: {buffett_val}% (bolsa vs PIB)',
                        'detail': f'El valor de la bolsa es {buffett_val/100:.1f}x el PIB. '
                                  f'Media histórica: ~100%. Máximo anterior: ~200% (2021).',
                        'implication': 'El mercado se ha DESCONECTADO de la economía real. '
                                       'Esto no es sostenible a largo plazo. '
                                       'Tu momentum puede seguir funcionando meses, '
                                       'pero el riesgo sistémico de corrección es muy alto.',
                        'methodology': f'Capitalización total del mercado / PIB de EE.UU. '
                                       f'Fuente: currentmarketvaluation.com. '
                                       f'Media: ~100%. Alerta si >200%, warning si >150%.',
                    })
                elif buffett_val > 150:
                    alerts.append({
                        'id': 'buffett_indicator',
                        'severity': 'warning',
                        'title': f'🟡 Indicador Buffett elevado: {buffett_val}%',
                        'detail': f'El mercado vale {buffett_val/100:.1f}x el PIB. Caro pero no extremo.',
                        'implication': 'Valoración alta. Retornos a largo plazo probablemente menores.',
                        'methodology': 'Market Cap / GDP. Warning si >150%.',
                    })
                else:
                    alerts.append({
                        'id': 'buffett_indicator',
                        'severity': 'ok',
                        'title': f'🟢 Indicador Buffett normal: {buffett_val}%',
                        'detail': f'Mercado alineado con la economía ({buffett_val/100:.1f}x PIB).',
                        'implication': 'Valoración razonable a nivel macro.',
                        'methodology': 'Market Cap / GDP < 150%.',
                    })
        except Exception:
            pass

        # Summary
        n_alerts = sum(1 for a in alerts if a['severity'] == 'alert')
        n_warnings = sum(1 for a in alerts if a['severity'] == 'warning')
        n_ok = sum(1 for a in alerts if a['severity'] == 'ok')

        return {
            'alerts': alerts,
            'summary': {
                'alerts': n_alerts,
                'warnings': n_warnings,
                'ok': n_ok,
                'status': 'alert' if n_alerts > 0 else ('warning' if n_warnings > 0 else 'ok'),
            }
        }

    # ─── Main builders ───

    def build_snapshot(self, include_system=False):
        """Build complete snapshot with all data."""
        print("📊 Building dashboard snapshot...", flush=True)

        print("  → Market data...", flush=True)
        market = self.fetch_market()

        print("  → Yield curve...", flush=True)
        yields = self.fetch_yields()

        print("  → FRED macro...", flush=True)
        fred = self.fetch_fred()

        print("  → Headlines...", flush=True)
        headlines = self.fetch_headlines()

        print("  → Sentiment (Fear & Greed)...", flush=True)
        fear_greed = self.fetch_fear_greed()

        print("  → ETF Flows...", flush=True)
        etf_flows = self.fetch_etf_flows()

        print("  → Regime classification...", flush=True)
        regime = self.classify_regime(market)

        # HMM-based regime detection
        hmm_regime = {}
        try:
            from ml.regime_hmm import RegimeHMM
            print("  → HMM regime detection...", flush=True)
            hmm = RegimeHMM(lookback='2y')
            hmm.fit()
            hmm_regime = hmm.predict_regime()
        except Exception:
            pass

        print("  → Model health checks...", flush=True)
        health = self.compute_model_health(market)

        print("  → Momentum ranking...", flush=True)
        stocks, sectors = self.fetch_momentum_ranking()

        # Portfolio optimization (Markowitz) on top stocks
        portfolio_opt = {}
        try:
            from ml.portfolio_optimizer import optimize_portfolio
            # Take #1 from each sector (diverse picks)
            top_per_sector = []
            seen = set()
            for s in stocks:
                if s['sector'] not in seen and len(top_per_sector) < 8:
                    top_per_sector.append(s['ticker'])
                    seen.add(s['sector'])
            if len(top_per_sector) >= 3:
                print(f"  → Portfolio optimization ({len(top_per_sector)} tickers)...",
                      flush=True)
                portfolio_opt = optimize_portfolio(top_per_sector)
        except Exception:
            pass

        system_analytics = {}
        if include_system:
            print("  → System analytics (O-U, graph)...", flush=True)
            system_analytics = self.fetch_system_analytics()

        self._snapshot = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'yields': yields,
            'fred': fred,
            'headlines': headlines,
            'fear_greed': fear_greed,
            'etf_flows': etf_flows,
            'regime': regime,
            'hmm_regime': hmm_regime,
            'health': health,
            'stocks': stocks,
            'sectors': sectors,
            'portfolio_opt': portfolio_opt,
            'system_analytics': system_analytics,
        }

        print("✅ Snapshot ready", flush=True)
        return self._snapshot

    def build_llm_prompt(self, snapshot=None):
        """Build LLM prompt from snapshot."""
        from strategy.prompt_template import build_prompt
        s = snapshot or self._snapshot
        if not s:
            s = self.build_snapshot()
        return build_prompt(s)

    def to_json(self, snapshot=None):
        """Serialize snapshot to JSON-safe dict."""
        import math
        s = snapshot or self._snapshot
        if not s:
            return {}

        def _sanitize(obj):
            """Recursively clean NaN/Inf and numpy types."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                v = float(obj)
                return None if (math.isnan(v) or math.isinf(v)) else v
            if isinstance(obj, float):
                return None if (math.isnan(obj) or math.isinf(obj)) else obj
            if isinstance(obj, np.ndarray):
                return _sanitize(obj.tolist())
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        return json.loads(json.dumps(_sanitize(s)))


if __name__ == '__main__':
    pipe = DashboardPipeline()
    snapshot = pipe.build_snapshot(include_system=False)
    print(json.dumps(pipe.to_json(), indent=2, ensure_ascii=False)[:3000])
