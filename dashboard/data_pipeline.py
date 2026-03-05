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
        'CPIAUCSL': 'CPI (Inflación)',
        'UNRATE': 'Desempleo',
        'FEDFUNDS': 'Fed Funds Rate',
        'T10Y2Y': 'Spread 10Y-2Y',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'DTWEXBGS': 'Trade Weighted USD',
        'DGS10': 'Yield 10Y (FRED)',
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
                hist = yf.Ticker(tk).history(period="6mo")
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
                    'history': [float(x) for x in close.tail(60).values],
                    'dates': [d.strftime('%Y-%m-%d') for d in close.tail(60).index],
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
        """Fetch FRED indicators from Supabase."""
        macro = {}
        try:
            from db.database_manager import DatabaseManager
            db = DatabaseManager()
            for code, name in self.FRED_INDICATORS.items():
                try:
                    resp = (db.client.table('macro_data')
                            .select('date,value')
                            .eq('indicator', code)
                            .order('date', desc=True)
                            .limit(5)
                            .execute())
                    if resp.data and len(resp.data) > 0:
                        latest = resp.data[0]
                        prev = resp.data[-1] if len(resp.data) > 1 else latest
                        macro[code] = {
                            'name': name,
                            'value': float(latest['value']),
                            'date': latest['date'],
                            'prev_value': float(prev['value']),
                            'prev_date': prev['date'],
                        }
                except Exception:
                    pass
        except Exception:
            pass
        return macro

    def fetch_momentum_ranking(self):
        """Fetch TOP N momentum stocks with fundamentals."""
        stocks = []
        try:
            from ml.fundamental_momentum import load_quarterly_data, compute_momentum_features
            qdf, _ = load_quarterly_data()
            mom = compute_momentum_features(qdf)
            top = mom.nlargest(self.TOP_N, 'momentum_score')

            # Fetch prices for each
            for _, row in top.iterrows():
                ticker = row['ticker']
                stock = {
                    'ticker': ticker,
                    'score': float(row['momentum_score']),
                    'rev_qoq': float(row.get('rev_last_qoq', 0)),
                    'eps_growth': float(row.get('eps_growth_total', 0)) if pd.notna(row.get('eps_growth_total')) else None,
                    'roic_trend': float(row.get('roic_trend', 0)),
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
                stocks.append(stock)
        except Exception:
            pass
        return stocks

    def fetch_system_analytics(self):
        """Run core system analytics (O-U, graph, entropy)."""
        analytics = {}
        try:
            from core.graph_builder import GraphBuilder
            from core.fundamental_filter import FundamentalFilter
            from core.heat_engine import HeatEngine

            gb = GraphBuilder()
            gb.load_data()
            gb.build()

            ff = FundamentalFilter(gb.tickers)
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
                if len(engine.refuge_signal) > 0:
                    analytics['refuge_signal'] = float(engine.refuge_signal[-1])

            # Calibrated params
            analytics['s'] = float(gb.s) if hasattr(gb, 's') else 0.5
            analytics['gamma'] = float(engine.gamma) if hasattr(engine, 'gamma') else 1.0

            # Reversion probabilities for top stocks
            for i, t in enumerate(gb.tickers):
                try:
                    prob = engine.compute_probability(i, horizon=5)
                    if prob:
                        analytics[f'prob_{t}'] = {
                            k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in prob.items()
                        }
                except Exception:
                    pass

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

        # ── 1. VIX Distribution Shift ──
        # Historical VIX: mean ~19, std ~7. If 6M avg is far from this, regime changed.
        if len(vix_hist) >= 40:
            vix_mean_recent = np.mean(vix_hist[-40:])  # ~2 months of data available
            vix_std_recent = np.std(vix_hist[-40:])
            hist_mean, hist_std = 19.0, 7.0

            z_shift = abs(vix_mean_recent - hist_mean) / hist_std
            if z_shift > 2.0:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'alert',
                    'title': '🔴 VIX fuera de rango histórico',
                    'detail': f'Media VIX reciente: {vix_mean_recent:.1f} (histórica: ~19). '
                              f'Desviación: {z_shift:.1f}σ. Nuestros umbrales se calibraron en media ~19.',
                    'implication': 'Los umbrales VIX<15/20/30 pueden NO ser válidos en este régimen. '
                                   'Un VIX con media ~28 haría que umbral=20 esté siempre activo → overfitting.',
                    'methodology': 'Se compara la media del VIX de los últimos 40 días con la media '
                                   'histórica (19). Alerta si |diferencia| > 2 × std histórica (7).',
                })
            elif z_shift > 1.0:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'warning',
                    'title': '🟡 VIX algo elevado sobre rango habitual',
                    'detail': f'Media VIX reciente: {vix_mean_recent:.1f} vs histórica ~19. '
                              f'Desviación: {z_shift:.1f}σ. Dentro de lo tolerable pero vigilar.',
                    'implication': 'Los umbrales aún son válidos pero estamos en la zona límite.',
                    'methodology': 'Media VIX 40d vs media histórica (19). Warning si >1σ, alerta si >2σ.',
                })
            else:
                alerts.append({
                    'id': 'vix_distribution',
                    'severity': 'ok',
                    'title': '🟢 Distribución VIX dentro de rango',
                    'detail': f'Media VIX reciente: {vix_mean_recent:.1f} ({z_shift:.1f}σ de la media histórica). Normal.',
                    'implication': 'Los umbrales calibrados en backtest siguen siendo válidos.',
                    'methodology': 'Media VIX 40d vs media histórica (19±7).',
                })

        # ── 2. VIX Gate Frequency ──
        # In backtest, VIX>MA20 triggers ~37% of the time. If very different, model breaks.
        if len(vix_hist) >= 20:
            ma20_vals = [np.mean(vix_hist[max(0,i-19):i+1]) for i in range(19, len(vix_hist))]
            vix_subset = vix_hist[19:]
            gate_active = sum(1 for v, m in zip(vix_subset, ma20_vals) if v > m)
            freq = gate_active / len(vix_subset) if vix_subset else 0
            expected = 0.37

            if abs(freq - expected) > 0.20:
                alerts.append({
                    'id': 'gate_frequency',
                    'severity': 'alert',
                    'title': f'🔴 Gate VIX>MA20 frecuencia anómala: {freq*100:.0f}%',
                    'detail': f'El gate se activa {freq*100:.0f}% del tiempo vs {expected*100:.0f}% esperado en backtest. '
                              f'Diferencia: {abs(freq-expected)*100:.0f} puntos.',
                    'implication': f'{"El gate NUNCA se apaga → refugio permanente = sin alpha." if freq > 0.57 else "El gate NUNCA se activa → sin protección cuando venga la crisis."}',
                    'methodology': f'Se cuentan los días con VIX>MA20 en los últimos {len(vix_subset)} días. '
                                   'Backtest 2004-2026 dio ~37%. Alerta si difiere >20pp.',
                })
            else:
                alerts.append({
                    'id': 'gate_frequency',
                    'severity': 'ok',
                    'title': f'🟢 Frecuencia gate VIX>MA20: {freq*100:.0f}% (esperado ~37%)',
                    'detail': 'La frecuencia de activación del gate es coherente con el backtest.',
                    'implication': 'El modelo opera en un régimen similar al de entrenamiento.',
                    'methodology': f'Calculado sobre {len(vix_subset)} días disponibles.',
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

    def build_snapshot(self, include_system=True):
        """Build complete snapshot with all data."""
        print("📊 Building dashboard snapshot...", flush=True)

        print("  → Market data...", flush=True)
        market = self.fetch_market()

        print("  → Yield curve...", flush=True)
        yields = self.fetch_yields()

        print("  → FRED macro...", flush=True)
        fred = self.fetch_fred()

        print("  → Regime classification...", flush=True)
        regime = self.classify_regime(market)

        print("  → Model health checks...", flush=True)
        health = self.compute_model_health(market)

        print("  → Momentum ranking...", flush=True)
        stocks = self.fetch_momentum_ranking()

        system_analytics = {}
        if include_system:
            print("  → System analytics (O-U, graph)...", flush=True)
            system_analytics = self.fetch_system_analytics()

        self._snapshot = {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'yields': yields,
            'fred': fred,
            'regime': regime,
            'health': health,
            'stocks': stocks,
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
        s = snapshot or self._snapshot
        if not s:
            return {}

        def clean(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj

        return json.loads(json.dumps(s, default=clean))


if __name__ == '__main__':
    pipe = DashboardPipeline()
    snapshot = pipe.build_snapshot(include_system=False)
    print(json.dumps(pipe.to_json(), indent=2, ensure_ascii=False)[:3000])
