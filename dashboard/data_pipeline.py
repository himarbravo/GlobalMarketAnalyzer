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
