"""
DASHBOARD API — Flask backend
================================
Serves the dashboard data via REST endpoints.

Usage:
    python dashboard/api.py
    → http://localhost:8050
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, render_template, Response
from dashboard.data_pipeline import DashboardPipeline

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

pipeline = DashboardPipeline()
_cached_snapshot = None


def _get_snapshot(force=False):
    global _cached_snapshot
    if _cached_snapshot is None or force:
        _cached_snapshot = pipeline.build_snapshot(include_system=False)
    return _cached_snapshot


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/snapshot')
def api_snapshot():
    snapshot = _get_snapshot()
    return jsonify(pipeline.to_json(snapshot))


@app.route('/api/refresh')
def api_refresh():
    snapshot = _get_snapshot(force=True)
    return jsonify(pipeline.to_json(snapshot))


@app.route('/api/prompt')
def api_prompt():
    snapshot = _get_snapshot()
    prompt = pipeline.build_llm_prompt(snapshot)
    return Response(prompt, mimetype='text/plain; charset=utf-8')


@app.route('/api/stocks')
def api_stocks():
    snapshot = _get_snapshot()
    return jsonify(snapshot.get('stocks', []))


@app.route('/api/regime')
def api_regime():
    snapshot = _get_snapshot()
    return jsonify(snapshot.get('regime', {}))


@app.route('/api/health')
def api_health():
    snapshot = _get_snapshot()
    return jsonify(snapshot.get('health', {}))


@app.route('/api/calendar-prompt')
def api_calendar_prompt():
    from strategy.prompt_template import build_calendar_prompt
    snapshot = _get_snapshot()
    prompt = build_calendar_prompt(snapshot)
    return Response(prompt, mimetype='text/plain; charset=utf-8')


@app.route('/api/glossary')
def api_glossary():
    from strategy.glossary import get_all_indicators
    return jsonify(get_all_indicators())


if __name__ == '__main__':
    print("📊 Starting Dashboard API on http://localhost:8050", flush=True)
    # Pre-load data
    _get_snapshot()
    app.run(host='0.0.0.0', port=8050, debug=True)
