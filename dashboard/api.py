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

from flask import Flask, jsonify, render_template, Response, request
from dashboard.data_pipeline import DashboardPipeline
from strategy.gemini_client import build_market_diagnosis, GeminiError

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


@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    """Generate Gemini diagnosis directly from current dashboard snapshot."""
    payload = request.get_json(silent=True) or {}
    snapshot = _get_snapshot(force=bool(payload.get('force_refresh', False)))
    prompt = pipeline.build_llm_prompt(snapshot)

    model = payload.get('model')
    try:
        diagnosis = build_market_diagnosis(prompt, model=model)
    except GeminiError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'Error interno: {exc}'}), 500

    return jsonify({
        'ok': True,
        'model': model or 'default',
        'diagnosis': diagnosis,
        'timestamp': snapshot.get('timestamp'),
    })


@app.route('/api/stocks')
def api_stocks():
    snapshot = _get_snapshot()
    return jsonify(snapshot.get('stocks', []))


@app.route('/api/sectors')
def api_sectors():
    snapshot = _get_snapshot()
    return jsonify(snapshot.get('sectors', {}))


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
