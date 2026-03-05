"""
DAILY SIGNAL — Momentum + VIX Gate
=====================================
Checks VIX and outputs today's action.
Designed for cron job or Telegram bot integration.

Usage:
    python strategy/daily_signal.py
    python strategy/daily_signal.py --telegram

Cron (21:30 CET, Mon-Fri):
    30 21 * * 1-5 cd /path/to/project && python strategy/daily_signal.py --telegram
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from datetime import datetime

# ── CONFIGURATION ──
VIX_THRESHOLD = 20
PORTFOLIO = ['AAPL', 'CRM', 'NVDA', 'AMGN', 'AMZN',
             'AVGO', 'UPS', 'V', 'ASML', 'META']
REFUGE = {'TLT': 0.5, 'GLD': 0.5}


def get_vix():
    """Get current VIX level."""
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="5d")
    if hist.empty:
        return None
    return float(hist['Close'].iloc[-1])


def get_prices(tickers):
    """Get latest prices for portfolio tickers."""
    prices = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            h = tk.history(period="2d")
            if not h.empty:
                prices[t] = {
                    'price': float(h['Close'].iloc[-1]),
                    'change': float(h['Close'].pct_change().iloc[-1]) if len(h) > 1 else 0,
                }
        except Exception:
            pass
    return prices


def generate_signal():
    """Generate today's trading signal."""
    vix_level = get_vix()
    if vix_level is None:
        return {'error': 'Could not fetch VIX'}

    is_defensive = vix_level >= VIX_THRESHOLD

    signal = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'vix': round(vix_level, 1),
        'threshold': VIX_THRESHOLD,
        'mode': 'REFUGIO' if is_defensive else 'MOMENTUM',
        'portfolio': list(REFUGE.keys()) if is_defensive else PORTFOLIO,
    }

    return signal


def format_message(signal):
    """Format signal as readable message."""
    if 'error' in signal:
        return f"⚠️ Error: {signal['error']}"

    vix = signal['vix']
    mode = signal['mode']
    date = signal['date']

    if mode == 'REFUGIO':
        emoji = '🔴'
        action = f"REFUGIO → 50% TLT + 50% GLD"
        reason = f"VIX={vix} ≥ {signal['threshold']}"
    else:
        emoji = '🟢'
        tickers = ', '.join(signal['portfolio'])
        action = f"MOMENTUM → {tickers}"
        reason = f"VIX={vix} < {signal['threshold']}"

    msg = f"""{emoji} SEÑAL DIARIA — {date}

{reason}
Acción: {action}

{'⚠️ Mercado en estrés. Mantener posiciones defensivas.' if mode == 'REFUGIO' else '✅ Mercado tranquilo. Mantener cartera de momentum.'}"""

    return msg


def send_telegram(message, token=None, chat_id=None):
    """Send message via Telegram bot."""
    import os
    token = token or os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("⚠️ Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars")
        return False

    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={'chat_id': chat_id, 'text': message})
    return resp.ok


def main():
    parser = argparse.ArgumentParser(description='Daily trading signal')
    parser.add_argument('--telegram', action='store_true', help='Send via Telegram')
    args = parser.parse_args()

    signal = generate_signal()
    message = format_message(signal)

    print(message)

    if args.telegram:
        if send_telegram(message):
            print("\n✅ Sent to Telegram")
        else:
            print("\n❌ Failed to send to Telegram")


if __name__ == '__main__':
    main()
