"""
Telegram bot report: dashboard snapshot + Gemini diagnosis + charts.

Usage:
    PYTHONPATH=. python strategy/telegram_premium_bot.py
    PYTHONPATH=. python strategy/telegram_premium_bot.py --telegram
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import requests

from dashboard.data_pipeline import DashboardPipeline
from strategy.gemini_client import GeminiError, build_market_diagnosis
from strategy.telegram_charting import (
    chart_refuge_comparison,
    chart_vix,
    chart_yield_curve,
    cleanup_paths,
)


def _format_exec_summary(snapshot: dict) -> str:
    regime = snapshot.get("regime", {})
    health = snapshot.get("health", {}).get("summary", {})
    vix = snapshot.get("market", {}).get("^VIX", {})
    ts = snapshot.get("timestamp")
    ts_text = ts or datetime.now().isoformat()
    return (
        "📊 GlobalMarketAnalyzer — Premium Brief\n"
        f"🕒 {ts_text}\n\n"
        f"Régimen VIX: {regime.get('vix_state', 'n/a')}\n"
        f"Crisis: {regime.get('crisis_type', 'n/a')}\n"
        f"VIX: {vix.get('price', 'n/a')}\n"
        f"Refugio sugerido: {regime.get('recommended_refuge', 'n/a')}\n"
        f"Salud modelo: 🔴 {health.get('alerts', 0)} | "
        f"🟡 {health.get('warnings', 0)} | 🟢 {health.get('ok', 0)}"
    )


def _send_message(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "text": text[:4096], "disable_web_page_preview": True},
        timeout=25,
    )
    return resp.ok


def _send_photo(token: str, chat_id: str, path: str, caption: str = "") -> bool:
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(path, "rb") as f:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"photo": f},
            timeout=35,
        )
    return resp.ok


def run(send_to_telegram: bool = False, include_gemini: bool = True) -> None:
    pipe = DashboardPipeline()
    snapshot = pipe.build_snapshot(include_system=False)
    prompt = pipe.build_llm_prompt(snapshot)

    diagnosis = "Gemini desactivado."
    if include_gemini:
        try:
            diagnosis = build_market_diagnosis(prompt)
        except GeminiError as exc:
            diagnosis = f"No se pudo generar diagnóstico Gemini: {exc}"

    summary = _format_exec_summary(snapshot)
    full_msg = f"{summary}\n\n🧠 Diagnóstico IA\n{diagnosis}"
    print(full_msg)

    if not send_to_telegram:
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Faltan TELEGRAM_BOT_TOKEN y/o TELEGRAM_CHAT_ID")

    if not _send_message(token, chat_id, summary):
        raise RuntimeError("No se pudo enviar resumen a Telegram")

    if not _send_message(token, chat_id, f"🧠 Diagnóstico IA\n{diagnosis}"):
        raise RuntimeError("No se pudo enviar diagnóstico a Telegram")

    charts: List[str] = []
    try:
        for builder in (chart_vix, chart_yield_curve, chart_refuge_comparison):
            p = builder(snapshot)
            if p:
                charts.append(p)
        for idx, p in enumerate(charts, start=1):
            _send_photo(token, chat_id, p, caption=f"Gráfica {idx}/3")
    finally:
        cleanup_paths(charts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram premium market bot")
    parser.add_argument("--telegram", action="store_true", help="Enviar a Telegram")
    parser.add_argument("--no-gemini", action="store_true", help="No llamar Gemini")
    args = parser.parse_args()
    run(send_to_telegram=args.telegram, include_gemini=not args.no_gemini)


if __name__ == "__main__":
    main()
