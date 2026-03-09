"""Gemini API client for dashboard/telegram diagnostics."""

from __future__ import annotations

import os
from typing import Optional

import requests


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    """Raised when Gemini API call fails."""


def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise GeminiError("Gemini no devolvió candidatos")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    if not text.strip():
        raise GeminiError("Gemini devolvió respuesta vacía")
    return text.strip()


def generate_text(prompt: str,
                  model: Optional[str] = None,
                  api_key: Optional[str] = None,
                  temperature: float = 0.25,
                  max_output_tokens: int = 1200,
                  timeout: int = 40) -> str:
    """Generate text using Gemini API."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiError("Falta GEMINI_API_KEY")

    model_name = model or DEFAULT_MODEL
    url = f"{GEMINI_API_BASE}/{model_name}:generateContent"

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    try:
        resp = requests.post(
            url,
            params={"key": key},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise GeminiError(f"Error de red con Gemini: {exc}") from exc

    if not resp.ok:
        detail = ""
        try:
            detail = (resp.json().get("error") or {}).get("message", "")
        except Exception:
            detail = resp.text[:300]
        raise GeminiError(f"Gemini API error {resp.status_code}: {detail}")

    return _extract_text(resp.json())


def build_market_diagnosis(prompt: str,
                           model: Optional[str] = None,
                           api_key: Optional[str] = None) -> str:
    """Specialized diagnosis wrapper for market dashboard prompt."""
    instruction = (
        "Actúa como CIO de un hedge fund macro. "
        "Responde en español claro y accionable con este formato exacto:\n"
        "1) Diagnóstico del régimen (3-5 líneas)\n"
        "2) Riesgos clave próximos 5 días (máx 5 bullets)\n"
        "3) Refugio/cartera táctica sugerida con pesos (%)\n"
        "4) 3 señales a vigilar mañana\n"
        "5) Alertas de contradicción entre indicadores\n\n"
        "No inventes datos fuera del prompt. Si falta algo, dilo explícitamente.\n\n"
        "DATOS DEL DASHBOARD:\n"
    )
    return generate_text(
        prompt=instruction + prompt,
        model=model,
        api_key=api_key,
        temperature=0.2,
        max_output_tokens=1400,
    )
