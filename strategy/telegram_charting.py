"""Chart helpers for Telegram market reports."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List


def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib no está disponible. Instálalo para enviar gráficas en Telegram."
        ) from exc


def _save_chart(fig, prefix: str) -> str:
    out = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png", delete=False)
    out.close()
    fig.savefig(out.name, dpi=160, bbox_inches="tight")
    fig.clf()
    return out.name


def chart_vix(snapshot: Dict) -> str | None:
    market = snapshot.get("market", {})
    vix = market.get("^VIX", {})
    hist = vix.get("history") or []
    if len(hist) < 20:
        return None

    plt = _import_plt()
    dates = (vix.get("dates") or list(range(len(hist))))[-120:]
    vals = hist[-120:]
    ma20 = []
    for i in range(len(vals)):
        if i < 19:
            ma20.append(None)
        else:
            ma20.append(sum(vals[i - 19:i + 1]) / 20)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, vals, color="#f85149", linewidth=2, label="VIX")
    ax.plot(dates, ma20, color="#58a6ff", linewidth=1.4, linestyle="--", label="MA20")
    ax.axhline(20, color="#e3b341", linewidth=1, linestyle=":", label="Gate 20")
    ax.set_title("VIX y MA20 (últimos 120 días)")
    ax.set_ylabel("Nivel")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    return _save_chart(fig, "gma_vix_")


def chart_yield_curve(snapshot: Dict) -> str | None:
    yields = snapshot.get("yields", {})
    labels = ["3M", "5Y", "10Y", "30Y"]
    points = [yields.get(lb, {}).get("yield") for lb in labels]
    if any(v is None for v in points):
        return None

    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(labels, points, marker="o", color="#e3b341", linewidth=2)
    ax.fill_between(labels, points, [min(points)] * len(points), alpha=0.1, color="#e3b341")
    ax.set_title("Curva de tipos actual")
    ax.set_ylabel("Yield (%)")
    ax.grid(alpha=0.2)
    return _save_chart(fig, "gma_curve_")


def chart_refuge_comparison(snapshot: Dict) -> str | None:
    market = snapshot.get("market", {})
    spy = market.get("SPY", {}).get("history", [])
    tlt = market.get("TLT", {}).get("history", [])
    gld = market.get("GLD", {}).get("history", [])
    if min(len(spy), len(tlt), len(gld)) < 60:
        return None

    plt = _import_plt()
    n = 120
    spy, tlt, gld = spy[-n:], tlt[-n:], gld[-n:]
    x = list(range(len(spy)))

    def norm(series: List[float]) -> List[float]:
        base = series[0]
        return [100.0 * v / base for v in series]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, norm(spy), linewidth=2, label="SPY", color="#58a6ff")
    ax.plot(x, norm(tlt), linewidth=2, label="TLT", color="#3fb950")
    ax.plot(x, norm(gld), linewidth=2, label="GLD", color="#e3b341")
    ax.set_title("Comparativa refugio (base=100, 120 días)")
    ax.set_ylabel("Índice base 100")
    ax.grid(alpha=0.2)
    ax.legend()
    return _save_chart(fig, "gma_refuge_")


def cleanup_paths(paths: List[str]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
