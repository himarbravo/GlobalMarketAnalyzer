"""
Epoch validation for LLM-style allocation policy.

What it does:
1) Builds regime templates from docs/LLM_BACKTEST_PROMPTS.md responses.
2) Runs monthly walk-forward across multiple historical epochs.
3) Estimates luck via randomization test (shuffle regime sequence).
4) Exports metrics and charts to output/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ASSETS = ["SPY", "QQQ", "IWM", "TLT", "SHY", "GLD", "UUP"]
EPOCHS: List[Tuple[str, str, str]] = [
    ("GFC-Recovery", "2008-01-01", "2012-12-31"),
    ("LowVol-Bull", "2013-01-01", "2019-12-31"),
    ("COVID-Regime", "2020-01-01", "2021-12-31"),
    ("Inflation-Shock", "2022-01-01", "2023-12-31"),
    ("Recent", "2024-01-01", "2026-03-09"),
]


@dataclass
class TemplateRow:
    regime: str
    weights: Dict[str, float]


def parse_templates(md_path: Path) -> Dict[str, Dict[str, float]]:
    text = md_path.read_text(encoding="utf-8")
    blocks = re.findall(r"RESPUESTA GEMINI:\s*```json\s*(\{.*?\})\s*```", text, flags=re.S)

    rows: List[TemplateRow] = []
    for blk in blocks:
        try:
            obj = json.loads(blk)
        except Exception:
            continue
        reg = str(obj.get("regime", "neutral")).lower()
        w = obj.get("weights", {})
        weights = {a: float(w.get(a, 0.0)) / 100.0 for a in ASSETS}
        s = sum(weights.values())
        if s > 0:
            weights = {k: v / s for k, v in weights.items()}
            rows.append(TemplateRow(regime=reg, weights=weights))

    if not rows:
        raise RuntimeError("No se pudieron construir templates desde el markdown.")

    templates: Dict[str, Dict[str, float]] = {}
    for reg in sorted(set(r.regime for r in rows)):
        arr = pd.DataFrame([r.weights for r in rows if r.regime == reg]).mean().to_dict()
        s = sum(arr.values())
        templates[reg] = {k: (v / s if s > 0 else 0.0) for k, v in arr.items()}

    # Safety fallback
    templates.setdefault("neutral", {a: 1 / len(ASSETS) for a in ASSETS})
    for r in ["bull", "stress", "bear"]:
        templates.setdefault(r, templates["neutral"])

    return templates


def fetch_data(start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
    import yfinance as yf

    px = yf.download(ASSETS, start=start, end=end, auto_adjust=True, progress=False)
    if px.empty:
        raise RuntimeError("Sin datos de precios para ASSETS.")
    close = px["Close"].dropna(how="all")

    vix_px = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if vix_px.empty:
        raise RuntimeError("Sin datos de VIX.")
    vix = vix_px["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix = vix.dropna()
    return close, vix


def classify_regime(vix: pd.Series, date: pd.Timestamp) -> str:
    h = vix.loc[:date].dropna()
    if len(h) < 21:
        return "neutral"
    lv = float(h.iloc[-1])
    ma20 = float(h.tail(20).mean())
    if lv >= 30:
        return "bear"
    if lv >= 22 or lv > ma20 * 1.08:
        return "stress"
    if lv <= 17 and lv < ma20:
        return "bull"
    return "neutral"


def perf_stats(curve: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    daily = curve.pct_change().dropna()
    dd = curve / curve.cummax() - 1.0
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
    cagr = float(curve.iloc[-1] ** (1 / years) - 1)
    vol = float(daily.std() * np.sqrt(252)) if len(daily) else 0.0
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    bench_cagr = float(benchmark.iloc[-1] ** (1 / years) - 1) if len(benchmark) else 0.0
    alpha = cagr - bench_cagr
    return {
        "total_return": float(curve.iloc[-1] - 1.0),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "alpha_vs_spy_cagr": alpha,
    }


def run_epoch(close: pd.DataFrame, vix: pd.Series, templates: Dict[str, Dict[str, float]], start: str, end: str):
    close_ep = close[(close.index >= start) & (close.index <= end)].dropna(how="all")
    if len(close_ep) < 40:
        return None
    rets = close_ep.pct_change().fillna(0.0)
    monthly = close_ep.resample("BMS").first().index

    reg_seq: List[str] = []
    weight_schedule = {}
    for dt in monthly:
        reg = classify_regime(vix, dt)
        reg_seq.append(reg)
        weight_schedule[dt] = pd.Series(templates[reg]).reindex(ASSETS).fillna(0.0)

    out = []
    value = 1.0
    for dt, row in rets.iterrows():
        eligible = [d for d in monthly if d <= dt]
        if not eligible:
            continue
        rb = eligible[-1]
        w = weight_schedule[rb]
        r = float((row.reindex(ASSETS).fillna(0.0) * w).sum())
        value *= (1.0 + r)
        out.append((dt, value))
    curve = pd.Series({d: v for d, v in out}).sort_index()

    spy = (1.0 + rets["SPY"].fillna(0.0)).cumprod().reindex(curve.index)
    stats = perf_stats(curve, spy)
    return curve, spy, stats, reg_seq, monthly


def randomization_test(
    rets: pd.DataFrame,
    monthly: pd.DatetimeIndex,
    reg_seq: List[str],
    templates: Dict[str, Dict[str, float]],
    observed_total_return: float,
    n_iter: int = 500,
) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    base = np.array(reg_seq, dtype=object)
    sims = []
    for _ in range(n_iter):
        sh = base.copy()
        rng.shuffle(sh)
        weights = {d: pd.Series(templates[sh[i]]).reindex(ASSETS).fillna(0.0) for i, d in enumerate(monthly)}
        val = 1.0
        for dt, row in rets.iterrows():
            elig = [d for d in monthly if d <= dt]
            if not elig:
                continue
            rb = elig[-1]
            val *= 1.0 + float((row.reindex(ASSETS).fillna(0.0) * weights[rb]).sum())
        sims.append(val - 1.0)
    arr = np.array(sims, dtype=float)
    pval = float((arr >= observed_total_return).mean())
    return {
        "random_mean_return": float(arr.mean()),
        "random_p95_return": float(np.percentile(arr, 95)),
        "random_p05_return": float(np.percentile(arr, 5)),
        "p_value_luck": pval,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md_path = root / "docs" / "LLM_BACKTEST_PROMPTS.md"
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = parse_templates(md_path)
    start = min(e[1] for e in EPOCHS)
    end = max(e[2] for e in EPOCHS)
    close, vix = fetch_data(start, end)

    rows = []
    fig = make_subplots(rows=1, cols=1)

    for name, s, e in EPOCHS:
        res = run_epoch(close, vix, templates, s, e)
        if not res:
            continue
        curve, spy, stats, reg_seq, monthly = res
        fig.add_trace(go.Scatter(x=curve.index, y=curve, name=f"{name} LLM", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=spy.index, y=spy, name=f"{name} SPY", line=dict(width=1.2, dash="dot")))

        rets_ep = close[(close.index >= s) & (close.index <= e)].pct_change().fillna(0.0)
        luck = randomization_test(
            rets_ep,
            monthly,
            reg_seq,
            templates,
            observed_total_return=stats["total_return"],
            n_iter=400,
        )
        rows.append({
            "epoch": name,
            "start": s,
            "end": e,
            **stats,
            **luck,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "llm_epoch_validation.csv", index=False)

    fig.update_layout(
        title="Validación por Épocas: LLM Policy vs SPY",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h"),
    )
    fig.write_html(str(out_dir / "llm_epoch_validation_curves.html"))

    summary = {
        "templates": templates,
        "epochs_tested": len(df),
        "results": df.to_dict(orient="records"),
        "outputs": {
            "csv": "output/llm_epoch_validation.csv",
            "chart": "output/llm_epoch_validation_curves.html",
        },
    }
    (out_dir / "llm_epoch_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
