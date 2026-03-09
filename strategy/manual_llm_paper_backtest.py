"""
Manual LLM paper-trading backtest from docs/LLM_BACKTEST_PROMPTS.md.

Reads embedded JSON responses, computes portfolio evolution, and writes
interactive charts to output/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ASSETS = ["SPY", "QQQ", "IWM", "TLT", "SHY", "GLD", "UUP"]


@dataclass
class Decision:
    date: pd.Timestamp
    regime: str
    confidence: float
    weights: Dict[str, float]


def parse_decisions(md_path: Path) -> List[Decision]:
    text = md_path.read_text(encoding="utf-8")
    blocks = re.findall(r"RESPUESTA GEMINI:\s*```json\s*(\{.*?\})\s*```", text, flags=re.S)
    decisions: List[Decision] = []
    for blk in blocks:
        try:
            obj = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if "date" not in obj or "weights" not in obj:
            continue
        w_raw = obj["weights"]
        weights = {a: float(w_raw.get(a, 0.0)) / 100.0 for a in ASSETS}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        decisions.append(
            Decision(
                date=pd.to_datetime(obj["date"]),
                regime=str(obj.get("regime", "neutral")),
                confidence=float(obj.get("confidence", 0.5)),
                weights=weights,
            )
        )
    decisions.sort(key=lambda x: x.date)
    return decisions


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(ASSETS, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    close = data["Close"].copy() if "Close" in data else data.copy()
    close = close.dropna(how="all")
    return close


def simulate_offline(decisions: List[Decision]) -> pd.DataFrame:
    regime_map = {
        "bull": {"SPY": 0.03, "QQQ": 0.04, "IWM": 0.025, "TLT": -0.01, "SHY": 0.003, "GLD": 0.008, "UUP": -0.002},
        "neutral": {"SPY": 0.01, "QQQ": 0.012, "IWM": 0.008, "TLT": 0.005, "SHY": 0.003, "GLD": 0.006, "UUP": 0.001},
        "stress": {"SPY": -0.025, "QQQ": -0.035, "IWM": -0.04, "TLT": 0.012, "SHY": 0.003, "GLD": 0.015, "UUP": 0.007},
        "bear": {"SPY": -0.03, "QQQ": -0.04, "IWM": -0.045, "TLT": 0.015, "SHY": 0.003, "GLD": 0.018, "UUP": 0.009},
    }
    rows = []
    value = 1.0
    spy = 1.0
    for d in decisions:
        ret_map = regime_map.get(d.regime, regime_map["neutral"])
        p_ret = sum(d.weights[a] * ret_map[a] for a in ASSETS)
        value *= (1.0 + p_ret)
        spy *= (1.0 + ret_map["SPY"])
        rows.append({"date": d.date, "portfolio": value, "spy": spy, "mode": "offline"})
    return pd.DataFrame(rows).set_index("date")


def simulate_with_prices(decisions: List[Decision], prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0)
    start = decisions[0].date
    end = decisions[-1].date + pd.Timedelta(days=31)
    rets = rets[(rets.index >= start) & (rets.index <= end)]
    if rets.empty:
        return pd.DataFrame()

    value = 1.0
    out = []
    for i, d in enumerate(decisions):
        seg_start = d.date
        seg_end = decisions[i + 1].date if i + 1 < len(decisions) else rets.index.max() + pd.Timedelta(days=1)
        seg = rets[(rets.index >= seg_start) & (rets.index < seg_end)]
        if seg.empty:
            continue
        w = pd.Series(d.weights).reindex(ASSETS).fillna(0.0)
        port_daily = seg[ASSETS].mul(w, axis=1).sum(axis=1)
        for dt, r in port_daily.items():
            value *= (1.0 + float(r))
            out.append({"date": dt, "portfolio": value, "spy": None, "mode": "market"})

    df = pd.DataFrame(out).drop_duplicates(subset="date").set_index("date").sort_index()
    if "SPY" in rets.columns and not df.empty:
        spy_curve = (1.0 + rets["SPY"].loc[df.index]).cumprod()
        df["spy"] = spy_curve.values
    return df


def metrics(curve: pd.Series) -> dict:
    if curve.empty:
        return {}
    daily = curve.pct_change().dropna()
    dd = curve / curve.cummax() - 1.0
    sharpe = (daily.mean() / daily.std()) * (252 ** 0.5) if daily.std() > 0 else 0.0
    return {
        "return_total": float(curve.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "sharpe_approx": float(sharpe),
    }


def render_charts(df: pd.DataFrame, decisions: List[Decision], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dd_port = df["portfolio"] / df["portfolio"].cummax() - 1.0
    dd_spy = df["spy"] / df["spy"].cummax() - 1.0 if "spy" in df and df["spy"].notna().any() else None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Equity Curve", "Drawdown"))
    fig.add_trace(go.Scatter(x=df.index, y=df["portfolio"], name="LLM Portfolio", line=dict(width=2)), row=1, col=1)
    if "spy" in df.columns and df["spy"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["spy"], name="SPY Buy&Hold", line=dict(width=1.6, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=dd_port, name="DD Portfolio", line=dict(width=1.8)), row=2, col=1)
    if dd_spy is not None:
        fig.add_trace(go.Scatter(x=df.index, y=dd_spy, name="DD SPY", line=dict(width=1.2, dash="dot")), row=2, col=1)
    fig.update_layout(title="LLM Manual Paper Trading", template="plotly_white", height=780)
    fig.write_html(str(out_dir / "llm_paper_trading_curve.html"))

    # Allocation heatmap by rebalance date
    z = []
    x = []
    for d in decisions:
        x.append(d.date.strftime("%Y-%m-%d"))
        z.append([d.weights[a] * 100.0 for a in ASSETS])
    heat = go.Figure(
        data=go.Heatmap(
            z=list(map(list, zip(*z))),
            x=x,
            y=ASSETS,
            colorscale="Blues",
            colorbar=dict(title="% peso"),
        )
    )
    heat.update_layout(title="Asignación de cartera por rebalanceo", template="plotly_white", height=480)
    heat.write_html(str(out_dir / "llm_paper_trading_allocations.html"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md_path = root / "docs" / "LLM_BACKTEST_PROMPTS.md"
    out_dir = root / "output"

    decisions = parse_decisions(md_path)
    if not decisions:
        raise RuntimeError("No se pudieron parsear decisiones del markdown.")

    start = decisions[0].date.strftime("%Y-%m-%d")
    end = (decisions[-1].date + pd.Timedelta(days=35)).strftime("%Y-%m-%d")

    mode = "market"
    try:
        prices = fetch_prices(start, end)
    except Exception:
        prices = pd.DataFrame()

    if prices.empty:
        df = simulate_offline(decisions)
        mode = "offline"
    else:
        df = simulate_with_prices(decisions, prices)
        if df.empty:
            df = simulate_offline(decisions)
            mode = "offline"

    m = metrics(df["portfolio"])
    bench = metrics(df["spy"]) if "spy" in df.columns and df["spy"].notna().any() else {}

    render_charts(df, decisions, out_dir)

    summary = {
        "mode": mode,
        "n_rebalances": len(decisions),
        "start": start,
        "end": end,
        "portfolio": m,
        "benchmark_spy": bench,
        "outputs": {
            "curve_html": "output/llm_paper_trading_curve.html",
            "alloc_html": "output/llm_paper_trading_allocations.html",
        },
    }
    (out_dir / "llm_paper_trading_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
