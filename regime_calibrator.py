"""
REGIME CALIBRATOR — GlobalMarketAnalyzer
=========================================
Carga datos por periodos concretos de mercado conocidos,
clasifica regímenes, y calibra los parámetros del modelo.

Periodos históricos usados:
  1. 2007-2009: Financial Crisis (crisis)
  2. 2010-2012: Recovery + EU debt crisis (stress → normal)
  3. 2013-2015: QE expansion (calm/hype)
  4. 2016-2018: Trump rally + Q4 2018 selloff (normal → stress)
  5. 2019-2020: COVID crash and recovery (normal → crisis → hype)
  6. 2021-2022: Post-COVID inflation + rate hikes (hype → stress)
  7. 2023-2024: AI boom (normal → hype)

Para cada régimen calibra:
  - λ_eq: múltiplo de equilibrio (PE median)
  - α_m: velocidad de difusión del dinero
  - κ: acoplamiento precio↔capital
  - β: sensibilidades macro
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time
import json
from pathlib import Path


# ── Periodos históricos conocidos ──
MARKET_PERIODS = {
    "financial_crisis": {
        "start": "2007-06-01", "end": "2009-06-30",
        "regime": "crisis",
        "description": "Subprime → Lehman → QE1",
    },
    "recovery_eu_crisis": {
        "start": "2010-01-01", "end": "2012-12-31",
        "regime": "stress",
        "description": "QE2, EU sovereign debt, US downgrade",
    },
    "qe_expansion": {
        "start": "2013-01-01", "end": "2015-12-31",
        "regime": "calm",
        "description": "QE3, low vol, steady growth",
    },
    "trump_rally": {
        "start": "2016-01-01", "end": "2018-12-31",
        "regime": "normal",
        "description": "Tax cuts, trade war, Q4 2018 selloff",
    },
    "covid": {
        "start": "2019-06-01", "end": "2020-12-31",
        "regime": "crisis",
        "description": "COVID crash (-34% in 23 days), V-recovery",
    },
    "post_covid_inflation": {
        "start": "2021-01-01", "end": "2022-12-31",
        "regime": "stress",
        "description": "Meme stocks, inflation, Fed hikes 0→5%",
    },
    "ai_boom": {
        "start": "2023-01-01", "end": "2024-12-31",
        "regime": "hype",
        "description": "ChatGPT, NVDA ×10, Mag7 dominance",
    },
}

# Tickers representativos por sector (presentes en todo el historial)
CALIBRATION_TICKERS = [
    # Tech megacap
    "AAPL", "MSFT", "GOOG", "AMZN",
    # Financials
    "JPM", "GS", "BAC",
    # Energy
    "XOM", "CVX",
    # Healthcare
    "JNJ", "PFE", "UNH",
    # Consumer
    "WMT", "PG", "KO",
    # Industrial
    "CAT", "HON", "GE",
    # ETFs
    "SPY", "QQQ",
]

# Macro tickers
MACRO_TICKERS = ["^VIX", "^GSPC", "^IXIC", "DX-Y.NYB",
                  "GC=F", "CL=F", "HG=F",
                  "^TNX", "^TYX", "^IRX"]


def download_period(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Descarga precios para un periodo concreto."""
    try:
        data = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False
        )
        if data.empty:
            return pd.DataFrame()
        # Extraer Close
        if isinstance(data.columns, pd.MultiIndex):
            closes = data.xs("Close", axis=1, level="Price")
        else:
            closes = data[["Close"]].copy()
            closes.columns = [tickers[0]] if len(tickers) == 1 else tickers
        return closes
    except Exception as e:
        print(f"    ✗ Error descargando {start}→{end}: {e}")
        return pd.DataFrame()


def compute_regime_metrics(prices: pd.DataFrame,
                           macro: pd.DataFrame) -> dict:
    """Calcula métricas del régimen para un periodo."""
    returns = prices.pct_change().dropna()

    # Volatilidad realizada (anualizada)
    vol = returns.std() * np.sqrt(252)

    # VIX medio
    vix_col = [c for c in macro.columns if "VIX" in c.upper() or c == "^VIX"]
    vix_mean = float(macro[vix_col[0]].mean()) if vix_col and not macro.empty else 20.0

    # Retorno total del periodo
    total_return = {}
    for col in prices.columns:
        p = prices[col].dropna()
        if len(p) > 1:
            total_return[col] = float(p.iloc[-1] / p.iloc[0] - 1)

    # Cross-sectional dispersion (diferencia entre mejor y peor)
    if total_return:
        dispersion = max(total_return.values()) - min(total_return.values())
    else:
        dispersion = 0.0

    # s(t) estimado: basado en VIX
    # s = 1 - clip((VIX-12)/30, 0, 1) — normalización simplificada
    s_mean = max(0.15, 1.0 - min(1.0, (vix_mean - 12) / 30))

    # λ proxy = precio medio / earnings proxy
    # Usamos P/E del SPY como proxy de λ del mercado
    spy_data = prices.get("SPY")
    if spy_data is not None and len(spy_data.dropna()) > 0:
        spy_last = float(spy_data.dropna().iloc[-1])
        # SPY earnings yield histórico ~6% → PE ~17
        # Ajustar por nivel de precios
        lambda_market = spy_last / 25  # proxy: SPY $400 → λ ≈ 16
    else:
        lambda_market = 18.0

    return {
        "vol_mean": float(vol.mean()),
        "vix_mean": vix_mean,
        "s_mean": s_mean,
        "total_returns": total_return,
        "dispersion": dispersion,
        "lambda_market_proxy": lambda_market,
        "n_observations": len(returns),
    }


def compute_diffusion_alpha(returns: pd.DataFrame, lag: int = 1) -> float:
    """Estima α de difusión a partir de autocorrelación de retornos."""
    autocorrs = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) > lag + 10:
            ac = float(r.autocorr(lag=lag))
            if not np.isnan(ac):
                autocorrs.append(ac)

    if not autocorrs:
        return 0.01

    mean_ac = np.mean(autocorrs)
    # α ≈ -log(autocorr) / Δt para O-U
    if mean_ac > 0 and mean_ac < 1:
        alpha = -np.log(mean_ac)
    elif mean_ac <= 0:
        alpha = 0.1  # mean-reversion fuerte
    else:
        alpha = 0.005  # muy poca mean-reversion
    return float(np.clip(alpha, 0.005, 0.25))


def compute_kappa(returns: pd.DataFrame, lambda_vals: dict) -> float:
    """Estima κ = velocidad de convergencia precio→capital."""
    # κ se estima como la pendiente de la regresión:
    # Δu(t+h) = κ · (λ_eq - λ(t)) + ε
    # Con datos limitados, usamos la reversión a la media del PE ratio
    # Proxy: autocorrelación de PE → velocidad de convergencia

    # Con solo precios, estimamos κ desde la velocidad de mean-reversion
    # de la ratio precio/media(precio) — proxy de λ
    kappas = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 60:
            continue
        # Price relative to 60d mean
        prices_cumul = (1 + r).cumprod()
        ratio = prices_cumul / prices_cumul.rolling(60).mean()
        ratio = ratio.dropna()
        if len(ratio) < 20:
            continue
        # Autocorrelación del ratio → velocidad de reversion
        ac = float(ratio.autocorr(lag=5))
        if not np.isnan(ac) and 0 < ac < 1:
            k = -np.log(ac) / 5
            kappas.append(k)

    return float(np.median(kappas)) if kappas else 0.02


def compute_macro_betas(returns: pd.DataFrame,
                         macro: pd.DataFrame) -> dict:
    """Calcula sensibilidades β a factores macro."""
    macro_returns = macro.pct_change().dropna()

    # Alinear fechas
    common = returns.index.intersection(macro_returns.index)
    if len(common) < 30:
        return {}

    r_aligned = returns.loc[common]
    m_aligned = macro_returns.loc[common]

    betas = {}
    for macro_col in m_aligned.columns:
        m = m_aligned[macro_col].values
        m_valid = ~np.isnan(m)
        if m_valid.sum() < 20:
            continue

        col_betas = []
        for asset_col in r_aligned.columns:
            a = r_aligned[asset_col].values
            valid = m_valid & ~np.isnan(a)
            if valid.sum() < 20:
                continue
            # Regresión simple
            x = m[valid]
            y = a[valid]
            if np.std(x) > 1e-10:
                beta = float(np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x))
                col_betas.append(beta)

        if col_betas:
            betas[macro_col] = {
                "mean_beta": float(np.mean(col_betas)),
                "std_beta": float(np.std(col_betas)),
            }

    return betas


def calibrate_all():
    """Ejecuta calibración completa por periodos."""
    print("=" * 65)
    print("  CALIBRACIÓN POR RÉGIMEN HISTÓRICO")
    print("=" * 65)

    all_results = {}

    for period_name, period_info in MARKET_PERIODS.items():
        start = period_info["start"]
        end = period_info["end"]
        regime = period_info["regime"]

        print(f"\n{'─' * 65}")
        print(f"  📊 {period_name}: {period_info['description']}")
        print(f"     {start} → {end} | Régimen esperado: {regime}")
        print(f"{'─' * 65}")

        # 1. Descargar precios
        print("  Descargando precios...", end=" ")
        prices = download_period(CALIBRATION_TICKERS, start, end)
        if prices.empty:
            print("✗ Sin datos")
            continue
        print(f"✓ {len(prices)} días, {len(prices.columns)} tickers")

        # 2. Descargar macro
        print("  Descargando macro...", end=" ")
        macro = download_period(MACRO_TICKERS, start, end)
        print(f"✓ {len(macro)} días" if not macro.empty else "✗ Sin datos")
        time.sleep(1)  # Rate limiting

        # 3. Calcular retornos
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            print("  ⚠ Muy pocos datos, saltando")
            continue

        # 4. Métricas del régimen
        metrics = compute_regime_metrics(prices, macro)

        # 5. Calibrar α (difusión)
        alpha = compute_diffusion_alpha(returns)
        print(f"  α (difusión) = {alpha:.4f}")

        # 6. Calibrar κ (acoplamiento precio↔capital)
        kappa = compute_kappa(returns, {})
        print(f"  κ (acoplamiento) = {kappa:.4f}")

        # 7. Sensibilidades macro β
        betas = compute_macro_betas(returns, macro)
        if betas:
            vix_beta = betas.get("^VIX", {}).get("mean_beta", None)
            print(f"  β_VIX = {vix_beta:.4f}" if vix_beta else "  β_VIX = N/A")

        # 8. λ_eq del régimen
        lambda_eq = metrics["lambda_market_proxy"]
        print(f"  λ_eq (equilibrio) = {lambda_eq:.1f}")
        print(f"  s(t) medio = {metrics['s_mean']:.3f}")
        print(f"  VIX medio = {metrics['vix_mean']:.1f}")
        print(f"  Vol media anual = {metrics['vol_mean']:.1%}")

        # Top/bottom retornos
        tr = metrics["total_returns"]
        if tr:
            sorted_tr = sorted(tr.items(), key=lambda x: x[1], reverse=True)
            top3 = sorted_tr[:3]
            bot3 = sorted_tr[-3:]
            print(f"  Top 3: {', '.join(f'{t}={r:+.0%}' for t,r in top3)}")
            print(f"  Bot 3: {', '.join(f'{t}={r:+.0%}' for t,r in bot3)}")

        all_results[period_name] = {
            "regime": regime,
            "start": start,
            "end": end,
            "alpha": alpha,
            "kappa": kappa,
            "lambda_eq": lambda_eq,
            "s_mean": metrics["s_mean"],
            "vix_mean": metrics["vix_mean"],
            "vol_mean": metrics["vol_mean"],
            "dispersion": metrics["dispersion"],
            "n_observations": metrics["n_observations"],
            "betas": betas,
            "total_returns": tr,
        }

    # ── Resumen por régimen ──
    print("\n" + "=" * 65)
    print("  PARÁMETROS CALIBRADOS POR RÉGIMEN")
    print("=" * 65)

    regime_params = {}
    for regime in ["crisis", "stress", "normal", "calm", "hype"]:
        periods = {k: v for k, v in all_results.items() if v["regime"] == regime}
        if not periods:
            continue

        alphas = [v["alpha"] for v in periods.values()]
        kappas = [v["kappa"] for v in periods.values()]
        lambdas = [v["lambda_eq"] for v in periods.values()]
        s_vals = [v["s_mean"] for v in periods.values()]
        vix_vals = [v["vix_mean"] for v in periods.values()]

        params = {
            "alpha": float(np.mean(alphas)),
            "kappa": float(np.mean(kappas)),
            "lambda_eq": float(np.mean(lambdas)),
            "s_mean": float(np.mean(s_vals)),
            "vix_mean": float(np.mean(vix_vals)),
            "n_periods": len(periods),
            "periods": list(periods.keys()),
        }
        regime_params[regime] = params

        print(f"\n  {regime.upper()}")
        print(f"    α = {params['alpha']:.4f} | κ = {params['kappa']:.4f} | "
              f"λ_eq = {params['lambda_eq']:.1f}")
        print(f"    s = {params['s_mean']:.3f} | VIX = {params['vix_mean']:.1f} | "
              f"N periodos = {params['n_periods']}")

    # Guardar resultados
    output = {
        "calibration_date": pd.Timestamp.now().isoformat(),
        "periods": all_results,
        "regime_params": regime_params,
    }

    # Serializar para JSON (convertir numpy types)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    output_path = Path("calibration_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  💾 Resultados guardados en {output_path}")

    return output


if __name__ == "__main__":
    calibrate_all()
