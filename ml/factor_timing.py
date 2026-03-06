"""
FACTOR TIMING — Regime-conditional stock selection
=====================================================
Uses the HMM regime to adjust which 'factor style' to prioritize:
  - BULL  → momentum (high growth, price appreciation)
  - BEAR  → quality  (high ROIC, low debt, stable earnings)
  - NEUTRAL → dividend yield + low volatility

Usage:
    from ml.factor_timing import apply_factor_timing
    adjusted = apply_factor_timing(stocks, regime='bear')
"""

import numpy as np


# Factor weights by regime [momentum, quality, value, low_vol]
REGIME_WEIGHTS = {
    'bull':    {'momentum': 0.50, 'quality': 0.20, 'value': 0.10, 'low_vol': 0.20},
    'neutral': {'momentum': 0.25, 'quality': 0.30, 'value': 0.20, 'low_vol': 0.25},
    'bear':    {'momentum': 0.05, 'quality': 0.40, 'value': 0.35, 'low_vol': 0.20},
}


def _score_momentum(stock):
    """Score based on price momentum and EPS growth."""
    score = 0
    eps = stock.get('eps_growth', 0) or 0
    rev = stock.get('rev_growth', 0) or 0

    # EPS growth
    if eps > 50:
        score += 3
    elif eps > 20:
        score += 2
    elif eps > 0:
        score += 1
    elif eps < -20:
        score -= 2

    # Revenue growth
    if rev > 20:
        score += 2
    elif rev > 5:
        score += 1
    elif rev < -5:
        score -= 1

    return score


def _score_quality(stock):
    """Score based on ROIC, margin stability, low volatility."""
    score = 0
    roic = stock.get('roic_trend', 0) or 0
    vol = stock.get('volatility', 50) or 50

    # ROIC trend positive = quality
    if roic > 5:
        score += 3
    elif roic > 0:
        score += 1
    elif roic < -5:
        score -= 2

    # Low volatility = quality
    if vol < 25:
        score += 2
    elif vol < 35:
        score += 1
    elif vol > 60:
        score -= 2

    # Positive EPS (not losing money)
    eps = stock.get('eps_growth', 0) or 0
    if eps > 0:
        score += 1
    elif eps < -50:
        score -= 3

    return score


def _score_value(stock):
    """Score based on low valuation (price beaten down, but fundamentals OK)."""
    score = 0
    price_chg = stock.get('price_chg_20d', 0) or 0
    rev = stock.get('rev_growth', 0) or 0
    eps = stock.get('eps_growth', 0) or 0

    # Price beaten down
    if price_chg < -10:
        score += 2
    elif price_chg < -5:
        score += 1

    # But fundamentals are OK (not a value trap)
    if rev > 0 and eps > 0:
        score += 2  # growing but cheap
    elif rev > 0:
        score += 1

    return score


def _score_low_vol(stock):
    """Score based on low volatility and stability."""
    vol = stock.get('volatility', 50) or 50
    score = max(0, (60 - vol) / 10)  # 0-6 scale, lower vol = higher score
    return round(score, 1)


def apply_factor_timing(stocks, regime='neutral'):
    """
    Reweight stocks based on regime-specific factor preferences.

    Args:
        stocks: list of stock dicts (from momentum ranking)
        regime: 'bull', 'neutral', or 'bear' (from HMM)

    Returns:
        dict with:
            regime: current regime
            factor_weights: which factors are emphasized
            ranked_stocks: stocks re-ranked by regime-adjusted score
            regime_advice: text advice for current regime
    """
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['neutral'])

    # Score each stock on all factors
    scored = []
    for s in stocks:
        m = _score_momentum(s)
        q = _score_quality(s)
        v = _score_value(s)
        lv = _score_low_vol(s)

        # Weighted composite score
        composite = (weights['momentum'] * m +
                     weights['quality'] * q +
                     weights['value'] * v +
                     weights['low_vol'] * lv)

        scored.append({
            'ticker': s.get('ticker', '?'),
            'sector': s.get('sector', '?'),
            'composite_score': round(composite, 2),
            'momentum_score': m,
            'quality_score': q,
            'value_score': v,
            'low_vol_score': lv,
            'original_score': s.get('score', 0),
        })

    # Sort by composite score
    scored.sort(key=lambda x: x['composite_score'], reverse=True)

    # Regime-specific advice
    advice = {
        'bull': ("BULL: Priorizar momentum — empresas con EPS/revenue acelerando. "
                 "Reducir defensivos. Tolerar mayor volatilidad por mayor upside."),
        'neutral': ("NEUTRAL: Equilibrio entre momentum y calidad. "
                    "Preferir empresas con dividendo y baja volatilidad."),
        'bear': ("BEAR: Priorizar calidad y valor — ROIC alto, baja deuda, "
                 "fundamentales sólidos pese a precio castigado. "
                 "Evitar momentum puro (las caídas se aceleran). "
                 "Buscar value que NO sea trampa (rev > 0)."),
    }

    return {
        'regime': regime,
        'factor_weights': weights,
        'ranked_stocks': scored[:15],  # top 15
        'advice': advice.get(regime, advice['neutral']),
    }


if __name__ == '__main__':
    import json
    # Simulate some stocks
    test_stocks = [
        {'ticker': 'AAPL', 'sector': 'Tech', 'eps_growth': 18, 'rev_growth': 40,
         'roic_trend': 5, 'volatility': 22, 'price_chg_20d': -3, 'score': 6},
        {'ticker': 'NVDA', 'sector': 'Tech', 'eps_growth': 98, 'rev_growth': 80,
         'roic_trend': 10, 'volatility': 55, 'price_chg_20d': -8, 'score': 5},
        {'ticker': 'V', 'sector': 'Financials', 'eps_growth': 17, 'rev_growth': 12,
         'roic_trend': 8, 'volatility': 20, 'price_chg_20d': 2, 'score': 4},
        {'ticker': 'BA', 'sector': 'Industrials', 'eps_growth': -287, 'rev_growth': -5,
         'roic_trend': -15, 'volatility': 45, 'price_chg_20d': -12, 'score': 1},
        {'ticker': 'AMGN', 'sector': 'Health', 'eps_growth': 111, 'rev_growth': 8,
         'roic_trend': 3, 'volatility': 28, 'price_chg_20d': -2, 'score': 4},
    ]

    for regime in ['bull', 'neutral', 'bear']:
        print(f"\n=== {regime.upper()} ===")
        result = apply_factor_timing(test_stocks, regime=regime)
        print(f"Advice: {result['advice']}")
        print(f"Weights: {result['factor_weights']}")
        for s in result['ranked_stocks']:
            print(f"  {s['ticker']}: composite={s['composite_score']:.2f} "
                  f"(mom={s['momentum_score']}, qual={s['quality_score']}, "
                  f"val={s['value_score']}, lv={s['low_vol_score']})")
