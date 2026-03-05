# 🏆 Estrategia Ganadora: Momentum + VIX Gate

## Resultados (backtest 5 años, 2020-2026)

| Estrategia | Ret total | Sharpe | MaxDD | 100€→ |
|---|---|---|---|---|
| B&H SPY | +131% | 0.76 | -33.7% | 231€ |
| Momentum puro (TOP 10) | +333% | 1.02 | -36.2% | 433€ |
| **Momentum + VIX>20** ⭐ | **+755%** | **2.52** | **-19.4%** | **855€** |

## Regla (así de simple)

```
Cada día a las 21:30 CET:
  - Si VIX < 20 → comprar TOP 10 momentum stocks (equal weight)
  - Si VIX ≥ 20 → todo a 50% TLT + 50% GLD

Cada trimestre (tras earnings):
  - Recalcular momentum scores → actualizar TOP 10
```

## Scripts

| Script | Descripción |
|---|---|
| `backtest.py` | Backtest completo 5 años con variantes de VIX threshold |
| `select_stocks.py` | Calcula momentum scores y selecciona TOP N stocks |
| `daily_signal.py` | Señal diaria: ¿momentum o refugio? (para cron/bot) |

## Variantes testeadas

| VIX Gate | Ret 5y | Sharpe | MaxDD | % en refugio |
|---|---|---|---|---|
| Sin gate (puro) | +333% | 1.02 | -36.2% | 0% |
| **VIX > 20** ⭐ | **+755%** | **2.52** | -19.4% | 44% |
| VIX > 25 | +793% | 2.18 | -20.5% | 22% |
| VIX > 30 | +790% | 1.95 | -23.5% | 9% |

## ⚠️ Advertencias

1. **Look-ahead bias**: el backtest usa los TOP 10 de hoy retroactivamente
2. **Sin costes de transacción**: ~1.2%/año en comisiones reales
3. **Concentración**: 10 stocks = alto riesgo idiosincrático
4. **Paper trading obligatorio**: mínimo 3-6 meses antes de dinero real
