# 🏆 Estrategia: VIX Gate + Momentum

# Uso rápido
```bash
# Primero activar el entorno (solo una vez por terminal)
source .venv/bin/activate

# Dashboard (abrir http://localhost:8050)
PYTHONPATH=. python dashboard/api.py

# Señal de hoy
PYTHONPATH=. python strategy/daily_signal.py

# Backtest completo
PYTHONPATH=. python strategy/backtest.py --vix 20 --top 10

# Actualizar portfolio (tras earnings)
PYTHONPATH=. python strategy/select_stocks.py --top 10

# Test en 19 periodos históricos
PYTHONPATH=. python strategy/walk_forward_backtest.py
```

## Regla

```
Cada día:
  - Si VIX_hoy > media(VIX últimos 20 días) → REFUGIO (50% TLT + 50% GLD)
  - Si VIX_hoy ≤ media(VIX últimos 20 días) → MOMENTUM (SPY o TOP 10 stocks)

Cada trimestre (tras earnings):
  - Recalcular momentum scores → actualizar TOP 10
```

## Resultados (21 años, 2004-2026)

### VIX Fijo vs Relativo

| Método | Mejor regla | Ret/año | Sharpe | MaxDD | Switches |
|---|---|---|---|---|---|
| B&H SPY | — | +10.6% | 0.56 | -55.2% | 0 |
| Fijo | VIX > 18 | +25.5% | 2.11 | -25.9% | 315 |
| Percentil | p60 de 60d | +39.1% | 2.85 | -13.8% | 579 |
| Media móvil | VIX > MA20 | +48.9% | 3.52 | -13.1% | 819 |

### Por periodos (19 test periods, gate gana 14/19 = 74%)

Gana en: GFC 2008, Euro debt, QE, China scare, Volmageddon, pre-COVID, COVID crash, 2022 bear, tariffs 2025  
Pierde en: V-recoveries (post-GFC, COVID recovery, meme stocks)

## ⚠️ Estimación realista

```
Backtest bruto (MA20):     +48.9%/año, Sharpe 3.52
  - Costes transacción:     -4.0%/año (39 switches)
  - Slippage + delay:       -2.0%/año
  - Overfitting correction: ×0.7
  ≈ Estimación realista:    ~18-25%/año, Sharpe 1.0-1.5
```

Comparable a un **buen fondo activo** (top 25%) sin comisiones de gestión.

## ⚠️ Debilidades conocidas

1. **Whipsaw**: muchos switches en mercados laterales → costes
2. **Flash crash**: VIX sube DESPUÉS de la caída, ya es tarde
3. **TLT falla en inflación** (2022: TLT -31%)
4. **Overfitting**: elegimos MA20 retroactivamente
5. **Delay ejecución**: ves VIX al cierre, ejecutas al open
6. **V-recoveries**: te pierdes el rebote si VIX sigue alto

## Scripts

| Script | Descripción |
|---|---|
| `backtest.py` | Backtest con VIX fijo, variantes de threshold |
| `select_stocks.py` | Ranking trimestral momentum desde Supabase |
| `daily_signal.py` | Señal diaria + Telegram bot |
| `walk_forward_backtest.py` | Test en 19 periodos históricos (2004-2026) |

## Próximos pasos

- [ ] Paper trading 6 meses (señales diarias sin dinero real)
- [ ] Walk-forward momentum stock selection (requiere más fundamentals data)
- [ ] Testear alternativas de refugio (SHY, cash, TIPS)
- [ ] Implementar banda de histéresis para reducir whipsaw
