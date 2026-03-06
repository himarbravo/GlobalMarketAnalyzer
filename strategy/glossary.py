"""
GLOSSARY — Interpretive dictionary for all market indicators
==============================================================
Each indicator has:
  - name: human-readable name
  - explanation: what it measures
  - states: what different levels mean for investment decisions
  - refuge_impact: how it affects the refuge decision (TLT vs GLD vs cash)
"""

MARKET_INDICATORS = {
    'VIX': {
        'name': 'Índice de Volatilidad CBOE (VIX)',
        'explanation': 'Mide la volatilidad implícita esperada del S&P 500 a 30 días. '
                       'Se calcula a partir de los precios de opciones. Es el "indicador del miedo".',
        'states': {
            '<15': ('🟢 CALMA', 'Mercado muy tranquilo. Buen momento para momentum y riesgo. '
                    'Cuidado: calma extrema puede preceder shocks repentinos.'),
            '15-20': ('🟡 NORMAL', 'Nivel medio histórico. Mercado funciona con normalidad. '
                      'Sin necesidad de refugio.'),
            '20-30': ('🟠 NERVIOSISMO', 'Mercado inquieto. Considerar reducir exposición o '
                      'refugio parcial. Puede ser oportunidad contrarian si es spike temporal.'),
            '>30': ('🔴 PÁNICO', 'Miedo extremo. Refugio total recomendado. '
                    'Históricamente, VIX >35 suele marcar suelo de mercado (comprar en 1-2 semanas).'),
        },
        'refuge_impact': 'Señal principal del gate. VIX alto → activar refugio. '
                         'Pero VIX spike (>15% sobre MA20) puede ser oportunidad contrarian.',
        'unit': 'puntos',
        'source': 'yfinance (^VIX)',
    },

    'VIX_VS_MA20': {
        'name': 'VIX vs su media de 20 días',
        'explanation': 'Compara el VIX actual con su promedio de los últimos 20 días. '
                       'Detecta PICOS relativos, no niveles absolutos.',
        'states': {
            '>15% sobre MA': ('⚠️ SPIKE', 'Pico de miedo. Históricamente precede rebotes en 1-2 semanas.'),
            'cerca de MA': ('🟡 NORMAL', 'VIX en línea con su reciente. Sin alarma.'),
            '<15% bajo MA': ('⚠️ COMPLACENCIA', 'Mucha calma. Cuidado con shock repentino.'),
        },
        'refuge_impact': 'Mejor predictor que VIX fijo. Nuestro backtest mostró Sharpe 3.52 con MA20 vs 2.11 con umbral fijo.',
        'unit': '%',
        'source': 'calculado',
    },

    'SPY': {
        'name': 'SPDR S&P 500 ETF',
        'explanation': 'ETF que replica el S&P 500. Índice amplio del mercado USA (500 empresas más grandes).',
        'states': {
            'subiendo + VIX bajo': ('🟢 BULL', 'Tendencia alcista sana. Mantener momentum.'),
            'subiendo + VIX alto': ('🟡 REBOTE', 'Posible rebote técnico en mercado estresado. Cautela.'),
            'cayendo + VIX bajo': ('🟡 CORRECCIÓN', 'Corrección ordenada. Suele recuperar rápido.'),
            'cayendo + VIX alto': ('🔴 CRASH', 'Posible crash. Refugio inmediato.'),
        },
        'refuge_impact': 'Activo principal en modo momentum. Es lo que compramos cuando no hay estrés.',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'QQQ': {
        'name': 'Invesco QQQ (Nasdaq 100)',
        'explanation': 'ETF del Nasdaq 100. Dominado por tech (AAPL, MSFT, NVDA, AMZN, META). '
                       'Más sensible a tipos de interés que SPY.',
        'states': {
            'QQQ > SPY': ('📈 TECH LIDERA', 'Apetito por growth. Tipos bajos o bajando.'),
            'QQQ < SPY': ('📉 ROTACIÓN', 'Rotación fuera de tech. Tipos subiendo o risk-off.'),
        },
        'refuge_impact': 'Si QQQ cae más que SPY, el mercado está castigando growth → tipos subiendo → TLT peligroso.',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'IWM': {
        'name': 'iShares Russell 2000 (Small Caps)',
        'explanation': 'ETF de empresas pequeñas USA. Indicador de apetito por riesgo: '
                       'small caps suben más que large caps cuando hay confianza.',
        'states': {
            'IWM > SPY': ('🟢 RISK-ON', 'Dinero fluyendo a riesgo. Señal bull.'),
            'IWM < SPY': ('🟠 RISK-OFF', 'Fly-to-quality. Inversores prefieren empresas grandes/seguras.'),
        },
        'refuge_impact': 'Si IWM cae mucho más que SPY, es señal de aversión al riesgo generalizada.',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'TLT': {
        'name': 'iShares 20+ Year Treasury Bond',
        'explanation': 'ETF de bonos del Tesoro USA a largo plazo (>20 años). '
                       'Refugio clásico en crisis de PÁNICO. PELIGROSO en crisis de INFLACIÓN.',
        'states': {
            'subiendo': ('🛡️ REFUGIO ACTIVO', 'Yields bajando, flight-to-safety. TLT funciona como refugio.'),
            'cayendo': ('⚠️ REFUGIO ROTO', 'Yields subiendo o inflación. TLT PIERDE dinero. '
                        'NO usar como refugio. Ejemplo: 2022 TLT cayó -31%.'),
        },
        'refuge_impact': 'SOLO funciona cuando yields bajan (pánico/recesión). '
                         'En inflación/subida de tipos, ES EL PEOR REFUGIO POSIBLE.',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'SHY': {
        'name': 'iShares 1-3 Year Treasury Bond',
        'explanation': 'Bonos corto plazo. Prácticamente cash. Volatilidad mínima. '
                       'Refugio universal que funciona SIEMPRE pero rinde poco (~4-5% anual).',
        'states': {
            'siempre': ('🟢 ESTABLE', 'Casi nunca pierde más de 1%. El refugio más seguro.'),
        },
        'refuge_impact': 'Refugio de última instancia cuando NO confías ni en TLT ni en GLD. '
                         'Especialmente útil en crisis de inflación donde TLT cae.',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'GLD': {
        'name': 'SPDR Gold Shares',
        'explanation': 'ETF de oro físico. Refugio en incertidumbre geopolítica y devaluación monetaria. '
                       'Menos fiable en tipos reales altos (oro compite con bonos que pagan interés).',
        'states': {
            'subiendo fuerte': ('🥇 REFUGIO ACTIVO', 'Demanda de protección. Incertidumbre global.'),
            'estable/bajando': ('🟡 NEUTRAL', 'Sin demanda especial de refugio en oro.'),
        },
        'refuge_impact': 'Buen refugio en geopolítica (tariffs, guerras) y cuando la Fed baja tipos. '
                         'Menos eficaz si tipos reales están altos (bonos rinden más que oro).',
        'unit': 'USD',
        'source': 'yfinance',
    },

    'UUP': {
        'name': 'Invesco DB US Dollar Index Bullish Fund',
        'explanation': 'Proxy del dólar USA. Cuando sube = flight-to-safety global '
                       '(inversores compran USD como refugio).',
        'states': {
            'subiendo': ('💵 FLIGHT-TO-SAFETY', 'Inversores globales comprando USD. Risk-off.'),
            'bajando': ('💵 RISK-ON', 'Dólar débil = apetito por riesgo, bueno para emergentes y commodities.'),
        },
        'refuge_impact': 'Dólar fuerte + VIX alto = pánico clásico → TLT funciona. '
                         'Dólar débil + VIX alto = inflación/devaluación → GLD mejor que TLT.',
        'unit': 'USD',
        'source': 'yfinance',
    },
}

YIELD_INDICATORS = {
    'YIELD_3M': {
        'name': 'Yield 3 meses (T-Bill)',
        'explanation': 'Rendimiento de letras del tesoro a 3 meses. Proxy del tipo de interés de la Fed. '
                       'Es el "risk-free rate" — lo que puedes ganar sin riesgo.',
        'states': {
            '>5%': ('🔴 RESTRICTIVO', 'Fed muy dura. Presión sobre todo tipo de activos.'),
            '3-5%': ('🟡 NORMAL', 'Tipos moderados.'),
            '<2%': ('🟢 EXPANSIVO', 'Fed estimulando. Bueno para growth y momentum.'),
        },
        'refuge_impact': 'Si 3M yield > 10Y yield = curva invertida = recesión señalada.',
        'unit': '%',
        'source': 'yfinance (^IRX)',
    },

    'YIELD_10Y': {
        'name': 'Yield 10 años (Treasury)',
        'explanation': 'Rendimiento del bono a 10 años. La tasa de referencia más importante: '
                       'hipotecas, valoraciones de empresas, todo depende de este número.',
        'states': {
            'subiendo rápido (>0.3% en 20d)': ('⚠️ TYPES ↑', 'TLT va a CAER. NO usar TLT como refugio. '
                                                 'Growth stocks bajo presión (se descuentan peor).'),
            'bajando rápido (<-0.3% en 20d)': ('✅ TYPES ↓', 'TLT SUBE. Flight-to-safety activo. '
                                                 'TLT es buen refugio.'),
            'estable': ('🟡 NEUTRAL', 'Sin dirección clara en tipos.'),
        },
        'refuge_impact': 'LA variable más importante para decidir si TLT es refugio o trampa. '
                         'Yields subiendo = TLT cae = NO TLT. Yields bajando = TLT sube = SÍ TLT.',
        'unit': '%',
        'source': 'yfinance (^TNX)',
    },

    'YIELD_CURVE_SPREAD': {
        'name': 'Spread 10Y - 3M (Curva de tipos)',
        'explanation': 'Diferencia entre el yield a 10 años y el yield a 3 meses. '
                       'La señal de recesión más fiable que existe.',
        'states': {
            '<-0.5%': ('🔴 INVERTIDA', 'Recesión señalada en 6-18 meses. 100% track record desde 1970. '
                       'ATENCIÓN: la recesión NO empieza cuando se invierte, sino cuando se des-invierte.'),
            '-0.5% a 0%': ('🟠 PLANA', 'Casi invertida. El mercado de bonos no espera crecimiento.'),
            '0% a 1.5%': ('🟡 NORMAL', 'Curva normal. Sin señal de recesión.'),
            '>1.5%': ('🟢 EMPINADA', 'Economía acelerando. Buen momento para riesgo.'),
        },
        'refuge_impact': 'Curva invertida → recesión viene → eventualmente yields bajarán → TLT subirá. '
                         'Pero el timing es difícil — puede tardar 6-18 meses.',
        'unit': '%',
        'source': 'calculado',
    },
}

FRED_INDICATORS = {
    'CPIAUCSL': {
        'name': 'CPI — Índice de Precios al Consumo',
        'explanation': 'Mide la inflación en USA. Si crece >3% anual, la Fed sube tipos para frenarla.',
        'states': {
            'YoY >4%': ('🔴 INFLACIÓN ALTA', 'Fed subiendo tipos agresivamente. TLT CAEA. '
                        'Growth stocks bajo presión. Value y commodities mejor.'),
            'YoY 2-3%': ('🟢 OBJETIVO', 'Inflación controlada. Fed puede relajar.'),
            'YoY <2%': ('🟡 DEFLACIÓN', 'Posible recesión. Fed baja tipos → bueno para TLT.'),
        },
        'refuge_impact': 'Inflación alta = TLT es TRAMPA (pierde vs inflación). GLD o TIPS mejor.',
        'unit': 'índice',
        'source': 'FRED via Supabase',
    },

    'UNRATE': {
        'name': 'Tasa de Desempleo USA',
        'explanation': 'Porcentaje de la fuerza laboral que busca empleo. '
                       'La Fed baja tipos cuando sube mucho (para estimular).',
        'states': {
            '<4%': ('🟢 PLENO EMPLEO', 'Economía fuerte. Empresas ganan dinero.'),
            '4-6%': ('🟡 MODERADO', 'Economía enfriándose. Vigilar tendencia.'),
            '>6%': ('🔴 RECESIÓN', 'Fed va a bajar tipos agresivamente → TLT sube mucho.'),
        },
        'refuge_impact': 'Desempleo subiendo rápido → Fed baja tipos → TLT es BUEN refugio. '
                         'El mejor momento para comprar TLT es cuando desempleo empieza a subir.',
        'unit': '%',
        'source': 'FRED via Supabase',
    },

    'FEDFUNDS': {
        'name': 'Fed Funds Rate',
        'explanation': 'Tipo de interés de referencia de la Reserva Federal. '
                       'Cuando sube, todo se encarece. Cuando baja, todo se abarata.',
        'states': {
            'subiendo': ('⚠️ RESTRICTIVO', 'Fed endureciendo. Malo para growth, TLT, y deuda alta.'),
            'estable': ('🟡 PAUSA', 'Fed esperando datos. Mercado en espera.'),
            'bajando': ('✅ EXPANSIVO', 'Fed estimulando. Bueno para TLT, growth, y riesgo.'),
        },
        'refuge_impact': 'Si Fed baja tipos → TLT SUBE (buen refugio). Si sube → TLT CAE (mal refugio).',
        'unit': '%',
        'source': 'FRED via Supabase',
    },

    'T10Y2Y': {
        'name': 'Spread 10Y-2Y (FRED)',
        'explanation': 'Diferencial de yields entre bonos a 10 y 2 años desde FRED. '
                       'Complementa el spread 10Y-3M de yfinance.',
        'states': {
            'negativo': ('🔴 INVERTIDA', 'Recesión señalada. Ver YIELD_CURVE_SPREAD.'),
            'positivo': ('🟡 NORMAL', 'Sin señal de recesión.'),
        },
        'refuge_impact': 'Confirma o contradice la señal de la curva de yfinance.',
        'unit': '%',
        'source': 'FRED via Supabase',
    },

    'BAMLH0A0HYM2': {
        'name': 'High Yield Spread (ICE BofA)',
        'explanation': 'Diferencia de rendimiento entre bonos basura (high yield) y bonos del Tesoro. '
                       'Mide el ESTRÉS CREDITICIO: cuánto extra exige el mercado por prestar a empresas riesgosas.',
        'states': {
            '<3%': ('🟢 CALMA', 'Crédito fácil. Empresas se financian barato. Sin estrés.'),
            '3-5%': ('🟡 NORMAL', 'Nivel histórico medio.'),
            '5-8%': ('🟠 ESTRÉS', 'Mercado de crédito tenso. Empresas con deuda en riesgo.'),
            '>8%': ('🔴 PÁNICO', 'Crisis crediticia. Riesgo de defaults masivos. Tipo 2008.'),
        },
        'refuge_impact': 'HY spread >6% = estrés sistémico. En este caso, CASH es rey — '
                         'ni TLT ni GLD, solo SHY y money market. Posible contagio a todo.',
        'unit': '%',
        'source': 'FRED via Supabase',
    },

    'DTWEXBGS': {
        'name': 'Trade Weighted USD (Broad)',
        'explanation': 'Índice del dólar ponderado por comercio internacional. '
                       'Más representativo que DXY porque incluye más divisas.',
        'states': {
            'subiendo': ('💵 USD FUERTE', 'Capital fluyendo a USA. Malo para emergentes y exportadores.'),
            'bajando': ('💵 USD DÉBIL', 'Apetito por riesgo global. Bueno para commodities y emergentes.'),
        },
        'refuge_impact': 'Confirma o contradice la señal de UUP/DXY.',
        'unit': 'índice',
        'source': 'FRED via Supabase',
    },
}

SYSTEM_ANALYTICS = {
    'Z_SCORE_OU': {
        'name': 'Z-Score Ornstein-Uhlenbeck',
        'explanation': 'Mide cuánto se desvía un activo de su precio de equilibrio O-U. '
                       'El modelo O-U predice que los precios tienden a revertir a la media.',
        'states': {
            'z > +2': ('🔴 SOBRECOMPRADO', 'Muy por encima del equilibrio. '
                       'Alta probabilidad de caída a medio plazo SI no hay cambio fundamental.'),
            '+1 < z < +2': ('🟡 CARO', 'Por encima del equilibrio. Algo elevado.'),
            '-1 < z < +1': ('🟢 JUSTO', 'Cerca del equilibrio. Precio razonable.'),
            '-2 < z < -1': ('🟡 BARATO', 'Por debajo del equilibrio. Potencial de subida.'),
            'z < -2': ('🔴 SOBREVENDIDO', 'Muy por debajo del equilibrio. '
                       'Alta probabilidad de rebote SI no hay deterioro fundamental.'),
        },
        'refuge_impact': 'Si muchos activos tienen z>+2 simultáneamente, el mercado está caro en '
                         'agregado → mayor riesgo de corrección. Si muchos z<-2 → posible suelo.',
        'unit': 'desviaciones estándar',
        'source': 'core/heat_engine.py',
    },

    'REFUGE_SIGNAL': {
        'name': 'Señal de Refugio del Modelo O-U',
        'explanation': 'Combinación de flujos de capital, velocidad macro, y stress level '
                       'calculada por el heat engine. Rango -1 a +1.',
        'states': {
            '>+0.5': ('🔴 SALIR', 'Modelo dice: capital saliendo de equity. Ir a refugio.'),
            '-0.5 a +0.5': ('🟡 NEUTRAL', 'Sin señal clara de flujo.'),
            '<-0.5': ('🟢 ENTRAR', 'Capital entrando a equity. Risk-on.'),
        },
        'refuge_impact': 'Señal directa del modelo. Complementa al VIX gate.',
        'unit': 'score [-1, +1]',
        'source': 'core/heat_engine.py',
    },

    'S_FRACTIONAL': {
        'name': 'Exponente Fraccional (s)',
        'explanation': 'Parámetro del Laplaciano fraccional L^s del grafo. '
                       'Controla la velocidad de difusión del capital entre activos.',
        'states': {
            's ≈ 0.5': ('🟢 EFICIENTE', 'Difusión rápida. Mercado eficiente. Arbitraje funciona.'),
            's → 1.0': ('🔴 LENTO', 'Difusión lenta. Capital "atrapado". Estrés sistémico.'),
        },
        'refuge_impact': 's alto = mercado disfuncional = cuidado, la liquidez puede secarse.',
        'unit': 'adimensional [0.5, 1.0]',
        'source': 'core/graph_builder.py',
    },

    'GAMMA_INERTIA': {
        'name': 'Inercia Calibrada (γ)',
        'explanation': 'Parámetro de inercia del modelo O-U de 2do orden. '
                       'γ=1 = sin inercia (O-U puro). γ>1 = el dinero tiene momentum.',
        'states': {
            'γ ≈ 1': ('🟢 SIN INERCIA', 'Mercado reacciona rápido. Mean-reversion domina.'),
            'γ > 5': ('🟠 MOMENTUM', 'Tendencias fuertes y persistentes. No luchar contra la tendencia.'),
            'γ > 20': ('🔴 BURBUJA/CRASH', 'Inercia extrema. El mercado se mueve por momentum, no por valor.'),
        },
        'refuge_impact': 'γ alto + tendencia bajista = crash en desarrollo. No comprar dips.',
        'unit': 'adimensional',
        'source': 'core/heat_engine.py',
    },

    'VON_NEUMANN_ENTROPY': {
        'name': 'Entropía de Von Neumann',
        'explanation': 'Mide la diversidad de estructura en la matriz de correlación. '
                       'Baja entropía = todos se mueven juntos (contagio). Alta = diversificación real.',
        'states': {
            '<0.3': ('🔴 CONTAGIO', 'Activos muy correlacionados. Diversificación no funciona. '
                     'Si uno cae, todos caen. Refugio necesario.'),
            '0.3-0.7': ('🟡 PARCIAL', 'Algo de estructura. Diversificación parcial.'),
            '>0.7': ('🟢 DIVERSIFICADO', 'Activos independientes. Stock picking tiene sentido. '
                     'Puedes ser selectivo sin refugio total.'),
        },
        'refuge_impact': 'Entropía baja = todos caerán juntos = refugio total necesario. '
                         'Entropía alta = solo algunos en problemas = refugio parcial suficiente.',
        'unit': 'bits [0, 1]',
        'source': 'core/reversibility.py',
    },
}

STOCK_METRICS = {
    'MOMENTUM_SCORE': {
        'name': 'Score de Momentum Fundamental',
        'explanation': 'Puntuación 0-5 basada en: revenue growth, EPS growth, mejora de márgenes, '
                       'FCF improvement, deleveraging. Cada criterio que cumple suma 1 punto.',
        'scoring': {
            '5': 'Todos los fundamentales mejorando. Empresa en racha excepcional.',
            '4': 'Mayoría de fundamentales mejorando. Empresa sólida.',
            '3': 'Mejora moderada. Vigilar tendencia.',
            '2': 'Mejora limitada. Posible estancamiento.',
            '0-1': 'Fundamentales débiles o deteriorándose. Considerar salir.',
        },
        'unit': 'puntos [0-5]',
        'source': 'ml/fundamental_momentum.py',
    },

    'REV_QOQ': {
        'name': 'Revenue Quarter-over-Quarter',
        'explanation': 'Crecimiento de ingresos del último trimestre vs el anterior. '
                       'Métrica más importante de crecimiento orgánico.',
        'unit': '%',
        'source': 'Supabase (fundamentals_quarterly)',
    },

    'EPS_GROWTH': {
        'name': 'EPS Growth (Total Period)',
        'explanation': 'Crecimiento del beneficio por acción sobre todo el periodo disponible. '
                       'Incluye eficiencia operativa y buybacks.',
        'unit': '%',
        'source': 'Supabase (fundamentals_quarterly)',
    },

    'ROIC_TREND': {
        'name': 'ROIC Trend',
        'explanation': 'Tendencia del retorno sobre capital invertido. '
                       'Positivo = empresa mejorando su eficiencia en el uso de capital.',
        'unit': 'adimensional',
        'source': 'Supabase (fundamentals_quarterly)',
    },

    'VOL_20D': {
        'name': 'Volatilidad Anualizada (20d)',
        'explanation': 'Desviación estándar de retornos diarios × √252. '
                       'Mide cuánto se mueve el precio día a día.',
        'states': {
            '<20%': 'Baja volatilidad. Stock estable (utilities, consumer staples).',
            '20-40%': 'Volatilidad normal para growth stocks.',
            '>40%': 'Muy volátil. Alto riesgo. Puede subir o bajar mucho en poco tiempo.',
        },
        'unit': '%',
        'source': 'yfinance (calculado)',
    },

    'FROM_HIGH': {
        'name': 'Distancia al máximo de 3 meses',
        'explanation': 'Cuánto ha caído el stock desde su máximo reciente. '
                       'Proxy de la tendencia a corto plazo.',
        'states': {
            '0% a -5%': 'Cerca de máximos. Tendencia alcista intacta.',
            '-5% a -10%': 'Corrección leve. Normal en mercados sanos.',
            '-10% a -20%': 'Corrección significativa. Puede ser oportunidad si fundamentales bien.',
            '<-20%': 'Bear market para este stock. Investigar por qué.',
        },
        'unit': '%',
        'source': 'yfinance (calculado)',
    },
}

CRISIS_TYPES = {
    'PANIC': {
        'name': 'Crisis de Pánico',
        'pattern': 'VIX↑ + Yields↓ + USD↑',
        'examples': 'COVID marzo 2020, GFC sept 2008, 9/11',
        'what_happens': 'Inversores venden TODO y compran USD y bonos del Tesoro.',
        'best_refuge': 'TLT ✅ + GLD ✅',
        'worst_refuge': 'Quedarse en equity',
        'duration': '2-6 semanas típicamente. V-recovery posible.',
    },
    'INFLATION': {
        'name': 'Crisis de Inflación',
        'pattern': 'VIX↑ + Yields↑ + TLT↓',
        'examples': '2022 (Fed hiking), años 70',
        'what_happens': 'Fed sube tipos para frenar inflación. TODO cae: acciones Y bonos.',
        'best_refuge': 'SHY (cash) ✅ + Commodities',
        'worst_refuge': 'TLT ❌ (cae con acciones)',
        'duration': 'Meses a años. Gradual, no sudden.',
    },
    'GEOPOLITICAL': {
        'name': 'Crisis Geopolítica',
        'pattern': 'VIX↑ + GLD↑ + yields mixtos',
        'examples': 'Tariffs 2025, guerra Ucrania, guerra comercial US-China',
        'what_happens': 'Incertidumbre política, no macro pura. Sectores afectados selectivamente.',
        'best_refuge': 'GLD ✅ + TLT depende',
        'worst_refuge': 'Empresas expuestas a la zona de conflicto',
        'duration': 'Variable. Puede resolverse rápido o escalar.',
    },
    'LIQUIDITY': {
        'name': 'Crisis de Liquidez',
        'pattern': 'HY spread↑↑ + VIX↑ + todo cae',
        'examples': 'SVB 2023, LTCM 1998, Bear Stearns 2008',
        'what_happens': 'El sistema financiero se atasca. Nadie puede vender ni comprar.',
        'best_refuge': 'CASH ✅ (SHY, money market)',
        'worst_refuge': 'Todo lo demás (incluso oro puede caer por liquidaciones forzadas)',
        'duration': 'Días a semanas. La Fed interviene rápido.',
    },
}


def get_all_indicators():
    """Return all indicator categories as a combined dict."""
    return {
        'market': MARKET_INDICATORS,
        'yields': YIELD_INDICATORS,
        'fred': FRED_INDICATORS,
        'system': SYSTEM_ANALYTICS,
        'stocks': STOCK_METRICS,
        'crises': CRISIS_TYPES,
    }


def explain(indicator_key, category=None):
    """Get explanation for a specific indicator."""
    all_cats = [MARKET_INDICATORS, YIELD_INDICATORS, FRED_INDICATORS,
                SYSTEM_ANALYTICS, STOCK_METRICS]
    for cat in all_cats:
        if indicator_key in cat:
            return cat[indicator_key]
    return None


if __name__ == '__main__':
    # Print summary of all indicators
    all_ind = get_all_indicators()
    for cat_name, cat in all_ind.items():
        print(f"\n{'═'*50}")
        print(f"  {cat_name.upper()}")
        print(f"{'═'*50}")
        for key, ind in cat.items():
            name = ind.get('name', key)
            expl = ind.get('explanation', '')[:80]
            print(f"  {key:<20} {name}")
            print(f"  {'':20} {expl}...")
