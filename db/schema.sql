-- ============================================================
-- SCHEMA — GlobalMarketAnalyzer (Supabase / PostgreSQL)
-- ============================================================
-- Ejecuta en: Dashboard → SQL Editor → New query → pega y run
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- 1. ASSETS (catálogo maestro de activos)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS assets (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    asset_type  TEXT NOT NULL DEFAULT 'equity'
                    CHECK (asset_type IN ('equity','etf','crypto','bond','commodity','index')),
    currency    TEXT DEFAULT 'USD',
    exchange    TEXT,
    country     TEXT DEFAULT 'US',
    is_active   BOOLEAN DEFAULT TRUE,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- 2. PRICES (serie temporal OHLCV + indicadores técnicos)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL REFERENCES assets(ticker),
    date        DATE NOT NULL,

    -- ── OHLCV ──
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION NOT NULL,
    adj_close   DOUBLE PRECISION,
    volume      BIGINT,

    -- ── MEDIAS MÓVILES ──
    sma_5       DOUBLE PRECISION,
    sma_10      DOUBLE PRECISION,
    sma_20      DOUBLE PRECISION,
    sma_50      DOUBLE PRECISION,
    sma_100     DOUBLE PRECISION,
    sma_200     DOUBLE PRECISION,
    ema_9       DOUBLE PRECISION,
    ema_12      DOUBLE PRECISION,
    ema_21      DOUBLE PRECISION,
    ema_26      DOUBLE PRECISION,
    ema_50      DOUBLE PRECISION,

    -- ── RSI ──
    rsi_14      DOUBLE PRECISION,
    rsi_7       DOUBLE PRECISION,       -- RSI rápido (scalping / short-term)

    -- ── MACD ──
    macd        DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist   DOUBLE PRECISION,

    -- ── STOCHASTIC OSCILLATOR ──
    stoch_k     DOUBLE PRECISION,       -- %K (14 periodos)
    stoch_d     DOUBLE PRECISION,       -- %D (media de 3 periodos de %K)

    -- ── WILLIAMS %R ──
    williams_r  DOUBLE PRECISION,       -- Periodo 14

    -- ── ADX (Average Directional Index) ──
    adx         DOUBLE PRECISION,       -- Fuerza de tendencia (14 periodos)
    plus_di     DOUBLE PRECISION,       -- +DI
    minus_di    DOUBLE PRECISION,       -- -DI

    -- ── BOLLINGER BANDS ──
    bb_upper    DOUBLE PRECISION,
    bb_middle   DOUBLE PRECISION,
    bb_lower    DOUBLE PRECISION,
    bb_width    DOUBLE PRECISION,       -- (upper - lower) / middle
    bb_pct      DOUBLE PRECISION,       -- Posición del precio dentro de las bandas (0-1)

    -- ── KELTNER CHANNELS ──
    keltner_upper DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,

    -- ── ATR (Average True Range) ──
    atr_14      DOUBLE PRECISION,
    atr_7       DOUBLE PRECISION,       -- ATR corto

    -- ── VOLATILIDAD ──
    vol_5d      DOUBLE PRECISION,       -- Realizada 5 días (anualizada)
    vol_10d     DOUBLE PRECISION,       -- Realizada 10 días
    vol_20d     DOUBLE PRECISION,       -- Realizada 20 días (estándar)
    vol_60d     DOUBLE PRECISION,       -- Realizada trimestral

    -- ── VOLUMEN ──
    obv             DOUBLE PRECISION,   -- On-Balance Volume
    vwap            DOUBLE PRECISION,   -- Volume Weighted Avg Price (intraday aprox)
    volume_sma_20   DOUBLE PRECISION,   -- Media de volumen 20d
    volume_ratio    DOUBLE PRECISION,   -- Volumen actual / media 20d (>1 = alto)
    mfi             DOUBLE PRECISION,   -- Money Flow Index (14 periodos)
    cmf             DOUBLE PRECISION,   -- Chaikin Money Flow (20 periodos)

    -- ── ICHIMOKU CLOUD ──
    ichimoku_tenkan   DOUBLE PRECISION, -- Tenkan-sen (9 periodos)
    ichimoku_kijun    DOUBLE PRECISION, -- Kijun-sen (26 periodos)
    ichimoku_senkou_a DOUBLE PRECISION, -- Senkou Span A
    ichimoku_senkou_b DOUBLE PRECISION, -- Senkou Span B (52 periodos)

    -- ── PIVOTES ──
    pivot       DOUBLE PRECISION,       -- Pivot Point (H+L+C)/3
    pivot_r1    DOUBLE PRECISION,       -- Resistencia 1
    pivot_s1    DOUBLE PRECISION,       -- Soporte 1
    pivot_r2    DOUBLE PRECISION,       -- Resistencia 2
    pivot_s2    DOUBLE PRECISION,       -- Soporte 2

    -- ── RETORNOS (%) ──
    returns_1d  DOUBLE PRECISION,
    returns_2d  DOUBLE PRECISION,
    returns_3d  DOUBLE PRECISION,
    returns_5d  DOUBLE PRECISION,
    returns_10d DOUBLE PRECISION,
    returns_20d DOUBLE PRECISION,
    returns_60d DOUBLE PRECISION,
    returns_120d DOUBLE PRECISION,
    returns_252d DOUBLE PRECISION,      -- Retorno anual

    -- ── RIESGO ──
    sharpe_20d  DOUBLE PRECISION,       -- Sharpe ratio ventana 20d
    sharpe_60d  DOUBLE PRECISION,       -- Sharpe ratio ventana 60d
    sortino_20d DOUBLE PRECISION,       -- Sortino ratio (solo downside vol)
    max_drawdown_20d DOUBLE PRECISION,  -- Max drawdown últimos 20 días (%)
    max_drawdown_60d DOUBLE PRECISION,  -- Max drawdown últimos 60 días (%)

    -- ── PRECIO RELATIVO ──
    dist_sma_20  DOUBLE PRECISION,      -- Distancia % al SMA20
    dist_sma_50  DOUBLE PRECISION,      -- Distancia % al SMA50
    dist_sma_200 DOUBLE PRECISION,      -- Distancia % al SMA200
    dist_high_52w DOUBLE PRECISION,     -- Distancia % al máximo 52 semanas
    dist_low_52w  DOUBLE PRECISION,     -- Distancia % al mínimo 52 semanas
    high_52w    DOUBLE PRECISION,       -- Máximo 52 semanas
    low_52w     DOUBLE PRECISION,       -- Mínimo 52 semanas

    -- ── GAPS ──
    gap_pct     DOUBLE PRECISION,       -- Gap de apertura vs cierre anterior (%)

    UNIQUE (ticker, date)
);


-- ─────────────────────────────────────────────────────────────
-- 3. FUNDAMENTALS (datos trimestrales por empresa)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fundamentals (
    id               BIGSERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL REFERENCES assets(ticker),
    report_date      DATE NOT NULL,
    fiscal_quarter   TEXT,

    -- Valoración
    market_cap       DOUBLE PRECISION,
    enterprise_value DOUBLE PRECISION,
    pe_ratio         DOUBLE PRECISION,
    forward_pe       DOUBLE PRECISION,
    peg_ratio        DOUBLE PRECISION,
    pb_ratio         DOUBLE PRECISION,
    ps_ratio         DOUBLE PRECISION,
    ev_ebitda        DOUBLE PRECISION,
    ev_revenue       DOUBLE PRECISION,
    price_to_fcf     DOUBLE PRECISION,

    -- Cuenta de resultados
    revenue          DOUBLE PRECISION,
    revenue_growth   DOUBLE PRECISION,
    earnings         DOUBLE PRECISION,
    earnings_growth  DOUBLE PRECISION,
    ebitda           DOUBLE PRECISION,
    gross_margin     DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    net_margin       DOUBLE PRECISION,
    eps              DOUBLE PRECISION,
    eps_growth       DOUBLE PRECISION,

    -- Cash flow
    operating_cash_flow DOUBLE PRECISION,
    free_cash_flow      DOUBLE PRECISION,
    capex               DOUBLE PRECISION,

    -- Balance
    total_assets     DOUBLE PRECISION,
    total_debt       DOUBLE PRECISION,
    cash             DOUBLE PRECISION,
    net_debt         DOUBLE PRECISION,
    debt_to_equity   DOUBLE PRECISION,
    current_ratio    DOUBLE PRECISION,
    quick_ratio      DOUBLE PRECISION,
    working_capital  DOUBLE PRECISION,

    -- Eficiencia
    roe              DOUBLE PRECISION,
    roa              DOUBLE PRECISION,
    roic             DOUBLE PRECISION,
    asset_turnover   DOUBLE PRECISION,
    inventory_turnover DOUBLE PRECISION,

    -- Dividendos y accionistas
    dividend_yield      DOUBLE PRECISION,
    payout_ratio        DOUBLE PRECISION,
    buyback_yield       DOUBLE PRECISION,

    -- Riesgo
    beta                DOUBLE PRECISION,
    shares_outstanding  DOUBLE PRECISION,
    float_shares        DOUBLE PRECISION,
    insider_pct         DOUBLE PRECISION,
    institutional_pct   DOUBLE PRECISION,

    -- Analistas
    target_mean_price   DOUBLE PRECISION,
    target_high_price   DOUBLE PRECISION,
    target_low_price    DOUBLE PRECISION,
    recommendation      TEXT,               -- 'buy','hold','sell','strong_buy'
    num_analysts        INTEGER,

    UNIQUE (ticker, report_date)
);


-- ─────────────────────────────────────────────────────────────
-- 4. MACRO INDICATORS (diarios)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS macro_indicators (
    date        DATE PRIMARY KEY,

    -- Volatilidad y miedo
    vix             DOUBLE PRECISION,   -- CBOE VIX
    vix_3m          DOUBLE PRECISION,   -- VIX 3 meses
    vvix            DOUBLE PRECISION,   -- Vol del VIX (volatilidad de volatilidad)

    -- Renta fija USA (curva de tipos)
    yield_3m        DOUBLE PRECISION,
    yield_2y        DOUBLE PRECISION,
    yield_5y        DOUBLE PRECISION,
    yield_10y       DOUBLE PRECISION,
    yield_30y       DOUBLE PRECISION,
    yield_spread_10y_2y  DOUBLE PRECISION,  -- 10Y - 2Y (recesión si negativo)
    yield_spread_10y_3m  DOUBLE PRECISION,  -- 10Y - 3M

    -- Divisas
    dxy             DOUBLE PRECISION,   -- Dollar Index
    eurusd          DOUBLE PRECISION,
    usdjpy          DOUBLE PRECISION,
    gbpusd          DOUBLE PRECISION,

    -- Materias primas
    gold            DOUBLE PRECISION,   -- $/oz
    silver          DOUBLE PRECISION,   -- $/oz
    oil_wti         DOUBLE PRECISION,   -- $/bbl
    oil_brent       DOUBLE PRECISION,   -- $/bbl
    natural_gas     DOUBLE PRECISION,
    copper          DOUBLE PRECISION,   -- Indicador económico adelantado

    -- Índices principales
    sp500           DOUBLE PRECISION,
    nasdaq          DOUBLE PRECISION,
    dow_jones       DOUBLE PRECISION,
    russell_2000    DOUBLE PRECISION,

    -- Internacionales
    stoxx_600       DOUBLE PRECISION,   -- Europa
    nikkei          DOUBLE PRECISION,   -- Japón
    shanghai        DOUBLE PRECISION,   -- China
    msci_em         DOUBLE PRECISION,   -- Emergentes

    -- Crypto
    btc_usd         DOUBLE PRECISION,
    eth_usd         DOUBLE PRECISION,

    -- Indicadores de amplitud
    sp500_above_200d DOUBLE PRECISION,  -- % de componentes sobre SMA200

    -- Política monetaria
    fed_rate        DOUBLE PRECISION,

    -- FRED: Inflación
    cpi_value       DOUBLE PRECISION,   -- CPI Urban All Items (index)
    cpi_yoy         DOUBLE PRECISION,   -- CPI Year-over-Year %

    -- FRED: Crédito
    credit_spread_bbb DOUBLE PRECISION, -- BBB Option-Adjusted Spread (%)

    -- Tipos de interés bancos centrales
    rate_eu         DOUBLE PRECISION,   -- ECB Deposit Facility Rate
    rate_jp         DOUBLE PRECISION,   -- Japan Call Money Rate
    rate_uk         DOUBLE PRECISION,   -- UK Sterling Overnight Rate

    -- PMI / Business Confidence (mensual, >50 = expansión)
    pmi_us          DOUBLE PRECISION,
    pmi_eu          DOUBLE PRECISION,
    pmi_jp          DOUBLE PRECISION,
    pmi_cn          DOUBLE PRECISION,
    pmi_uk          DOUBLE PRECISION,

    -- PIB crecimiento real (trimestral)
    gdp_us          DOUBLE PRECISION,
    gdp_eu          DOUBLE PRECISION,
    gdp_jp          DOUBLE PRECISION,
    gdp_uk          DOUBLE PRECISION,

    -- Desempleo (mensual, %)
    unemp_us        DOUBLE PRECISION,
    unemp_eu        DOUBLE PRECISION,
    unemp_jp        DOUBLE PRECISION,

    updated_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- 5. SIGNALS (historial de señales de análisis)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id                 BIGSERIAL PRIMARY KEY,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    ticker             TEXT NOT NULL REFERENCES assets(ticker),
    date               DATE NOT NULL,
    signal             TEXT NOT NULL CHECK (signal IN ('BUY','SELL','HOLD','WATCH','AVOID')),
    confidence         DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 100),
    price              DOUBLE PRECISION,
    strategy           TEXT,
    regime             TEXT,
    timeframe          TEXT DEFAULT '1d',
    technical_score    DOUBLE PRECISION,
    fundamental_score  DOUBLE PRECISION,
    sentiment_score    DOUBLE PRECISION,
    macro_score        DOUBLE PRECISION,
    rationale          TEXT
);


-- ─────────────────────────────────────────────────────────────
-- 6. NEWS SENTIMENT
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_sentiment (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    ticker          TEXT REFERENCES assets(ticker),
    date            DATE NOT NULL,
    headline        TEXT NOT NULL,
    source          TEXT,
    sentiment_score DOUBLE PRECISION CHECK (sentiment_score BETWEEN -100 AND 100),
    summary         TEXT,
    category        TEXT
);


-- ─────────────────────────────────────────────────────────────
-- 7. EARNINGS CALENDAR
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL REFERENCES assets(ticker),
    earnings_date   DATE NOT NULL,
    fiscal_quarter  TEXT,

    eps_estimate    DOUBLE PRECISION,
    eps_actual      DOUBLE PRECISION,
    eps_surprise    DOUBLE PRECISION,   -- (actual - estimate) / |estimate| * 100

    revenue_estimate   DOUBLE PRECISION,
    revenue_actual     DOUBLE PRECISION,
    revenue_surprise   DOUBLE PRECISION,

    report_time     TEXT,               -- 'BMO' (before market open), 'AMC' (after market close)

    -- Impacto en precio
    price_1d_before DOUBLE PRECISION,
    price_1d_after  DOUBLE PRECISION,
    price_move_pct  DOUBLE PRECISION,   -- Movimiento % post-earnings

    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, earnings_date)
);


-- ─────────────────────────────────────────────────────────────
-- 8. SECTOR ROTATION (seguimiento de flujos por sector)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sector_performance (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    sector_etf  TEXT NOT NULL,          -- XLK, XLF, XLE, etc.
    sector_name TEXT,
    close       DOUBLE PRECISION,
    returns_1d  DOUBLE PRECISION,
    returns_5d  DOUBLE PRECISION,
    returns_20d DOUBLE PRECISION,
    returns_60d DOUBLE PRECISION,
    volume      BIGINT,
    relative_strength DOUBLE PRECISION, -- RS vs SPY
    rank_1d     INTEGER,                -- Ranking diario (1 = mejor)
    rank_20d    INTEGER,                -- Ranking mensual

    UNIQUE (sector_etf, date)
);


-- ============================================================
-- ÍNDICES (optimización de queries temporales)
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_prices_date
    ON prices (date DESC);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_close
    ON prices (ticker, date DESC, close);

CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker
    ON fundamentals (ticker, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_date
    ON signals (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_signals_date
    ON signals (date DESC);

CREATE INDEX IF NOT EXISTS idx_news_ticker_date
    ON news_sentiment (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_ticker
    ON earnings_calendar (ticker, earnings_date DESC);

CREATE INDEX IF NOT EXISTS idx_sector_perf_date
    ON sector_performance (date DESC);

CREATE INDEX IF NOT EXISTS idx_sector_perf_etf
    ON sector_performance (sector_etf, date DESC);

CREATE INDEX IF NOT EXISTS idx_macro_date
    ON macro_indicators (date DESC);


-- ============================================================
-- ROW LEVEL SECURITY (desactivado para uso backend personal)
-- ============================================================

ALTER TABLE assets              DISABLE ROW LEVEL SECURITY;
ALTER TABLE prices              DISABLE ROW LEVEL SECURITY;
ALTER TABLE fundamentals        DISABLE ROW LEVEL SECURITY;
ALTER TABLE macro_indicators    DISABLE ROW LEVEL SECURITY;
ALTER TABLE signals             DISABLE ROW LEVEL SECURITY;
ALTER TABLE news_sentiment      DISABLE ROW LEVEL SECURITY;
ALTER TABLE earnings_calendar   DISABLE ROW LEVEL SECURITY;
ALTER TABLE sector_performance  DISABLE ROW LEVEL SECURITY;
