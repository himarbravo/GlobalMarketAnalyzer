"""
LGBM DIRECTION PREDICTOR — LightGBM classifier for SPY weekly direction
==========================================================================
Predicts whether SPY goes UP or DOWN in the next 5 trading days.
Uses macro, sentiment, and technical features already available in the pipeline.

Usage:
    from ml.lgbm_direction import LGBMDirectionPredictor
    model = LGBMDirectionPredictor()
    model.train()
    signal = model.predict_current()
"""

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta


class LGBMDirectionPredictor:
    """Predicts 5-day SPY direction using LightGBM."""

    FEATURE_NAMES = [
        'spy_ret_1d', 'spy_ret_5d', 'spy_ret_20d',
        'spy_vol_20d', 'spy_rsi_14',
        'vix_level', 'vix_change_5d', 'vix_ma20_ratio',
        'spread_10y2y', 'spread_10y3m',
        'qqq_ret_5d', 'iwm_ret_5d', 'tlt_ret_5d', 'gld_ret_5d',
        'spy_above_ma50', 'spy_above_ma200',
        'volume_ratio_5_20',
    ]

    def __init__(self, train_years=3):
        self.train_years = train_years
        self.model = None
        self.feature_importance = None

    def _fetch_data(self):
        """Fetch and build feature matrix from market data."""
        period = f'{self.train_years + 1}y'  # extra year for warmup

        # Fetch tickers
        tickers = {
            'SPY': yf.Ticker('SPY').history(period=period),
            'QQQ': yf.Ticker('QQQ').history(period=period),
            'IWM': yf.Ticker('IWM').history(period=period),
            'TLT': yf.Ticker('TLT').history(period=period),
            'GLD': yf.Ticker('GLD').history(period=period),
            '^VIX': yf.Ticker('^VIX').history(period=period),
            '^TNX': yf.Ticker('^TNX').history(period=period),
            '^IRX': yf.Ticker('^IRX').history(period=period),  # 3M yield
        }

        spy = tickers['SPY']['Close']
        vix = tickers['^VIX']['Close']
        tnx = tickers['^TNX']['Close']
        irx = tickers['^IRX']['Close']

        # Align all
        df = pd.DataFrame({
            'spy': spy,
            'qqq': tickers['QQQ']['Close'],
            'iwm': tickers['IWM']['Close'],
            'tlt': tickers['TLT']['Close'],
            'gld': tickers['GLD']['Close'],
            'vix': vix,
            'tnx': tnx,
            'irx': irx,
            'spy_vol': tickers['SPY']['Volume'],
        }).ffill().dropna()

        # --- Features ---
        # Returns
        df['spy_ret_1d'] = df['spy'].pct_change(1)
        df['spy_ret_5d'] = df['spy'].pct_change(5)
        df['spy_ret_20d'] = df['spy'].pct_change(20)
        df['qqq_ret_5d'] = df['qqq'].pct_change(5)
        df['iwm_ret_5d'] = df['iwm'].pct_change(5)
        df['tlt_ret_5d'] = df['tlt'].pct_change(5)
        df['gld_ret_5d'] = df['gld'].pct_change(5)

        # Volatility
        df['spy_vol_20d'] = df['spy_ret_1d'].rolling(20).std() * np.sqrt(252)

        # RSI 14
        delta = df['spy'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['spy_rsi_14'] = 100 - (100 / (1 + rs))

        # VIX features
        df['vix_level'] = df['vix']
        df['vix_change_5d'] = df['vix'].pct_change(5)
        df['vix_ma20'] = df['vix'].rolling(20).mean()
        df['vix_ma20_ratio'] = df['vix'] / df['vix_ma20']

        # Yield spreads
        df['spread_10y2y'] = df['tnx'] - df['irx']  # approx 10Y-3M
        df['spread_10y3m'] = df['tnx'] - df['irx']

        # Moving averages
        df['spy_ma50'] = df['spy'].rolling(50).mean()
        df['spy_ma200'] = df['spy'].rolling(200).mean()
        df['spy_above_ma50'] = (df['spy'] > df['spy_ma50']).astype(int)
        df['spy_above_ma200'] = (df['spy'] > df['spy_ma200']).astype(int)

        # Volume ratio
        df['vol_5d'] = df['spy_vol'].rolling(5).mean()
        df['vol_20d'] = df['spy_vol'].rolling(20).mean()
        df['volume_ratio_5_20'] = df['vol_5d'] / df['vol_20d'].replace(0, 1)

        # --- Target: SPY up in 5 days ---
        df['target'] = (df['spy'].shift(-5) > df['spy']).astype(int)

        # Clean
        df = df.dropna()

        # Remove last 5 rows (no target)
        df = df.iloc[:-5]

        return df

    def train(self, test_ratio=0.2):
        """
        Train the model with walk-forward split.

        Returns:
            dict with accuracy, classification report, feature importance
        """
        print("Fetching data...", flush=True)
        df = self._fetch_data()

        features = df[self.FEATURE_NAMES].values
        target = df['target'].values
        dates = df.index

        # Walk-forward split: last test_ratio for test
        split_idx = int(len(features) * (1 - test_ratio))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        dates_test = dates[split_idx:]

        print(f"Training on {len(X_train)} days, testing on {len(X_test)} days...",
              flush=True)

        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_train,
                                 feature_name=self.FEATURE_NAMES)
        val_data = lgb.Dataset(X_test, label=y_test,
                               feature_name=self.FEATURE_NAMES, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

        callbacks = [lgb.log_evaluation(period=0)]
        self.model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)

        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        fi = sorted(zip(self.FEATURE_NAMES, importance),
                    key=lambda x: x[1], reverse=True)
        self.feature_importance = fi

        # Simulated returns
        spy_test_returns = df['spy_ret_1d'].values[split_idx:split_idx + len(y_pred)]
        # Strategy: go long when predict UP, cash when predict DOWN
        strategy_returns = np.where(y_pred == 1, spy_test_returns, 0)
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_bh = (1 + spy_test_returns).cumprod()

        sharpe_strategy = (strategy_returns.mean() / strategy_returns.std()
                           * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        sharpe_bh = (spy_test_returns.mean() / spy_test_returns.std()
                     * np.sqrt(252)) if spy_test_returns.std() > 0 else 0

        self._results = {
            'accuracy': round(float(acc), 4),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_start': str(dates_test[0].date()),
            'test_end': str(dates_test[-1].date()),
            'up_pct_actual': round(float(y_test.mean()), 3),
            'up_pct_predicted': round(float(y_pred.mean()), 3),
            'feature_importance': [(f, round(float(v), 1)) for f, v in fi[:10]],
            'strategy_return': round(float(cum_strategy[-1] - 1), 4),
            'bh_return': round(float(cum_bh[-1] - 1), 4),
            'strategy_sharpe': round(float(sharpe_strategy), 3),
            'bh_sharpe': round(float(sharpe_bh), 3),
        }

        print(f"✅ Accuracy: {acc:.1%} (baseline: {y_test.mean():.1%} up days)",
              flush=True)
        print(f"   Strategy: {self._results['strategy_return']:+.1%} "
              f"(B&H: {self._results['bh_return']:+.1%})", flush=True)
        print(f"   Sharpe: {sharpe_strategy:.2f} (B&H: {sharpe_bh:.2f})", flush=True)

        return self._results

    def predict_current(self):
        """
        Predict current 5-day direction.

        Returns:
            dict with signal, confidence, top features
        """
        if self.model is None:
            return {'error': 'Model not trained. Call train() first.'}

        # Fetch fresh data
        df = self._fetch_data()
        latest = df[self.FEATURE_NAMES].iloc[-1:].values

        prob_up = float(self.model.predict(latest)[0])
        signal = 'UP' if prob_up > 0.5 else 'DOWN'
        confidence = prob_up if prob_up > 0.5 else (1 - prob_up)

        # Top contributing features for this prediction
        top_features = []
        for feat, imp in self.feature_importance[:5]:
            val = float(df[feat].iloc[-1])
            top_features.append({'feature': feat, 'value': round(val, 4),
                                 'importance': round(imp, 1)})

        return {
            'signal': signal,
            'confidence': round(float(confidence), 3),
            'prob_up': round(float(prob_up), 3),
            'top_features': top_features,
            'model_accuracy': self._results.get('accuracy', 0),
        }


if __name__ == '__main__':
    import json
    model = LGBMDirectionPredictor(train_years=3)
    results = model.train()
    print("\n=== Training Results ===")
    print(json.dumps(results, indent=2))
    print("\n=== Current Signal ===")
    signal = model.predict_current()
    print(json.dumps(signal, indent=2))
