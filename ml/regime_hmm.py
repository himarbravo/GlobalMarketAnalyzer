"""
HMM REGIME DETECTOR — Hidden Markov Model for market regimes
==============================================================
Uses a 3-state Gaussian HMM trained on VIX, yield spread, and
SPY returns to classify the market into bull/neutral/bear regimes
with transition probabilities.

Usage:
    from ml.regime_hmm import RegimeHMM
    hmm = RegimeHMM()
    hmm.fit()
    result = hmm.predict_regime()
"""

import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    """3-state Hidden Markov Model for market regime detection."""

    REGIME_NAMES = {0: 'bull', 1: 'neutral', 2: 'bear'}

    def __init__(self, lookback='2y'):
        """
        Args:
            lookback: training data period (default 2 years)
        """
        self.lookback = lookback
        self.model = None
        self.features = None
        self.regime_map = {}  # state_id → regime_name

    def _fetch_features(self):
        """Fetch and build feature matrix: [SPY_ret, VIX_level, yield_spread]."""
        import pandas as pd

        # SPY returns
        spy = yf.Ticker('SPY').history(period=self.lookback)
        spy_ret = spy['Close'].pct_change().dropna()

        # VIX level
        vix = yf.Ticker('^VIX').history(period=self.lookback)
        vix_level = vix['Close']

        # 10Y-2Y yield spread
        tnx = yf.Ticker('^TNX').history(period=self.lookback)['Close']  # 10Y
        two_y = yf.Ticker('2YY=F').history(period=self.lookback)['Close']  # 2Y

        # Align all series
        df = pd.DataFrame({
            'spy_ret': spy_ret,
            'vix': vix_level,
            'tnx': tnx,
            'two_y': two_y,
        }).ffill().dropna()

        # Build feature matrix
        df['spread'] = df['tnx'] - df['two_y']
        df['spy_vol_20d'] = df['spy_ret'].rolling(20).std() * np.sqrt(252)

        features = df[['spy_ret', 'vix', 'spread']].dropna().values
        self.feature_index = df[['spy_ret', 'vix', 'spread']].dropna().index

        return features

    def fit(self):
        """Train the HMM on historical data."""
        features = self._fetch_features()

        # Normalize features for numerical stability
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0)
        self.stds[self.stds < 1e-10] = 1.0
        features_norm = (features - self.means) / self.stds

        # Fit 3-state Gaussian HMM
        self.model = GaussianHMM(
            n_components=3,
            covariance_type='full',
            n_iter=200,
            random_state=42,
            verbose=False,
        )
        self.model.fit(features_norm)
        self.features = features_norm

        # Map states to names using mean SPY return
        states = self.model.predict(features_norm)
        state_returns = {}
        for s in range(3):
            mask = states == s
            if mask.sum() > 0:
                state_returns[s] = features[mask, 0].mean()  # mean SPY return

        # Sort: highest return = bull, lowest = bear
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1], reverse=True)
        names = ['bull', 'neutral', 'bear']
        self.regime_map = {s: names[i] for i, (s, _) in enumerate(sorted_states)}

    def predict_regime(self):
        """
        Predict current regime and transition probabilities.

        Returns:
            dict with keys:
                current_regime: 'bull', 'neutral', or 'bear'
                probabilities: {regime: probability} for current state
                transition_1w: {regime: probability} expected in 1 week
                regime_history: last 20 days of regimes
                stats: {regime: {mean_ret, vol, avg_duration_days}}
        """
        if self.model is None:
            return {'error': 'Model not fitted. Call fit() first.'}

        # Current state probabilities
        state_probs = self.model.predict_proba(self.features)
        current_probs = state_probs[-1]

        current_state = np.argmax(current_probs)
        current_regime = self.regime_map.get(current_state, 'unknown')

        # Named probabilities
        named_probs = {self.regime_map[i]: round(float(current_probs[i]), 3)
                       for i in range(3)}

        # Transition probabilities (1 week ≈ 5 trading days)
        trans = self.model.transmat_
        trans_5d = np.linalg.matrix_power(trans, 5)
        trans_1w = {self.regime_map[j]: round(float(trans_5d[current_state, j]), 3)
                    for j in range(3)}

        # Regime history (last 20 days)
        states = self.model.predict(self.features)
        last_20 = [self.regime_map.get(int(s), '?') for s in states[-20:]]

        # Stats per regime
        raw_features = self.features * self.stds + self.means
        stats = {}
        for s in range(3):
            mask = states == s
            if mask.sum() > 5:
                ret = raw_features[mask, 0]
                name = self.regime_map[s]
                # Average duration: count consecutive days in regime
                changes = np.diff(states)
                runs = np.split(np.where(states == s)[0],
                                np.where(np.diff(np.where(states == s)[0]) > 1)[0] + 1)
                avg_dur = np.mean([len(r) for r in runs if len(r) > 0])

                stats[name] = {
                    'mean_daily_return': round(float(ret.mean()), 5),
                    'annual_return': round(float(ret.mean() * 252), 3),
                    'annual_vol': round(float(ret.std() * np.sqrt(252)), 3),
                    'avg_duration_days': round(float(avg_dur), 1),
                    'pct_time': round(float(mask.sum() / len(states)), 3),
                }

        return {
            'current_regime': current_regime,
            'probabilities': named_probs,
            'transition_1w': trans_1w,
            'regime_history': last_20,
            'stats': stats,
        }


if __name__ == '__main__':
    import json
    hmm = RegimeHMM()
    print('Fitting HMM on 2 years of data...')
    hmm.fit()
    result = hmm.predict_regime()
    print(json.dumps(result, indent=2))
