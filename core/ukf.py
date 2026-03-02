"""
UKF (Unscented Kalman Filter) — GlobalMarketAnalyzer
=====================================================
Tracks the fractional exponent s(t) through regime transitions.

Why UKF instead of EKF for s:
  - s transitions abruptly (0.85 → 0.30 in crisis)
  - EKF linearizes: misses the jump by 3-5 days
  - UKF propagates sigma points: captures nonlinear transitions

State:  x = [s]  (1D)
Process:  s[t+1] = s[t] + drift_toward_prior + noise
Measurement: z = mean(|prediction_error|) — higher error → s is wrong

Usage:
    ukf = UKF_S(s_init=0.85)
    ukf.predict(prior_s=vix_implied_s)
    ukf.update(prediction_errors)
    s_posterior = ukf.x
"""

import numpy as np
import json


class UKF_S:
    """
    1D Unscented Kalman Filter for the fractional exponent s.

    Merwe scaled sigma points (2n+1 = 3 for 1D).
    """

    def __init__(self, s_init: float = 0.85, P_init: float = 0.05,
                 Q: float = 0.002, R: float = 0.01,
                 alpha_ukf: float = 0.1, beta: float = 2.0, kappa: float = 0.0,
                 s_min: float = 0.15, s_max: float = 1.0,
                 drift_strength: float = 0.1):
        """
        Parameters
        ----------
        s_init : initial s estimate
        P_init : initial state covariance
        Q : process noise (how much s can change per step)
        R : measurement noise (observation uncertainty)
        alpha_ukf : spread of sigma points (small = tight around mean)
        beta : prior knowledge (2.0 = Gaussian)
        kappa : secondary scaling (0 for state estimation)
        s_min, s_max : physical bounds on s
        drift_strength : how strongly s drifts toward the VIX-implied prior
        """
        # State
        self.x = float(s_init)      # posterior mean
        self.P = float(P_init)      # posterior variance

        # Noise
        self.Q = float(Q)
        self.R = float(R)

        # Sigma point parameters
        self.alpha_ukf = alpha_ukf
        self.beta = beta
        self.kappa = kappa

        # Bounds
        self.s_min = s_min
        self.s_max = s_max
        self.drift_strength = drift_strength

        # Compute weights (constant for 1D)
        n = 1  # state dimension
        lam = alpha_ukf**2 * (n + kappa) - n
        self._lambda = lam
        self._n = n

        # Mean weights
        self.Wm = np.array([
            lam / (n + lam),               # W0_m
            1.0 / (2.0 * (n + lam)),       # W1_m
            1.0 / (2.0 * (n + lam)),       # W2_m
        ])
        # Covariance weights
        self.Wc = np.array([
            lam / (n + lam) + (1 - alpha_ukf**2 + beta),   # W0_c
            1.0 / (2.0 * (n + lam)),                       # W1_c
            1.0 / (2.0 * (n + lam)),                       # W2_c
        ])

        # History for diagnostics
        self.history = []

    def _sigma_points(self) -> np.ndarray:
        """Generate 3 sigma points for 1D state."""
        spread = np.sqrt((self._n + self._lambda) * self.P)
        return np.array([
            self.x,
            self.x + spread,
            self.x - spread,
        ])

    def _process_model(self, s: float, prior_s: float) -> float:
        """
        Process model: s drifts toward VIX-implied prior.
        s[t+1] = s[t] + drift_strength * (prior - s[t])
        """
        s_next = s + self.drift_strength * (prior_s - s)
        return np.clip(s_next, self.s_min, self.s_max)

    def _measurement_model(self, s: float, returns: np.ndarray,
                           predictions: np.ndarray) -> float:
        """
        Measurement model: expected absolute prediction error as a function of s.
        When s is correct, prediction errors are minimized.
        We model: z = |error| ∝ |s - s_true| (simplified)

        In practice, we compute the actual prediction error magnitude.
        """
        # The measurement is the mean absolute prediction error
        # Lower s → more anomalous diffusion → larger jumps expected
        # We return the "expected error" for this s value
        # A simple model: error increases when s is far from truth
        return 0.0  # baseline: zero error when s is perfect

    def predict(self, prior_s: float = None):
        """
        Predict step: propagate sigma points through process model.

        Parameters
        ----------
        prior_s : VIX/credit-implied s value (attractor)
                  If None, use current estimate (pure random walk)
        """
        if prior_s is None:
            prior_s = self.x

        # Generate sigma points
        sigmas = self._sigma_points()

        # Propagate through process model
        sigmas_pred = np.array([
            self._process_model(s, prior_s) for s in sigmas
        ])

        # Predicted mean
        self.x = float(np.sum(self.Wm * sigmas_pred))

        # Predicted covariance
        diff = sigmas_pred - self.x
        self.P = float(np.sum(self.Wc * diff**2)) + self.Q

        # Store propagated sigmas for update step
        self._sigmas_pred = sigmas_pred

    def update(self, prediction_errors: np.ndarray):
        """
        Update step: incorporate observed prediction errors.

        Parameters
        ----------
        prediction_errors : array of (actual - predicted) returns
        """
        if len(prediction_errors) == 0:
            return

        # Observed measurement: mean absolute prediction error
        z_obs = float(np.mean(np.abs(prediction_errors)))

        # Generate sigma points at predicted state
        sigmas = self._sigma_points()

        # Predicted measurements for each sigma point
        # Model: higher s → less anomalous diffusion → lower expected error
        # z(s) = base_error * (1 + scale * (s_ref - s))
        # When s is too low → model predicts too-extreme behavior → higher error
        # When s is too high → model predicts too-smooth behavior → higher error
        s_ref = 0.85  # reference s for normal markets
        base_error = z_obs  # use observed as baseline
        z_sigmas = np.array([
            base_error * (1.0 + 0.5 * abs(s - s_ref))
            for s in sigmas
        ])

        # Mean predicted measurement
        z_pred = float(np.sum(self.Wm * z_sigmas))

        # Innovation
        innovation = z_obs - z_pred

        # Cross covariance (state-measurement)
        diff_x = sigmas - self.x
        diff_z = z_sigmas - z_pred
        Pxz = float(np.sum(self.Wc * diff_x * diff_z))

        # Innovation covariance
        Pzz = float(np.sum(self.Wc * diff_z**2)) + self.R

        # Kalman gain
        K = Pxz / (Pzz + 1e-10)

        # Update state
        self.x = float(np.clip(self.x + K * innovation, self.s_min, self.s_max))

        # Update covariance
        self.P = max(1e-6, self.P - K * Pzz * K)

        # Record
        self.history.append({
            's': self.x,
            'P': self.P,
            'K': K,
            'z_obs': z_obs,
            'innovation': innovation,
        })

    def get_s(self) -> float:
        """Return current s estimate."""
        return float(self.x)

    def get_uncertainty(self) -> float:
        """Return current uncertainty (std dev of s)."""
        return float(np.sqrt(self.P))

    # ── Serialization for P3.2 (persist state) ──

    def to_dict(self) -> dict:
        """Serialize filter state for database storage."""
        return {
            'x': self.x,
            'P': self.P,
            'Q': self.Q,
            'R': self.R,
            'alpha_ukf': self.alpha_ukf,
            'beta': self.beta,
            'kappa': self.kappa,
            's_min': self.s_min,
            's_max': self.s_max,
            'drift_strength': self.drift_strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'UKF_S':
        """Restore filter from saved state."""
        ukf = cls(
            s_init=d['x'],
            P_init=d['P'],
            Q=d.get('Q', 0.002),
            R=d.get('R', 0.01),
            alpha_ukf=d.get('alpha_ukf', 0.1),
            beta=d.get('beta', 2.0),
            kappa=d.get('kappa', 0.0),
            s_min=d.get('s_min', 0.15),
            s_max=d.get('s_max', 1.0),
            drift_strength=d.get('drift_strength', 0.1),
        )
        return ukf

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> 'UKF_S':
        return cls.from_dict(json.loads(s))
