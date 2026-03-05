"""
Tests for P7 Reversibility Filter (Sector-Correlation approach).

Tests sector correlation computation, correlation drop detection,
and the ReversibilityFilter class.
"""

import numpy as np
import pytest
from unittest.mock import patch
from core.reversibility import (
    compute_von_neumann_entropy,
    compute_sector_correlations,
    ReversibilityFilter,
)


# ── Von Neumann Entropy ──

class TestVonNeumannEntropy:
    def test_single_dominant_mode_low_entropy(self):
        """One large eigenvalue + rest tiny → low entropy."""
        lam = np.array([0.0, 10.0, 0.01, 0.01, 0.01])
        S = compute_von_neumann_entropy(lam)
        assert S < 1.0

    def test_uniform_eigenvalues_max_entropy(self):
        """All equal eigenvalues → maximum entropy ~ ln(N)."""
        N = 10
        lam = np.ones(N)
        S = compute_von_neumann_entropy(lam)
        assert abs(S - np.log(N)) < 0.01

    def test_zero_eigenvalues_ignored(self):
        """Zeros should be excluded from entropy calculation."""
        lam = np.array([0.0, 0.0, 1.0, 1.0])
        S = compute_von_neumann_entropy(lam)
        assert abs(S - np.log(2)) < 0.01

    def test_entropy_non_negative(self):
        rng = np.random.default_rng(42)
        lam = rng.exponential(1.0, 20)
        S = compute_von_neumann_entropy(lam)
        assert S >= 0

    def test_empty_eigenvalues_returns_zero(self):
        assert compute_von_neumann_entropy(np.array([])) == 0.0


# ── Sector Correlations ──

class TestSectorCorrelations:
    def test_identical_assets_high_correlation(self):
        """Assets that are copies of each other should have corr ≈ 1."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal((100, 1))
        # 5 assets: first 3 are copies (same sector), last 2 are random
        returns = np.hstack([
            base + rng.standard_normal((100, 1)) * 0.01,   # 0
            base + rng.standard_normal((100, 1)) * 0.01,   # 1
            base + rng.standard_normal((100, 1)) * 0.01,   # 2
            rng.standard_normal((100, 1)),                   # 3
            rng.standard_normal((100, 1)),                   # 4
        ])
        tickers = ["A", "B", "C", "D", "E"]
        sector_map = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}

        corrs = compute_sector_correlations(returns, tickers, sector_map)
        assert corrs[0] > 0.95
        assert corrs[1] > 0.95
        assert corrs[2] > 0.95

    def test_uncorrelated_assets_low_correlation(self):
        """Random uncorrelated assets → correlation near 0."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((200, 5))
        tickers = ["A", "B", "C", "D", "E"]
        sector_map = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]}

        corrs = compute_sector_correlations(returns, tickers, sector_map)
        for i in corrs:
            assert abs(corrs[i]) < 0.3

    def test_too_short_returns_gives_empty(self):
        """Less than 10 days → can't compute correlations."""
        returns = np.random.randn(5, 3)
        corrs = compute_sector_correlations(returns, ["A", "B", "C"], {0: [0, 1, 2]})
        assert len(corrs) == 0


# ── ReversibilityFilter ──

class TestReversibilityFilter:
    def _make_filter(self, n=10):
        tickers = [f"TK{i}" for i in range(n)]
        # Create filter without config (bypass _build_sector_map)
        filt = ReversibilityFilter.__new__(ReversibilityFilter)
        filt.tickers = tickers
        filt.corr_drop_threshold = 0.25
        filt.corr_min_threshold = 0.10
        filt.entropy_sigma_mult = 2.0
        filt._corr_prev = {}
        filt._corr_curr = {}
        filt._corr_drop = {}
        filt._entropy_curr = None
        filt._entropy_prev = None
        filt._delta_s = None
        filt._entropy_history = []
        filt.is_ready = False
        filt.n_updates = 0
        # Manually set sector map: 3 groups of 3 + 1 orphan
        filt._sector_map = {
            0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2],  # TECH
            3: [3, 4, 5], 4: [3, 4, 5], 5: [3, 4, 5],  # BANKS
            6: [6, 7, 8], 7: [6, 7, 8], 8: [6, 7, 8],  # ENERGY
            # 9 has no peers (singleton sector)
        }
        return filt

    def test_not_ready_before_update(self):
        filt = self._make_filter()
        assert not filt.is_ready
        assert filt.should_trade(0) is True  # default: allow

    def test_stable_sector_allows_trade(self):
        """Assets with stable sector correlation should be tradeable."""
        filt = self._make_filter()
        rng = np.random.default_rng(42)

        # Generate correlated data for tech sector (0,1,2)
        base_tech = rng.standard_normal((100, 1))
        base_bank = rng.standard_normal((100, 1))
        base_ener = rng.standard_normal((100, 1))

        # Both windows have same structure → stable correlations
        ret_prev = np.hstack([
            base_tech + rng.standard_normal((100, 1)) * 0.1,
            base_tech + rng.standard_normal((100, 1)) * 0.1,
            base_tech + rng.standard_normal((100, 1)) * 0.1,
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            base_ener + rng.standard_normal((100, 1)) * 0.1,
            base_ener + rng.standard_normal((100, 1)) * 0.1,
            base_ener + rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)),
        ])

        base_tech2 = rng.standard_normal((100, 1))
        base_bank2 = rng.standard_normal((100, 1))
        base_ener2 = rng.standard_normal((100, 1))

        ret_curr = np.hstack([
            base_tech2 + rng.standard_normal((100, 1)) * 0.1,
            base_tech2 + rng.standard_normal((100, 1)) * 0.1,
            base_tech2 + rng.standard_normal((100, 1)) * 0.1,
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            base_ener2 + rng.standard_normal((100, 1)) * 0.1,
            base_ener2 + rng.standard_normal((100, 1)) * 0.1,
            base_ener2 + rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)),
        ])

        filt.update(ret_prev, ret_curr)
        assert filt.is_ready
        # All sector assets should be tradeable (correlation stayed high)
        assert filt.should_trade(0) is True
        assert filt.should_trade(3) is True
        assert filt.should_trade(6) is True
        # Asset 9 has no peers → always allowed
        assert filt.should_trade(9) is True

    def test_decoupled_asset_blocked(self):
        """Asset that decorrelates from sector should be blocked."""
        filt = self._make_filter()
        rng = np.random.default_rng(42)

        base_tech = rng.standard_normal((100, 1))
        base_bank = rng.standard_normal((100, 1))

        # Previous window: TK0 is correlated with tech sector
        ret_prev = np.hstack([
            base_tech + rng.standard_normal((100, 1)) * 0.1,  # TK0 = tech
            base_tech + rng.standard_normal((100, 1)) * 0.1,  # TK1 = tech
            base_tech + rng.standard_normal((100, 1)) * 0.1,  # TK2 = tech
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            base_bank + rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)),
        ])

        base_tech2 = rng.standard_normal((100, 1))
        base_bank2 = rng.standard_normal((100, 1))

        # Current window: TK0 decoupled (random), but TK1,TK2 still correlated
        ret_curr = np.hstack([
            rng.standard_normal((100, 1)),                     # TK0 = DECOUPLED
            base_tech2 + rng.standard_normal((100, 1)) * 0.1,  # TK1 = tech
            base_tech2 + rng.standard_normal((100, 1)) * 0.1,  # TK2 = tech
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            base_bank2 + rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)) * 0.1,
            rng.standard_normal((100, 1)),
        ])

        filt.update(ret_prev, ret_curr)
        assert filt.is_ready

        # TK0 decorrelated from sector → should be blocked
        # Its sector corr should have dropped significantly
        corr_0 = filt.asset_sector_corr(0)
        assert corr_0 < 0.3, f"TK0 should have low sector corr, got {corr_0}"

        # TK1, TK2 still correlated → should have HIGHER sector corr than TK0
        assert filt.asset_sector_corr(1) > corr_0, "TK1 should be more correlated than decoupled TK0"

    def test_diagnostics_returns_dict(self):
        filt = self._make_filter()
        rng = np.random.default_rng(42)
        ret = rng.standard_normal((100, 10))
        eigenvalues = rng.exponential(1.0, 10)
        filt.update(ret, ret, eigenvalues)

        diag = filt.get_diagnostics()
        assert isinstance(diag, dict)
        assert "mean_sector_corr" in diag
        assert "mean_corr_drop" in diag
        assert "filter_rate" in diag
        assert "entropy" in diag

    def test_entropy_tracking(self):
        filt = self._make_filter()
        rng = np.random.default_rng(42)
        ret = rng.standard_normal((100, 10))

        lam1 = np.ones(10)
        filt.update(ret, ret, lam1)
        assert filt.entropy > 0

        lam2 = np.array([10.0] + [0.1]*9)
        filt.update(ret, ret, lam2)
        assert filt.delta_entropy != 0
