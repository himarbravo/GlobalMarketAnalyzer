"""
Tests for P7 Reversibility Filter.

Tests modal overlap, Von Neumann entropy, effective overlap,
and the ReversibilityFilter class.
"""

import numpy as np
import pytest
from core.reversibility import (
    compute_modal_overlap,
    compute_von_neumann_entropy,
    compute_effective_overlap,
    ReversibilityFilter,
)


class TestModalOverlap:
    """Tests for compute_modal_overlap()."""

    def test_identical_eigenvectors_gives_overlap_one(self):
        """Same eigenvectors → overlap = 1.0 for all modes."""
        N = 10
        # Random orthogonal matrix
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)

        overlaps = compute_modal_overlap(Q, Q)
        np.testing.assert_allclose(overlaps, 1.0, atol=1e-10)

    def test_random_rotation_gives_low_overlap(self):
        """Completely different eigenvectors → overlap ≈ 0."""
        N = 20
        A1 = np.random.randn(N, N)
        Q1, _ = np.linalg.qr(A1)
        A2 = np.random.randn(N, N)
        Q2, _ = np.linalg.qr(A2)

        overlaps = compute_modal_overlap(Q1, Q2)
        # Mean overlap should be low (≈ 1/N for random orthogonal)
        assert np.mean(overlaps) < 0.3

    def test_sign_flip_gives_overlap_one(self):
        """Eigenvectors can have arbitrary sign → overlap should be 1."""
        N = 10
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)
        Q_flipped = -Q  # flip all signs

        overlaps = compute_modal_overlap(Q, Q_flipped)
        np.testing.assert_allclose(overlaps, 1.0, atol=1e-10)

    def test_partial_rotation_gives_intermediate_overlap(self):
        """One mode rotated, rest stable → one low overlap, rest high."""
        N = 10
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)
        Q_mod = Q.copy()
        # Rotate mode 3 by mixing with mode 4
        theta = np.pi / 3  # 60 degrees
        Q_mod[:, 3] = np.cos(theta) * Q[:, 3] + np.sin(theta) * Q[:, 4]
        Q_mod[:, 4] = -np.sin(theta) * Q[:, 3] + np.cos(theta) * Q[:, 4]

        overlaps = compute_modal_overlap(Q, Q_mod)
        # Mode 3 and 4 should have lower overlap
        assert overlaps[3] < 0.5
        assert overlaps[4] < 0.5
        # Other modes should remain high
        for k in [0, 1, 2, 5, 6, 7, 8, 9]:
            assert overlaps[k] > 0.9

    def test_n_modes_parameter(self):
        """n_modes limits the output size."""
        N = 10
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)

        overlaps = compute_modal_overlap(Q, Q, n_modes=5)
        assert len(overlaps) == 5

    def test_degenerate_eigenvalues_use_subspace(self):
        """Degenerate eigenvalues → subspace overlap instead of per-mode."""
        N = 10
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)

        # Rotate within a degenerate subspace (modes 2,3)
        theta = np.pi / 4
        Q_rot = Q.copy()
        Q_rot[:, 2] = np.cos(theta) * Q[:, 2] + np.sin(theta) * Q[:, 3]
        Q_rot[:, 3] = -np.sin(theta) * Q[:, 2] + np.cos(theta) * Q[:, 3]

        # With degenerate eigenvalues for modes 2,3 → subspace overlap = 1
        eigenvalues = np.array([0, 0.5, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        overlaps = compute_modal_overlap(Q, Q_rot, eigenvalues_new=eigenvalues)

        # Modes 2,3 should have high overlap (subspace didn't change)
        assert overlaps[2] > 0.9
        assert overlaps[3] > 0.9


class TestVonNeumannEntropy:
    """Tests for compute_von_neumann_entropy()."""

    def test_single_dominant_mode_low_entropy(self):
        """One big eigenvalue, rest tiny → low entropy."""
        eigenvalues = np.array([0, 100, 0.01, 0.01, 0.01])
        S = compute_von_neumann_entropy(eigenvalues)
        assert S < 0.5

    def test_uniform_eigenvalues_max_entropy(self):
        """All eigenvalues equal → maximum entropy."""
        N = 20
        eigenvalues = np.ones(N)
        eigenvalues[0] = 0  # zero mode
        S = compute_von_neumann_entropy(eigenvalues)
        S_max = np.log(N - 1)  # max entropy = ln(n)
        assert abs(S - S_max) < 0.01

    def test_zero_eigenvalues_ignored(self):
        """Zero eigenvalues (nullspace) should not affect entropy."""
        eigenvalues_with_zeros = np.array([0, 0, 1.0, 2.0, 3.0])
        eigenvalues_without = np.array([1.0, 2.0, 3.0])
        S_with = compute_von_neumann_entropy(eigenvalues_with_zeros)
        S_without = compute_von_neumann_entropy(eigenvalues_without)
        assert abs(S_with - S_without) < 1e-10

    def test_entropy_non_negative(self):
        """Entropy should always be >= 0."""
        for _ in range(10):
            eigenvalues = np.abs(np.random.randn(20))
            eigenvalues[0] = 0
            S = compute_von_neumann_entropy(eigenvalues)
            assert S >= 0

    def test_empty_eigenvalues_returns_zero(self):
        """All zeros → entropy = 0."""
        eigenvalues = np.zeros(5)
        S = compute_von_neumann_entropy(eigenvalues)
        assert S == 0.0


class TestEffectiveOverlap:
    """Tests for compute_effective_overlap()."""

    def test_uniform_participation_gives_mean(self):
        """If asset participates equally in all modes → O_eff = mean(overlaps)."""
        N = 10
        overlaps = np.random.rand(N)
        # Eigenvector with equal participation: all φ_{ki} = 1/√N
        eigvecs = np.ones((N, N)) / np.sqrt(N)

        o_eff = compute_effective_overlap(eigvecs, overlaps, asset_idx=0)
        expected = np.mean(overlaps)
        assert abs(o_eff - expected) < 0.01

    def test_single_mode_participation(self):
        """If asset only participates in mode k → O_eff = O_k."""
        N = 10
        overlaps = np.random.rand(N)
        eigvecs = np.zeros((N, N))
        eigvecs[3, 5] = 1.0  # asset 3 only in mode 5

        o_eff = compute_effective_overlap(eigvecs, overlaps, asset_idx=3)
        assert abs(o_eff - overlaps[5]) < 1e-10


class TestReversibilityFilter:
    """Tests for ReversibilityFilter class."""

    def _make_laplacian_eigenstuff(self, N=20, seed=42):
        """Helper: create a random graph Laplacian and its eigendecomposition."""
        rng = np.random.RandomState(seed)
        W = rng.randn(N, N)
        W = (W + W.T) / 2  # symmetric
        np.fill_diagonal(W, 0)
        D = np.diag(np.abs(W).sum(axis=1))
        L = D - W
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvalues, eigenvectors

    def test_not_ready_before_two_updates(self):
        """Filter should not be ready until 2nd update."""
        rf = ReversibilityFilter()
        assert not rf.is_ready

        evals, evecs = self._make_laplacian_eigenstuff(seed=1)
        rf.update(evecs, evals)
        assert not rf.is_ready  # only 1 update

        evals2, evecs2 = self._make_laplacian_eigenstuff(seed=2)
        rf.update(evecs2, evals2)
        assert rf.is_ready  # now ready

    def test_stable_graph_identified(self):
        """Same graph twice → is_graph_stable = True."""
        rf = ReversibilityFilter()
        evals, evecs = self._make_laplacian_eigenstuff(seed=42)

        rf.update(evecs, evals)
        rf.update(evecs, evals)  # same graph
        assert rf.is_graph_stable
        assert rf.mean_overlap > 0.9

    def test_mutated_graph_detected(self):
        """Completely different graph → is_graph_stable may be False, low overlap."""
        rf = ReversibilityFilter()
        evals1, evecs1 = self._make_laplacian_eigenstuff(seed=1)
        evals2, evecs2 = self._make_laplacian_eigenstuff(seed=99)

        rf.update(evecs1, evals1)
        rf.update(evecs2, evals2)
        # Overlap should be low
        assert rf.mean_overlap < 0.5

    def test_should_trade_default_allows(self):
        """Before filter is ready, should_trade returns True."""
        rf = ReversibilityFilter()
        assert rf.should_trade(0) is True

    def test_should_trade_stable_allows(self):
        """Stable graph → should_trade returns True for all assets."""
        rf = ReversibilityFilter()
        evals, evecs = self._make_laplacian_eigenstuff(seed=42)
        rf.update(evecs, evals)
        rf.update(evecs, evals)

        for i in range(20):
            assert rf.should_trade(i) is True

    def test_diagnostics_returns_dict(self):
        """get_diagnostics should return a well-formed dict."""
        rf = ReversibilityFilter()
        evals, evecs = self._make_laplacian_eigenstuff(seed=42)
        rf.update(evecs, evals)
        rf.update(evecs, evals)

        diag = rf.get_diagnostics()
        assert isinstance(diag, dict)
        assert "is_ready" in diag
        assert "mean_overlap" in diag
        assert "filter_rate" in diag
        assert diag["is_ready"] is True
        assert diag["filter_rate"] == 0.0  # stable graph, nothing filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
