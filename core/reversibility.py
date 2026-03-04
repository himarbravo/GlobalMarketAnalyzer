"""
REVERSIBILITY FILTER — GlobalMarketAnalyzer (P7)
===================================================
Distinguishes temporary dislocations (tradeable) from structural
changes (untradeable) using two graph-theoretic indicators:

  1. Modal Overlap:   O_k = |<φ_k(old)|φ_k(new)>|²
     → Local check: has this asset's neighborhood changed?

  2. Von Neumann Entropy:  S = -Σ λ̃_k ln λ̃_k
     → Global check: is the market structure mutating?

Usage:
    from core.reversibility import ReversibilityFilter

    rev = ReversibilityFilter()
    # On each graph rebuild:
    rev.update(gb.eigenvectors, gb.eigenvalues)
    # Check before trading:
    if rev.is_ready:
        if not rev.is_graph_stable:
            ... # global instability, go defensive
        if not rev.should_trade(asset_idx):
            ... # this asset's modes rotated, skip

Reference: MATHEMATICS.md Sections 13.1–13.6
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Default thresholds ──
OVERLAP_THRESHOLD = 0.3     # O_eff,i > this → asset's modes are stable
ENTROPY_SIGMA_MULT = 2.0    # |ΔS| > this × σ_S → graph in transition
DEGENERACY_TOL = 0.05       # eigenvalues within this relative gap → degenerate
K_MODES_DEFAULT = None      # number of modes to track (None = sqrt(N))


def compute_modal_overlap(eigvecs_old: np.ndarray,
                          eigvecs_new: np.ndarray,
                          eigenvalues_new: np.ndarray = None,
                          n_modes: int = None) -> np.ndarray:
    """
    Per-mode overlap between two eigendecompositions.

    O_k = |<φ_k(old)|φ_k(new)>|²

    For degenerate eigenvalues (λ_k ≈ λ_{k+1}), uses subspace overlap
    to avoid false positives from arbitrary rotation within the degenerate
    subspace.

    Args:
        eigvecs_old: (N, N) previous eigenvectors
        eigvecs_new: (N, N) current eigenvectors
        eigenvalues_new: (N,) current eigenvalues (for degeneracy detection)
        n_modes: number of modes to compute (default: all)

    Returns:
        overlaps: (n_modes,) array with O_k ∈ [0, 1]
    """
    N = eigvecs_old.shape[1]
    if n_modes is None:
        n_modes = N

    n_modes = min(n_modes, N)
    overlaps = np.zeros(n_modes)

    # Identify degenerate groups if eigenvalues provided
    degenerate_groups = []
    if eigenvalues_new is not None:
        k = 0
        while k < n_modes:
            group = [k]
            while (k + 1 < n_modes and
                   eigenvalues_new[k + 1] > 1e-10 and
                   abs(eigenvalues_new[k + 1] - eigenvalues_new[k]) /
                   eigenvalues_new[k + 1] < DEGENERACY_TOL):
                k += 1
                group.append(k)
            if len(group) > 1:
                degenerate_groups.append(group)
            k += 1

    # Mark which modes are in degenerate groups
    in_degenerate = set()
    for group in degenerate_groups:
        in_degenerate.update(group)

    # Compute per-mode overlap
    for k in range(n_modes):
        if k in in_degenerate:
            continue  # handled below by subspace overlap

        # Standard overlap: |<φ_old|φ_new>|²
        dot = np.dot(eigvecs_old[:, k], eigvecs_new[:, k])
        overlaps[k] = dot ** 2

    # Subspace overlap for degenerate groups
    for group in degenerate_groups:
        idx = np.array(group)
        # Projectors: P = Σ_k |φ_k><φ_k|
        # Subspace overlap = Tr(P_old · P_new) / dim
        P_old = eigvecs_old[:, idx] @ eigvecs_old[:, idx].T
        P_new = eigvecs_new[:, idx] @ eigvecs_new[:, idx].T
        sub_overlap = np.trace(P_old @ P_new) / len(group)
        # Assign same overlap to all modes in the group
        for k in group:
            if k < n_modes:
                overlaps[k] = sub_overlap

    return overlaps


def compute_von_neumann_entropy(eigenvalues: np.ndarray) -> float:
    """
    Von Neumann graph entropy.

    S(G) = -Σ λ̃_k ln λ̃_k    where λ̃_k = λ_k / Σλ_j

    Low entropy: few modes dominate → organized market (sectors clear)
    High entropy: all modes equal → random/unstructured market

    Args:
        eigenvalues: (N,) eigenvalues of the graph Laplacian (≥ 0)

    Returns:
        S: scalar entropy value
    """
    # Skip zero eigenvalues (the nullspace / conservation mode)
    lam = eigenvalues[eigenvalues > 1e-12]
    if len(lam) == 0:
        return 0.0

    # Normalize to probability distribution
    lam_norm = lam / lam.sum()

    # Entropy: -Σ p_k ln p_k
    entropy = -np.sum(lam_norm * np.log(lam_norm))
    return float(entropy)


def compute_effective_overlap(eigvecs: np.ndarray,
                              overlaps: np.ndarray,
                              asset_idx: int,
                              n_modes: int = None) -> float:
    """
    Effective overlap for a specific asset.

    O_eff,i = Σ_k φ_{ki}² · O_k

    Each asset participates in modes with weight φ_{ki}².
    The effective overlap is a weighted average of mode overlaps,
    weighted by how much the asset contributes to each mode.

    Args:
        eigvecs: (N, N) current eigenvectors
        overlaps: (n_modes,) per-mode overlaps
        asset_idx: index of the asset
        n_modes: number of modes to use (default: all available)

    Returns:
        O_eff: scalar ∈ [0, 1]
    """
    n = len(overlaps) if n_modes is None else min(n_modes, len(overlaps))

    # Weight = squared participation of asset i in mode k
    weights = eigvecs[asset_idx, :n] ** 2
    w_sum = weights.sum()

    if w_sum < 1e-12:
        return 1.0  # asset doesn't participate in any mode → assume stable

    return float(np.dot(weights, overlaps[:n]) / w_sum)


class ReversibilityFilter:
    """
    Tracks graph eigenstate evolution and provides trade filtering.

    Usage:
        rev = ReversibilityFilter()
        rev.update(eigvecs, eigvals)    # call on each graph rebuild
        rev.is_graph_stable             # global stability check
        rev.should_trade(i)             # per-asset check
    """

    def __init__(self, overlap_threshold: float = OVERLAP_THRESHOLD,
                 entropy_sigma_mult: float = ENTROPY_SIGMA_MULT,
                 n_modes: int = K_MODES_DEFAULT):
        self.overlap_threshold = overlap_threshold
        self.entropy_sigma_mult = entropy_sigma_mult
        self.n_modes = n_modes  # None = auto (sqrt(N))

        # State
        self._eigvecs_prev = None
        self._eigvals_prev = None
        self._eigvecs_curr = None
        self._eigvals_curr = None

        # Computed values
        self._overlaps = None           # (N,) per-mode overlap
        self._entropy_curr = None       # current S
        self._entropy_prev = None       # previous S
        self._delta_s = None            # ΔS = S(now) - S(prev)
        self._entropy_history = []      # rolling history for σ_S
        self._asset_overlaps = {}       # cache: asset_idx → O_eff

        self.is_ready = False
        self.n_updates = 0

    def update(self, eigvecs: np.ndarray, eigvals: np.ndarray):
        """
        Update with new eigenstate from graph_builder.build().

        Call this AFTER each graph rebuild. The filter becomes active
        after the 2nd update (needs old + new to compute overlap).
        """
        # Shift current → previous
        self._eigvecs_prev = self._eigvecs_curr
        self._eigvals_prev = self._eigvals_curr
        self._entropy_prev = self._entropy_curr

        # Store new
        self._eigvecs_curr = eigvecs.copy()
        self._eigvals_curr = eigvals.copy()

        # Compute entropy
        self._entropy_curr = compute_von_neumann_entropy(eigvals)
        self._entropy_history.append(self._entropy_curr)

        # Keep rolling window of entropy (max 50 refits = ~1000 trading days)
        if len(self._entropy_history) > 50:
            self._entropy_history = self._entropy_history[-50:]

        self.n_updates += 1

        # Need at least 2 updates to compute overlap
        if self._eigvecs_prev is None:
            self.is_ready = False
            self._overlaps = None
            self._delta_s = None
            self._asset_overlaps = {}
            return

        # Compute modal overlap (top K modes only)
        N = eigvecs.shape[1]
        k = self.n_modes or max(5, int(np.sqrt(N)))
        self._n_modes_used = k

        self._overlaps = compute_modal_overlap(
            self._eigvecs_prev, self._eigvecs_curr,
            eigenvalues_new=self._eigvals_curr,
            n_modes=k
        )

        # Compute ΔS
        self._delta_s = self._entropy_curr - self._entropy_prev

        # Clear asset overlap cache (modes changed)
        self._asset_overlaps = {}

        self.is_ready = True

        # Log summary
        mean_overlap = np.mean(self._overlaps) if self._overlaps is not None else 0
        logger.info(
            f"  P7 Reversibility: ΔS={self._delta_s:+.4f}, "
            f"mean_overlap={mean_overlap:.3f}, "
            f"graph_stable={self.is_graph_stable}"
        )

    @property
    def is_graph_stable(self) -> bool:
        """
        Global stability check: |ΔS| < σ_S threshold.

        Returns True if the overall graph structure is stable
        (entropy production is within normal bounds).
        """
        if not self.is_ready or self._delta_s is None:
            return True  # assume stable if no data

        # Need enough history to estimate σ_S
        if len(self._entropy_history) < 3:
            return True  # not enough data, assume stable

        sigma_s = np.std(self._entropy_history)
        if sigma_s < 1e-10:
            return True  # no variance, stable

        return abs(self._delta_s) < self.entropy_sigma_mult * sigma_s

    @property
    def entropy(self) -> float:
        """Current graph entropy."""
        return self._entropy_curr if self._entropy_curr is not None else 0.0

    @property
    def delta_entropy(self) -> float:
        """Entropy change since last update."""
        return self._delta_s if self._delta_s is not None else 0.0

    @property
    def mean_overlap(self) -> float:
        """Mean modal overlap across all modes."""
        if self._overlaps is None:
            return 1.0
        return float(np.mean(self._overlaps))

    def asset_overlap(self, asset_idx: int) -> float:
        """
        Effective overlap for a specific asset.

        Cached per update cycle to avoid recomputation.
        """
        if not self.is_ready or self._overlaps is None:
            return 1.0  # assume stable if no data

        if asset_idx not in self._asset_overlaps:
            k = getattr(self, '_n_modes_used', None)
            self._asset_overlaps[asset_idx] = compute_effective_overlap(
                self._eigvecs_curr, self._overlaps, asset_idx, n_modes=k
            )

        return self._asset_overlaps[asset_idx]

    def should_trade(self, asset_idx: int,
                     overlap_threshold: float = None) -> bool:
        """
        Should we trade this asset based on reversibility analysis?

        Combines global (entropy) and local (overlap) checks.

        Returns:
            True if the asset's dislocation appears reversible.
        """
        if not self.is_ready:
            return True  # no data → allow trading (conservative)

        # Global check: is the graph in transition?
        if not self.is_graph_stable:
            return False

        # Local check: is this asset's neighborhood stable?
        threshold = overlap_threshold or self.overlap_threshold
        return self.asset_overlap(asset_idx) >= threshold

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging/debugging."""
        n_tradeable = 0
        n_total = 0
        if self.is_ready and self._eigvecs_curr is not None:
            N = self._eigvecs_curr.shape[0]
            n_total = N
            for i in range(N):
                if self.should_trade(i):
                    n_tradeable += 1

        return {
            "is_ready": self.is_ready,
            "n_updates": self.n_updates,
            "entropy": round(self.entropy, 4),
            "delta_entropy": round(self.delta_entropy, 4),
            "mean_overlap": round(self.mean_overlap, 3),
            "is_graph_stable": self.is_graph_stable,
            "n_tradeable": n_tradeable,
            "n_total": n_total,
            "filter_rate": round(1 - n_tradeable / n_total, 3) if n_total > 0 else 0,
        }
