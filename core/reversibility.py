"""
REVERSIBILITY FILTER — GlobalMarketAnalyzer (P7)
===================================================
Distinguishes temporary dislocations (tradeable) from structural
changes (untradeable) using sector-based correlation stability.

Core idea:
  For each asset, measure its average correlation with sector peers
  in the current window vs the previous window. If the correlation
  drops significantly → the asset is decoupling from its group →
  its z-score equilibrium has shifted → don't trade it.

Also tracks Von Neumann Graph Entropy as a global stability measure.

Usage:
    from core.reversibility import ReversibilityFilter

    rev = ReversibilityFilter(tickers, returns_df)
    rev.update(returns_current_window, returns_prev_window)
    if not rev.should_trade(i):
        ...  # asset decoupled, skip

Reference: MATHEMATICS.md Section 13 / GitHub Issue #15
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Default thresholds ──
CORR_DROP_THRESHOLD = 0.25   # if intra-sector corr drops > this → decoupled
CORR_MIN_THRESHOLD = 0.10    # if absolute intra-sector corr < this → unstable
ENTROPY_SIGMA_MULT = 2.0     # |ΔS| > this × σ_S → graph in transition
MIN_PEERS = 2                # need at least 2 sector peers to compute


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
    lam = eigenvalues[eigenvalues > 1e-12]
    if len(lam) == 0:
        return 0.0
    lam_norm = lam / lam.sum()
    entropy = -np.sum(lam_norm * np.log(lam_norm))
    return float(entropy)


def compute_sector_correlations(returns: np.ndarray,
                                tickers: list[str],
                                sector_map: dict[int, list[int]]
                                ) -> dict[int, float]:
    """
    For each asset, compute its mean correlation with sector peers.

    Args:
        returns: (T, N) array of returns for the window
        tickers: list of ticker names
        sector_map: dict mapping asset_idx → list of peer indices (same sector)

    Returns:
        dict mapping asset_idx → mean correlation with peers (float in [-1, 1])
    """
    if len(returns) < 10:
        return {}

    # Compute full correlation matrix once
    # Handle NaN: use pairwise complete observations approach
    N = returns.shape[1]
    corr = np.corrcoef(returns.T)

    result = {}
    for i, peers in sector_map.items():
        if len(peers) < MIN_PEERS:
            continue
        # Mean correlation of asset i with its peers (excluding itself)
        peer_corrs = [corr[i, j] for j in peers if j != i and np.isfinite(corr[i, j])]
        if peer_corrs:
            result[i] = float(np.mean(peer_corrs))

    return result


class ReversibilityFilter:
    """
    Tracks sector correlation stability between graph rebuilds.

    On each update:
    - Computes intra-sector correlations for current and previous windows
    - Flags assets whose sector correlation dropped significantly
    - Tracks graph entropy for global stability

    Usage:
        rev = ReversibilityFilter(tickers)
        rev.update(returns_prev_window, returns_curr_window, eigenvalues)
        rev.should_trade(i)  # True if asset i's sector is stable
    """

    def __init__(self, tickers: list[str] = None,
                 corr_drop_threshold: float = CORR_DROP_THRESHOLD,
                 corr_min_threshold: float = CORR_MIN_THRESHOLD,
                 entropy_sigma_mult: float = ENTROPY_SIGMA_MULT):
        self.tickers = tickers or []
        self.corr_drop_threshold = corr_drop_threshold
        self.corr_min_threshold = corr_min_threshold
        self.entropy_sigma_mult = entropy_sigma_mult

        # Sector mapping: asset_idx → list of peer indices
        self._sector_map: dict[int, list[int]] = {}

        # State
        self._corr_prev: dict[int, float] = {}   # previous window sector correlations
        self._corr_curr: dict[int, float] = {}   # current window sector correlations
        self._corr_drop: dict[int, float] = {}   # drop per asset (prev - curr)
        self._entropy_curr = None
        self._entropy_prev = None
        self._delta_s = None
        self._entropy_history = []

        self.is_ready = False
        self.n_updates = 0

        if tickers:
            self._build_sector_map()

    def _build_sector_map(self):
        """Build mapping from asset index to its sector peers."""
        import config
        # Group indices by sector
        sector_groups: dict[str, list[int]] = {}
        for i, ticker in enumerate(self.tickers):
            sector = config.get_sector(ticker)
            if sector == "UNKNOWN":
                continue
            sector_groups.setdefault(sector, []).append(i)

        # For each asset, store its peer list
        for sector, indices in sector_groups.items():
            if len(indices) >= MIN_PEERS:
                for i in indices:
                    self._sector_map[i] = indices

        logger.debug(f"P7: {len(self._sector_map)} assets have sector peers "
                     f"across {len(sector_groups)} sectors")

    def set_tickers(self, tickers: list[str]):
        """Update tickers and rebuild sector map."""
        self.tickers = tickers
        self._build_sector_map()
        # Reset state
        self._corr_prev = {}
        self._corr_curr = {}
        self._corr_drop = {}
        self.is_ready = False
        self.n_updates = 0

    def update(self, returns_prev: np.ndarray, returns_curr: np.ndarray,
               eigenvalues: np.ndarray = None):
        """
        Update with two windows of returns and optional eigenvalues.

        Args:
            returns_prev: (T_prev, N) returns from the PREVIOUS window
            returns_curr: (T_curr, N) returns from the CURRENT window
            eigenvalues: (N,) eigenvalues from the current graph build (for entropy)
        """
        self.n_updates += 1

        # Shift previous correlations
        self._corr_prev = self._corr_curr.copy()

        # Compute sector correlations for both windows
        corr_prev_window = compute_sector_correlations(
            returns_prev, self.tickers, self._sector_map
        )
        corr_curr_window = compute_sector_correlations(
            returns_curr, self.tickers, self._sector_map
        )
        self._corr_curr = corr_curr_window

        # Compute drop: how much did intra-sector correlation decrease?
        self._corr_drop = {}
        for i in corr_curr_window:
            if i in corr_prev_window:
                self._corr_drop[i] = corr_prev_window[i] - corr_curr_window[i]

        # Entropy tracking
        self._entropy_prev = self._entropy_curr
        if eigenvalues is not None:
            self._entropy_curr = compute_von_neumann_entropy(eigenvalues)
            self._entropy_history.append(self._entropy_curr)
            if len(self._entropy_history) > 50:
                self._entropy_history = self._entropy_history[-50:]
            if self._entropy_prev is not None:
                self._delta_s = self._entropy_curr - self._entropy_prev

        self.is_ready = len(self._corr_drop) > 0

        # Log summary
        if self.is_ready:
            drops = list(self._corr_drop.values())
            n_decoupled = sum(1 for d in drops if d > self.corr_drop_threshold)
            curr_corrs = list(corr_curr_window.values())
            n_weak = sum(1 for c in curr_corrs if c < self.corr_min_threshold)
            logger.info(
                f"  P7 Sector: mean_corr={np.mean(curr_corrs):.3f}, "
                f"mean_drop={np.mean(drops):+.3f}, "
                f"decoupled={n_decoupled}/{len(drops)}, "
                f"weak={n_weak}/{len(curr_corrs)}"
            )

    @property
    def is_graph_stable(self) -> bool:
        """Global stability check via entropy."""
        if self._delta_s is None:
            return True
        if len(self._entropy_history) < 3:
            return True
        sigma_s = np.std(self._entropy_history)
        if sigma_s < 1e-10:
            return True
        return abs(self._delta_s) < self.entropy_sigma_mult * sigma_s

    @property
    def entropy(self) -> float:
        return self._entropy_curr if self._entropy_curr is not None else 0.0

    @property
    def delta_entropy(self) -> float:
        return self._delta_s if self._delta_s is not None else 0.0

    def asset_sector_corr(self, asset_idx: int) -> float:
        """Current intra-sector correlation for an asset."""
        return self._corr_curr.get(asset_idx, 1.0)  # default: assume stable

    def asset_corr_drop(self, asset_idx: int) -> float:
        """How much the intra-sector correlation dropped since last window."""
        return self._corr_drop.get(asset_idx, 0.0)  # default: no drop

    def should_trade(self, asset_idx: int) -> bool:
        """
        Should we trade this asset?

        Blocked if:
        1. Global: graph entropy spike (structural transition)
        2. Local: sector correlation dropped > threshold (decoupled)
        3. Local: sector correlation is very weak (< min threshold)
        """
        if not self.is_ready:
            return True  # no data → allow

        # Global check
        if not self.is_graph_stable:
            return False

        # Skip assets without sector peers (ETFs, crypto, etc.)
        if asset_idx not in self._sector_map:
            return True  # can't assess → allow

        # Check 1: correlation dropped too much?
        drop = self._corr_drop.get(asset_idx, 0.0)
        if drop > self.corr_drop_threshold:
            return False

        # Check 2: correlation is too weak overall?
        corr = self._corr_curr.get(asset_idx, 1.0)
        if corr < self.corr_min_threshold:
            return False

        return True

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for logging/debugging."""
        n_total = len(self._sector_map)
        n_tradeable = sum(1 for i in self._sector_map if self.should_trade(i))
        curr_corrs = list(self._corr_curr.values())
        drops = list(self._corr_drop.values())

        return {
            "is_ready": self.is_ready,
            "n_updates": self.n_updates,
            "entropy": round(self.entropy, 4),
            "delta_entropy": round(self.delta_entropy, 4),
            "is_graph_stable": self.is_graph_stable,
            "mean_sector_corr": round(np.mean(curr_corrs), 3) if curr_corrs else 0,
            "mean_corr_drop": round(np.mean(drops), 3) if drops else 0,
            "n_tradeable": n_tradeable,
            "n_with_peers": n_total,
            "filter_rate": round(1 - n_tradeable / max(n_total, 1), 3),
        }
