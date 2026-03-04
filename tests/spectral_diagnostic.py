#!/usr/bin/env python3
"""
P7 Spectral Diagnostic — Tests whether modal overlap and entropy
can predict trade success (reversible vs irreversible dislocations).

Uses cached data from crisis_backtest runs. No Supabase needed if cache exists.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.heat_engine import HeatEngine
from core.fundamental_filter import FundamentalFilter

# ── Periods to test (same as crisis_backtest) ──
PERIODS = {
    "Volmageddon":    ("2017-09-01", "2018-06-30"),
    "COVID":          ("2019-09-01", "2020-09-30"),
    "Fed 2022":       ("2021-09-01", "2023-03-31"),
    "Bull 2019":      ("2019-01-01", "2019-09-30"),
    "AI Rally 23-24": ("2023-04-01", "2024-06-30"),
}


def von_neumann_entropy(L):
    """Von Neumann entropy of graph: S = -Tr(ρ ln ρ) where ρ = L/Tr(L)."""
    evals = np.linalg.eigvalsh(L)
    evals = np.maximum(evals, 1e-12)  # avoid log(0)
    trace = evals.sum()
    if trace < 1e-10:
        return 0.0
    rho = evals / trace
    return -np.sum(rho * np.log(rho))


def eigenvalue_gap_analysis(eigenvalues, max_k=15):
    """Find natural number of clusters from eigenvalue gaps."""
    evals = eigenvalues[eigenvalues > 1e-8]  # skip zero eigenvalue
    if len(evals) < 3:
        return 1, []

    gaps = []
    for i in range(min(len(evals) - 1, max_k)):
        ratio = evals[i + 1] / max(evals[i], 1e-10)
        gaps.append((i + 1, ratio, evals[i], evals[i + 1]))

    # Best gap = largest ratio
    best_k = max(gaps, key=lambda x: x[1])[0] + 1  # +1 because we include zero mode
    return best_k, gaps


def spectral_clusters(eigenvectors, eigenvalues, k):
    """Assign assets to k clusters using spectral embedding (numpy only)."""
    # Use first k eigenvectors (skip the constant mode φ₀)
    X = eigenvectors[:, 1:k + 1]  # embedding matrix (N × k)
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    X_norm = X / norms

    # Simple k-means (numpy only, no sklearn needed)
    N = X_norm.shape[0]
    rng = np.random.RandomState(42)
    # Initialize centroids randomly
    centroids = X_norm[rng.choice(N, k, replace=False)]
    for _ in range(50):  # 50 iterations
        # Assign to nearest centroid
        dists = np.array([[np.linalg.norm(X_norm[i] - centroids[j]) for j in range(k)] for i in range(N)])
        labels = np.argmin(dists, axis=1)
        # Update centroids
        new_centroids = np.array([X_norm[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j] for j in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels


def modal_overlap(phi_old, phi_new, k_max=10):
    """Compute overlap |<φ_old|φ_new>|² for first k_max modes."""
    overlaps = []
    n_modes = min(k_max, phi_old.shape[1], phi_new.shape[1])
    for k in range(1, n_modes):  # skip mode 0 (constant)
        dot = np.abs(np.dot(phi_old[:, k], phi_new[:, k]))
        overlaps.append(dot ** 2)
    return overlaps


def ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf shrinkage estimator for covariance/correlation.
    X: (T, N) matrix of returns. Returns shrunk correlation matrix."""
    T, N = X.shape
    # Handle NaN: replace with 0
    X = np.nan_to_num(X, nan=0.0)
    # Standardize
    means = X.mean(axis=0)
    X_c = X - means
    stds = X_c.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X_s = X_c / stds

    # Sample correlation
    S = (X_s.T @ X_s) / T
    np.fill_diagonal(S, 1.0)  # ensure diagonal is 1

    # Target: identity (uncorrelated)
    F = np.eye(N)

    # Optimal shrinkage intensity (simplified Ledoit-Wolf)
    d2 = np.sum((S - F) ** 2) / N
    # Simplified b2 estimate (faster than O(N²) loop)
    b2 = min(1.0 / T, d2)

    # Shrinkage intensity
    alpha = min(b2 / max(d2, 1e-10), 1.0)
    alpha = max(alpha, 0.01)  # minimum 1% shrinkage

    # Shrunk estimator
    S_shrunk = alpha * F + (1 - alpha) * S
    return S_shrunk, alpha


def compute_rolling_graph(gb, returns_df, window=60, step=20, use_shrinkage=False):
    """Build graphs at different time points using rolling windows."""
    N = returns_df.shape[1]
    dates = returns_df.index
    results = []

    for t in range(window, len(dates), step):
        window_data = returns_df.iloc[t - window:t].values

        if use_shrinkage:
            corr, alpha = ledoit_wolf_shrinkage(window_data)
        else:
            corr_df = pd.DataFrame(window_data).corr().fillna(0)
            corr = corr_df.values

        # Build W and L
        W = np.abs(corr)
        np.fill_diagonal(W, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W

        evals, evecs = np.linalg.eigh(L)
        evals = np.maximum(evals, 0.0)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        entropy = von_neumann_entropy(L)

        results.append({
            "date": dates[t],
            "eigenvalues": evals,
            "eigenvectors": evecs,
            "entropy": entropy,
            "L": L,
            "W": W,
        })

    return results


def run_diagnostic(period_name, start, end):
    """Run full spectral diagnostic for one period."""
    print(f"\n{'═' * 80}")
    print(f"  SPECTRAL DIAGNOSTIC: {period_name}")
    print(f"  Period: {start} → {end}")
    print(f"{'═' * 80}")

    db = DatabaseManager()
    gb = GraphBuilder(db)
    gb.load_data(start_date=start, end_date=end)

    if len(gb.prices) < 60:
        print(f"  ⚠ Insufficient data ({len(gb.prices)} rows)")
        return None

    gb.build()
    tickers = gb.tickers
    N = len(tickers)

    print(f"  N={N} tickers, T={len(gb.prices)} days")

    # ── 1. Eigenvalue Spectrum ──
    evals = gb.eigenvalues
    print(f"\n  ── 1. EIGENVALUE SPECTRUM ──")
    print(f"  λ₀={evals[0]:.4f}, λ₁={evals[1]:.4f}, λ₂={evals[2]:.4f}, ... λ_N={evals[-1]:.4f}")

    k_best, gaps = eigenvalue_gap_analysis(evals)
    print(f"  Best k* = {k_best} clusters (from eigenvalue gap)")
    print(f"  Top 10 gaps:")
    for k, ratio, ev_low, ev_high in sorted(gaps, key=lambda x: -x[1])[:10]:
        print(f"    k={k:2d}: λ_{k}={ev_low:.4f} → λ_{k+1}={ev_high:.4f}  ratio={ratio:.2f}")

    # ── 2. Spectral Clusters ──
    k_use = min(k_best, 8)  # cap at 8 clusters
    if k_use < 2:
        k_use = 3
    labels = spectral_clusters(gb.eigenvectors, evals, k_use)

    print(f"\n  ── 2. SPECTRAL CLUSTERS (k={k_use}) ──")
    for c in range(k_use):
        members = [tickers[i] for i in range(N) if labels[i] == c]
        tag = f"Cluster {c} ({len(members)})"
        print(f"  {tag}: {', '.join(members[:10])}{'...' if len(members) > 10 else ''}")

    # ── 3. Von Neumann Entropy ──
    S = von_neumann_entropy(gb.L)
    print(f"\n  ── 3. VON NEUMANN ENTROPY ──")
    print(f"  S(G) = {S:.4f}")

    # ── 4. Rolling Modal Overlap + Entropy ──
    returns_df = gb.returns
    if returns_df is None or len(returns_df) < 80:
        print("  ⚠ Insufficient returns for rolling analysis")
        return None

    print(f"\n  ── 4. ROLLING MODAL OVERLAP (window=60d, step=20d) ──")
    snapshots = compute_rolling_graph(gb, returns_df, window=60, step=20)

    if len(snapshots) < 2:
        print("  ⚠ Not enough snapshots")
        return None

    print(f"  {len(snapshots)} snapshots computed")
    print(f"\n  {'Date':<12} {'S(G)':>7} {'ΔS':>7} {'O₁':>5} {'O₂':>5} {'O₃':>5} {'O_avg':>6}")
    print(f"  {'─' * 56}")

    entropy_timeline = []
    overlap_timeline = []

    for i in range(1, len(snapshots)):
        s_prev = snapshots[i - 1]
        s_curr = snapshots[i]

        delta_s = s_curr["entropy"] - s_prev["entropy"]

        # Modal overlap (first 5 non-trivial modes)
        overlaps = modal_overlap(s_prev["eigenvectors"], s_curr["eigenvectors"], k_max=6)
        o_avg = np.mean(overlaps) if overlaps else 0

        entropy_timeline.append({"date": s_curr["date"], "S": s_curr["entropy"], "dS": delta_s})
        overlap_timeline.append({"date": s_curr["date"], "overlaps": overlaps, "o_avg": o_avg})

        date_str = str(s_curr["date"].date()) if hasattr(s_curr["date"], 'date') else str(s_curr["date"])
        o_str = "  ".join(f"{o:.2f}" for o in overlaps[:3])
        print(f"  {date_str:<12} {s_curr['entropy']:>7.4f} {delta_s:>+7.4f} {o_str}  {o_avg:>5.3f}")

    # ── 5. Summary Statistics ──
    all_overlaps = [e["o_avg"] for e in overlap_timeline]
    all_dS = [e["dS"] for e in entropy_timeline]

    print(f"\n  ── 5. SUMMARY ──")
    print(f"  Overlap mean: {np.mean(all_overlaps):.3f} ± {np.std(all_overlaps):.3f}")
    print(f"  Overlap range: [{min(all_overlaps):.3f}, {max(all_overlaps):.3f}]")
    print(f"  ΔS mean: {np.mean(all_dS):+.4f} ± {np.std(all_dS):.4f}")
    print(f"  ΔS range: [{min(all_dS):+.4f}, {max(all_dS):+.4f}]")

    # Low overlap = potential structural change
    low_overlap_dates = [e["date"] for e, o in zip(overlap_timeline, all_overlaps) if o < 0.5]
    high_dS_dates = [e["date"] for e in entropy_timeline if abs(e["dS"]) > np.std(all_dS)]

    print(f"\n  ⚠ Low overlap (<0.5) dates: {len(low_overlap_dates)}")
    for d in low_overlap_dates[:5]:
        print(f"    {d}")
    print(f"  ⚠ High |ΔS| (>σ) dates: {len(high_dS_dates)}")
    for d in high_dS_dates[:5]:
        print(f"    {d}")

    return {
        "period": period_name,
        "n_tickers": N,
        "k_clusters": k_use,
        "entropy": S,
        "overlap_mean": np.mean(all_overlaps),
        "overlap_std": np.std(all_overlaps),
        "dS_std": np.std(all_dS),
        "n_low_overlap": len(low_overlap_dates),
        "n_high_dS": len(high_dS_dates),
    }


if __name__ == "__main__":
    results = []
    for name, (start, end) in PERIODS.items():
        try:
            r = run_diagnostic(name, start, end)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print(f"\n{'═' * 80}")
        print(f"  COMPARATIVE SUMMARY")
        print(f"{'═' * 80}")
        print(f"  {'Period':<20} {'N':>4} {'k*':>3} {'S(G)':>6} {'O_avg':>6} {'O_σ':>5} {'ΔS_σ':>6} {'LowO':>5} {'HiΔS':>5}")
        print(f"  {'─' * 62}")
        for r in results:
            print(f"  {r['period']:<20} {r['n_tickers']:>4} {r['k_clusters']:>3} "
                  f"{r['entropy']:>6.3f} {r['overlap_mean']:>6.3f} {r['overlap_std']:>5.3f} "
                  f"{r['dS_std']:>6.4f} {r['n_low_overlap']:>5} {r['n_high_dS']:>5}")
