"""
3D WATER-LANDSCAPE VISUALIZER with TIME SLIDER
=================================================
Interactive 3D visualization with date scrubber:
  - X, Y: spectral embedding (Φ₁, Φ₂)
  - Z height = K (capital terrain)
  - Color = δ = λ - λ_eq (mispricing: red=overvalued, green=undervalued)
  - Node size = |m| (money assigned)
  - Edges from W
  - SLIDER: scrub through dates to see fields evolve

python visualize_3d.py
"""

import numpy as np
import plotly.graph_objects as go
import logging
from database_manager import DatabaseManager
from graph_builder import GraphBuilder
from fundamental_filter import FundamentalFilter
from heat_engine import HeatEngine

logging.basicConfig(level=logging.WARNING)


def build_visualization():
    print("⏳ Building pipeline...")
    db = DatabaseManager()
    ff = FundamentalFilter(db)
    ff.compute_all()
    gb = GraphBuilder(db)
    gb.load_data()
    gb.build()
    engine = HeatEngine(gb, ff)
    engine.solve(calibrate=True)
    print(f"   N={gb.N}, edges={np.sum(gb.W != 0)}, s={gb.s:.3f}")

    tickers = gb.tickers
    N = gb.N
    Phi = gb.eigenvectors

    # ── Layout: spectral embedding ──
    x_pos = Phi[:, 1] / (np.max(np.abs(Phi[:, 1])) + 1e-10)
    y_pos = Phi[:, 2] / (np.max(np.abs(Phi[:, 2])) + 1e-10)

    # ── Time series of fields ──
    m_all = engine.u_real  # (T, N) money
    lambda_field = getattr(engine, 'lambda_field', np.ones_like(m_all))
    lambda_eq = getattr(engine, 'lambda_eq', 18.0)

    # K(t) terrain
    cf = getattr(engine, 'capital_field', None)
    if cf is not None and hasattr(cf, 'c_daily'):
        K_all = cf.c_daily.values  # (T, N)
    else:
        K_all = np.ones_like(m_all)

    # Dates
    dates = gb.returns.index
    T = len(dates)

    # Sample every N days for slider (too many frames = slow)
    step = max(1, T // 60)  # ~60 frames
    frame_indices = list(range(0, T, step))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    print(f"   {len(frame_indices)} frames (step={step}d)")

    # ── Edges (static, top 20% by weight) ──
    W = gb.W
    edge_x, edge_y, edge_z_base = [], [], []
    threshold = np.percentile(np.abs(W[W != 0]), 85) if np.any(W != 0) else 0.1
    edge_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(W[i, j]) > threshold:
                edge_pairs.append((i, j))
                edge_x.extend([x_pos[i], x_pos[j], None])
                edge_y.extend([y_pos[i], y_pos[j], None])

    # ── Global normalization across all time for consistent scales ──
    K_abs_all = np.abs(K_all)
    K_global_max = np.nanmax(K_abs_all) + 1e-10
    m_abs_all = np.abs(m_all)
    m_global_max = np.nanmax(m_abs_all) + 1e-10
    delta_all = lambda_field - lambda_eq
    delta_std = np.nanstd(delta_all) + 1e-10

    # ── Build frames ──
    frames = []
    slider_steps = []

    for idx, t in enumerate(frame_indices):
        date_str = str(dates[t].date())

        # Fields at time t
        m_t = m_all[t]
        K_t = K_all[t] if t < len(K_all) else K_all[-1]
        lam_t = lambda_field[t] if t < len(lambda_field) else lambda_field[-1]
        delta_t = lam_t - lambda_eq

        # Normalize
        K_norm = np.abs(K_t) / K_global_max
        z_pos = K_norm
        m_norm = np.abs(m_t) / m_global_max
        node_size = 6 + 22 * m_norm
        delta_clip = np.clip(delta_t / delta_std, -3, 3)

        # Edge Z coordinates for this frame
        edge_z = []
        for i, j in edge_pairs:
            edge_z.extend([z_pos[i], z_pos[j], None])

        # Hover text
        hover = []
        for i, tk in enumerate(tickers):
            sector = gb.sector_map.get(tk, "?")
            F = ff.scores.get(tk, 0)
            hover.append(
                f"<b>{tk}</b> ({sector})<br>"
                f"Fecha: {date_str}<br>"
                f"m = {m_t[i]:.3f}<br>"
                f"K = {K_t[i]:.3f}<br>"
                f"λ = {lam_t[i]:.2f}<br>"
                f"δ = {delta_t[i]:+.2f}<br>"
                f"F = {F:.3f}"
            )

        # Labels for big nodes only
        labels = [tk if m_norm[i] > 0.25 else '' for i, tk in enumerate(tickers)]

        frame = go.Frame(
            data=[
                go.Scatter3d(  # Edges
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.12)', width=0.4),
                    hoverinfo='none',
                ),
                go.Scatter3d(  # Nodes
                    x=x_pos, y=y_pos, z=z_pos,
                    mode='markers+text',
                    marker=dict(
                        size=node_size,
                        color=delta_clip,
                        colorscale='RdYlGn_r',
                        cmin=-3, cmax=3,
                        colorbar=dict(title='δ = λ-λ_eq', x=1.02, len=0.6),
                        opacity=0.9,
                        line=dict(width=0.3, color='white'),
                    ),
                    text=labels,
                    textposition='top center',
                    textfont=dict(size=8, color='white'),
                    hovertext=hover,
                    hoverinfo='text',
                ),
            ],
            name=date_str,
        )
        frames.append(frame)

        slider_steps.append(dict(
            args=[[date_str], dict(frame=dict(duration=100, redraw=True), mode='immediate')],
            label=date_str,
            method='animate',
        ))

    # ── Initial frame ──
    init_frame = frames[-1]  # start at latest date

    fig = go.Figure(
        data=init_frame.data,
        frames=frames,
    )

    # ── Layout ──
    regime = getattr(engine, 'current_regime', 'unknown')
    fig.update_layout(
        title=dict(
            text=f"Water-Landscape 3D — {regime} | s={gb.s:.3f} | λ_eq={lambda_eq:.1f}",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title='Φ₁ (partición principal)',
            yaxis_title='Φ₂ (segunda partición)',
            zaxis_title='K (Capital = Terreno)',
            bgcolor='rgb(12,12,25)',
            xaxis=dict(gridcolor='rgba(80,80,120,0.3)', showbackground=False),
            yaxis=dict(gridcolor='rgba(80,80,120,0.3)', showbackground=False),
            zaxis=dict(gridcolor='rgba(80,80,120,0.3)', showbackground=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor='rgb(12,12,25)',
        font=dict(color='white'),
        margin=dict(l=0, r=80, t=50, b=80),
        showlegend=False,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate')]),
                ],
            ),
        ],
        sliders=[dict(
            active=len(slider_steps) - 1,
            steps=slider_steps,
            x=0.05, len=0.9,
            xanchor='left',
            y=0,
            yanchor='top',
            pad=dict(b=10, t=30),
            currentvalue=dict(
                prefix='Fecha: ',
                visible=True,
                xanchor='center',
                font=dict(size=14, color='white'),
            ),
            transition=dict(duration=100),
            font=dict(color='white'),
            bgcolor='rgba(50,50,80,0.5)',
            activebgcolor='rgba(100,100,200,0.8)',
        )],
    )

    output_path = "landscape_3d.html"
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"\n✅ Visualización 3D con timeline guardada en: {output_path}")
    print(f"   Abrir con: open {output_path}")

    try:
        import webbrowser
        webbrowser.open(output_path)
    except Exception:
        pass


if __name__ == "__main__":
    build_visualization()
