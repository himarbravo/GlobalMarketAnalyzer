"""
3D WATER-LANDSCAPE VISUALIZER v3
==================================
Sector-based layout for clarity:
  - Nodes arranged in circles BY SECTOR
  - Z = K (market cap as fallback for tickers without fundamentals)
  - Color = δ = λ - λ_eq
  - Size = |m|
  - Edges from W
  - Date slider

python visualize_3d.py
"""

import numpy as np
import plotly.graph_objects as go
import logging
from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.fundamental_filter import FundamentalFilter
from core.heat_engine import HeatEngine

logging.basicConfig(level=logging.WARNING)


def sector_layout(tickers, sector_map, sectors):
    """Arrange nodes in sector circles, spread apart."""
    N = len(tickers)
    x = np.zeros(N)
    y = np.zeros(N)

    # Group tickers by sector
    sector_groups = {}
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "OTHER")
        sector_groups.setdefault(s, []).append(i)

    n_sectors = len(sector_groups)
    sector_names = sorted(sector_groups.keys())

    for si, sec_name in enumerate(sector_names):
        members = sector_groups[sec_name]
        n_members = len(members)

        # Sector center on a big circle
        angle = 2 * np.pi * si / n_sectors
        cx = 3.0 * np.cos(angle)
        cy = 3.0 * np.sin(angle)

        # Members arranged in a small circle around sector center
        for mi, idx in enumerate(members):
            if n_members == 1:
                x[idx] = cx
                y[idx] = cy
            else:
                a = 2 * np.pi * mi / n_members
                r = 0.4 + 0.15 * n_members  # bigger sectors = bigger circles
                x[idx] = cx + r * np.cos(a)
                y[idx] = cy + r * np.sin(a)

    return x, y, sector_names, sector_groups


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

    tickers = gb.tickers
    N = gb.N
    W = gb.W
    dates = gb.returns.index
    T = len(dates)

    print(f"   N={N}, edges={np.sum(W != 0)}, s={gb.s:.3f}")

    # ── Layout: sector circles ──
    x_pos, y_pos, sector_names, sector_groups = sector_layout(
        tickers, gb.sector_map, gb.sectors)

    # ── K terrain: use market_cap as fallback ──
    cf = getattr(engine, 'capital_field', None)
    if cf is not None and hasattr(cf, 'c_daily'):
        K_all = cf.c_daily.values.copy()
    else:
        K_all = np.zeros((T, N))

    # Fallback: for tickers with K=0, use market_cap from fundamentals
    mcap_resp = db.client.table("fundamentals").select(
        "ticker, market_cap"
    ).order("report_date", desc=True).execute()

    mcap_map = {}
    if mcap_resp.data:
        for row in mcap_resp.data:
            t = row["ticker"]
            mc = row.get("market_cap")
            if t not in mcap_map and mc is not None and mc > 0:
                mcap_map[t] = mc

    # Fill K=0 and K=NaN with normalized market_cap
    mcap_values = np.array([mcap_map.get(t, 0) for t in tickers])
    mcap_max = max(mcap_values.max(), 1)
    mcap_norm = mcap_values / mcap_max

    for t_idx in range(T):
        for i in range(N):
            if K_all[t_idx, i] == 0 or np.isnan(K_all[t_idx, i]):
                K_all[t_idx, i] = mcap_norm[i] * 0.01  # scale to match K range

    print(f"   K>0 after fallback: {np.sum(K_all[-1] > 0)}/{N}")

    # ── Fields ──
    m_all = engine.u_real
    lambda_field = getattr(engine, 'lambda_field', np.ones_like(m_all))
    lambda_eq = getattr(engine, 'lambda_eq', 18.0)

    # ── Edges (top 30% — more visible) ──
    edge_pairs = []
    w_abs = np.abs(W)
    threshold = np.percentile(w_abs[w_abs > 0], 70) if np.any(w_abs > 0) else 0
    for i in range(N):
        for j in range(i + 1, N):
            if w_abs[i, j] > threshold:
                edge_pairs.append((i, j, W[i, j]))

    print(f"   {len(edge_pairs)} edges to draw")

    # ── Global normalization ──
    K_last = K_all[-1]
    K_p95 = np.percentile(K_last[K_last > 0], 95) if np.any(K_last > 0) else 1
    K_p5 = np.percentile(K_last[K_last > 0], 5) if np.any(K_last > 0) else 0

    m_global_max = np.nanmax(np.abs(m_all)) + 1e-10
    delta_std = np.nanstd(lambda_field - lambda_eq) + 1e-10

    # ── Sector label annotations ──
    annotations = []
    for sec_name, members in sector_groups.items():
        cx = np.mean([x_pos[i] for i in members])
        cy = np.mean([y_pos[i] for i in members])
        # Place label slightly above the sector cluster
        annotations.append(dict(
            x=cx, y=cy,
            text=f"<b>{sec_name}</b>",
            showarrow=False,
            font=dict(size=9, color='yellow'),
        ))

    # ── Sample frames ──
    step = max(1, T // 30)
    frame_indices = list(range(0, T, step))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    print(f"   {len(frame_indices)} frames")

    # ── Build frames ──
    frames = []
    slider_steps = []

    for idx, t_idx in enumerate(frame_indices):
        date_str = str(dates[t_idx].date())

        m_t = m_all[t_idx]
        K_t = K_all[min(t_idx, len(K_all) - 1)]
        lam_t = lambda_field[min(t_idx, len(lambda_field) - 1)]
        delta_t = lam_t - lambda_eq

        # Z = log(K) scaled for spread
        K_safe = np.maximum(np.abs(K_t), 1e-8)
        z_pos = np.log10(K_safe + 1e-8)
        z_pos = z_pos - z_pos.min()
        z_max = z_pos.max() + 1e-10
        z_pos = z_pos / z_max  # normalize to [0, 1]

        # Size ~ |m|
        m_norm = np.abs(m_t) / m_global_max
        node_size = 4 + 14 * m_norm

        # Color = δ
        delta_clip = np.clip(delta_t / delta_std, -2.5, 2.5)

        # Edges
        edge_x, edge_y, edge_z = [], [], []
        edge_line_colors = []
        for i, j, w in edge_pairs:
            edge_x.extend([x_pos[i], x_pos[j], None])
            edge_y.extend([y_pos[i], y_pos[j], None])
            edge_z.extend([z_pos[i], z_pos[j], None])

        # Hover
        hover = []
        for i, tk in enumerate(tickers):
            sector = gb.sector_map.get(tk, "?")
            F = ff.scores.get(tk, 0)
            cls = ff.classifications.get(tk, "neutral")
            hover.append(
                f"<b>{tk}</b> [{sector}]<br>"
                f"📅 {date_str}<br>"
                f"💰 m = {m_t[i]:.4f}<br>"
                f"🏗️ K = {K_t[i]:.6f}<br>"
                f"📊 λ = m/K = {lam_t[i]:.2f}<br>"
                f"📈 δ = λ-λ_eq = {delta_t[i]:+.2f}<br>"
                f"⭐ F = {F:.3f} ({cls})"
            )

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='rgba(80,120,255,0.12)', width=0.8),
                    hoverinfo='none',
                ),
                go.Scatter3d(
                    x=x_pos, y=y_pos, z=z_pos,
                    mode='markers+text',
                    marker=dict(
                        size=node_size,
                        color=delta_clip,
                        colorscale=[[0, '#22cc55'], [0.5, '#eeee44'], [1, '#ee2233']],
                        cmin=-2.5, cmax=2.5,
                        colorbar=dict(
                            title=dict(text='δ=λ−λ_eq', font=dict(size=10)),
                            x=1.01, len=0.4, y=0.7,
                            tickvals=[-2, 0, 2],
                            ticktext=['Barato ↓', 'Equilibrio', 'Caro ↑'],
                        ),
                        opacity=0.92,
                        line=dict(width=0.4, color='white'),
                    ),
                    text=tickers,
                    textposition='top center',
                    textfont=dict(size=6, color='rgba(200,210,255,0.85)'),
                    hovertext=hover,
                    hoverinfo='text',
                ),
            ],
            name=date_str,
        )
        frames.append(frame)
        slider_steps.append(dict(
            args=[[date_str], dict(frame=dict(duration=150, redraw=True), mode='immediate')],
            label=date_str, method='animate',
        ))

    # ── Figure ──
    fig = go.Figure(data=frames[-1].data, frames=frames)

    regime = getattr(engine, 'current_regime', '?')
    fig.update_layout(
        title=dict(
            text=(f"💧 Water-Landscape — {N} nodos × {len(edge_pairs)} aristas | "
                  f"{regime} | s={gb.s:.3f} | λ_eq={lambda_eq:.1f}"),
            font=dict(size=13, color='white'),
        ),
        scene=dict(
            xaxis_title='Sectores (layout circular)',
            yaxis_title='',
            zaxis_title='K (Capital = Terreno)',
            bgcolor='rgb(8,8,18)',
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False),
            zaxis=dict(gridcolor='rgba(50,50,100,0.3)', showbackground=True,
                       backgroundcolor='rgb(12,12,25)'),
            camera=dict(eye=dict(x=0.8, y=0.8, z=1.5)),  # top-down-ish view
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1.5, z=0.5),
        ),
        paper_bgcolor='rgb(8,8,18)',
        font=dict(color='white'),
        margin=dict(l=0, r=50, t=45, b=90),
        showlegend=False,
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=-0.08, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True)]),
                dict(label='⏸', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=len(slider_steps) - 1,
            steps=slider_steps,
            x=0.05, len=0.9, xanchor='left',
            y=-0.04, yanchor='top',
            pad=dict(b=5, t=35),
            currentvalue=dict(prefix='📅 ', visible=True, xanchor='center',
                              font=dict(size=12, color='#88bbff')),
            font=dict(color='white', size=7),
            bgcolor='rgba(20,20,50,0.6)',
            activebgcolor='rgba(60,80,180,0.8)',
        )],
    )

    output_path = "landscape_3d.html"
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"\n✅ Guardado: {output_path}")
    print(f"   open {output_path}")

    try:
        import webbrowser
        webbrowser.open(output_path)
    except Exception:
        pass


if __name__ == "__main__":
    build_visualization()
