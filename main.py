import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
import os

# --- EXTENSIBILIDAD: CARGA DINÁMICA ---
def load_csv_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def get_available_csvs():
    """Detecta todos los archivos CSV en el directorio raíz."""
    return [f for f in os.listdir('.') if f.endswith('.csv')]

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente de Mercado Matutino",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos Clean & Dark
st.markdown("""
<style>
    .big-signal { font-size: 2rem; font-weight: bold; padding: 10px; border-radius: 10px; text-align: center; }
    .buy-signal { background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; border: 1px solid #4CAF50; }
    .sell-signal { background-color: rgba(244, 67, 54, 0.2); color: #F44336; border: 1px solid #F44336; }
    .metric-box { background-color: #1E1E1E; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    h3 { border-bottom: 1px solid #333; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- CARGAR DATOS ---
signals_df = load_csv_data(config.SIGNALS_FILE)
history_df = load_csv_data(config.DATA_FILE)

# --- SIDEBAR: SYSTEM STATUS ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    # AI Status
    ai_provider = config.AI_CONFIG["PROVIDER"]
    st.markdown(f"**🧠 AI Brain:** `{ai_provider}` ({config.AI_CONFIG['MODEL']})")
    
    # Telegram Status
    tg_status = "✅ ONLINE" if config.TELEGRAM_CONFIG["ENABLED"] else "🔕 OFFLINE"
    st.markdown(f"**📲 Telegram:** {tg_status}")
    if not config.TELEGRAM_CONFIG["ENABLED"]:
        st.caption("Configura el Token en `config.py` para activar alertas.")

    st.divider()
    st.markdown("### Última Actualización")
    if not signals_df.empty and 'Date' in signals_df.columns:
        st.info(signals_df['Date'].iloc[0])
    
    if st.button("🔄 Refrescar Datos"):
        st.rerun()

# --- TABS PRINCIPALES ---
tab_advisor, tab_explorer = st.tabs(["🎯 Asesor Diario", "📁 Explorador de Datos (.nb / CSV)"])

with tab_advisor:
    if not signals_df.empty:
        # --- RESUMEN DE MERCADO (AI BRIEF) ---
        if 'Market_Context' in signals_df.columns:
            market_narrative = signals_df['Market_Context'].iloc[0]
            st.info(f"🧠 **Morning Brief**: {market_narrative}")

        # Determinar estado general visual
        # Usamos VIX o lógica simple si no hay columna "Status" explícita
        if "AVOID" in signals_df['Signal'].values:
            status_color = "🔴"
            status_text = "RIESGO ELEVADO"
        else:
            status_color = "🟢"
            status_text = "MERCADO OPERABLE"

        st.markdown(f"### {status_color} {status_text}")
        st.divider()

        # --- BLOQUE 1: MEJORES OPORTUNIDADES (Top Confidence) ---
        st.subheader("⭐ Top Picks (Alta Confianza)")
        # Filtrar solo entradas (BUYs) con alta confianza
        top_picks = signals_df[(signals_df['Signal'].str.contains('BUY')) & (signals_df['Confidence_Score'] >= 60)]
        
        if not top_picks.empty:
            cols = st.columns(min(len(top_picks), 3))
            for i, (_, row) in enumerate(top_picks.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-box" style="border-left: 5px solid #4CAF50;">
                        <h3>{row['Ticker']} <span style="font-size: 0.8em; float: right;">${row['Price']}</span></h3>
                        <p style="font-size: 1.2em; font-weight: bold; color: #4CAF50;">{row['Signal']}</p>
                        <p>Confidence: <b>{row['Confidence_Score']}/100</b></p>
                        <hr style="margin: 5px 0;">
                        <p style="font-size: 0.9em; font-style: italic;">"{row['AI_Insight']}"</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ El algoritmo no detecta oportunidades de alta probabilidad hoy.")

        st.divider()

        # --- BLOQUE 2: CARTERA COMPLETA ---
        st.subheader("📋 Análisis de Cartera (Conv: 50% Tech, 30% Fund, 20% News)")
        
        # Preparar dataframe para mostrar
        display_cols = ['Ticker', 'Price', 'Signal', 'Confidence_Score', 'Tech_Score', 'Fund_Score', 'Sentiment_Score', 'News_Summary', 'AI_Insight']
        # Asegurarnos de que las columnas existan (por retrocompatibilidad)
        available_cols = [c for c in display_cols if c in signals_df.columns]
        display_df = signals_df[available_cols].copy()

        # Renombrar para visualización más limpia
        display_df.rename(columns={'Confidence_Score': 'Total_Conviction', 'Sentiment_Score': 'News_Sent'}, inplace=True)

        # Colorear señales
        def color_signal(val):
            if 'BUY' in str(val): return 'color: #4CAF50; font-weight: bold'
            if 'SELL' in str(val): return 'color: #F44336; font-weight: bold'
            return 'color: #FFC107'
        
        st.dataframe(
            display_df.style.applymap(color_signal, subset=['Signal'])
                   .bar(subset=['Total_Conviction'], color='#2196F3', vmin=0, vmax=100)
                   .bar(subset=['Tech_Score'], color='#00E5FF', vmin=0, vmax=100)
                   .bar(subset=['Fund_Score'], color='#76FF03', vmin=0, vmax=100),
            use_container_width=True
        )

        st.divider()
        
        # --- BLOQUE 3: DETALLE TÉCNICO & FUNDAMENTAL ---
        col_list, col_chart = st.columns([1, 2])
        
        with col_list:
            st.subheader("🔍 Inspector")
            selected_ticker = st.radio("Selecciona Activo:", signals_df['Ticker'].unique())
            
            # Mostrar datos crudos del ticker seleccionado
            sel_row = signals_df[signals_df['Ticker'] == selected_ticker].iloc[0]
            st.write("**Datos Clave:**")
            st.write(sel_row.dropna().to_dict())

        with col_chart:
            st.subheader(f"Gráfico: {selected_ticker}")
            if not history_df.empty:
                ticker_data = history_df[['Date', selected_ticker]].dropna().copy()
                ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
                ticker_line = ticker_data[selected_ticker]
                ticker_sma = ticker_data[selected_ticker].rolling(window=config.STRATEGY["TREND_WINDOW"]).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_line, name="Precio", line=dict(color="#00E5FF")))
                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_sma, name=f"SMA {config.STRATEGY['TREND_WINDOW']}", line=dict(color="#FFAB00", dash="dash")))
                
                fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("⚠️ No se encontraron señales analizadas. Ejecuta 'python market_analyst.py' para generar el informe.")

with tab_explorer:
    st.subheader("📁 Explorador de Archivos Locales")
    st.caption("Visualiza cualquier dato exportado desde tu Notebook (.nb) u otras fuentes.")
    
    available_files = get_available_csvs()
    selected_file = st.selectbox("Selecciona un archivo para visualizar:", available_files)
    
    if selected_file:
        df_generic = pd.read_csv(selected_file)
        st.dataframe(df_generic.head(10), use_container_width=True)
        
        # Intentar graficar automáticamente si hay una columna 'Date' o similar
        cols = df_generic.columns.tolist()
        date_col = next((c for c in cols if 'date' in c.lower() or 'fecha' in c.lower()), None)
        numeric_cols = df_generic.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            st.markdown("#### 📊 Previsualización Gráfica")
            plot_cols = st.multiselect("Columnas a graficar:", numeric_cols, default=numeric_cols[:2])
            
            if plot_cols:
                if date_col:
                    fig_gen = px.line(df_generic, x=date_col, y=plot_cols, template="plotly_dark")
                else:
                    fig_gen = px.line(df_generic, y=plot_cols, template="plotly_dark")
                
                fig_gen.update_layout(height=400)
                st.plotly_chart(fig_gen, use_container_width=True)
        else:
            st.info("Este archivo no contiene datos numéricos para graficar.")

