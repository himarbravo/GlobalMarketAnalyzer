"""
STREAMLIT DASHBOARD - Strategy Testing & Analysis
==================================================
Interactive dashboard for:
1. Strategy performance comparison
2. Backtest results visualization
3. Current portfolio recommendations
4. Macro flow analysis

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
try:
    from unified_system import UnifiedIntelligenceSystem
    from strategy_backtest_comparator import StrategyBacktester
    from database_manager import DatabaseManager
    import yfinance as yf
except ImportError as e:
    st.error(f"Error importing modules: {e}")

# Page config
st.set_page_config(
    page_title="Global Market Analyzer - Strategy Testing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .strategy-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ddd;
        margin: 0.5rem 0;
    }
    .metric-positive {
        color: #28a745;
        font-weight: bold;
    }
    .metric-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Control Panel")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Current Analysis", "Strategy Backtesting", "Macro Flows", "Compare Strategies"]
)

st.sidebar.markdown("---")

# Strategy selection for current analysis
if mode == "Current Analysis":
    strategy_mode = st.sidebar.selectbox(
        "Strategy",
        ["auto", "value", "growth", "risk_off"]
    )
    
    offense_weight = st.sidebar.slider(
        "Offense Weight (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=10,
        help="Percentage allocated to offensive (growth) assets"
    )

# Main content
st.markdown('<div class="main-header">📊 Global Market Analyzer</div>', unsafe_allow_html=True)
st.markdown("### Strategy Testing & Portfolio Optimization Dashboard")

# ============================================================================
# MODE 1: CURRENT ANALYSIS
# ============================================================================
if mode == "Current Analysis":
    st.header("🔍 Current Market Analysis & Portfolio")
    
    with st.spinner("Running unified intelligence system..."):
        try:
            # Run unified system
            system = UnifiedIntelligenceSystem(
                strategy_mode=strategy_mode,
                offense_weight=offense_weight
            )
            
            # Capture output (simplified - just run)
            portfolio_df, signals_df = system.run_full_analysis()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Active Strategy", strategy_mode.upper())
            
            with col2:
                offense_pct = portfolio_df[portfolio_df['type'] == 'OFFENSE']['allocation'].sum()
                st.metric("Offense Allocation", f"{offense_pct:.1f}%")
            
            with col3:
                defense_pct = portfolio_df[portfolio_df['type'] == 'DEFENSE']['allocation'].sum()
                st.metric("Defense Allocation", f"{defense_pct:.1f}%")
            
            # Portfolio visualization
            st.subheader("💼 Optimized Portfolio")
            
            if len(portfolio_df) > 0:
                # Pie chart
                fig = px.pie(
                    portfolio_df,
                    values='allocation',
                    names='ticker',
                    title='Portfolio Allocation',
                    color='type',
                    color_discrete_map={'OFFENSE': '#ff6b6b', 'DEFENSE': '#4ecdc4', 'NEUTRAL': '#95e1d3'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    portfolio_df[['ticker', 'type', 'allocation', 'conviction']].style.format({
                        'allocation': '{:.1f}%',
                        'conviction': '{:.1f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No portfolio generated")
            
            # All signals
            st.subheader("📈 All Asset Signals")
            if len(signals_df) > 0:
                # Get EURUSD for display
                try:
                    eur_rate = yf.download("EURUSD=X", period="1d", progress=False)['Adj Close'].iloc[-1]
                    signals_df['price_eur'] = signals_df['price'] / eur_rate
                except:
                    signals_df['price_eur'] = signals_df['price'] # Fallback
                
                st.dataframe(
                    signals_df[['ticker', 'signal', 'conviction', 'price_eur', 'regime']].style.format({
                        'conviction': '{:.1f}',
                        'price_eur': '€{:.2f}'
                    }),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            st.exception(e)

# ============================================================================
# MODE 2: STRATEGY BACKTESTING
# ============================================================================
elif mode == "Strategy Backtesting":
    st.header("⏮️ Strategy Backtesting")
    
    st.markdown("""
    Compare performance of **Value**, **Growth**, and **Risk-Off** strategies 
    across historical market conditions.
    """)
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest on historical data... This may take a few minutes."):
            try:
                backtester = StrategyBacktester()
                backtester.run_full_backtest()
                
                # Load results
                results_df = pd.DataFrame(backtester.results)
                
                if len(results_df) > 0:
                    st.success("✅ Backtest completed!")
                    
                    # Summary metrics
                    st.subheader("📊 Overall Performance")
                    
                    cols = st.columns(3)
                    
                    for idx, strategy_name in enumerate(results_df['strategy'].unique()):
                        subset = results_df[results_df['strategy'] == strategy_name]
                        avg_return = subset['portfolio_return_pct'].mean()
                        
                        with cols[idx]:
                            st.metric(
                                strategy_name,
                                f"{avg_return:+.1f}%",
                                delta=f"Win rate: {(subset['portfolio_return_pct'] > 0).sum() / len(subset) * 100:.0f}%"
                            )
                    
                    # Performance by scenario
                    st.subheader("📅 Performance by Scenario")
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    for strategy in results_df['strategy'].unique():
                        strategy_data = results_df[results_df['strategy'] == strategy]
                        
                        fig.add_trace(go.Bar(
                            name=strategy,
                            x=strategy_data['scenario'],
                            y=strategy_data['portfolio_return_pct'],
                            text=[f"{val:+.1f}%" for val in strategy_data['portfolio_return_pct']],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Strategy Returns by Period",
                        xaxis_title="Scenario",
                        yaxis_title="Portfolio Return (%)",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(
                        results_df[['scenario', 'strategy', 'regime', 'vix', 'portfolio_return_pct']].style.format({
                            'vix': '{:.1f}',
                            'portfolio_return_pct': '{:+.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                else:
                    st.warning("No results generated")
                    
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                st.exception(e)
    
    # Load existing results if available
    try:
        existing_results = pd.read_csv('strategy_backtest_results.csv')
        
        if len(existing_results) > 0:
            st.info("💡 Previous backtest results found. Click 'Run Backtest' to generate new results.")
            
            with st.expander("View Previous Results"):
                st.dataframe(existing_results, use_container_width=True)
                
    except FileNotFoundError:
        st.info("No previous backtest results. Click 'Run Backtest' to start.")

# ============================================================================
# MODE 3: MACRO FLOWS
# ============================================================================
elif mode == "Macro Flows":
    st.header("💰 Macro Money Flow Analysis")
    
    st.markdown("""
    Visualize how money flows between asset classes based on macro conditions:
    - **Interest Rates** → Bonds → Growth vs Value
    - **Dollar Strength** → Gold → Exporters
    - **VIX** → Risk assets vs Safe havens
    - **Credit Spreads** → High yield vs Treasuries
    """)
    
    with st.spinner("Analyzing macro flows..."):
        try:
            from macro_flow_analyzer import MacroFlowAnalyzer
            
            # Download recent data
            tickers = ['SPY', 'TLT', 'GLD', 'HYG', 'IEF', '^VIX']
            data = yf.download(tickers, period='1y', progress=False, auto_adjust=False)['Adj Close']
            data = data.ffill()
            
            # Analyze
            analyzer = MacroFlowAnalyzer(data)
            flow_summary = analyzer.get_money_flow_summary()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Interest Rates Flow")
                rates_info = flow_summary['rates']
                st.metric("Trend", rates_info['trend'].upper())
                if 'impact' in rates_info and 'explicacion' in rates_info['impact']:
                    st.info(rates_info['impact']['explicacion'])
            
            with col2:
                st.subheader("⚡ Risk Flow (VIX)")
                risk_info = flow_summary['risk']
                st.metric("Mode", risk_info['mode'].upper(), delta=f"VIX: {risk_info.get('vix', 'N/A'):.1f}")
                if 'impact' in risk_info and 'explicacion' in risk_info['impact']:
                    st.info(risk_info['impact']['explicacion'])
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("💵 Dollar Flow")
                dollar_info = flow_summary['dollar']
                st.metric("Strength", dollar_info['trend'].upper())
                if 'impact' in dollar_info and 'explicacion' in dollar_info['impact']:
                    st.info(dollar_info['impact']['explicacion'])
            
            with col4:
                st.subheader("🏦 Credit Flow")
                credit_info = flow_summary['credit']
                st.metric("State", credit_info['state'].upper())
                if 'impact' in credit_info and 'explicacion' in credit_info['impact']:
                    st.info(credit_info['impact']['explicacion'])
            
            # Overall regime
            st.subheader("🎯 Overall Market Regime")
            st.markdown(f"### {flow_summary['overall_regime']}")
            
        except Exception as e:
            st.error(f"Error analyzing flows: {e}")
            st.exception(e)

# ============================================================================
# MODE 4: COMPARE STRATEGIES
# ============================================================================
elif mode == "Compare Strategies":
    st.header("⚖️ Strategy Comparison")
    
    st.markdown("""
    Compare the three investment philosophies:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Value (Buffett)")
        st.markdown("""
        **Philosophy**: Buy cheap, hold long
        
        **Best in**:
        - Crisis / Recessions
        - High VIX periods
        - Low correlation markets
        
        **Returns**:
        - Normal: 8-12% annual
        - Crisis: Up to 20%+
        """)
    
    with col2:
        st.subheader("🚀 Growth (ARK)")
        st.markdown("""
        **Philosophy**: Innovation at any price
        
        **Best in**:
        - Bull markets
        - Low interest rates
        - Risk-on environments
        
        **Returns**:
        - Bull: 30-50% annual
        - Bear: -40% to -60%
        """)
    
    with col3:
        st.subheader("🛡️ Risk-Off (Dalio)")
        st.markdown("""
        **Philosophy**: Capital preservation
        
        **Best in**:
        - VIX > 25
        - Banking crises
        - Wars / Uncertainty
        
        **Returns**:
        - Crisis: +10% to +30%
        - Bull: -5% to +5%
        """)

# Footer
st.markdown("---")
st.markdown("**Global Market Analyzer** | Unified Intelligence System | Strategy Testing Dashboard")
