"""
MACRO FLOW ANALYZER
===================
Analiza flujos de dinero basados en relaciones causa-efecto.

Grafos de Flujo:
- Tipos de interés → Bonos (inverso) → Empresas endeudadas
- Dollar → Oro (inverso) → Export companies
- VIX → Risk assets (inverso) → Safe havens

Ejemplos:
- Fed sube tipos → TLT cae → Growth tech sufre (val future earnings menos)
- Dollar sube → Oro cae → Exportadoras sufren (AAPL, gold miners)
- VIX sube → Stocks caen → TLT/GLD suben
"""

import pandas as pd
import numpy as np
import yfinance as yf

class MacroFlowAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def analyze_rates_flow(self):
        """
        Analiza el flujo de tipos de interés.
        TLT price down = rates up → Penaliza growth, favorece value/financials
        """
        try:
            tlt = self.data['TLT']
            tlt_change_30d = (tlt.iloc[-1] / tlt.iloc[-30] - 1) * 100
            
            # IEF (mid-term) para confirmar tendencia
            ief = self.data['IEF'] if 'IEF' in self.data.columns else tlt
            ief_change_30d = (ief.iloc[-1] / ief.iloc[-30] - 1) * 100
            
            # Interpretar
            if tlt_change_30d < -3:  # TLT down 3%+ = rates rising
                trend = "rising"
                impact = {
                    'beneficiados': ['JPM', 'BAC', 'WFC', 'XLF'],  # Bancos
                    'perjudicados': ['NVDA', 'TSLA', 'ARK', 'growth'],  # Growth
                    'explicacion': "Tipos subiendo → Bancos ganan más spread. Growth tech pierde (valuaciones futuras valen menos)."
                }
            elif tlt_change_30d > 3:  # TLT up 3%+ = rates falling
                trend = "falling"
                impact = {
                    'beneficiados': ['NVDA', 'TSLA', 'Growth tech'],
                    'perjudicados': ['JPM', 'BAC', 'Financials'],
                    'explicacion': "Tipos bajando → Growth tech se beneficia (cheap money, valuaciones futuras valen más)."
                }
            else:
                trend = "stable"
                impact = {'explicacion': "Tipos estables. Sin flujo claro actualmente."}
            
            return {
                'trend': trend,
                'tlt_change_30d': tlt_change_30d,
                'impact': impact
            }
        except Exception as e:
            return {'trend': 'unknown', 'error': str(e)}
    
    def analyze_dollar_flow(self):
        """
        Analiza el flujo del dólar.
        Dollar up → Oro down, exportadoras sufren (AAPL, semis)
        Dollar down → Oro up, exportadoras ganan
        """
        try:
            # Use UUP (Dollar ETF) as proxy, or DXY if available
            # For now, infer from GLD (inverse correlation)
            gld = self.data['GLD']
            gld_change_30d = (gld.iloc[-1] / gld.iloc[-30] - 1) * 100
            
            # GLD up usually means dollar down (inverse)
            if gld_change_30d > 5:
                dollar_trend = "weakening"
                impact = {
                    'beneficiados': ['GLD', 'GDX', 'AAPL', 'NVDA', 'Exportadoras'],
                    'perjudicados': ['Importadoras', 'Domestic retail'],
                    'explicacion': "Dólar débil → Oro sube. Exportadoras tech (AAPL, NVDA) más competitivas internacionalmente."
                }
            elif gld_change_30d < -5:
                dollar_trend = "strengthening"
                impact = {
                    'beneficiados': ['Importadoras', 'Domestic retail', 'Dollar-denominated debt holders'],
                    'perjudicados': ['GLD', 'GDX', 'Exportadoras', 'Emerging markets'],
                    'explicacion': "Dólar fuerte → Oro cae. Exportadoras sufren (productos USA más caros abroad)."
                }
            else:
                dollar_trend = "stable"
                impact = {'explicacion': "Dólar estable. Sin flujo significativo."}
            
            return {
                'trend': dollar_trend,
                'gld_change_30d': gld_change_30d,
                'impact': impact
            }
        except Exception as e:
            return {'trend': 'unknown', 'error': str(e)}
    
    def analyze_risk_flow(self):
        """
        Analiza el flujo de riesgo (VIX → Assets).
        VIX up → Risk-off (stocks down, safe havens up)
        VIX down → Risk-on (stocks up, safe havens flat)
        """
        try:
            vix = self.data['^VIX'].iloc[-1]
            vix_30d_ago = self.data['^VIX'].iloc[-30]
            vix_change = vix - vix_30d_ago
            
            if vix > 25:
                risk_mode = "risk-off"
                impact = {
                    'beneficiados': ['TLT', 'GLD', 'SHY', 'XLP', 'XLU'],
                    'perjudicados': ['NVDA', 'TSLA', 'IWM', 'BTC-USD', 'growth'],
                    'explicacion': f"VIX alto ({vix:.1f}) → FEAR. Dinero huye a bonos/oro. Growth tech cae más."
                }
            elif vix < 15:
                risk_mode = "risk-on"
                impact = {
                    'beneficiados': ['Growth tech', 'Small caps', 'BTC-USD', 'Emerging markets'],
                    'perjudicados': ['TLT', 'GLD', 'Defensive sectors'],
                    'explicacion': f"VIX bajo ({vix:.1f}) → GREED. Dinero va a riesgo. Safe havens underperform."
                }
            else:
                risk_mode = "neutral"
                impact = {'explicacion': f"VIX neutral ({vix:.1f}). Balance entre riesgo y seguridad."}
            
            return {
                'mode': risk_mode,
                'vix': vix,
                'vix_change_30d': vix_change,
                'impact': impact
            }
        except Exception as e:
            return {'mode': 'unknown', 'error': str(e)}
    
    def analyze_credit_flow(self):
        """
        Analiza el flujo de crédito (HYG/TLT ratio).
        Ratio falling → Credit stress → Crisis potential
        Ratio rising → Credit healthy → Risk-on
        """
        try:
            hyg = self.data['HYG']  # High yield bonds
            tlt = self.data['TLT']  # Safe treasuries
            
            ratio = hyg / tlt
            ratio_change_3d = (ratio.iloc[-1] / ratio.iloc[-3] - 1) * 100
            ratio_change_30d = (ratio.iloc[-1] / ratio.iloc[-30] - 1) * 100
            
            if ratio_change_30d < -5:
                credit_state = "stress"
                impact = {
                    'beneficiados': ['TLT', 'SHY', 'GLD'],
                    'perjudicados': ['Junk bonds', 'Risky assets', 'Leveraged companies'],
                    'explicacion': "Credit spreads widening (HYG vs TLT). Institucionales vendiendo riesgo. CRISIS POTENTIAL."
                }
            elif ratio_change_30d > 5:
                credit_state = "healthy"
                impact = {
                    'beneficiados': ['HYG', 'Junk bonds', 'Growth', 'Leverage plays'],
                    'perjudicados': ['Safe havens'],
                    'explicacion': "Credit spreads tightening. Confianza institucional. Risk-on environment."
                }
            else:
                credit_state = "stable"
                impact = {'explicacion': "Credit neutral. Sin stress significativo."}
            
            return {
                'state': credit_state,
                'ratio_change_3d': ratio_change_3d,
                'ratio_change_30d': ratio_change_30d,
                'impact': impact
            }
        except Exception as e:
            return {'state': 'unknown', 'error': str(e)}
    
    def get_money_flow_summary(self):
        """
        Combina todos los flujos para dar un panorama completo.
        """
        rates = self.analyze_rates_flow()
        dollar = self.analyze_dollar_flow()
        risk = self.analyze_risk_flow()
        credit = self.analyze_credit_flow()
        
        summary = {
            'rates': rates,
            'dollar': dollar,
            'risk': risk,
            'credit': credit
        }
        
        # Determine overall market regime
        risk_off_signals = 0
        risk_on_signals = 0
        
        if risk['mode'] == 'risk-off':
            risk_off_signals += 2
        elif risk['mode'] == 'risk-on':
            risk_on_signals += 2
        
        if credit['state'] == 'stress':
            risk_off_signals += 2
        elif credit['state'] == 'healthy':
            risk_on_signals += 1
        
        if rates['trend'] == 'rising':
            risk_off_signals += 1
        elif rates['trend'] == 'falling':
            risk_on_signals += 1
        
        if risk_off_signals > risk_on_signals:
            overall_regime = "RISK-OFF"
        elif risk_on_signals > risk_off_signals:
            overall_regime = "RISK-ON"
        else:
            overall_regime = "MIXED"
        
        summary['overall_regime'] = overall_regime
        
        return summary
    
    def print_flow_analysis(self):
        """
        Print human-readable flow analysis.
        """
        summary = self.get_money_flow_summary()
        
        print("\n" + "="*70)
        print("💰 MACRO MONEY FLOW ANALYSIS")
        print("="*70)
        
        print(f"\n🎯 Overall Regime: {summary['overall_regime']}")
        
        print("\n📊 1. TIPOS DE INTERÉS")
        rates = summary['rates']
        print(f"   Tendencia: {rates['trend'].upper()}")
        if 'impact' in rates and 'explicacion' in rates['impact']:
            print(f"   {rates['impact']['explicacion']}")
            if 'beneficiados' in rates['impact']:
                print(f"   ✅ Beneficiados: {', '.join(rates['impact']['beneficiados'][:3])}")
                print(f"   ❌ Perjudicados: {', '.join(rates['impact']['perjudicados'][:3])}")
        
        print("\n💵 2. DÓLAR")
        dollar = summary['dollar']
        print(f"   Tendencia: {dollar['trend'].upper()}")
        if 'impact' in dollar and 'explicacion' in dollar['impact']:
            print(f"   {dollar['impact']['explicacion']}")
        
        print("\n⚡ 3. RIESGO (VIX)")
        risk = summary['risk']
        print(f"   Modo: {risk['mode'].upper()} (VIX: {risk.get('vix', 'N/A'):.1f})")
        if 'impact' in risk and 'explicacion' in risk['impact']:
            print(f"   {risk['impact']['explicacion']}")
        
        print("\n🏦 4. CRÉDITO (HYG/TLT)")
        credit = summary['credit']
        print(f"   Estado: {credit['state'].upper()}")
        if 'impact' in credit and 'explicacion' in credit['impact']:
            print(f"   {credit['impact']['explicacion']}")
        
        print("\n" + "="*70 + "\n")
        
        return summary
