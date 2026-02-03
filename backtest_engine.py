import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

class BacktestEngine:
    """
    Walk-forward backtesting engine.
    Simulates running the analysis at historical dates and measures accuracy.
    """
    
    def __init__(self, tickers, start_date='2015-01-01', end_date=None):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
    def generate_random_dates(self, n=100):
        """Generate N random trading dates between start and end."""
        date_range = (self.end_date - self.start_date).days
        random_dates = []
        
        for _ in range(n):
            random_days = random.randint(0, date_range)
            test_date = self.start_date + timedelta(days=random_days)
            
            # Ensure it's a weekday
            while test_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                test_date += timedelta(days=1)
            
            # Ensure we have enough history (2 years) and enough future (1 month)
            if test_date >= self.start_date + timedelta(days=730) and \
               test_date <= self.end_date - timedelta(days=30):
                random_dates.append(test_date)
        
        return sorted(random_dates)
    
    def run_historical_analysis_at_date(self, target_date):
        """
        Simulates running the analysis at target_date using only data before that date.
        Returns signals and convictions for all tickers.
        """
        import config
        from regime_detector import RegimeDetector
        from factor_analyzer import FactorAnalyzer
        
        # 1. Download data UP TO target_date (no peeking into future)
        end_date_str = target_date.strftime('%Y-%m-%d')
        start_date_str = (target_date - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years history
        
        print(f"  Descargando datos hasta {end_date_str}...")
        raw_data = yf.download(self.tickers, start=start_date_str, end=end_date_str, progress=False, auto_adjust=False)
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Adj Close']
        else:
            data = raw_data[['Adj Close']].rename(columns={' Close': self.tickers[0]})
        
        # Forward fill missing values
        data = data.fillna(method='ffill', limit=5)
        
        # 2. Train regime detector with historical data only
        regime_detector = RegimeDetector()
        spy_returns = data['SPY'].pct_change().dropna()
        regime_detector.train(spy_returns)
        regime_name, regime_conf, regime_params = regime_detector.predict_current_regime(spy_returns.tail(60))
        
        # 3. Initialize factor analyzer
        factor_analyzer = FactorAnalyzer(data)
        
        # 4. For each ticker, calculate signals (simplified version)
        results = {}
        
        for ticker in self.tickers:
            if ticker in ['^VIX', 'SPY']:  # Skip market indicators
                continue
                
            try:
                series = data[ticker].dropna()
                if len(series) < 100:
                    continue
                
                price = series.iloc[-1]
                
                # Technical analysis (simplified)
                sma50 = series.tail(50).mean()
                returns = series.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                
                # RSI calculation
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]
                
                # Factor alpha
                asset_returns = series.pct_change().dropna()
                factor_result = factor_analyzer.decompose_asset(asset_returns)
                alpha = factor_result['alpha_annual_pct'] if factor_result else 0
                
                # Simple conviction (trend + rsi + alpha)
                trend_score = 100 if price > sma50 else 0
                rsi_score = 100 - abs(rsi_val - 50)  # Neutral RSI = highest score
                alpha_score = 50 + (alpha * 2)  # Scale alpha
                
                conviction = (trend_score * 0.4 + rsi_score * 0.3 + alpha_score * 0.3)
                conviction = max(0, min(100, conviction))
                
                # Signal
                if conviction >= 65:
                    signal = "BUY"
                elif conviction <= 35:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                results[ticker] = {
                    'signal': signal,
                    'conviction': conviction,
                    'price_at_signal': price,
                    'alpha': alpha
                }
                
            except Exception as e:
                print(f"    Error con {ticker}: {e}")
                continue
        
        return results, regime_name
    
    def measure_accuracy(self, target_date, signals, horizon_days=21):
        """
        Compare predictions vs reality after horizon_days.
        Returns accuracy metrics.
        """
        future_date = target_date + timedelta(days=horizon_days + 10)  # +10 buffer for weekends
        
        # Download actual prices
        actual_data = yf.download(
            list(signals.keys()),
            start=target_date.strftime('%Y-%m-%d'),
            end=future_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if isinstance(actual_data.columns, pd.MultiIndex):
            actual_prices = actual_data['Adj Close']
        else:
            actual_prices = actual_data[['Adj Close']]
        
        # Calculate accuracy
        results = {}
        
        for ticker, sig_data in signals.items():
            try:
                if ticker not in actual_prices.columns:
                    continue
                
                prices = actual_prices[ticker].dropna()
                if len(prices) < 2:
                    continue
                
                initial_price = prices.iloc[0]
                final_price = prices.iloc[-1]
                actual_return = (final_price / initial_price - 1) * 100
                
                signal = sig_data['signal']
                
                # Determine if prediction was correct
                hit = False
                if signal == "BUY" and actual_return > 2:  # 2% threshold
                    hit = True
                elif signal == "SELL" and actual_return < -2:
                    hit = True
                elif signal == "HOLD" and abs(actual_return) < 2:
                    hit = True
                
                results[ticker] = {
                    'signal': signal,
                    'actual_return': actual_return,
                    'hit': hit,
                    'conviction': sig_data['conviction'],
                    'alpha': sig_data['alpha']
                }
                
            except Exception as e:
                continue
        
        return results
    
    def run_monte_carlo_backtest(self, n_tests=100, horizon_days=21):
        """
        Run backtest on N random historical dates.
        Returns aggregate statistics.
        """
        test_dates = self.generate_random_dates(n_tests)
        all_results = []
        
        print(f"\\n🎲 Iniciando Monte Carlo Backtest ({n_tests} fechas random)...\\n")
        
        for i, date in enumerate(test_dates):
            print(f"[{i+1}/{len(test_dates)}] Fecha: {date.strftime('%Y-%m-%d')}")
            
            try:
                # Run analysis at this date
                signals, regime = self.run_historical_analysis_at_date(date)
                
                if not signals:
                    print("  ⚠️ Sin señales generadas")
                    continue
                
                # Measure accuracy
                accuracy_results = self.measure_accuracy(date, signals, horizon_days)
                
                for ticker, result in accuracy_results.items():
                    all_results.append({
                        'date': date,
                        'ticker': ticker,
                        'regime': regime,  # regime is now a string
                        **result
                    })
                
                hits = sum(1 for r in accuracy_results.values() if r['hit'])
                total = len(accuracy_results)
                print(f"  ✅ Accuracy: {hits}/{total} ({hits/total*100:.1f}%)\\n")
                
            except Exception as e:
                print(f"  ❌ Error: {e}\\n")
                continue
        
        # Aggregate statistics
        df = pd.DataFrame(all_results)
        
        if len(df) == 0:
            print("⚠️ No se obtuvieron resultados")
            return None
        
        print("\\n" + "="*60)
        print("📊 RESULTADOS DEL BACKTEST")
        print("="*60)
        
        overall_accuracy = df['hit'].mean()
        print(f"\\n🎯 Accuracy Global: {overall_accuracy*100:.1f}%")
        
        # By signal type
        print("\\n📈 Accuracy por Tipo de Señal:")
        for signal in ['BUY', 'SELL','HOLD']:
            sig_df = df[df['signal'] == signal]
            if len(sig_df) > 0:
                acc = sig_df['hit'].mean()
                print(f"  {signal:8s}: {acc*100:.1f}% ({len(sig_df)} casos)")
        
        # By regime
        print("\\n🌡️ Accuracy por Régimen:")
        for regime in df['regime'].unique():
            reg_df = df[df['regime'] == regime]
            acc = reg_df['hit'].mean()
            print(f"  {regime:8s}: {acc*100:.1f}% ({len(reg_df)} casos)")
        
        # Best/worst performers
        ticker_acc = df.groupby('ticker')['hit'].agg(['mean', 'count'])
        ticker_acc = ticker_acc[ticker_acc['count'] >= 5]  # Min 5 tests
        
        if len(ticker_acc) > 0:
            print("\\n🏆 Top Accuracy por Ticker (min 5 tests):")
            top_tickers = ticker_acc.sort_values('mean', ascending=False).head(5)
            for ticker, row in top_tickers.iterrows():
                print(f"  {ticker:8s}: {row['mean']*100:.1f}% ({int(row['count'])} tests)")
        
        # Save results
        df.to_csv('backtest_results.csv', index=False)
        print("\\n💾 Resultados guardados en: backtest_results.csv")
        
        return df


if __name__ == "__main__":
    # Test with a subset of tickers
    import config
    
    test_tickers = ['SPY', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BTC-USD', 'GLD']
    
    engine = BacktestEngine(
        tickers=test_tickers,
        start_date='2018-01-01',  # 7 years history
        end_date='2025-12-31'
    )
    
    # Run Monte Carlo with 50 random dates
    results = engine.run_monte_carlo_backtest(n_tests=50, horizon_days=21)
