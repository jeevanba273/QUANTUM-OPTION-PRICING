import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import time
from datetime import datetime, timedelta
from app.models.classical_models import BlackScholes, MonteCarlo
from app.models.quantum_models import QuantumAmplitudeEstimation
from app.models.advanced_quantum_models import AdvancedQuantumOptionPricing

class MarketValidation:
    """
    Validate option pricing models against real market data.
    Support for both global and Indian markets.
    """
    
    @staticmethod
    def fetch_market_data(ticker, start_date, end_date, market='global'):
        """
        Fetch historical stock data for global or Indian markets.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            market: 'global' or 'india'
            
        Returns:
            DataFrame with historical stock data
        """
        # Add ".NS" suffix for Indian stocks if not already present
        if market == 'india' and not ticker.endswith('.NS') and ticker.lower() not in ['nifty', 'banknifty', 'finnifty']:
            ticker = f"{ticker}.NS"
            
        # Handle special index cases for Indian market
        if market == 'india' and ticker.lower() == 'nifty':
            ticker = "^NSEI"  # NIFTY 50 index
        elif market == 'india' and ticker.lower() == 'banknifty':
            ticker = "^NSEBANK"  # NIFTY Bank index
        elif market == 'india' and ticker.lower() == 'finnifty':
            ticker = "NIFTY_FIN_SERVICE.NS"  # Financial Services index
        
        # Implement retry logic with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Use auto_adjust=True to avoid warnings
                stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                
                # Check if data is empty
                if stock_data.empty:
                    print(f"No data found for {ticker}")
                    return pd.DataFrame()
                    
                # Add a delay to avoid rate limiting in subsequent calls
                time.sleep(2)
                
                return stock_data
            
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch data for {ticker} after {max_retries} attempts")
                    return pd.DataFrame()
    
    @staticmethod
    def fetch_indian_option_chain(symbol):
        """
        Fetch option chain data from NSE for Indian stocks.
        
        Args:
            symbol: Stock symbol (without .NS suffix)
            
        Returns:
            Dictionary containing option chain data
        """
        try:
            # NSE-specific headers to mimic browser request
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
                'accept-language': 'en,gu;q=0.9,hi;q=0.8',
                'accept-encoding': 'gzip, deflate, br'
            }
            
            # Map symbol to proper NSE symbol if needed
            symbol_map = {
                'nifty': 'NIFTY',
                'banknifty': 'BANKNIFTY',
                'finnifty': 'FINNIFTY'
            }
            
            nse_symbol = symbol_map.get(symbol.lower(), symbol.upper())
            
            # NSE API URL for option chain data
            # For indices
            if nse_symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                url = f"https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}"
            else:
                # For stocks
                url = f"https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}"
            
            session = requests.Session()
            # First request to get cookies
            cookie_request = session.get("https://www.nseindia.com/", headers=headers, timeout=10)
            
            # Second request to get the data
            response = session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                option_chain = response.json()
                return option_chain
            else:
                print(f"Error fetching Indian option chain: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching Indian option chain: {e}")
            return None
    
    @staticmethod
    def fetch_option_chain(ticker, market='global'):
        """
        Fetch current option chain data with market-specific handling.
        
        Args:
            ticker: Stock ticker symbol
            market: 'global' or 'india'
            
        Returns:
            Tuple of (expiry dates, option chains dict)
        """
        if market == 'india':
            # For Indian markets, use NSE API
            # Remove .NS suffix if present for the API call
            clean_ticker = ticker.replace('.NS', '')
            
            # Handle popular Indian indices
            if clean_ticker.lower() in ['nifty', 'banknifty', 'finnifty', '^nsei', '^nsebank']:
                # Normalize index names
                index_map = {
                    'nifty': 'NIFTY',
                    'banknifty': 'BANKNIFTY',
                    'finnifty': 'FINNIFTY',
                    '^nsei': 'NIFTY',
                    '^nsebank': 'BANKNIFTY'
                }
                symbol = index_map.get(clean_ticker.lower(), 'NIFTY')
                
                # Try to get option data
                option_data = MarketValidation.fetch_indian_option_chain(symbol)
                
                if option_data and 'records' in option_data:
                    # Extract expiry dates
                    expiry_dates = option_data['records'].get('expiryDates', [])
                    
                    # Organize data by expiry date
                    option_chains = {}
                    for date in expiry_dates:
                        # Filter data for this expiry
                        ce_options = []
                        pe_options = []
                        
                        for item in option_data['records'].get('data', []):
                            if item.get('expiryDate') == date:
                                if 'CE' in item:
                                    ce_options.append(item['CE'])
                                if 'PE' in item:
                                    pe_options.append(item['PE'])
                        
                        # Convert to DataFrame format similar to yfinance
                        ce_df = pd.DataFrame(ce_options) if ce_options else pd.DataFrame()
                        pe_df = pd.DataFrame(pe_options) if pe_options else pd.DataFrame()
                        
                        # Rename columns to match yfinance format
                        if not ce_df.empty:
                            ce_df = ce_df.rename(columns={
                                'strikePrice': 'strike',
                                'lastPrice': 'lastPrice',
                                'impliedVolatility': 'impliedVolatility'
                            })
                        
                        if not pe_df.empty:
                            pe_df = pe_df.rename(columns={
                                'strikePrice': 'strike',
                                'lastPrice': 'lastPrice',
                                'impliedVolatility': 'impliedVolatility'
                            })
                        
                        # Create an object with calls and puts properties
                        class OptionChain:
                            def __init__(self, calls, puts):
                                self.calls = calls
                                self.puts = puts
                        
                        option_chains[date] = OptionChain(ce_df, pe_df)
                    
                    return expiry_dates, option_chains
                
                print(f"No option data found for Indian index {symbol}")
                return [], {}
            
            else:
                # For individual stocks, try to use NSE API with stock symbol
                option_data = MarketValidation.fetch_indian_option_chain(clean_ticker)
                
                if option_data and 'records' in option_data:
                    # Extract expiry dates
                    expiry_dates = option_data['records'].get('expiryDates', [])
                    
                    # Organize data by expiry date
                    option_chains = {}
                    for date in expiry_dates:
                        # Filter data for this expiry
                        ce_options = []
                        pe_options = []
                        
                        for item in option_data['records'].get('data', []):
                            if item.get('expiryDate') == date:
                                if 'CE' in item:
                                    ce_options.append(item['CE'])
                                if 'PE' in item:
                                    pe_options.append(item['PE'])
                        
                        # Convert to DataFrame format similar to yfinance
                        ce_df = pd.DataFrame(ce_options) if ce_options else pd.DataFrame()
                        pe_df = pd.DataFrame(pe_options) if pe_options else pd.DataFrame()
                        
                        # Rename columns to match yfinance format
                        if not ce_df.empty:
                            ce_df = ce_df.rename(columns={
                                'strikePrice': 'strike',
                                'lastPrice': 'lastPrice',
                                'impliedVolatility': 'impliedVolatility'
                            })
                        
                        if not pe_df.empty:
                            pe_df = pe_df.rename(columns={
                                'strikePrice': 'strike',
                                'lastPrice': 'lastPrice',
                                'impliedVolatility': 'impliedVolatility'
                            })
                        
                        # Create an object with calls and puts properties
                        class OptionChain:
                            def __init__(self, calls, puts):
                                self.calls = calls
                                self.puts = puts
                        
                        option_chains[date] = OptionChain(ce_df, pe_df)
                    
                    return expiry_dates, option_chains
                
                # Fallback to yfinance for Indian stocks if NSE API fails
                try:
                    stock = yf.Ticker(f"{clean_ticker}.NS")
                    expiry_dates = stock.options
                    
                    if not expiry_dates:
                        print(f"No options available for {clean_ticker}")
                        return [], {}
                    
                    option_chains = {}
                    for date in expiry_dates:
                        try:
                            # Add delay to avoid rate limiting
                            time.sleep(2)
                            option_chains[date] = stock.option_chain(date)
                        except Exception as e:
                            print(f"Error fetching option chain for {clean_ticker} expiry {date}: {e}")
                    
                    return expiry_dates, option_chains
                    
                except Exception as e:
                    print(f"Error in fetch_option_chain for {clean_ticker}: {e}")
                    return [], {}
        
        else:
            # For global markets, use yfinance
            try:
                stock = yf.Ticker(ticker)
                expiry_dates = stock.options
                
                if not expiry_dates:
                    print(f"No options available for {ticker}")
                    return [], {}
                
                option_chains = {}
                for date in expiry_dates:
                    try:
                        # Add delay to avoid rate limiting
                        time.sleep(2)
                        option_chains[date] = stock.option_chain(date)
                    except Exception as e:
                        print(f"Error fetching option chain for {ticker} expiry {date}: {e}")
                
                return expiry_dates, option_chains
                
            except Exception as e:
                print(f"Error in fetch_option_chain for {ticker}: {e}")
                return [], {}
    
    @staticmethod
    def calculate_implied_volatility(option_price, S, K, T, r, option_type):
        """
        Calculate implied volatility using the bisection method.
        
        Args:
            option_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility estimate
        """
        # Skip calculation if price is invalid
        if option_price <= 0:
            return 0.3  # Default value
        
        def price_difference(sigma):
            if option_type.lower() == 'call':
                theoretical_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                theoretical_price = BlackScholes.put_price(S, K, T, r, sigma)
            # Ensure we are working with a scalar float
            return float(theoretical_price) - option_price
        
        max_iterations = 100
        precision = 1e-6
        a, b = 0.001, 5.0  # Reasonable volatility range
        
        # Check if solution is bracketed
        if price_difference(a) * price_difference(b) > 0:
            return 0.3  # Default to 30% volatility if not bracketed
        
        for i in range(max_iterations):
            c = (a + b) / 2
            diff_c = price_difference(c)
            if abs(diff_c) < precision:
                return c
            if price_difference(a) * diff_c < 0:
                b = c
            else:
                a = c
        return (a + b) / 2
    
    @staticmethod
    def to_float(x):
        """
        Convert x to float.
        
        If x is a single element Series, use its first element.
        """
        if isinstance(x, pd.Series):
            if len(x) > 0:
                return float(x.iloc[0])
            else:
                return 0.0
        
        # Handle None or NaN values
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
            
        return float(x)
    
    @staticmethod
    def process_option_data(ticker, expiry, S, r, hist_vol, option, option_type):
        """Helper method to process individual options"""
        try:
            # Convert option values to scalars using the helper
            K = MarketValidation.to_float(option['strike'])
            market_price = MarketValidation.to_float(option['lastPrice'])
            
            # Skip if market price is invalid
            if pd.isna(market_price) or market_price <= 0:
                return None
                
            # Calculate time to expiry in years
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            T = (expiry_date - datetime.now()).days / 365
            
            # Skip if expiry is in the past or too close
            if T <= 0.01:
                return None
            
            # Calculate implied volatility using the market price
            imp_vol = MarketValidation.calculate_implied_volatility(
                market_price, S, K, T, r, option_type)
            
            # Calculate prices using different models
            if option_type == 'call':
                bs_price = BlackScholes.call_price(S, K, T, r, hist_vol)
                mc_price = MonteCarlo.call_price(S, K, T, r, hist_vol)
                quantum_price = QuantumAmplitudeEstimation.european_call_price(
                    S, K, T, r, hist_vol, num_qubits=6, num_shots=1000)['price']
                advanced_quantum_price = AdvancedQuantumOptionPricing.european_option_price(
                    S, K, T, r, hist_vol, 'call', num_qubits=6)['price']
            else:
                bs_price = BlackScholes.put_price(S, K, T, r, hist_vol)
                mc_price = MonteCarlo.put_price(S, K, T, r, hist_vol)
                quantum_price = QuantumAmplitudeEstimation.european_put_price(
                    S, K, T, r, hist_vol, num_qubits=6, num_shots=1000)['price']
                advanced_quantum_price = AdvancedQuantumOptionPricing.european_option_price(
                    S, K, T, r, hist_vol, 'put', num_qubits=6)['price']
            
            # Calculate pricing errors (percentage)
            bs_error = abs(bs_price - market_price) / market_price * 100
            mc_error = abs(mc_price - market_price) / market_price * 100
            quantum_error = abs(quantum_price - market_price) / market_price * 100
            advanced_quantum_error = abs(advanced_quantum_price - market_price) / market_price * 100
            
            return {
                'ticker': ticker,
                'expiry': expiry,
                'strike': K,
                'type': option_type,
                'market_price': market_price,
                'stock_price': S,
                'time_to_expiry': T,
                'historical_volatility': hist_vol,
                'implied_volatility': imp_vol,
                'bs_price': bs_price,
                'mc_price': mc_price,
                'quantum_price': quantum_price,
                'advanced_quantum_price': advanced_quantum_price,
                'bs_error': bs_error,
                'mc_error': mc_error,
                'quantum_error': quantum_error,
                'advanced_quantum_error': advanced_quantum_error
            }
        except Exception as e:
            print(f"Error processing option {option_type} with strike {K if 'K' in locals() else 'unknown'}: {e}")
            return None
    
    @staticmethod
    def validate_against_market(tickers=['NIFTY', 'BANKNIFTY'], num_options=2, market='india'):
        """
        Validate quantum and classical models against real market option prices.
        
        Args:
            tickers: List of stock ticker symbols
            num_options: Number of options to sample per ticker
            market: 'global' or 'india'
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for ticker in tickers:
            try:
                # Fetch stock data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
                
                print(f"Fetching data for {ticker} in {market} market...")
                stock_data = MarketValidation.fetch_market_data(ticker, start_date, end_date, market=market)
                
                # Check if data is empty
                if stock_data.empty:
                    print(f"Skipping {ticker} - no stock data available")
                    continue
                
                # Get current stock price - ensure it's a scalar with error handling
                try:
                    if len(stock_data) > 0:
                        S = float(stock_data['Close'].iloc[-1])
                    else:
                        print(f"Empty stock data for {ticker}")
                        continue
                except IndexError:
                    print(f"IndexError for {ticker} - using default price")
                    S = 100.0  # Default price if data access fails
                
                # Calculate historical volatility with error handling
                try:
                    returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
                    hist_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.3
                except Exception as e:
                    print(f"Error calculating volatility for {ticker}: {e}")
                    hist_vol = 0.3  # Default volatility
                
                # Fetch option chain
                print(f"Fetching option chain for {ticker}...")
                expiry_dates, chains = MarketValidation.fetch_option_chain(ticker, market=market)
                
                # Risk-free rate (use appropriate Indian rate for INR)
                r = 0.065 if market == 'india' else 0.05
                
                # Process the options data for the first expiry date
                if len(expiry_dates) > 0:
                    expiry = expiry_dates[0]
                    chain = chains[expiry]
                    
                    # Process calls
                    if hasattr(chain, 'calls') and not chain.calls.empty:
                        calls = chain.calls
                        # Get the closest options to the current price
                        try:
                            calls_diff = abs(calls['strike'] - S)
                            # Sort and take top num_options
                            closest_indices = calls_diff.nsmallest(num_options).index
                            closest_calls = calls.loc[closest_indices]
                            
                            for idx, option in closest_calls.iterrows():
                                result = MarketValidation.process_option_data(
                                    ticker, expiry, S, r, hist_vol, option, 'call')
                                if result:
                                    results.append(result)
                        except Exception as e:
                            print(f"Error processing calls for {ticker}: {e}")
                    
                    # Process puts
                    if hasattr(chain, 'puts') and not chain.puts.empty:
                        puts = chain.puts
                        # Get the closest options to the current price
                        try:
                            puts_diff = abs(puts['strike'] - S)
                            # Sort and take top num_options
                            closest_indices = puts_diff.nsmallest(num_options).index
                            closest_puts = puts.loc[closest_indices]
                            
                            for idx, option in closest_puts.iterrows():
                                result = MarketValidation.process_option_data(
                                    ticker, expiry, S, r, hist_vol, option, 'put')
                                if result:
                                    results.append(result)
                        except Exception as e:
                            print(f"Error processing puts for {ticker}: {e}")
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                import traceback
                traceback.print_exc()
        
        return pd.DataFrame(results) if results else pd.DataFrame()