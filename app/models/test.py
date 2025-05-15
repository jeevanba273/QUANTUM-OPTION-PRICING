import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_yfinance_options(ticker='AAPL'):
    """
    Test function to check if yfinance is returning options data correctly.
    """
    print(f"\n===== Testing yfinance options data for {ticker} =====")
    
    try:
        # Fetch stock data
        print(f"Fetching stock data for {ticker}...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            print(f"No stock data found for {ticker}")
            return
            
        print(f"Downloaded {len(stock_data)} days of stock data")
        
        # Get the latest closing price, ensuring it's a scalar
        close_value = stock_data['Close'].iloc[-1]
        if isinstance(close_value, pd.Series):
            close_value = close_value.iloc[0]  # Get first value if it's a Series
        print(f"Latest closing price: ${float(close_value):.2f}")
        
        # Create ticker object and get options
        stock = yf.Ticker(ticker)
        
        # Get available options expiration dates
        expiry_dates = stock.options
        print(f"Available expiry dates: {expiry_dates}")
        
        if not expiry_dates or len(expiry_dates) == 0:
            print(f"No option expiry dates found for {ticker}")
            return
            
        # Get the first expiration date
        first_expiry = expiry_dates[0]
        print(f"\nExamining option chain for expiry date: {first_expiry}")
        
        # Get option chain
        opt_chain = stock.option_chain(first_expiry)
        
        # Check the type of the returned object
        print(f"Option chain type: {type(opt_chain)}")
        
        # Check if the calls and puts properties exist
        if hasattr(opt_chain, 'calls'):
            print("\nCALLS DATA:")
            calls = opt_chain.calls
            print(f"Type: {type(calls)}")
            print(f"Shape: {calls.shape}")
            print(f"Columns: {calls.columns.tolist()}")
            
            if not calls.empty:
                # Display first few calls
                print("\nFirst 3 calls:")
                print(calls.head(3))
                
                # Check close-to-ATM options
                print(f"\nOptions near current price (${float(close_value):.2f}):")
                # Get absolute difference between strike and current price
                strikes_diff = abs(calls['strike'] - float(close_value))
                # Find indices of the smallest differences
                closest_idx = strikes_diff.nsmallest(3).index
                # Get those rows
                near_atm = calls.loc[closest_idx]
                print(near_atm)
            else:
                print("Calls dataframe is empty")
        else:
            print("No 'calls' attribute found in option chain")
        
        # Check puts
        if hasattr(opt_chain, 'puts'):
            print("\nPUTS DATA:")
            puts = opt_chain.puts
            print(f"Type: {type(puts)}")
            print(f"Shape: {puts.shape}")
            print(f"Columns: {puts.columns.tolist()}")
            
            if not puts.empty:
                # Display first few puts
                print("\nFirst 3 puts:")
                print(puts.head(3))
            else:
                print("Puts dataframe is empty")
        else:
            print("No 'puts' attribute found in option chain")
            
    except Exception as e:
        print(f"Error testing options data for {ticker}: {e}")
        import traceback
        traceback.print_exc()

# List of tickers to test
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Run the test for each ticker
if __name__ == "__main__":
    print("YFINANCE OPTIONS DATA TEST")
    print("=========================")
    
    for ticker in tickers:
        test_yfinance_options(ticker)