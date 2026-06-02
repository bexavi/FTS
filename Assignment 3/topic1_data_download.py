"""
Topic 1: Exchange Rate Data Download
VECM vs. Transformers for Exchange Rate Prediction

This script downloads exchange rate data for analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTION 1: Using Yahoo Finance (Easiest - No API key needed)
# ============================================================================

def download_exchange_rates_yfinance():
    """
    Download major currency pairs using yfinance.
    Returns daily exchange rates.
    """
    print("Downloading exchange rate data from Yahoo Finance...")
    
    # Define currency pairs
    # Note: Yahoo Finance uses format like 'EURUSD=X' for EUR/USD
    tickers = {
        'EURUSD': 'EURUSD=X',  # Euro/USD
        'GBPUSD': 'GBPUSD=X',  # British Pound/USD
        'USDJPY': 'JPY=X',     # USD/Japanese Yen
        # Add more if needed:
        # 'USDCHF': 'CHF=X',   # USD/Swiss Franc
        # 'AUDUSD': 'AUDUSD=X', # Australian Dollar/USD
    }
    
    # Set date range (ensure ≥150 observations)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of daily data
    
    # Download data
    data_dict = {}
    for name, ticker in tickers.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not df.empty:
                # Use 'Close' price
                data_dict[name] = df['Close'].rename(name)
                print(f"  ✓ Downloaded {len(df)} observations for {name}")
            else:
                print(f"  ✗ No data for {ticker}")
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")
    
    # Combine into single DataFrame
    if data_dict:
        data = pd.DataFrame(data_dict)
        data = data.dropna()  # Remove missing values
        
        print(f"\n✓ Successfully downloaded data:")
        print(f"  Shape: {data.shape}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Variables: {list(data.columns)}")
        
        return data
    else:
        print("\n✗ Failed to download any data")
        return None


# ============================================================================
# OPTION 2: Using FRED API (More reliable, requires API key)
# ============================================================================

def download_exchange_rates_fred(api_key=None):
    """
    Download exchange rates from FRED (Federal Reserve Economic Data).
    Requires FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("Installing fredapi: pip install fredapi")
        return None
    
    if api_key is None:
        print("FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    print("Downloading exchange rate data from FRED...")
    
    fred = Fred(api_key=api_key)
    
    # FRED series IDs for exchange rates
    series_ids = {
        'EURUSD': 'DEXUSEU',  # U.S. / Euro Foreign Exchange Rate
        'GBPUSD': 'DEXUSUK',  # U.S. / U.K. Foreign Exchange Rate
        'USDJPY': 'DEXJPUS',  # Japan / U.S. Foreign Exchange Rate
    }
    
    data_dict = {}
    for name, series_id in series_ids.items():
        try:
            print(f"Downloading {name} ({series_id})...")
            series = fred.get_series(series_id, start='2010-01-01')
            if not series.empty:
                data_dict[name] = series.rename(name)
                print(f"  ✓ Downloaded {len(series)} observations")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if data_dict:
        data = pd.DataFrame(data_dict)
        data = data.dropna()
        print(f"\n✓ Successfully downloaded {len(data)} observations")
        return data
    else:
        return None


# ============================================================================
# OPTION 3: Using OANDA API (Professional forex data)
# ============================================================================

def download_exchange_rates_oanda(api_key=None):
    """
    Download from OANDA (requires API key).
    More professional forex data but requires registration.
    """
    # Implementation would require oandapyV20 or similar
    # This is optional - yfinance is sufficient for this assignment
    pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOPIC 1: Exchange Rate Data Download")
    print("=" * 60)
    
    # Try yfinance first (easiest)
    data = download_exchange_rates_yfinance()
    
    if data is not None and len(data) >= 150:
        # Save to CSV
        filename = 'exchange_rates_data.csv'
        data.to_csv(filename)
        print(f"\n✓ Data saved to {filename}")
        
        # Display summary statistics
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(data.describe())
        print("\nFirst few rows:")
        print(data.head())
        print("\nLast few rows:")
        print(data.tail())
        
        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum())
        
    else:
        print("\n✗ Insufficient data downloaded. Try FRED API instead.")
        print("  To use FRED:")
        print("  1. Get API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. Install: pip install fredapi")
        print("  3. Call: download_exchange_rates_fred(api_key='YOUR_KEY')")

