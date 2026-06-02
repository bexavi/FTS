"""
Topic 2: Inflation and Macroeconomic Data Download
DFA vs. LSTM for Inflation Prediction

This script downloads U.S. macroeconomic data for inflation analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Using FRED API (Recommended for macroeconomic data)
# ============================================================================

def download_inflation_data_fred(api_key=None):
    """
    Download U.S. macroeconomic indicators from FRED.
    Requires FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
    
    Returns:
        DataFrame with monthly macroeconomic data
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("Installing fredapi: pip install fredapi")
        print("Get API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    if api_key is None:
        print("FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\nTo use this script:")
        print("  1. Sign up at FRED: https://fred.stlouisfed.org/")
        print("  2. Get API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  3. Install: pip install fredapi")
        print("  4. Call: download_inflation_data_fred(api_key='YOUR_KEY')")
        return None
    
    print("Downloading macroeconomic data from FRED...")
    fred = Fred(api_key=api_key)
    
    # Define FRED series IDs for key macroeconomic indicators
    series_info = {
        'CPI': {
            'id': 'CPIAUCSL',
            'name': 'Consumer Price Index (All Urban Consumers)',
            'transform': 'level'  # Use level or calculate inflation rate
        },
        'Core_CPI': {
            'id': 'CPILFESL',
            'name': 'Core CPI (excluding food and energy)',
            'transform': 'level'
        },
        'Unemployment': {
            'id': 'UNRATE',
            'name': 'Unemployment Rate',
            'transform': 'level'
        },
        'Fed_Funds_Rate': {
            'id': 'FEDFUNDS',
            'name': 'Federal Funds Effective Rate',
            'transform': 'level'
        },
        'GDP': {
            'id': 'GDPC1',
            'name': 'Real Gross Domestic Product',
            'transform': 'level'
        },
        'Money_Supply_M2': {
            'id': 'M2SL',
            'name': 'M2 Money Stock',
            'transform': 'level'
        },
        'Oil_Price': {
            'id': 'DCOILWTICO',
            'name': 'Crude Oil Prices: West Texas Intermediate',
            'transform': 'level'
        },
        'Industrial_Production': {
            'id': 'INDPRO',
            'name': 'Industrial Production Index',
            'transform': 'level'
        },
    }
    
    data_dict = {}
    start_date = '1990-01-01'  # Ensures ≥150 monthly observations
    
    print(f"\nDownloading data from {start_date} to present...")
    
    for var_name, info in series_info.items():
        try:
            print(f"Downloading {var_name} ({info['id']})...")
            series = fred.get_series(info['id'], start=start_date)
            
            if not series.empty:
                data_dict[var_name] = series
                print(f"  ✓ {len(series)} observations")
            else:
                print(f"  ✗ No data available")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if data_dict:
        # Combine into DataFrame
        data = pd.DataFrame(data_dict)
        
        # Calculate inflation rate (YoY % change in CPI)
        if 'CPI' in data.columns:
            data['Inflation_Rate'] = data['CPI'].pct_change(12) * 100  # 12-month % change
        
        # Calculate GDP growth rate (if quarterly, convert to monthly or use different approach)
        if 'GDP' in data.columns:
            # GDP is quarterly, so we might need to handle this differently
            # Option: Use monthly GDP proxy or interpolate
            pass
        
        # Remove missing values
        data = data.dropna()
        
        print(f"\n✓ Successfully downloaded data:")
        print(f"  Shape: {data.shape}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Variables: {list(data.columns)}")
        
        return data
    else:
        print("\n✗ Failed to download any data")
        return None


# ============================================================================
# Alternative: Download pre-processed data from CSV (if available)
# ============================================================================

def load_inflation_data_from_csv(filename='inflation_data.csv'):
    """Load data from CSV file if already downloaded."""
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} observations from {filename}")
        return data
    except FileNotFoundError:
        print(f"✗ File {filename} not found")
        return None


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def calculate_inflation_rate(cpi_series, method='yoy'):
    """
    Calculate inflation rate from CPI.
    
    Parameters:
        cpi_series: Series of CPI values
        method: 'yoy' (year-over-year) or 'mom' (month-over-month)
    
    Returns:
        Inflation rate as percentage
    """
    if method == 'yoy':
        return cpi_series.pct_change(12) * 100
    elif method == 'mom':
        return cpi_series.pct_change(1) * 100
    else:
        raise ValueError("Method must be 'yoy' or 'mom'")


def prepare_data_for_analysis(data, target_var='Inflation_Rate'):
    """
    Prepare data for time series analysis.
    
    Parameters:
        data: DataFrame with macroeconomic variables
        target_var: Name of target variable (inflation rate)
    
    Returns:
        Cleaned DataFrame ready for analysis
    """
    # Make sure target variable exists
    if target_var not in data.columns:
        if 'CPI' in data.columns:
            data[target_var] = calculate_inflation_rate(data['CPI'])
        else:
            raise ValueError(f"Target variable {target_var} not found and CPI not available")
    
    # Remove rows with missing target
    data = data.dropna(subset=[target_var])
    
    # Optionally: Remove outliers (e.g., beyond 3 standard deviations)
    # This is optional and depends on your analysis
    
    return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOPIC 2: Inflation and Macroeconomic Data Download")
    print("=" * 60)
    
    # Check if API key is provided
    import sys
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        data = download_inflation_data_fred(api_key=api_key)
    else:
        print("\nUsage:")
        print("  python topic2_data_download.py YOUR_FRED_API_KEY")
        print("\nOr set API key in script and uncomment the line below:")
        # data = download_inflation_data_fred(api_key='YOUR_API_KEY_HERE')
        data = None
    
    if data is not None and len(data) >= 150:
        # Prepare data
        data = prepare_data_for_analysis(data)
        
        # Save to CSV
        filename = 'inflation_data.csv'
        data.to_csv(filename)
        print(f"\n✓ Data saved to {filename}")
        
        # Display summary
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
        
        # Check data frequency
        print(f"\nData frequency: {pd.infer_freq(data.index)}")
        print(f"Total observations: {len(data)}")
        
    else:
        print("\n" + "=" * 60)
        print("SETUP INSTRUCTIONS")
        print("=" * 60)
        print("1. Get FRED API key (free):")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\n2. Install required package:")
        print("   pip install fredapi pandas numpy")
        print("\n3. Run script with API key:")
        print("   python topic2_data_download.py YOUR_API_KEY")
        print("\nAlternative: Set API key in script and run directly")

