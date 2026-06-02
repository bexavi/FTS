"""
Topic 1: Step 1 - Exploratory Data Analysis
VECM vs. Transformers for Exchange Rate Prediction

This script performs exploratory data analysis including:
- Data loading and cleaning
- Visualizations
- Summary statistics
- Stationarity tests (ADF, KPSS)
- Cointegration tests (Johansen)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filename='exchange_rates_data.csv'):
    """
    Load exchange rate data from CSV file.
    If file doesn't exist, prompts user to run data download script first.
    """
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} observations from {filename}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Variables: {list(data.columns)}")
        return data
    except FileNotFoundError:
        print(f"✗ File {filename} not found!")
        print("\nPlease run topic1_data_download.py first to download data.")
        print("  Command: python topic1_data_download.py")
        return None

# ============================================================================
# 2. DATA SUMMARY STATISTICS
# ============================================================================

def summary_statistics(data):
    """Calculate and display summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    missing = data.isnull().sum()
    print(missing)
    
    if missing.sum() > 0:
        print(f"\n⚠ Warning: {missing.sum()} missing values found")
        print("  Consider handling missing values before analysis")
    
    return data.describe()

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

def plot_time_series(data, save_plots=True):
    """Plot time series for each exchange rate."""
    n_vars = len(data.columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, col in enumerate(data.columns):
        axes[i].plot(data.index, data[col], linewidth=1.5)
        axes[i].set_title(f'{col} Exchange Rate Over Time', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Exchange Rate')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('topic1_time_series_plots.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved time series plots to topic1_time_series_plots.png")
    plt.show()

def plot_returns(data, save_plots=True):
    """Plot returns (first differences of log prices)."""
    # Calculate log returns
    log_returns = np.log(data / data.shift(1)).dropna()
    
    n_vars = len(log_returns.columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, col in enumerate(log_returns.columns):
        axes[i].plot(log_returns.index, log_returns[col], linewidth=0.8, alpha=0.7)
        axes[i].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[i].set_title(f'{col} Log Returns', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Log Return')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('topic1_returns_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Saved returns plots to topic1_returns_plots.png")
    plt.show()
    
    return log_returns

def plot_correlation_matrix(data, save_plots=True):
    """Plot correlation matrix."""
    correlation = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Exchange Rate Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('topic1_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved correlation matrix to topic1_correlation_matrix.png")
    plt.show()
    
    return correlation

# ============================================================================
# 4. STATIONARITY TESTS
# ============================================================================

def adf_test(series, title="Series"):
    """
    Augmented Dickey-Fuller test for stationarity.
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    """
    result = adfuller(series.dropna())
    
    print(f"\n{'='*60}")
    print(f"ADF Test for {title}")
    print(f"{'='*60}")
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.6f}")
    
    if result[1] <= 0.05:
        print(f"✓ Result: Reject H0 - Series is STATIONARY (p-value = {result[1]:.4f})")
        return True
    else:
        print(f"✗ Result: Fail to reject H0 - Series is NON-STATIONARY (p-value = {result[1]:.4f})")
        return False

def kpss_test(series, title="Series"):
    """
    KPSS test for stationarity.
    H0: Series is stationary
    H1: Series has unit root (non-stationary)
    """
    try:
        result = kpss(series.dropna(), regression='ct')  # 'ct' for trend-stationary
    except:
        result = kpss(series.dropna(), regression='c')  # 'c' for level-stationary
    
    print(f"\n{'='*60}")
    print(f"KPSS Test for {title}")
    print(f"{'='*60}")
    print(f"KPSS Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[3].items():
        print(f"  {key}: {value:.6f}")
    
    if result[1] >= 0.05:
        print(f"✓ Result: Fail to reject H0 - Series is STATIONARY (p-value = {result[1]:.4f})")
        return True
    else:
        print(f"✗ Result: Reject H0 - Series is NON-STATIONARY (p-value = {result[1]:.4f})")
        return False

def test_stationarity(data):
    """
    Test stationarity for all series using ADF and KPSS tests.
    Returns dictionary with results.
    """
    print("\n" + "="*60)
    print("STATIONARITY TESTS")
    print("="*60)
    print("\nNote: Following methodology from W3 Linear Time Series Models (pages 13-14)")
    print("  ADF Test: H0 = unit root (non-stationary)")
    print("  KPSS Test: H0 = stationary")
    print("  Use both tests for complementary evidence\n")
    
    results = {}
    
    for col in data.columns:
        print(f"\n{'#'*60}")
        print(f"Testing: {col}")
        print(f"{'#'*60}")
        
        # ADF Test
        adf_stationary = adf_test(data[col], title=col)
        
        # KPSS Test
        kpss_stationary = kpss_test(data[col], title=col)
        
        # Store results
        results[col] = {
            'adf_stationary': adf_stationary,
            'kpss_stationary': kpss_stationary,
            'overall': adf_stationary and kpss_stationary  # Both must agree
        }
    
    # Summary
    print("\n" + "="*60)
    print("STATIONARITY TEST SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(results).T
    print(summary_df)
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    all_stationary = all([r['overall'] for r in results.values()])
    all_non_stationary = all([not r['overall'] for r in results.values()])
    
    if all_stationary:
        print("✓ All series are STATIONARY")
        print("  → Can use VAR model directly")
    elif all_non_stationary:
        print("✗ All series are NON-STATIONARY")
        print("  → Need to test for COINTEGRATION")
        print("  → If cointegrated: Use VECM")
        print("  → If not cointegrated: Use VAR on differenced data")
    else:
        print("⚠ Mixed results: Some series stationary, some non-stationary")
        print("  → Check individual series and consider differencing non-stationary ones")
    
    return results

# ============================================================================
# 5. COINTEGRATION TESTS
# ============================================================================

def johansen_cointegration_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration test.
    Following methodology from Multivariate Time Series Models (pages 19-20)
    
    Parameters:
        det_order: 0 = no constant, 1 = constant, -1 = constant and trend
        k_ar_diff: Lag order for VAR in differences
    """
    print("\n" + "="*60)
    print("JOHANSEN COINTEGRATION TEST")
    print("="*60)
    print("\nFollowing methodology from Multivariate Time Series Models (pages 19-20)")
    print("  Tests for long-run equilibrium relationships between non-stationary series")
    print(f"  Lag order: {k_ar_diff}")
    print(f"  Deterministic term: {det_order}\n")
    
    try:
        # Remove any missing values
        data_clean = data.dropna()
        
        # Run Johansen test
        result = coint_johansen(data_clean.values, det_order=det_order, k_ar_diff=k_ar_diff)
        
        n_vars = data_clean.shape[1]
        
        print(f"Number of variables: {n_vars}")
        print(f"Sample size: {len(data_clean)}")
        print(f"\nEigenvalues: {result.eig}")
        print(f"Trace statistics: {result.lr1}")
        print(f"Max eigenvalue statistics: {result.lr2}")
        
        # Critical values (5% significance)
        print(f"\nCritical Values (5% significance):")
        print(f"Trace test: {result.cvt[:, 1]}")  # Second column is 5%
        print(f"Max eigenvalue test: {result.cvm[:, 1]}")
        
        # Test results
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        
        cointegration_rank = None
        
        # Trace test
        print("\nTrace Test:")
        for i in range(n_vars):
            trace_stat = result.lr1[i]
            trace_cv = result.cvt[i, 1]  # 5% critical value
            reject = trace_stat > trace_cv
            
            if reject:
                print(f"  r <= {i}: Trace = {trace_stat:.4f} > CV = {trace_cv:.4f} → Reject H0")
                cointegration_rank = i + 1
            else:
                print(f"  r <= {i}: Trace = {trace_stat:.4f} < CV = {trace_cv:.4f} → Fail to reject H0")
                break
        
        # Max eigenvalue test
        print("\nMax Eigenvalue Test:")
        max_rank = None
        for i in range(n_vars):
            max_stat = result.lr2[i]
            max_cv = result.cvm[i, 1]  # 5% critical value
            reject = max_stat > max_cv
            
            if reject:
                print(f"  r = {i}: Max = {max_stat:.4f} > CV = {max_cv:.4f} → Reject H0")
                max_rank = i + 1
            else:
                print(f"  r = {i}: Max = {max_stat:.4f} < CV = {max_cv:.4f} → Fail to reject H0")
                if max_rank is None:
                    max_rank = i
                break
        
        # Final interpretation
        print(f"\n{'='*60}")
        print("INTERPRETATION")
        print(f"{'='='*60}")
        
        if cointegration_rank is None or cointegration_rank == 0:
            print("✗ No cointegration found")
            print("  → Series are not cointegrated")
            print("  → Use VAR on differenced data (not VECM)")
            print("  → Or consider differencing and using VAR")
        else:
            print(f"✓ Cointegration found!")
            print(f"  → Number of cointegrating relationships: {cointegration_rank}")
            print(f"  → VECM is the appropriate model")
            print(f"  → VECM will model both:")
            print(f"     - Long-run equilibrium relationships")
            print(f"     - Short-run adjustment dynamics")
        
        return {
            'cointegration_rank': cointegration_rank,
            'trace_statistics': result.lr1,
            'max_statistics': result.lr2,
            'eigenvalues': result.eig,
            'critical_values_trace': result.cvt,
            'critical_values_max': result.cvm,
            'result_object': result
        }
        
    except Exception as e:
        print(f"✗ Error running Johansen test: {e}")
        print("  Make sure data has no missing values and is non-stationary")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TOPIC 1: EXPLORATORY DATA ANALYSIS")
    print("Exchange Rate Prediction: VECM vs. Transformers")
    print("="*60)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    data = load_data()
    
    if data is None:
        print("\nPlease download data first by running:")
        print("  python topic1_data_download.py")
        exit(1)
    
    # Check data requirements
    if len(data) < 150:
        print(f"\n⚠ Warning: Only {len(data)} observations. Assignment requires ≥150.")
        print("  Consider downloading more historical data.")
    
    if len(data.columns) < 2:
        print(f"\n⚠ Warning: Only {len(data.columns)} variable(s). Assignment requires ≥2.")
        print("  Consider adding more currency pairs.")
    
    # Step 2: Summary statistics
    print("\n[STEP 2] Calculating summary statistics...")
    summary_stats = summary_statistics(data)
    
    # Step 3: Visualizations
    print("\n[STEP 3] Creating visualizations...")
    plot_time_series(data)
    log_returns = plot_returns(data)
    correlation = plot_correlation_matrix(data)
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    print(correlation)
    
    # Step 4: Stationarity tests
    print("\n[STEP 4] Testing for stationarity...")
    stationarity_results = test_stationarity(data)
    
    # Step 5: Cointegration test (only if series are non-stationary)
    all_non_stationary = all([not r['overall'] for r in stationarity_results.values()])
    
    if all_non_stationary:
        print("\n[STEP 5] Testing for cointegration...")
        print("  (Series are non-stationary, so cointegration test is appropriate)")
        cointegration_results = johansen_cointegration_test(data)
    else:
        print("\n[STEP 5] Skipping cointegration test...")
        print("  (Some series are stationary, so cointegration test not applicable)")
        print("  → Consider using VAR model directly")
        cointegration_results = None
    
    # Final summary
    print("\n" + "="*60)
    print("EDA SUMMARY")
    print("="*60)
    print(f"Dataset: {len(data)} observations, {len(data.columns)} variables")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"\nNext steps:")
    
    if cointegration_results and cointegration_results.get('cointegration_rank', 0) > 0:
        print("  ✓ Use VECM model (cointegration found)")
        print("  ✓ Implement Transformer model for comparison")
    elif all_non_stationary:
        print("  → Use VAR on differenced data (no cointegration)")
        print("  → Or reconsider: Are series truly non-stationary?")
    else:
        print("  → Use VAR model (series are stationary)")
        print("  → Or difference non-stationary series first")
    
    print("\n✓ EDA complete! Results saved to plots.")
    print("  Proceed to model implementation.")

