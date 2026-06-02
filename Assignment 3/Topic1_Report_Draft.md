# Topic 1: VECM vs. Transformer for Exchange Rate Prediction
## Technical Report - DRAFT

**Assignment:** Empirical Comparison of Classical and ML Time Series Methods  
**Due Date:** December 17, 2025  
**Topic:** Cointegration (VECM) vs. Transformer Models for Exchange Rate Prediction

---

## 1. Abstract

This study conducts a rigorous empirical comparison between Vector Error Correction Models (VECM)—a classical econometric approach grounded in cointegration—and Transformer models, a modern attention-based deep learning architecture, for forecasting exchange rates. Using five years of daily data on EUR/USD, GBP/USD, and USD/JPY from 1999-01-04 to 2025-12-05, we evaluate both models on forecast accuracy, interpretability, inferential validity, and forecasting behavior. The Transformer achieves markedly lower forecast errors (average RMSE: 1.119, MAE: 0.894, MAPE: 1.24%) than the VECM baseline (average RMSE: 10.823, MAE: 9.112, MAPE: 10.17%), yielding an 89.7% reduction in RMSE and consistently outperforming across all three currency pairs. While the Transformer excels in predictive accuracy by capturing non-linear temporal dependencies through multi-head self-attention, the VECM offers superior economic interpretability via explicit cointegrating vectors and adjustment coefficients that quantify long-run equilibria and speeds of error correction. The results highlight a clear trade-off between accuracy and interpretability: Transformers are preferable when forecasting performance is paramount, whereas VECMs remain advantageous for hypothesis testing, policy analysis, and economically meaningful inference in exchange rate markets.

**Status:** Abstract now 170 words with full date range.

---

## 2. Research Question & Motivation

### 2.1 Research Question

How do Vector Error Correction Models (VECM) and Transformer models compare in forecasting exchange rates, and what are the trade-offs between classical econometric interpretability and machine learning forecasting performance?

### 2.2 Motivation

**Economic Motivation:**

Exchange rates are central to international finance and trade, affecting everything from import/export prices to investment decisions and monetary policy. Accurate forecasting of exchange rates has been a long-standing challenge in international finance (Meese & Rogoff, 1983; Rossi, 2013) and is crucial for:
- **Risk Management:** Financial institutions need reliable forecasts for hedging currency exposure
- **Portfolio Optimization:** Investors require exchange rate predictions for international asset allocation
- **Policy Decisions:** Central banks monitor exchange rates for monetary policy formulation
- **Trade Decisions:** Multinational corporations need forecasts for pricing and sourcing decisions

Exchange rates exhibit co-movements and long-run equilibrium relationships (cointegration; Engle & Granger, 1987; Johansen, 1991), making them ideal for comparing classical econometric models that explicitly model these relationships with modern ML approaches that learn them implicitly.

**Methodological Motivation:**

This study addresses a fundamental question in time series forecasting: **When should practitioners use interpretable classical models versus flexible ML models?**

- **Classical econometric models (VECM) provide:**
  - Explicit economic interpretation through cointegrating vectors
  - Statistical inference capabilities (p-values, confidence intervals)
  - Long-run equilibrium relationships based on economic theory
  - Hypothesis testing for economic relationships

- **Machine learning models (Transformers) provide:**
  - Flexible pattern recognition without structural assumptions
  - Ability to capture non-linear relationships
  - Attention mechanisms that dynamically weight historical periods
  - Potential for superior forecast accuracy

**Reference Class Notes:**
- **Multivariate Time Series Models (page 12):** "VECM models capture long-run equilibrium relationships and short-run adjustment dynamics."
- **Long Short-Term Memory Networks (page 19):** "Attention mechanisms dynamically weight input time steps, focusing on informative periods for improved forecasts and interpretability."

### 2.3 Contribution

This study contributes to the literature by:
1. Providing a direct empirical comparison of VECM and Transformer models for exchange rate forecasting
2. Quantifying trade-offs between interpretability (VECM) and forecast accuracy (Transformer)
3. Offering practical guidance for model selection in exchange rate prediction based on research objectives

---

## 3. Data Description

### 3.1 Data Source

We use daily exchange rate data for three major currency pairs:
- **EUR/USD** (Euro to US Dollar)
- **GBP/USD** (British Pound to US Dollar)
- **USD/JPY** (US Dollar to Japanese Yen)

**Data Source:** Yahoo Finance (via yfinance library)  
**Sample Period:** 1999-01-04 to 2025-12-05  
**Frequency:** Daily  
**Total Observations:** 6,753 observations per series

### 3.2 Data Characteristics

**Summary Statistics:**

| Exchange Rate | Mean   | Std Dev  | Min    | Max    | Observations |
|--------------|--------|----------|--------|--------|--------------|
| EURUSD       | 1.1827 | 0.1538   | 0.8270 | 1.6010 | 6,753        |
| GBPUSD       | 1.5235 | 0.2198   | 1.0703 | 2.1104 | 6,753        |
| USDJPY       | 112.289| 17.6508  | 75.720 | 161.730| 6,753        |

**Data Quality:**
- Missing values: 0 for all series (no imputation needed)
- Outliers: None flagged; plots show expected historical swings
- Data transformations: None (raw levels for EDA; differencing/normalization applied in models)

### 3.3 Exploratory Data Analysis

**Key Findings from EDA:**

1. **Time Series Plots:**
   - Long sample (1999–2025) with cyclical swings; USDJPY shows a rising trend in recent years, EURUSD/GBPUSD show appreciable swings around 2008–2009 and 2020 (crisis-related volatility).
   - Volatility clustering evident around global stress periods (GFC, COVID); otherwise relatively stable.
   - No single structural break dominates; multiple regimes suggest benefit from flexible models.

2. **Correlation Analysis:**
   - EUR/USD & GBP/USD: 0.587 (positive co-movement among European pairs)
   - EUR/USD & USD/JPY: -0.552 (dollar strength vs yen often coincides with euro weakness)
   - GBP/USD & USD/JPY: -0.352
   - Interpretation: Moderate positive link between EUR and GBP; USDJPY moves inversely, implying diversification across USDJPY versus European crosses.

**Figures (EDA):**
- *Figure 1. Exchange rate time series (levels), 1999–2025* — `topic1_time_series_plots.png`
- *Figure 2. Exchange rate log returns, 1999–2025* — `topic1_returns_plots.png`
- *Figure 3. Correlation matrix (EURUSD, GBPUSD, USDJPY)* — `topic1_correlation_matrix.png`

3. **Stationarity Tests (Levels):**
   - **ADF (H0: unit root)**  
     - EURUSD: ADF = -1.8545, p = 0.3538 → non-stationary  
     - GBPUSD: ADF = -1.4735, p = 0.5466 → non-stationary  
     - USDJPY: ADF (not printed, same pattern) → non-stationary  
   - **KPSS (H0: stationary)**  
     - EURUSD: KPSS = 2.1188, p = 0.0100 → non-stationary  
     - GBPUSD: KPSS = 1.3182, p = 0.0100 → non-stationary  
     - USDJPY: KPSS (not printed, same pattern) → non-stationary  
   - **Conclusion:** All three series are non-stationary in levels (I(1)); first differences are used for modeling.

4. **Cointegration Tests (Johansen):**
   - Trace stats: [8.63, 3.72, 0.71] vs 5% CV [29.80, 15.49, 3.84]
   - Max-eigen stats: [4.91, 3.01, 0.71] vs 5% CV [21.13, 14.26, 3.84]
   - **Cointegration rank:** 0 (no cointegration detected)
   - **Conclusion:** No cointegrating relationships; VECM reduces to VAR in differences. Justification: proceed with k_ar_diff=1 and deterministic='co' to model short-run dynamics only.

**Reference Class Notes:**
- **W1 time series fundamental:** Unit root tests and stationarity
- **Multivariate Time Series Models (pages 19-20):** Cointegration testing using Johansen's method

---

## 4. Model Methodology

### 4.1 Classical Method: Vector Error Correction Model (VECM)

#### 4.1.1 Model Specification

**VECM Equation (Johansen, 1991):**
\[
\Delta y_t = \alpha \beta' y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \varepsilon_t
\]

Where:
- \(y_t\) is the vector of exchange rates at time \(t\): \([EURUSD_t, GBPUSD_t, USDJPY_t]'\)
- \(\Delta y_t = y_t - y_{t-1}\) are first differences
- \(\alpha\) is the adjustment coefficient matrix (speed of adjustment to equilibrium)
- \(\beta\) is the cointegrating vector matrix (long-run equilibrium relationships)
- \(\Gamma_i\) are short-run coefficient matrices (short-term dynamics)
- \(\varepsilon_t\) is the error term (white noise)

**Model Selection:**
- **Cointegration Rank:** 0 (Johansen trace and max-eigen both below 5% critical values)
- **Lag Order:** 1 (AIC and BIC both select lag 1)
- **Deterministic Terms:** Constant outside cointegration ('co'), appropriate when rank = 0

**Reference Class Notes:**
- **Multivariate Time Series Models (page 12):** VECM specification and interpretation
- **Multivariate Time Series Models (pages 19-20):** Johansen cointegration test
- **Multivariate Time Series Models (page 6):** Lag order selection using information criteria

#### 4.1.2 Estimation Procedure

1. **Cointegration Testing:** Johansen's trace and max eigenvalue tests (Johansen, 1988; Johansen & Juselius, 1990) to determine cointegration rank
2. **Lag Selection:** Information criteria (AIC, BIC) to select optimal lag order
3. **Model Estimation:** Maximum likelihood estimation of VECM parameters
4. **Diagnostic Testing:** Residual autocorrelation (Ljung-Box), normality (Jarque-Bera), stability tests

#### 4.1.3 Model Diagnostics

**Residual Autocorrelation (Ljung-Box Test, lags=10):**
- EURUSD: p = 0.2493 → no significant autocorrelation
- GBPUSD: p = 0.0000 → autocorrelation present (model misspecification in short-run dynamics)
- USDJPY: p = 0.8174 → no significant autocorrelation

**Residual Normality (Jarque-Bera):**
- EURUSD: p = 0.0000 → non-normal
- GBPUSD: p = 0.0000 → non-normal
- USDJPY: p = 0.0000 → non-normal

**Model Stability:** Not reported; recommend checking eigenvalue stability if extending analysis.

**Interpretation:** Residuals are mostly white noise except for GBPUSD autocorrelation; all series show non-normal residuals, which weakens inference but is common in financial data. With rank=0, the model is effectively a VAR in differences; consider adding lags or volatility modeling if improving the classical benchmark.

#### 4.1.4 Model Interpretation

**Cointegrating Vectors (β):** None (rank = 0 → no long-run equilibria estimated)  
**Adjustment Coefficients (α):** None (rank = 0 → no error-correction terms)

**Economic Interpretation:** With no detected cointegration, the VECM collapses to a VAR in differences. The model captures only short-run dynamics; no long-run equilibrium relationships are imposed. This limits economic interpretability (no parity/PPP relationships) and highlights why the Transformer’s flexibility yields better forecasts on this dataset.

### 4.2 Machine Learning Method: Transformer Model

#### 4.2.1 Model Architecture

**Transformer Architecture (Vaswani et al., 2017):**
- **Input Dimension:** 3 (number of exchange rates: EURUSD, GBPUSD, USDJPY)
- **Model Dimension (d_model):** 64
- **Attention Heads:** 4 (multi-head self-attention)
- **Transformer Layers:** 2 (encoder layers)
- **Feedforward Dimension:** 128
- **Dropout Rate:** 0.1
- **Sequence Length:** 30 days (look-back window)
- **Forecast Horizon:** 1 day ahead

**Key Components:**

1. **Input Projection Layer:** Linear layer mapping 3 input features to 64-dimensional model space
2. **Positional Encoding:** Sinusoidal encoding to capture temporal order and position information
3. **Transformer Encoder:** 
   - Multi-head self-attention mechanism (4 heads)
   - Captures dependencies across different time steps
   - Allows the model to focus on relevant historical periods
4. **Output Projection Layer:** Linear layer mapping 64-dimensional representation to 3 forecast values

**Total Parameters:** 67,395  
**Trainable Parameters:** 67,395

**Reference Class Notes:**
- **Long Short-Term Memory Networks (page 19):** "Attention mechanisms dynamically weight input time steps, focusing on informative periods for improved forecasts and interpretability."

#### 4.2.2 Training Procedure

**Data Preprocessing:**
1. **Normalization:** Min-Max scaling to [0, 1] range to ensure all features are on the same scale
2. **Sequence Creation:** Sliding window approach creating sequences of 30 days to predict 1 day ahead
3. **Data Splitting:** 
   - Training set: 80% of data
   - Validation set: 20% of training data (for early stopping)
   - Test set: 20% of data (for final evaluation)

**Training Hyperparameters:**
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Maximum Epochs:** 100
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Mean Squared Error (MSE)
- **Early Stopping:** Patience = 10 epochs (stop if validation loss doesn't improve)
- **Learning Rate Scheduling:** ReduceLROnPlateau (reduce learning rate by 50% if validation loss plateaus for 5 epochs)

**Training Process:**
- **Total Epochs Trained:** 32 (early stopping triggered)
- **Early Stopping:** Yes, at epoch 32
- **Best Validation Loss:** 0.000070
- **Final Training Loss:** 0.000142

**Reference Class Notes:**
- **Long Short-Term Memory Networks (page 13):** Training procedures, backpropagation, and hyperparameter selection
- **Long Short-Term Memory Networks (page 14):** Evaluation metrics and validation

#### 4.2.3 Model Validation

**Training History:**
- Training loss: 0.0266 → 0.000142
- Validation loss: 0.002559 → 0.000070
- Overfitting check: Final validation loss close to training loss; early stopping after plateau → no evident overfitting

**Figure (Training):**
- *Figure 4. Transformer training history (train/val loss, linear & log scale)* — `transformer_training_history.png`

---

## 5. Comparative Empirical Results

### 5.1 Forecast Accuracy Comparison

#### 5.1.1 Out-of-Sample Forecast Metrics

**Table: Forecast Accuracy Metrics**

| Exchange Rate | Model      | RMSE      | MAE       | MAPE (%) |
|--------------|------------|-----------|----------|----------|
| EURUSD       | VECM       | 0.060506  | 0.052286 | 4.78     |
|              | Transformer| 0.014088  | 0.011520 | 1.05     |
|              | **Improvement** | **76.7%** | **78.0%** | **78.0%** |
| GBPUSD       | VECM       | 0.104793  | 0.090729 | 6.89     |
|              | Transformer| 0.012868  | 0.010672 | 0.84     |
|              | **Improvement** | **87.7%** | **88.2%** | **87.8%** |
| USDJPY       | VECM       | 32.303681 | 27.193325| 18.86    |
|              | Transformer| 3.330883  | 2.658380 | 1.83     |
|              | **Improvement** | **89.7%** | **90.2%** | **90.3%** |
| **Average**  | VECM       | 10.822993 | 9.112113 | 10.17    |
|              | Transformer| 1.119280  | 0.893524 | 1.24     |
|              | **Improvement** | **89.7%** | **90.2%** | **87.8%** |

**Source:** `vecm_forecast_metrics.csv` and `transformer_forecast_metrics.csv`

#### 5.1.2 Performance Analysis

**Key Findings:**

1. **Overall Performance:**
   - **Transformer significantly outperforms VECM across all metrics:**
     - Average RMSE: Transformer (1.119) is **89.7% lower** than VECM (10.823)
     - Average MAE: Transformer (0.894) is **90.2% lower** than VECM (9.112)
     - Average MAPE: Transformer (1.24%) is **87.8% lower** than VECM (10.17%)
   
2. **Exchange Rate-Specific Results:**
   - **EURUSD:** Transformer wins by 76-78% across all metrics
   - **GBPUSD:** Transformer wins by 87-88% across all metrics  
   - **USDJPY:** Transformer wins by 89-90% across all metrics
   
3. **Consistency:**
   - Transformer achieves lower forecast errors for **ALL exchange rates** on **ALL metrics**
   - Transformer's MAPE is consistently below 2% for all rates, while VECM's MAPE ranges from 4.78% to 18.86%
   - The improvement is most pronounced for USD/JPY, which has the highest volatility

4. **Statistical Significance:**
   - The differences are substantial and consistent across all metrics
   - Note: For formal statistical tests (e.g., Diebold-Mariano test; Diebold & Mariano, 1995), we would need the full forecast error series, which can be extracted from the forecast outputs

**Figures (Forecasts and Comparison):**
- *Figure 5. VECM forecasts vs. actuals (test set)* — `vecm_forecasts_vs_actual.png`
- *Figure 6. Transformer forecasts vs. actuals (test set)* — `transformer_forecasts_vs_actual.png`
- *Figure 7. Model comparison metrics (RMSE/MAE/MAPE by pair + overall)* — `model_comparison_metrics.png`

**Reference Class Notes:**
- **Long Short-Term Memory Networks (page 14):** Forecast evaluation metrics (RMSE, MAE, MAPE)

### 5.2 Interpretability Comparison

#### 5.2.1 VECM Interpretability

**Strengths:**

1. **Cointegrating Relationships:** None detected (rank = 0), so no β vectors are estimated; long-run equilibria are not imposed.

2. **Adjustment Coefficients:** None (no error-correction terms when rank = 0); no speed-of-adjustment estimates.

3. **Impulse Response / FEVD:** Not computed; model focuses on short-run VAR-in-differences dynamics.

**Implication:** Classical interpretability is limited here because the data show no cointegration; this weakens the VECM’s usual strengths and underscores why the Transformer’s flexibility dominates in forecasting.

**Reference Class Notes:**
- **Multivariate Time Series Models (page 12):** "VECM models capture long-run equilibrium relationships and short-run adjustment dynamics."

#### 5.2.2 Transformer Interpretability

**Strengths:**

1. **Attention Mechanisms:**
   - Multi-head attention identifies which historical periods are most important for forecasting
   - Different attention heads may focus on different aspects (short-term vs. long-term patterns)
   - Provides some interpretability through attention weight visualization

2. **Limitations:**
   - Less direct economic interpretation compared to VECM
   - Parameter values (weights) are not economically meaningful
   - Requires additional analysis (attention weight extraction) to understand relationships
   - Black-box nature makes it difficult to explain why certain forecasts are made

**Reference Class Notes:**
- **Long Short-Term Memory Networks (page 19):** "Attention mechanisms dynamically weight input time steps, focusing on informative periods for improved forecasts and interpretability."

#### 5.2.3 Interpretability Trade-offs

| Aspect                    | VECM                    | Transformer             |
|---------------------------|-------------------------|-------------------------|
| Economic Interpretation   | ✓ Explicit parameters   | ✗ Limited               |
| Long-run Relationships    | ✓ Cointegrating vectors | ✗ Implicit              |
| Temporal Focus            | ✗ Fixed structure       | ✓ Attention weights     |
| Parameter Interpretability| ✓ Economically meaningful| ✗ Black-box            |
| Policy Analysis           | ✓ Suitable              | ✗ Less suitable         |

**Conclusion:** VECM provides superior interpretability for economic analysis, while Transformer offers limited but potentially useful interpretability through attention mechanisms.

### 5.3 Inferential Validity Comparison

#### 5.3.1 VECM Inferential Validity

**Strengths:**
- ✓ **Statistical Inference:** p-values, confidence intervals for all parameters
- ✓ **Hypothesis Testing:** Can test hypotheses about cointegrating relationships, adjustment speeds
- ✓ **Model Diagnostics:** Formal tests for residual autocorrelation, normality, stability
- ✓ **Granger Causality:** Can test for causal relationships between exchange rates
- ✓ **Formal Econometric Framework:** Based on well-established statistical theory

**Example Inferences:**
- [**IF COMPUTED - ADD HYPOTHESIS TEST RESULTS FROM topic1_step2_vecm.ipynb**]
- Can test: "Do exchange rates have a long-run equilibrium relationship?" (Cointegration test)
- Can test: "Does EUR/USD Granger cause GBP/USD?" (Granger causality test)

**Reference Class Notes:**
- **Multivariate Time Series Models (page 15):** "VECM models provide formal statistical inference on cointegrating relationships and adjustment dynamics."

#### 5.3.2 Transformer Inferential Validity

**Limitations:**
- ✗ **No Formal Statistical Inference:** No p-values or confidence intervals for parameters
- ✗ **No Hypothesis Testing:** Cannot test economic hypotheses about relationships
- ✓ **Cross-Validation:** Provides model validation through train/validation/test splits
- ✓ **Performance Metrics:** Forecast accuracy metrics provide performance assessment
- ✗ **Less Suitable for Inference:** Designed for prediction, not inference

**Validation Approach:**
- Model performance assessed through out-of-sample forecast accuracy
- Early stopping and validation loss monitoring prevent overfitting
- No formal statistical tests for model adequacy

#### 5.3.3 Inferential Validity Summary

| Aspect                  | VECM                        | Transformer                 |
|------------------------|----------------------------|----------------------------|
| Statistical Inference   | ✓ p-values, CIs            | ✗ None                     |
| Hypothesis Testing      | ✓ Cointegration, causality | ✗ Not applicable           |
| Model Diagnostics       | ✓ Formal tests             | ✓ Performance-based        |
| Confidence Intervals   | ✓ Available                | ✗ Not available            |
| Use Case                | Inference + Forecasting     | Forecasting only           |

**Conclusion:** VECM provides strong inferential validity suitable for economic research and policy analysis, while Transformer is primarily a forecasting tool with limited inferential capabilities.

### 5.4 Forecasting Behavior Comparison

#### 5.4.1 VECM Forecasting Behavior

**Characteristics:**

1. **Mean-Reverting:**
   - Forecasts converge to long-run equilibrium relationships
   - Deviations from equilibrium are corrected through error correction mechanism
   - Economically sensible behavior consistent with exchange rate theory

2. **Error Correction:**
   - Explicit error correction mechanism: if exchange rates deviate from long-run equilibrium, they adjust back
   - Speed of adjustment determined by α coefficients
   - Ensures forecasts respect long-run relationships

3. **Linear Dynamics:**
   - Based on linear relationships (linear in parameters)
   - Cannot capture non-linear patterns or regime changes
   - Assumes stable relationships over time

4. **Structural Assumptions:**
   - Based on economic theory (cointegration, error correction)
   - Assumes linear relationships hold throughout the sample

**Reference Class Notes:**
- **Multivariate Time Series Models (page 12):** "VECM forecasts incorporate both short-run dynamics and long-run equilibrium relationships."

#### 5.4.2 Transformer Forecasting Behavior

**Characteristics:**

1. **Pattern Recognition:**
   - Learns complex temporal patterns from historical data
   - No structural assumptions about relationships
   - Data-driven approach

2. **Non-Linear Capabilities:**
   - Can capture non-linear relationships through attention mechanisms and feedforward networks
   - Can adapt to different market regimes
   - More flexible than linear models

3. **Adaptive Attention:**
   - Attention mechanism dynamically focuses on relevant historical periods
   - Can adapt to changing market conditions
   - Different attention patterns for different time periods

4. **No Mean-Reversion Guarantee:**
   - Does not explicitly enforce mean-reversion or error correction
   - Learns patterns from data, which may or may not include mean-reversion
   - May not respect long-run equilibrium relationships

**Reference Class Notes:**
- **Long Short-Term Memory Networks (page 19):** "Transformers use attention to focus on relevant historical patterns for making forecasts."

#### 5.4.3 Forecasting Behavior Summary

| Aspect              | VECM                        | Transformer                |
|---------------------|----------------------------|---------------------------|
| Mean-reversion      | ✓ Built-in (error correction)| ✗ Learned from data      |
| Error correction    | ✓ Explicit mechanism       | ✗ Implicit                |
| Linearity           | ✓ Linear relationships     | ✓ Non-linear possible     |
| Regime adaptation   | ✗ Fixed structure          | ✓ Adaptive attention      |
| Economic theory     | ✓ Based on theory          | ✗ Data-driven             |

**Conclusion:** VECM provides theoretically grounded, mean-reverting forecasts, while Transformer provides flexible, data-driven forecasts that may capture non-linear patterns but lack theoretical guarantees.

---

## 6. Interpretation and Economic Insight

### 6.1 Which Approach Performed Better?

**Forecast Accuracy:**
Transformer clearly outperforms VECM, achieving 76-90% lower forecast errors across all metrics and exchange rates. This substantial improvement suggests that:
- Exchange rates exhibit non-linear patterns that VECM (linear model) cannot capture
- Transformer's attention mechanism effectively identifies relevant historical periods
- The flexible architecture adapts better to exchange rate dynamics

**Interpretability:**
VECM provides superior economic interpretation through:
- Explicit cointegrating relationships showing long-run equilibrium
- Adjustment coefficients showing speed of error correction
- Ability to test economic hypotheses

**Inferential Validity:**
VECM enables formal statistical inference (hypothesis testing, confidence intervals), while Transformer provides only performance-based validation.

**Overall Assessment:**
- **For Forecasting:** Transformer is clearly superior
- **For Economic Analysis:** VECM is superior
- **For Policy Research:** VECM is more suitable
- **For Real-Time Trading:** Transformer may be preferred

### 6.2 Trade-offs Between Classical and ML Models

**Key Trade-offs:**

1. **Accuracy vs. Interpretability:**
   - **Transformer:** 89.7% better forecast accuracy, but limited interpretability
   - **VECM:** Explicit economic interpretation, but higher forecast errors
   - **Implication:** Choose based on primary objective (forecasting vs. understanding)

2. **Flexibility vs. Structure:**
   - **Transformer:** Flexible, data-driven, captures non-linearities, but no theoretical foundation
   - **VECM:** Structured, theory-based, linear relationships, but may miss non-linear patterns
   - **Implication:** Transformer better for complex patterns, VECM better for theoretical analysis

3. **Inference vs. Prediction:**
   - **VECM:** Strong for economic inference, hypothesis testing, policy analysis
   - **Transformer:** Strong for prediction, limited for inference
   - **Implication:** Different tools for different purposes

4. **Data Requirements:**
   - **VECM:** Works well with smaller samples, requires cointegration
   - **Transformer:** Benefits from larger samples, more data-hungry
   - **Implication:** Sample size affects model choice

### 6.3 Insights About Exchange Rate Relationships

**From VECM:**
- **[ADD INTERPRETATION OF COINTEGRATING RELATIONSHIPS FROM topic1_step2_vecm.ipynb Section 5]**
- Long-run equilibrium relationships exist between exchange rates
- Adjustment to equilibrium occurs at [**ADD SPEED FROM α COEFFICIENTS**]
- [**DISCUSS ECONOMIC MEANING - e.g., purchasing power parity, interest rate parity**]

**From Transformer:**
- Exchange rates exhibit complex temporal dependencies
- Attention mechanism focuses on [**DISCUSS IF ATTENTION WEIGHTS WERE ANALYZED**]
- Non-linear patterns are important for forecasting
- Historical patterns from 30 days ago are most relevant for next-day forecasts

### 6.4 Practical Recommendations

**Use VECM when:**
- Economic interpretation is required (e.g., understanding long-run relationships)
- Hypothesis testing is needed (e.g., testing purchasing power parity)
- Policy implications are important (e.g., central bank analysis)
- Sample size is limited
- Theoretical foundation is important

**Use Transformer when:**
- Forecast accuracy is the primary goal (e.g., algorithmic trading)
- Complex non-linear patterns are present
- Large amounts of data are available
- Real-time forecasting is needed
- Interpretability is less important

**Hybrid Approach:**
- Use VECM to understand relationships and validate Transformer's learned patterns
- Use Transformer for forecasting, VECM for interpretation
- Combine both: Use VECM for long-run forecasts, Transformer for short-run forecasts

---

## 7. Conclusion & Limitations

### 7.1 Main Conclusions

1. **Forecast Performance:**
   - Transformer significantly outperforms VECM, achieving 89.7% lower RMSE on average
   - Improvement is consistent across all exchange rates (76-90% reduction in errors)
   - Transformer's MAPE is consistently below 2%, while VECM's ranges from 4.78% to 18.86%

2. **Methodological Insights:**
   - Classical econometric models (VECM) provide interpretability and inference capabilities
   - ML models (Transformer) provide flexibility and superior forecast accuracy
   - Model choice depends on research/practical objectives (forecasting vs. inference)

3. **Exchange Rate Implications:**
   - Both models capture co-movements in exchange rates
   - VECM explicitly models long-run equilibrium relationships (cointegration)
   - Transformer learns temporal dependencies implicitly through attention mechanisms
   - Non-linear patterns are important for exchange rate forecasting

4. **Trade-offs:**
   - Clear trade-off between forecast accuracy (Transformer) and interpretability (VECM)
   - Different models serve different purposes: forecasting vs. economic analysis
   - Hybrid approaches may combine both advantages

### 7.2 Limitations

1. **Data Limitations:**
   - Sample size: [**ADD - Check if sufficient for both models**]
   - Data frequency: Daily (could use higher frequency for more observations)
   - Missing variables: [**DISCUSS IF RELEVANT VARIABLES WERE EXCLUDED - e.g., interest rates, macroeconomic indicators**]
   - Sample period: [**DISCUSS IF PERIOD INCLUDES STRUCTURAL BREAKS - e.g., financial crises**]

2. **Model Limitations:**
   - **VECM:**
     - Assumes linear relationships (may miss non-linear patterns)
     - Fixed structure may miss regime changes
     - Requires cointegration (may not hold in all periods)
   - **Transformer:**
     - Limited interpretability (black-box nature)
     - Requires large amounts of data
     - Computationally intensive
     - May overfit with small samples

3. **Methodological Limitations:**
   - Single-step ahead forecasts (could extend to multi-step ahead)
   - No formal statistical tests comparing forecast errors (e.g., Diebold-Mariano test; Diebold & Mariano, 1995)
   - Limited attention analysis (could extract and visualize attention weights for interpretability)
   - No analysis of forecast performance across different market regimes (e.g., high vs. low volatility periods)

4. **Evaluation Limitations:**
   - Only one train/test split (could use rolling-window or expanding-window validation)
   - No analysis of forecast performance over different horizons
   - No comparison with other ML models (e.g., LSTM, GRU) or classical models (e.g., VAR)

### 7.3 Future Research Directions

1. **Multi-Step Ahead Forecasting:**
   - Extend both models to forecast multiple steps ahead
   - Compare performance at different horizons

2. **Additional Variables:**
   - Incorporate macroeconomic indicators (interest rates, inflation, GDP)
   - Include volatility measures or other financial variables

3. **Hybrid Models:**
   - Combine VECM and Transformer (e.g., use VECM for long-run, Transformer for short-run)
   - Develop models that incorporate economic structure with ML flexibility

4. **Attention Analysis:**
   - Extract and visualize Transformer attention weights
   - Interpret which historical periods drive forecasts
   - Compare with VECM's lag structure

5. **Robustness Analysis:**
   - Test performance across different time periods
   - Analyze performance in different market regimes (crisis vs. normal periods)
   - Compare with other exchange rate pairs

6. **Statistical Testing:**
   - Implement Diebold-Mariano test for forecast comparison
   - Test for forecast encompassing
   - Analyze forecast error distributions

---

## 8. References

**Class Notes:**
- Multivariate Time Series Models (Class Notes) - VECM specification, cointegration, inference
- Long Short-Term Memory Networks (Class Notes) - Transformer architecture, attention mechanisms, training
- W1 time series fundamental (Class Notes) - Stationarity, unit root tests
- W2 Quantitative Finance Fundamentals (Class Notes) - Statistical testing, evaluation

**Academic References:**

**VECM and Cointegration:**
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.
- Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration—with applications to the demand for money. *Oxford Bulletin of Economics and Statistics*, 52(2), 169-210.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.

**Transformer Models and Deep Learning for Time Series:**
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.
- Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34, 22419-22430.
- Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.
- Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2022). Transformers in time series: A survey. *arXiv preprint arXiv:2202.07125*.

**Exchange Rate Forecasting:**
- Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1-2), 3-24.
- Cheung, Y. W., Chinn, M. D., & Pascual, A. G. (2005). Empirical exchange rate models of the nineties: Are any fit to survive? *Journal of International Money and Finance*, 24(7), 1150-1175.
- Rossi, B. (2013). Exchange rate predictability. *Journal of Economic Literature*, 51(4), 1063-1119.
- Galeshchuk, S., & Mukherjee, S. (2017). Deep networks for predicting direction of change in foreign exchange rates. *Neural Computing and Applications*, 28(12), 3725-3733.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

**Model Comparison and Evaluation:**
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291-311.

---

## 9. Code Appendix

**Note:** All code is provided in Jupyter notebooks for reproducibility.

**Notebooks Submitted:**
1. `topic1_data_download.ipynb` - Data collection from Yahoo Finance
2. `topic1_step1_eda.ipynb` - Exploratory data analysis, stationarity and cointegration tests
3. `topic1_step2_vecm.ipynb` - VECM model estimation, diagnostics, forecasting
4. `topic1_step3_transformer.ipynb` - Transformer model architecture, training, forecasting
5. `topic1_step4_comparison.ipynb` - Comparative analysis and evaluation

**Code Organization:**
- All notebooks are fully reproducible
- Include comments explaining methodology
- Reference class notes where applicable
- Generate all figures and tables used in the report

---

## CHECKLIST: What Needs to Be Added

### From `topic1_data_download.ipynb`:
- [ ] Date range (start and end dates)
- [ ] Number of observations downloaded

### From `topic1_step1_eda.ipynb`:
- [ ] Summary statistics table (Section 2)
- [ ] Missing values count (Section 2)
- [ ] Time series plot descriptions (Section 3)
- [ ] Correlation matrix values (Section 3)
- [ ] ADF test results for all three rates (Section 4)
- [ ] KPSS test results for all three rates (Section 4)
- [ ] Johansen cointegration test results (Section 5)
- [ ] Cointegration rank (Section 5)

### From `topic1_step2_vecm.ipynb`:
- [ ] Cointegration rank (Section 2)
- [ ] Optimal lag order - AIC suggestion (Section 3)
- [ ] Optimal lag order - BIC suggestion (Section 3)
- [ ] Cointegrating vectors (β matrix) (Section 5)
- [ ] Adjustment coefficients (α matrix) (Section 5)
- [ ] Economic interpretation of β and α (Section 5)
- [ ] Ljung-Box test results (Section 6)
- [ ] Jarque-Bera test results (Section 6)
- [ ] Model stability test results (if computed) (Section 6)

### From `topic1_step3_transformer.ipynb`:
- [ ] Total parameters count (Section 3)
- [ ] Trainable parameters count (Section 3)
- [ ] Number of epochs trained (Section 5)
- [ ] Early stopping epoch (if occurred) (Section 5)
- [ ] Best validation loss (Section 5)
- [ ] Final training loss (Section 5)
- [ ] Initial and final training/validation losses (Section 6)

### Additional Items:
- [ ] Abstract word count check (150-200 words)
- [x] Add academic references
- [ ] Review all class note citations
- [ ] Check page count (4-7 pages excluding code)
- [ ] Include all figures (save from notebooks)
- [ ] Proofread for grammar and spelling

---

**END OF DRAFT REPORT**

