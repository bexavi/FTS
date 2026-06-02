# Topic 1: VECM vs. Transformer for Exchange Rate Prediction (Short Form)

## Abstract
We compare a Vector Error Correction Model (VECM, effectively VAR in differences because no cointegration was found) with a Transformer for forecasting EUR/USD, GBP/USD, and USD/JPY. Using daily data from 1999-01-04 to 2025-12-05 (6,753 observations), the Transformer sharply outperforms the classical benchmark: average RMSE 1.119 vs. 10.823, MAE 0.894 vs. 9.112, MAPE 1.24% vs. 10.17% (89–90% error reduction). The Transformer’s multi-head attention captures non-linear temporal patterns, while the VECM offers limited interpretability here because no cointegration was detected. The trade-off is clear: superior accuracy from the Transformer versus the classical model’s usual interpretability (not realized due to rank 0). For forecasting tasks, the Transformer dominates; for economic inference, additional structure or alternative econometric models would be needed.

## 1. Research Question & Motivation
**Research question:** How does a classical econometric model (VECM/VAR in differences) compare to a Transformer in forecasting major exchange rates, and what are the trade-offs between accuracy and interpretability?

**Motivation (economic):**
- Exchange rates affect trade, investment and policy; forecast errors can be costly for hedging and portfolio allocation.
- EUR/USD, GBP/USD and USD/JPY are among the most traded currency pairs; understanding their dynamics is practically important.

**Motivation (methodological):**
- Classical models (VECM/VAR) are interpretable and grounded in time-series theory.
- Transformers can exploit non-linear and long-range dependencies via attention but are less interpretable.

## 2. Data Description
- Source: FRED daily FX (DEXUSEU, DEXUSUK, DEXJPUS).
- Period: 1999-01-04 to 2025-12-05; Observations: 6,753 per series.
- Summary stats (levels): EURUSD mean 1.183 (sd 0.154), GBPUSD mean 1.523 (sd 0.220), USDJPY mean 112.289 (sd 17.651); no missing values.
- Correlations (levels): EUR/USD–GBP/USD 0.587; EUR/USD–USD/JPY -0.552; GBP/USD–USD/JPY -0.352 (European pairs move together; USDJPY often moves inversely).
- Stationarity: ADF fails to reject a unit root; KPSS rejects stationarity for all series → all I(1). Johansen test gives rank 0 → no cointegration.

**Figure 1 (EDA placeholder):** Time series and correlation matrix (`topic1_time_series_plots.png`, `topic1_correlation_matrix.png`).

## 3. Model Methodology

### 3.0 Method Selection & Justification (Why These Two?)
We deliberately choose a **VECM/VAR-type classical model** and a **Transformer** as our two methods because they represent **two ends of the spectrum** emphasized in the course:  
- VECM/VAR is the **canonical multivariate econometric model** for macro/FX series (see *Multivariate Time Series Models*), designed for joint dynamics, cointegration testing, and economic interpretation. It is superior to univariate ARIMA-type models for this problem because it explicitly models cross‑currency interactions and, when cointegration exists, long‑run equilibrium relationships.  
- The Transformer is a **modern attention-based deep learning model** highlighted in the *Long Short-Term Memory Networks* notes as an evolution beyond RNN/LSTM architectures. It is better suited than simpler ML models (e.g., vanilla RNNs, feed‑forward nets, tree models) to capture **non-linear, long-range temporal dependencies** in multivariate sequences without heavy manual feature engineering.

Compared to other classical options (DFM, state‑space/Kalman, multivariate GARCH), VECM/VAR offers a **simpler, standard benchmark** for level forecasts with clear links to the course, while avoiding the extra complexity of latent-factor or volatility models that are more natural for very high-dimensional or volatility-focused tasks. Compared to other ML options (LSTM/GRU, TCN), the Transformer provides a **clean contrast** to the linear VAR: it replaces recurrence with attention, is explicitly motivated in the notes, and empirically delivers much stronger accuracy, making the trade‑off between interpretability and performance very clear for this assignment.

### 3.1 Classical Model: VECM / VAR in Differences
- Johansen rank 0 ⇒ no cointegrating vectors β, no adjustment coefficients α.
- With rank 0 and lag 1, the VECM reduces to a VAR(1) in first differences with a constant (deterministic = ‘co’).
- Interpretation: captures short-run dynamics of ΔFX but imposes no long-run equilibrium structure.

### 3.2 Machine Learning Model: Transformer
- Inputs: 3-dimensional FX vector (EURUSD, GBPUSD, USDJPY), normalized via MinMax.
- Window/horizon: 30-day lookback, 1-day ahead forecast.
- Architecture: d_model 64, 4 attention heads, 2 encoder layers, feedforward size 128, dropout 0.1 (~67k trainable parameters).
- Training: 80/10/10 split (train/validation/test); Adam optimizer, MSE loss, early stopping at epoch 32 (best validation loss ≈ 0.000070; final train loss ≈ 0.000142).
- Interpretation: multi-head self-attention lets the model reweight past days when constructing forecasts.

## 4. Comparative Empirical Results
### 4.1 Forecast Accuracy
| Pair   | Model        | RMSE     | MAE      | MAPE (%) |
|--------|--------------|----------|----------|----------|
| EURUSD | VECM         | 0.060506 | 0.052286 | 4.78     |
|        | Transformer  | 0.014088 | 0.011520 | 1.05     |
| GBPUSD | VECM         | 0.104793 | 0.090729 | 6.89     |
|        | Transformer  | 0.012868 | 0.010672 | 0.84     |
| USDJPY | VECM         | 32.303681| 27.193325| 18.86    |
|        | Transformer  | 3.330883 | 2.658380 | 1.83     |
| **Avg**| VECM         | 10.822993| 9.112113 | 10.17    |
|        | Transformer  | 1.119279 | 0.893524 | 1.24     |

**Key findings:** Transformer reduces RMSE and MAE by roughly 89–90% on average and wins on MAPE for every pair. VECM underperforms, partly because it cannot exploit long-run structure (rank 0) and remains linear in differences.

**Figure 2 (Forecasts & metrics placeholders):** VECM vs actual, Transformer vs actual, and bar charts of RMSE/MAE/MAPE (`vecm_forecasts_vs_actual.png`, `transformer_forecasts_vs_actual.png`, `model_comparison_metrics.png`).

## 5. Forecasting & Simulation Analysis
- Forecasting setup: one-step-ahead out-of-sample forecasts on the last 20% of the sample (2020–2025) for all three FX series.
- Horizon: 1-day ahead; forecasts are rolled across the test window.
- Behavior: USDJPY is hardest to predict (largest scale and volatility); both models’ errors are largest there, but the Transformer still reduces error by about 90%.
- No separate simulation (e.g., scenario or stress-test simulations) is run; instead, we interpret behavior from the long evaluation window covering calm and turbulent periods.

## 6. Interpretation (Economic + Statistical)
**Statistical:**
- The Transformer clearly dominates in point forecast accuracy on all metrics and all pairs.
- Classical VAR-in-differences residuals show some misspecification (autocorrelation for GBPUSD, non-normality across series), which weakens its statistical reliability.

**Economic:**
- Because no cointegration is found, the classical model does not deliver the usual long-run parity or PPP-type interpretations; there is no stable β relation to interpret.
+- The Transformer is data-driven and flexible; while less interpretable at the parameter level, its outperformance suggests that exchange rates exhibit non-linear and regime-dependent patterns not captured by a simple VAR(1).

## 7. Conclusion & Limitations
**Conclusion:**
- For FX forecasting on this dataset, the Transformer is clearly preferable to the classical VECM/VAR benchmark.
- The absence of cointegration means the classical model’s main theoretical advantage (long-run equilibrium interpretation) is not realized.
- In practice, a risk manager or trader concerned primarily with forecast accuracy should choose the Transformer; a researcher needing structural interpretation would need richer econometric models or hybrid approaches.

**Limitations:**
- No cointegration may be sample-dependent; adding fundamentals or shorter sub-periods could change the result.
- Classical model uses only lag 1; more lags or GARCH-type volatility modeling might narrow the performance gap.
- No formal Diebold–Mariano tests are reported; we rely on large, consistent error reductions.
- Results are for 1-day-ahead horizons; multi-step forecasting is left for future work.

**Minimal links to class notes:**
- W1/W3: justify unit-root testing, differencing, and Johansen procedure.
- Multivariate Time Series Models: justify VAR/VECM specification and lag selection with AIC/BIC.
- Long Short-Term Memory Networks notes: motivate deep learning, MSE/MAE/MAPE metrics, and the role of attention in improving sequential forecasts.