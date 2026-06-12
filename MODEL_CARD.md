# Model Card: Demand Forecasting

## Model Comparison
* **SARIMA Baseline:** Simple, performs well with clear seasonality, but struggles with rapid trend shifts.
* **Meta Prophet:** Great at handling macro holiday effects and shifting trends automatically.
* **XGBoost Regressor:** Extremely powerful at mapping complex relationships using lag and rolling features.

## Evaluation Metric
We optimized for **MAPE (Mean Absolute Percentage Error)** because it represents error as a percentage, making it intuitive for retail business stakeholders to evaluate inventory forecasting accuracy.
$$MAPE = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{Actual_t - Forecast_t}{Actual_t} \right| \times 100\%$$

## Final Selection
MAPE of SARIMA = 0.3712
MAPE of Prophet = 40.6340
MAPE of XGBoost = 0.5360

Among these, SARIMA has the lowest MAPE of 37.12% ($0.3712$), meaning, SARIMA is the clear winner here.
It is incredibly common in time-series analysis for classical statistical models like SARIMA to outperform complex machine learning models like XGBoost when forecasting a single item. SARIMA natively captures autocorrelation and seasonality, whereas tree-based models like XGBoost require a massive amount of historical data and window features to map those same mathematical waves.