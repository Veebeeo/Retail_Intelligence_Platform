import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.statsmodels
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

# 1. Setup & Data Loading
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# Setup MLflow Experiment
mlflow.set_experiment("Retail_Demand_Forecasting")

def load_data():
    query = "SELECT * FROM ml_weekly_features ORDER BY week ASC"
    df = pd.read_sql(query, engine)
    df['week'] = pd.to_datetime(df['week'])
    return df

def main():
    df = load_data()
    
    # Pick the top SKU by volume to run our experiment on
    top_sku = df.groupby('stock_code')['weekly_sales'].sum().idxmax()
    print(f"Running forecasting experiments for Top SKU: {top_sku}")
    
    sku_data = df[df['stock_code'] == top_sku].copy().reset_index(drop=True)
    
    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(sku_data) * 0.8)
    train_df = sku_data.iloc[:split_idx]
    test_df = sku_data.iloc[split_idx:]
    
    print(f"Train periods: {len(train_df)}, Test periods: {len(test_df)}")

    # EXPERIMENT 1: SARIMA Baseline
    with mlflow.start_run(run_name="SARIMA_Baseline"):
        print("Training SARIMA...")
        # Simple seasonal order guess for demonstration
        model_sarima = SARIMAX(train_df['weekly_sales'], order=(1,1,1), seasonal_order=(1,1,0,4))
        results_sarima = model_sarima.fit(disp=False)
        
        preds_sarima = results_sarima.forecast(steps=len(test_df))
        
        # Calculate Metrics
        mape = mean_absolute_percentage_error(test_df['weekly_sales'], preds_sarima)
        rmse = np.sqrt(mean_squared_error(test_df['weekly_sales'], preds_sarima))
        
        # Log to MLflow
        mlflow.log_param("model_type", "SARIMA")
        mlflow.log_param("order", "(1,1,1)")
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("RMSE", rmse)
        mlflow.statsmodels.log_model(results_sarima, "model")
        print(f"SARIMA completed. MAPE: {mape:.4f}")

    # EXPERIMENT 2: Meta Prophet

    with mlflow.start_run(run_name="Prophet_Model"):
        print("Training Prophet...")
        # Prophet requires 'ds' and 'y' column names
        prophet_train = train_df[['week', 'weekly_sales']].rename(columns={'week': 'ds', 'weekly_sales': 'y'})
        prophet_test = test_df[['week', 'weekly_sales']].rename(columns={'week': 'ds', 'weekly_sales': 'y'})
        
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_prophet.fit(prophet_train)
        
        forecast = model_prophet.predict(prophet_test[['ds']])
        preds_prophet = forecast['yhat'].values
        
        mape = mean_absolute_percentage_error(test_df['weekly_sales'], preds_prophet)
        rmse = np.sqrt(mean_squared_error(test_df['weekly_sales'], preds_prophet))
        
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("RMSE", rmse)
        # Prophet model logged as generic python function artifact
        mlflow.sharable = False 
        print(f"Prophet completed. MAPE: {mape:.4f}")

    # EXPERIMENT 3: XGBoost with Lag Features

    with mlflow.start_run(run_name="XGBoost_Regressor"):
        print("Training XGBoost...")
        # Define features and target from Phase 2
        features = ['lag_1_week', 'lag_2_week', 'rolling_4_wk_avg', 'month']
        target = 'weekly_sales'
        
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]
        
        model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        model_xgb.fit(X_train, y_train)
        
        preds_xgb = model_xgb.predict(X_test)
        
        mape = mean_absolute_percentage_error(y_test, preds_xgb)
        rmse = np.sqrt(mean_squared_error(y_test, preds_xgb))
        
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("RMSE", rmse)
        mlflow.xgboost.log_model(model_xgb, "model")
        print(f"XGBoost completed. MAPE: {mape:.4f}")

if __name__ == "__main__":
    main()