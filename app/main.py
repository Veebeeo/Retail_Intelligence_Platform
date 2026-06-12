from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Smart Retail Intelligence API",
    description="Production endpoints for Demand Forecasting and Customer Segmentation ",
    version="1.0"
)

#Connect to Database

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    
    if user and password and db:
        DATABASE_URL = f"postgresql://{user}:{password}@postgres_db:5432/{db}"
    else:
        raise ValueError("CRITICAL ERROR: Database credentials could not be found anywhere!")

engine = create_engine(DATABASE_URL)

#Define request schemas using Pydantic 
class ForecastRequest(BaseModel):
    stock_code : str
    horizon_weeks : int = 4

class SegmentRequest(BaseModel):
    customer_id: float

@app.get("/")
def read_root():
    return {"status":"healthy", "message": "Smart Retail Intelligence Platform API is live."}

@app.post("/forecast")
def get_forecast(payload: ForecastRequest):
    '''Predicts next N weeks of demand for a given SKU based on historical
    averages and our champion SARIM baseline growth trend.'''

    query = f"SELECT weekly_sales FROM ml_weekly_features WHERE stock_code = '{payload.stock_code}' ORDER BY week DESC LIMIT 4"
    df = pd.read_sql(query,engine)

    if df.empty:
        raise HTTPException(status_code = 404, detail=f"Stock code '{payload.stock_code}' not found in engineered features")
    
    recent_avg = df['weekly_sales'].mean() #Use recent historical baseline to generate future steps

    predictions = []
    for week in range(1, payload.horizon_weeks + 1):
        predicted_value = max(0, recent_avg * (1 + (0.02 * week)))
        predictions.append({
            "week_horizon" : week,
            "predicted_quantity" : round(predicted_value, 2)

        })
    return {
        "stock_code": payload.stock_code,
        "forecast_horizon_weeks": payload.horizon_weeks,
        "predictions": predictions
    }

@app.post("/segment")
def get_customer_segment(payload: SegmentRequest):
    """
    Returns the designated RFM cluster segment along with an actionable 
    marketing strategy for the specified customer.
    """
    query = f"SELECT recency, frequency, monetary, segment_label FROM customer_segments WHERE customer_id = {payload.customer_id}" 
    df = pd.read_sql(query, engine)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {payload.customer_id} not found.")
    
    customer_data = df.iloc[0]
    label = customer_data['segment_label']
    
    # Inject action-oriented corporate strategies based on labels
    strategies = {
        "Champions": "Offer early access to new product rollouts. Reward with exclusive loyalty incentives.",
        "Leal Customers": "Upsell premium bundles or subscription tiers. Solicit reviews or referrals.",
        "About to Sleep": "Send limited-time personalized discount hooks to incentivize re-engagement.",
        "Hibernating / At-Risk": "Launch an aggressive win-back campaign via email offering steep value matches."
    }
    
    return {
        "customer_id": payload.customer_id,
        "metrics": {
            "days_since_last_purchase": int(customer_data['recency']),
            "total_lifetime_orders": int(customer_data['frequency']),
            "total_monetary_spend": round(float(customer_data['monetary']), 2)
        },
        "segmentation": {
            "assigned_bucket": label,
            "strategic_next_action": strategies.get(label, "Maintain standard outreach pipelines.")
        }
    }

