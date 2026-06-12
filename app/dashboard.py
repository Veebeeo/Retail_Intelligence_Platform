import streamlit as str
import requests
import pandas as pd

str.set_page_config(page_title="Smart Retail Intelligence Engine", layout="wide")

str.title("Smart Retail Intelligence Platform Dashboard")
str.markdown("---")

API_BASE_URL = "https://retail-intelligence-api-rxib.onrender.com"

# Create two clean columns for our distinct business tools
col1, col2 = str.columns(2)

with col1:
    str.header("Inventory Demand Forecasting")
    str.write("Predict next-week transactional SKU demand values instantly.")
    
    # User inputs for forecasting endpoint
    stock_code_input = str.text_input("Enter SKU / Stock Code:", value="85123A")
    horizon_input = str.slider("Forecast Horizon (Weeks):", min_value=1, max_value=12, value=4)
    
    if str.button("Generate Demand Forecast"):
        with str.spinner("Querying inference pipeline..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/forecast", 
                    json={"stock_code": stock_code_input, "horizon_weeks": horizon_input}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    str.success(f"Forecast fetched successfully for SKU: {data['stock_code']}")
                    
                    # Formulate dataframe from response payload
                    forecast_df = pd.DataFrame(data['predictions'])
                    forecast_df.columns = ['Week Horizon', 'Predicted Stock Quantity']
                    
                    # Display metrics and chart views
                    str.dataframe(forecast_df, use_container_width=True)
                    str.line_chart(data=forecast_df.set_index('Week Horizon'))
                else:
                    str.error(f"API Error: {response.json().get('detail', 'Unknown error occurred.')}")
            except Exception as e:
                str.error(f"Failed to connect to the live FastAPI server: {e}")

with col2:
    str.header("RFM Customer Segmentation Lookup")
    str.write("Retrieve targeted marketing tactical actions for consumer accounts.")
    
    # User inputs for segmentation lookup endpoint
    customer_id_input = str.number_input("Enter Valid Customer ID:", value=17841.0, step=1.0)
    
    if str.button("Fetch Customer Profile"):
        with str.spinner("Searching customer clusters..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/segment",
                    json={"customer_id": customer_id_input}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    metrics = data['metrics']
                    segmentation = data['segmentation']
                    
                    str.metric(label="Assigned Customer Cohort", value=segmentation['assigned_bucket'])
                    
                    # Highlight the strategic business next step beautifully in a blockquote
                    str.markdown(f"> **Strategic Next Step:** {segmentation['strategic_next_action']}")
                    
                    # Render specific customer transactional behaviors
                    str.subheader("Behavioral Background Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric Indicator": ["Days Since Last Purchase", "Total Lifetime Unique Orders", "Total Monetary Spend (£)"],
                        "Value Value": [metrics['days_since_last_purchase'], metrics['total_lifetime_orders'], metrics['total_monetary_spend']]
                    })
                    str.table(metrics_df)
                else:
                    str.error(f"API Error: {response.json().get('detail', 'Unknown error occurred.')}")
            except Exception as e:
                str.error(f"Failed to connect to the live FastAPI server: {e}")