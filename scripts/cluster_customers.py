import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Setup & Connection
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

def load_transaction_data():
    query = "SELECT customer_id, invoice_no, invoice_date, quantity, unit_price FROM transactions"
    df = pd.read_sql(query, engine)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['total_cost'] = df['quantity'] * df['unit_price']
    return df

def build_rfm_features(df):
    print("Calculating RFM metrics...")
    # Reference date is set to 1 day after the absolute last invoice in the dataset
    snapshot_date = df['invoice_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (snapshot_date - x.max()).days, # Recency
        'invoice_no': 'nunique',                                  # Frequency
        'total_cost': 'sum'                                       # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    return rfm

def assign_business_labels(df):
    """
    Maps K-Means cluster numbers to intuitive strategic business labels 
    based on their average RFM characteristics.
    """
    # We will sort clusters by average monetary value to systematically assign labels
    cluster_profile = df.groupby('cluster')['monetary'].mean().sort_values().index
    
    # Mapping logic: Lowest spending cluster to highest spending cluster
    labels = {
        cluster_profile[0]: "Hibernating / At-Risk",
        cluster_profile[1]: "About to Sleep",
        cluster_profile[2]: "Leal Customers",
        cluster_profile[3]: "Champions"
    }
    
    df['segment_label'] = df['cluster'].map(labels)
    return df

def main():
    # Load and process data
    raw_df = load_transaction_data()
    rfm_df = build_rfm_features(raw_df)
    
    # 2. Preprocessing for K-Means (Log transform to fix skewness + Scale)
    print("Preprocessing data...")
    rfm_log = rfm_df[['recency', 'frequency', 'monetary']].copy()
    
    # Add a small constant to prevent log(0)
    rfm_log['recency'] = np.log(rfm_log['recency'] + 1)
    rfm_log['frequency'] = np.log(rfm_log['frequency'] + 1)
    rfm_log['monetary'] = np.log(rfm_log['monetary'] + 1)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 3. Apply K-Means (We'll use K=4 for highly interpretable retail segments)
    print("Training K-Means Clustering Engine...")
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Assign readable strategies to the clusters
    rfm_segmented = assign_business_labels(rfm_df)
    
    # 4. Save segments back to PostgreSQL
    print("Saving segments to database...")
    rfm_segmented.to_sql('customer_segments', engine, if_exists='replace', index=False)
    
    print("\nCustomer Segmentation complete! Summary Profiles:")
    summary = rfm_segmented.groupby('segment_label').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'customer_count'})
    print(summary)

if __name__ == "__main__":
    main()