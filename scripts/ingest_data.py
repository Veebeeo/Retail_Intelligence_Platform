import pandas as pd
from database.models import engine

def load_and_clean_data(file_path):
    print("Reading Excel file")
    # Reading the 2010-2011 sheet
    df = pd.read_excel(file_path, sheet_name="Year 2010-2011")
    
    print("Initial shape:", df.shape)
    
    # 1. Drop rows without Customer IDs 
    df = df.dropna(subset=['Customer ID'])
    
    # 2. Filter out cancelled orders (Invoice numbers starting with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # 3. Filter out negative quantities and zero pricing
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # Rename columns to match our SQLAlchemy model
    df = df.rename(columns={
        'Invoice': 'invoice_no',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'Price': 'unit_price',
        'Customer ID': 'customer_id',
        'Country': 'country'
    })
    
    print("Cleaned shape:", df.shape)
    return df

def ingest_to_db(df):
    print("Pushing data to PostgreSQL...")
    # Using pandas to_sql for efficient bulk insertion
    df.to_sql('transactions', engine, if_exists='append', index=False)
    print("Data ingestion complete!")

if __name__ == "__main__":
    FILE_PATH = "data/online_retail_II.xlsx" # Update with exact file name
    clean_df = load_and_clean_data(FILE_PATH)
    ingest_to_db(clean_df)