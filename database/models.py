from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    invoice_no = Column(String, index=True)
    stock_code = Column(String, index=True)
    description = Column(String)
    quantity = Column(Integer)
    invoice_date = Column(DateTime, index=True)
    unit_price = Column(Float)
    customer_id = Column(Float, index=True) # Stored as float because Pandas reads missing IDs as NaN
    country = Column(String)

Base.metadata.create_all(bind=engine)