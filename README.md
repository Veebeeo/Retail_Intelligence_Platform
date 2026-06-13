# Retail Intelligence & Demand Planning Platform

A production-grade, multi-container MLOps and Data Engineering ecosystem designed to process over 500K+ transactional records. The platform ingests raw retail data, tracks data-drift and experiments using MLflow, runs predictive time-series and clustering pipelines, and serves real-time inference via a high-performance FastAPI microservice to a reactive Streamlit business dashboard.

The entire architecture is containerized using Docker, ensuring reproducible environments across local development and seamless horizontal scaling during cloud deployment.

---

## Live Production Links

The platform is fully operational and deployed across the public cloud topology. You can interact with the live ecosystem instantly:

* **Production Business UI:** [https://retailintelligence-platform.streamlit.app/](https://retailintelligence-platform.streamlit.app/)
* **Production REST API Gateway (Swagger docs):** [https://retail-intelligence-api-rxib.onrender.com/docs](https://retail-intelligence-api-rxib.onrender.com/docs)

---

## System Architecture

The platform is designed around decoupled microservices interacting within an isolated overlay network, separating the concerns of data storage, business logic inference, and user presentation.

* **Data Warehouse Tier (PostgreSQL):** A persistent relational database optimized with indices on indexing nodes (`stock_code`, `customer_id`) to accelerate complex analytical joins and slice transactions across time windows.
* **Application Services Tier (FastAPI & Uvicorn):** An asynchronous REST API executing container-contained predictive tasks. It acts as an abstraction layer between the presentation dashboard and the cold-storage layer, serving validated Pydantic payloads.
* **Analytics Layer (MLflow Tracking Server):** An experimental registry tracking model metadata, hyperparameter sweeps, performance metrics, and evaluation plots across historical script runs.
* **Presentation Layer (Streamlit UI):** A stateful frontend rendering analytical matrices, predictive trends, and marketing recommendation profiles directly to stakeholders.

* <img width="1918" height="380" alt="graphviz (6)" src="https://github.com/user-attachments/assets/f83d2327-9629-4134-9158-3b557742f490" />


### Repository Structure

```text
retail-intelligence-platform/
├── app/
│   ├── __init__.py
│   ├── dashboard.py           # Streamlit Web Presentation Dashboard
│   └── main.py                # FastAPI Application and Inference Routes
├── database/
│   ├── __init__.py
│   └── models.py              # Declarative SQLAlchemy ORM Data Schemas
├── scripts/
│   ├── __init__.py
│   ├── ingest_data.py         # ETL pipeline parsing raw transactional logs
│   ├── ingest_features.py     # Feature engineering pipeline for time-series matrices
│   └── cluster_customers.py   # Analytical pipeline calculating RFM segments
├── .gitignore                 # Excludes raw data, virtual environments, and local credentials
├── Dockerfile                 # Multi-stage compilation production configuration
├── docker-compose.yml         # Container ecosystem orchestration matrix
├── requirements.txt           # Consolidated framework and tracking dependencies
└── README.md                  # Comprehensive technical dossier

```

# Screenshots

<img width="1920" height="1080" alt="Screenshot 2026-06-13 213859" src="https://github.com/user-attachments/assets/8d370feb-681e-4032-ba06-eda8381a792f" />

<img width="1920" height="1080" alt="Screenshot 2026-06-13 213928" src="https://github.com/user-attachments/assets/b058234d-8a0d-4635-8829-0f2480fa724f" />

# Core Data & Feature Engineering Pipeline

Before any model training or clustering occurs, the platform executes a multi-stage **ETL (Extract, Transform, Load)** pipeline that converts raw, noisy transactional data into structured tables optimized for machine learning algorithms.

## 1. Ingestion and Cleaning (`ingest_data.py`)

### Raw Ingestion

* Reads the bulk dataset.
* Standardizes schema types.
* Drops structural anomalies (e.g., records with missing `Customer ID`).

### Financial Correction

* Filters duplicate records.
* Handles canceled orders (invoices prefixed with `C`) by adjusting net volume and sales records.
* Prevents artificial revenue inflation.

### Database Target

* Streams normalized transactions directly into the production PostgreSQL `transactions` table using SQLAlchemy chunked inserts.

---

## 2. Time-Series Feature Engineering (`ingest_features.py`)

To feed predictive demand models, transactional data is transformed from granular purchases into chronological, equally spaced weekly frequency blocks.

### Market-Share Isolation

* Groups transactions by `stock_code`.
* Selects the top 100 highest-volume SKUs.
* Focuses computational resources on critical inventory drivers.

### Temporal Aggregation

* Groups data by week (`freq='W'`).
* Summarizes net quantity into a uniform `weekly_sales` time-series column.

### Lag & Window Features

#### Lag Features

* **Lag 1:** Sales shifted by 1 week (`t-1`)
* **Lag 2:** Sales shifted by 2 weeks (`t-2`)

These provide short-term memory and capture momentum patterns.

#### Rolling Average

* **4-Week Rolling Mean**
* Captures the previous month's performance.
* Smooths out erratic weekly spikes.

#### Seasonality Index

* Extracts the calendar month integer.
* Captures periodic and holiday-driven demand fluctuations.

---

# Machine Learning Framework & Pipelines

## 1. Demand Forecasting (Time-Series)

The platform evaluates both classical statistical models and machine learning approaches to predict future inventory requirements.

### Model Evaluation & Rationale

#### XGBoost Regressor

Uses engineered lag and calendar-based features.

**Hyperparameters**

```python
n_estimators = 100
max_depth = 5
learning_rate = 0.1
```

**Performance**

```text
MAPE = 53.60%
```

**Observations**

* Captures complex non-linear relationships.
* Struggles to model long-term trends and seasonality when historical data is limited.

---

#### SARIMA (Seasonal ARIMA)

A statistical forecasting model that directly captures:

* Autoregressive patterns
* Long-term trends
* Seasonal demand cycles

**Configuration**

```text
(p,d,q) = (1,1,1)
(P,D,Q,s) = (1,1,1,52)
```

**Performance**

```text
MAPE = 37.12%
```

---

### Experiment Tracking

All training runs, parameters, and metrics are tracked using **MLflow**.

---

### Production Model Selection

| Model   | MAPE   |
| ------- | ------ |
| XGBoost | 53.60% |
| SARIMA  | 37.12% |

SARIMA significantly outperformed XGBoost by capturing deep seasonal retail patterns and was therefore selected as the production forecasting model.

To reduce deployment complexity and container size:

* Heavy ML dependencies such as `xgboost` were removed from production requirements.
* SARIMA became the default forecasting engine.

---

## 2. Customer Segmentation (RFM Clustering)

The platform uses an unsupervised clustering engine to segment customers based on purchasing behavior.

Implemented in:

```text
cluster_customers.py
```

### RFM Metrics

#### Recency

Number of days since the customer's most recent purchase.

#### Frequency

Total number of unique invoices generated by the customer.

#### Monetary

Total customer spending across all transactions.

---

### Clustering Optimization

Before clustering:

1. RFM values undergo logarithmic transformation.
2. Features are standardized using `StandardScaler`.

### K-Means Selection

The optimal cluster count was determined using:

* Elbow Method
* Silhouette Coefficient Analysis

```text
Optimal K = 4
```

---

### Customer Segments & Marketing Strategies

#### Cluster 0 — Champions / Core Loyalists

**Characteristics**

* High frequency
* High spending
* Recent purchases

**Strategy**

* Early access to product launches
* Premium loyalty rewards
* VIP campaigns

---

#### Cluster 1 — At-Risk High Spenders

**Characteristics**

* Historically high spenders
* High recency (inactive)

**Strategy**

* Win-back campaigns
* Personalized discounts
* Re-engagement offers

---

#### Cluster 2 — New / Recent Buyers

**Characteristics**

* Low recency
* Moderate frequency
* Moderate spending

**Strategy**

* Welcome onboarding sequences
* Cross-sell and upsell recommendations

---

#### Cluster 3 — Low-Value Occasional Customers

**Characteristics**

* High recency
* Low frequency
* Low spending

**Strategy**

* Automated engagement campaigns
* Clearance and promotional offers

---

# Production Deployment & Cloud Infrastructure

The platform follows a distributed cloud architecture that separates workloads into independently scalable layers.

---

## 1. Multi-Stage Docker Build Optimization

A secure multi-stage Docker build is used to separate compilation dependencies from runtime execution.

### Stage 1 — Builder

Uses a heavyweight base image containing:

```text
build-essential
libpq-dev
```

Purpose:

* Compile Python wheels
* Build analytical libraries

---

### Stage 2 — Runtime

Uses:

```text
python:3.11-slim
```

Only compiled dependencies are copied from the builder stage.

**Benefits**

* Smaller image size
* Faster deployments
* Reduced attack surface
* Elimination of unnecessary build tools

---

## 2. Environment Variable Protection

Database credentials are loaded dynamically from environment variables rather than hardcoded into source code.

```python
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")

    if user and password and db:
        DATABASE_URL = (
            f"postgresql://{user}:{password}@postgres_db:5432/{db}"
        )
    else:
        raise ValueError(
            "CRITICAL ERROR: Database credentials could not be found anywhere!"
        )
```

---

## 3. Distributed Cloud Deployment Topology

### Database Layer — Render PostgreSQL

* Managed PostgreSQL instance
* Relational indexing
* Handles transactional workloads

---

### API Layer — Render Web Service

* Deploys FastAPI backend
* Builds directly from Dockerfile
* Dynamically binds container ports

---

### Client Layer — Streamlit Community Cloud

* Pulls directly from GitHub
* Communicates with backend via REST APIs
* Hosts the analytics dashboard

---

# 🛠️ Running the Project Locally

## Prerequisites

* Docker Desktop
* Python 3.11+

---

## Clone the Repository

```bash
git clone https://github.com/Veebeeo/Retail_Intelligence_Platform.git

cd Retail_Intelligence_Platform
```

---

## Environment Configuration

Create a `.env` file in the project root:

```env
POSTGRES_USER=retail_user
POSTGRES_PASSWORD=retail_password
POSTGRES_DB=retail_db

DATABASE_URL=postgresql://retail_user:retail_password@localhost:5430/retail_db
```

---

## Start the Infrastructure

```bash
docker compose up --build -d
```

This command:

* Builds the FastAPI image
* Downloads PostgreSQL 15
* Creates the internal network
* Starts all services

---

## Verify Container Health

```bash
docker compose ps
```

Ensure both containers show:

```text
Up
```

Expected services:

```text
retail-intelligence-postgres_db-1
retail-intelligence-web_api-1
```

---

# Data Initialization & Pipeline Execution

## Create Virtual Environment

### Windows

```bash
python -m venv venv

.\venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run ETL & ML Pipelines

### 1. Populate Transaction Data

```bash
python -m scripts.ingest_data
```

---

### 2. Generate Customer Segments

```bash
python -m scripts.cluster_customers
```

---

### 3. Create Forecasting Features

```bash
python -m scripts.ingest_features
```

---

# Accessing the Platform in Your System

## FastAPI Swagger Documentation

```text
http://localhost:8001/docs
```

Use the Swagger UI to test API endpoints interactively.

---

## Streamlit Dashboard

Start the dashboard:

```bash
streamlit run app/dashboard.py
```

Open:

```text
http://localhost:8501
```

to access the complete Retail Intelligence analytics platform.

---

# Key Features

* End-to-End Retail ETL Pipeline
* Transaction Cleaning & Validation
* Weekly Time-Series Feature Engineering
* SARIMA Demand Forecasting
* MLflow Experiment Tracking
* RFM-Based Customer Segmentation
* K-Means Clustering
* FastAPI REST Backend
* PostgreSQL Data Warehouse
* Streamlit Analytics Dashboard
* Dockerized Infrastructure
* Render Cloud Deployment
* Environment-Based Configuration Management
