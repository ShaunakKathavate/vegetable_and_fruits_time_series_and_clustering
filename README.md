# vegetable_and_fruits_time_series_and_clustering
AI-powered agricultural market intelligence system for forecasting vegetable and fruit prices using Time Series Analysis, Machine Learning, and Deep Learning. The project also performs commodity clustering to identify seasonal and volatility-based market patterns, with interactive visualizations and forecasting dashboards.

# Vegetable and Fruit Price Prediction & Commodity Clustering

AI-powered agricultural market intelligence system for forecasting vegetable and fruit prices using Time Series Analysis, Machine Learning, and Deep Learning techniques.

---

## Project Overview

This project predicts future prices of vegetables and fruits using historical market data and performs clustering of commodities based on pricing behavior, seasonality, and volatility.

The project combines:
- Time Series Forecasting
- Machine Learning
- Deep Learning
- Commodity Clustering
- Data Visualization
- Dashboard Development

---

## Objectives

- Forecast future prices of vegetables and fruits
- Analyze seasonal and trend patterns
- Identify highly volatile commodities
- Cluster commodities with similar pricing behavior
- Build an interactive dashboard for visualization and predictions

---

## Features

### Time Series Forecasting
- ARIMA
- SARIMA
- XGBoost
- LSTM

### Commodity Clustering
- K-Means Clustering
- Hierarchical Clustering
- PCA Visualization

### Data Analysis
- Seasonal decomposition
- Trend analysis
- Volatility analysis
- Correlation analysis

### Dashboard
- Interactive visualizations
- Forecast plots
- Commodity comparison
- Cluster visualization

---

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| Programming | Python |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, xgboost |
| Deep Learning | tensorflow, keras |
| Time Series | statsmodels, prophet |
| Dashboard | streamlit |
| Version Control | Git & GitHub |

---

## Project Structure

```bash
vegetable_price_prediction/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── clustering.py
│   └── utils.py
│
├── models/
├── app/
│   └── streamlit_app.py
│
├── outputs/
├── requirements.txt
├── README.md
└── main.py
Dataset

Possible dataset sources:

Agmarknet
data.gov.in
Kaggle agricultural datasets

Dataset includes:

Commodity name
Date
Market
State
Minimum price
Maximum price
Modal price
Workflow
Data Collection
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Time Series Forecasting
Commodity Clustering
Model Evaluation
Dashboard Development
Deployment
Machine Learning Models
Forecasting Models
ARIMA
SARIMA
XGBoost Regressor
LSTM Neural Networks
Clustering Models
K-Means
Hierarchical Clustering
DBSCAN
Evaluation Metrics
Forecasting Metrics
MAE
RMSE
MAPE
Clustering Metrics
Silhouette Score
Davies-Bouldin Score
Future Improvements
Weather data integration
Festival/event impact analysis
Real-time API integration
Hybrid forecasting models
Market-wise recommendation system
Price anomaly detection
Installation
git clone https://github.com/your-username/vegetable-price-prediction.git

cd vegetable-price-prediction

pip install -r requirements.txt
Run the Project
streamlit run app/streamlit_app.py
Expected Outcomes
Accurate agricultural price forecasting
Commodity behavior analysis
Seasonal trend insights
Interactive analytical dashboard
Author

Shaunak Kathavate

LinkedIn: https://www.linkedin.com/in/shaunak-kathavate-7322321a4/

GitHub: https://github.com/ShaunakKathavate
