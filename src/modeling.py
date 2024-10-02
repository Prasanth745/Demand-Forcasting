# src/modeling.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_arima_model(data):
    model = ARIMA(data['Quantity'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def train_prophet_model(data):
    df_prophet = data[['Date', 'Quantity']].rename(columns={'Date': 'ds', 'Quantity': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

def xgboost_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model
