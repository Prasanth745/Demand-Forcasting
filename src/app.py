# src/app.py
import streamlit as st
import pandas as pd
from utils import load_data
from eda import perform_eda
from modeling import train_arima_model, train_prophet_model



def main():
    st.title("Demand Forecasting System")
    transactional_data, customer_data, product_data = load_data()

    # Perform EDA
    perform_eda(transactional_data)

    # Select product
    stock_code = st.selectbox("Select Stock Code", transactional_data['StockCode'].unique())
    num_weeks = st.number_input("Number of Weeks for Forecast", min_value=1, max_value=15, value=15)

    # Forecasting logic
    if st.button("Forecast"):
        # Filter data for the selected product
        product_data = transactional_data[transactional_data['StockCode'] == stock_code]

        # Train models and forecast
        arima_model = train_arima_model(product_data)
        forecast = arima_model.forecast(steps=num_weeks)

        st.write("Forecast for the next weeks:")
        st.line_chart(forecast)

        # Option to download CSV
        forecast_df = pd.DataFrame(forecast, columns=['Forecast'])
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

if __name__ == "__main__":
    main()
