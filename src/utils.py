# src/utils.py
import pandas as pd

def load_data():
    data1 = pd.read_csv('Transactional_data_retail_01.csv')
    data2 = pd.read_csv('Transactional_data_retail_02.csv')
    customer_data = pd.read_csv('CustomerDemographics.csv')
    product_data = pd.read_csv('ProductInfo.csv')
    
    # Combine transactional data
    transactional_data = pd.concat([data1, data2])
    return transactional_data, customer_data, product_data
