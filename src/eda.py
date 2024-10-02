# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(transactional_data):
    # Summary statistics
    print(transactional_data.describe())

    # Top 10 stock codes by quantity sold
    top_10_products = transactional_data.groupby('StockCode')['Quantity'].sum().nlargest(10).reset_index()
    print(top_10_products)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='StockCode', y='Quantity', data=top_10_products)
    plt.title('Top 10 Products by Quantity Sold')
    plt.xticks(rotation=45)
    plt.show()
