#!/usr/bin/env python3
"""
Sample Database Creator for Repository
Creates a smaller sample database for the repository (under 10MB).
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_database():
    """Create a smaller sample database for the repository."""
    print("Creating sample database for repository...")
    
    # Create database connection
    conn = sqlite3.connect('sample_data.sqlite')
    
    # Generate smaller datasets
    n_rows = 10000  # Much smaller than the 1M rows
    
    print("Creating sales data...")
    # 1. SALES DATA (for histograms and bar charts)
    sales_data = pd.DataFrame({
        'amount': np.random.lognormal(6.5, 0.8, n_rows),  # Log-normal distribution
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food', 'Sports', 'Home'], n_rows, p=[0.3, 0.25, 0.15, 0.2, 0.05, 0.05]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_rows),
        'customer_type': np.random.choice(['Individual', 'Business', 'Government'], n_rows, p=[0.6, 0.3, 0.1]),
        'date': pd.date_range('2020-01-01', periods=n_rows, freq='h'),
        'profit_margin': np.random.beta(2, 5, n_rows) * 100,  # 0-100%
        'quantity': np.random.poisson(3, n_rows) + 1,  # 1-20 items
        'rating': np.random.normal(4.2, 0.8, n_rows).clip(1, 5)  # 1-5 stars
    })
    
    print("Creating measurements data...")
    # 2. MEASUREMENTS DATA (for polar plots)
    measurements_data = pd.DataFrame({
        'angle': np.random.uniform(0, 360, n_rows),  # 0-360 degrees
        'radius': np.random.exponential(5, n_rows),  # Exponential distribution
        'temperature': np.random.normal(20, 10, n_rows),  # Temperature in Celsius
        'pressure': np.random.normal(1013, 50, n_rows),  # Atmospheric pressure
        'humidity': np.random.uniform(30, 90, n_rows)  # Humidity percentage
    })
    
    print("Creating time series data...")
    # 3. TIME SERIES DATA (for time series plots)
    dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    time_series_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(1000) / 24) + np.random.normal(0, 2, 1000),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(1000) / 12) + np.random.normal(0, 5, 1000),
        'pressure': 1013 + 20 * np.sin(2 * np.pi * np.arange(1000) / 48) + np.random.normal(0, 10, 1000)
    })
    
    print("Creating categorical summaries...")
    # 4. CATEGORICAL SUMMARIES (for bar charts)
    category_summary = sales_data.groupby('category').agg({
        'amount': ['sum', 'mean', 'count'],
        'profit_margin': 'mean',
        'rating': 'mean'
    }).round(2)
    category_summary.columns = ['total_amount', 'avg_amount', 'count', 'avg_profit', 'avg_rating']
    category_summary = category_summary.reset_index()
    
    region_summary = sales_data.groupby('region').agg({
        'amount': ['sum', 'mean', 'count'],
        'profit_margin': 'mean'
    }).round(2)
    region_summary.columns = ['total_amount', 'avg_amount', 'count', 'avg_profit']
    region_summary = region_summary.reset_index()
    
    print("Creating distribution data...")
    # 5. DISTRIBUTION DATA (for distribution plots)
    distributions_data = pd.DataFrame({
        'normal_dist': np.random.normal(0, 1, n_rows),
        'exponential_dist': np.random.exponential(1, n_rows),
        'uniform_dist': np.random.uniform(-2, 2, n_rows),
        'bimodal_dist': np.concatenate([
            np.random.normal(-2, 0.5, n_rows//2),
            np.random.normal(2, 0.5, n_rows//2)
        ]),
        'skewed_dist': np.random.lognormal(0, 1, n_rows),
        'age': np.random.normal(35, 15, n_rows).clip(18, 80),
        'income': np.random.lognormal(10, 0.5, n_rows),
        'satisfaction_score': np.random.normal(7, 1.5, n_rows).clip(1, 10)
    })
    
    print("Creating correlation data...")
    # 6. CORRELATION DATA (for scatter plots)
    x = np.random.normal(0, 1, n_rows)
    correlations_data = pd.DataFrame({
        'x': x,
        'y_positive': x + np.random.normal(0, 0.5, n_rows),  # Positive correlation
        'y_negative': -x + np.random.normal(0, 0.5, n_rows),  # Negative correlation
        'y_no_correlation': np.random.normal(0, 1, n_rows),  # No correlation
        'size': np.random.uniform(10, 100, n_rows),
        'color': np.random.uniform(0, 1, n_rows)
    })
    
    print("Writing tables to database...")
    # Write all tables to database
    sales_data.to_sql('sales', conn, if_exists='replace', index=False)
    measurements_data.to_sql('measurements', conn, if_exists='replace', index=False)
    time_series_data.to_sql('time_series', conn, if_exists='replace', index=False)
    category_summary.to_sql('category_summary', conn, if_exists='replace', index=False)
    region_summary.to_sql('region_summary', conn, if_exists='replace', index=False)
    distributions_data.to_sql('distributions', conn, if_exists='replace', index=False)
    correlations_data.to_sql('correlations', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("\n✅ Sample database created successfully!")
    print(f"📊 Total data: {n_rows:,} rows across multiple tables")
    print(f"📁 Database file: sample_data.sqlite")
    
    print("\n📋 Available tables:")
    print("- sales: Main sales data with amounts, categories, regions")
    print("- measurements: Scientific data with angles, radii, temperatures")
    print("- time_series: Hourly time series data")
    print("- category_summary: Aggregated sales by category")
    print("- region_summary: Aggregated sales by region")
    print("- distributions: Various statistical distributions")
    print("- correlations: Data for scatter plots")
    
    print("\n🎯 Ready to test plotting scripts!")
    print("💡 Use 'sample_data.sqlite' instead of 'data.sqlite' in your scripts")

if __name__ == "__main__":
    create_sample_database() 