#!/usr/bin/env python3
"""
Comprehensive Database Creator
Creates a large SQLite database with 1M+ rows of diverse data for testing all plotting scenarios.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_comprehensive_database():
    """Create a comprehensive database with 1M+ rows of diverse data."""
    
    print("Creating comprehensive test database...")
    
    # Create database
    conn = sqlite3.connect('data.sqlite')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 1M rows of diverse data
    n_rows = 1_000_000
    
    print(f"Generating {n_rows:,} rows of data...")
    
    # 1. SALES DATA (for histograms and bar charts)
    print("Creating sales data...")
    sales_data = pd.DataFrame({
        'amount': np.random.lognormal(6.5, 0.8, n_rows),  # Log-normal distribution
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food', 'Sports', 'Home'], n_rows, p=[0.3, 0.25, 0.15, 0.2, 0.05, 0.05]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_rows),
        'customer_type': np.random.choice(['Individual', 'Business', 'Government'], n_rows, p=[0.6, 0.3, 0.1]),
        'date': pd.date_range('2020-01-01', periods=n_rows, freq='H'),
        'profit_margin': np.random.beta(2, 5, n_rows) * 100,  # 0-100%
        'quantity': np.random.poisson(3, n_rows) + 1,  # 1-20 items
        'rating': np.random.normal(4.2, 0.8, n_rows).clip(1, 5)  # 1-5 stars
    })
    
    # 2. SCIENTIFIC MEASUREMENTS (for polar plots)
    print("Creating scientific measurements...")
    angles = np.linspace(0, 360, 1000)
    # Create complex polar patterns
    radii_complex = (
        10 + 
        5 * np.sin(np.radians(angles * 2)) + 
        3 * np.cos(np.radians(angles * 3)) + 
        2 * np.sin(np.radians(angles * 5)) +
        np.random.normal(0, 0.5, 1000)
    )
    
    measurements_data = pd.DataFrame({
        'angle': angles,
        'radius': radii_complex,
        'temperature': 20 + 10 * np.sin(np.radians(angles)) + np.random.normal(0, 2, 1000),
        'pressure': 1013 + 50 * np.cos(np.radians(angles * 2)) + np.random.normal(0, 10, 1000),
        'humidity': 50 + 30 * np.sin(np.radians(angles * 1.5)) + np.random.normal(0, 5, 1000)
    })
    
    # 3. TIME SERIES DATA (for line plots)
    print("Creating time series data...")
    dates = pd.date_range('2020-01-01', periods=10000, freq='H')
    time_series_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, 10000),
        'humidity': 60 + 20 * np.cos(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 5, 10000),
        'pressure': 1013 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 3, 10000),
        'wind_speed': np.random.exponential(5, 10000),
        'precipitation': np.random.exponential(0.5, 10000)
    })
    
    # 4. CATEGORICAL DATA (for bar charts)
    print("Creating categorical summaries...")
    category_summary = sales_data.groupby('category').agg({
        'amount': ['sum', 'mean', 'count'],
        'profit_margin': 'mean',
        'rating': 'mean'
    }).round(2)
    category_summary.columns = ['total_sales', 'avg_sale', 'transaction_count', 'avg_profit_margin', 'avg_rating']
    category_summary = category_summary.reset_index()
    
    region_summary = sales_data.groupby('region').agg({
        'amount': ['sum', 'count'],
        'profit_margin': 'mean'
    }).round(2)
    region_summary.columns = ['total_sales', 'transaction_count', 'avg_profit_margin']
    region_summary = region_summary.reset_index()
    
    # 5. DISTRIBUTION DATA (for histograms)
    print("Creating distribution data...")
    distribution_data = pd.DataFrame({
        'normal_dist': np.random.normal(100, 20, n_rows),
        'exponential_dist': np.random.exponential(50, n_rows),
        'uniform_dist': np.random.uniform(0, 200, n_rows),
        'bimodal_dist': np.concatenate([
            np.random.normal(50, 10, n_rows // 2),
            np.random.normal(150, 15, n_rows // 2)
        ]),
        'skewed_dist': np.random.lognormal(4, 1, n_rows),
        'age': np.random.normal(35, 12, n_rows).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_rows),
        'satisfaction_score': np.random.beta(2, 2, n_rows) * 10
    })
    
    # 6. CORRELATION DATA (for scatter plots)
    print("Creating correlation data...")
    correlation_data = pd.DataFrame({
        'x': np.random.normal(0, 1, 10000),
        'y_positive': np.random.normal(0, 1, 10000) + 0.8 * np.random.normal(0, 1, 10000),
        'y_negative': np.random.normal(0, 1, 10000) - 0.6 * np.random.normal(0, 1, 10000),
        'y_no_correlation': np.random.normal(0, 1, 10000),
        'size': np.random.uniform(10, 100, 10000),
        'color': np.random.choice(['A', 'B', 'C', 'D'], 10000)
    })
    
    # Write all tables to database
    print("Writing tables to database...")
    
    tables = {
        'sales': sales_data,
        'measurements': measurements_data,
        'time_series': time_series_data,
        'category_summary': category_summary,
        'region_summary': region_summary,
        'distributions': distribution_data,
        'correlations': correlation_data
    }
    
    for table_name, data in tables.items():
        print(f"Writing {table_name} table ({len(data):,} rows)...")
        data.to_sql(table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("\n✅ Comprehensive database created successfully!")
    print(f"📊 Total data: {n_rows:,} rows across multiple tables")
    print("\n📋 Available tables:")
    print("- sales: Main sales data with amounts, categories, regions")
    print("- measurements: Scientific data with angles, radii, temperatures")
    print("- time_series: Hourly time series data")
    print("- category_summary: Aggregated sales by category")
    print("- region_summary: Aggregated sales by region")
    print("- distributions: Various statistical distributions")
    print("- correlations: Data for scatter plots")
    
    print("\n🎯 Ready to test plotting scripts!")

if __name__ == "__main__":
    create_comprehensive_database() 