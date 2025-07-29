#!/usr/bin/env python3
"""
Create SQLite Database from CSV Files
====================================

Converts CSV files to SQLite database for better querying and analysis capabilities.
"""

import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def csv_to_sqlite(csv_file, db_file, table_name=None):
    """Convert CSV file to SQLite database."""
    print(f"[CONVERT] Converting {csv_file} to SQLite database...")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"[OK] Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Create database connection
        conn = sqlite3.connect(db_file)
        
        # Use provided table name or derive from CSV filename
        if table_name is None:
            table_name = Path(csv_file).stem
        
        # Write to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        print(f"[OK] Created table '{table_name}' with {len(columns)} columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        print(f"[OK] Database saved as: {db_file}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {csv_file}: {e}")
        return False

def create_sample_database():
    """Create a comprehensive sample database with multiple tables."""
    print("[SETUP] Creating comprehensive sample database...")
    
    # Create database
    db_file = "../test_data/sample_database.sqlite"
    conn = sqlite3.connect(db_file)
    
    # Create sample_data table
    sample_df = pd.read_csv("../test_data/sample_data.csv")
    sample_df.to_sql("sample_data", conn, if_exists='replace', index=False)
    
    # Create sample_time_data table
    time_df = pd.read_csv("../test_data/sample_time_data.csv")
    time_df.to_sql("sample_time_data", conn, if_exists='replace', index=False)
    
    # Create additional sample tables for analysis
    # Sales data
    sales_data = {
        'id': range(1, 101),
        'product_name': [f'Product_{i}' for i in range(1, 101)],
        'category': ['Electronics', 'Furniture', 'Clothing'] * 33 + ['Electronics'],
        'price': np.random.uniform(10, 1000, 100),
        'quantity': np.random.randint(1, 50, 100),
        'region': ['North', 'South', 'East', 'West'] * 25,
        'sales_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'profit_margin': np.random.uniform(0.1, 0.4, 100)
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_sql("sales", conn, if_exists='replace', index=False)
    
    # Customer data
    customer_data = {
        'customer_id': range(1, 51),
        'name': [f'Customer_{i}' for i in range(1, 51)],
        'email': [f'customer{i}@example.com' for i in range(1, 51)],
        'age': np.random.randint(18, 80, 50),
        'income': np.random.uniform(30000, 150000, 50),
        'region': ['North', 'South', 'East', 'West'] * 12 + ['North', 'South'],
        'loyalty_score': np.random.uniform(0, 100, 50)
    }
    customer_df = pd.DataFrame(customer_data)
    customer_df.to_sql("customers", conn, if_exists='replace', index=False)
    
    # Product data
    product_data = {
        'product_id': range(1, 21),
        'name': [f'Product_{i}' for i in range(1, 21)],
        'category': ['Electronics', 'Furniture', 'Clothing', 'Books', 'Sports'] * 4,
        'price': np.random.uniform(20, 500, 20),
        'cost': np.random.uniform(10, 300, 20),
        'supplier': [f'Supplier_{i}' for i in range(1, 21)],
        'rating': np.random.uniform(1, 5, 20)
    }
    product_df = pd.DataFrame(product_data)
    product_df.to_sql("products", conn, if_exists='replace', index=False)
    
    # Time series data
    time_series_data = {
        'timestamp': pd.date_range('2024-01-01 00:00:00', periods=1000, freq='H'),
        'temperature': np.random.normal(20, 5, 1000),
        'humidity': np.random.uniform(30, 80, 1000),
        'pressure': np.random.uniform(1000, 1020, 1000),
        'sensor_id': np.random.randint(1, 6, 1000)
    }
    time_series_df = pd.DataFrame(time_series_data)
    time_series_df.to_sql("sensor_data", conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"[OK] Created comprehensive database: {db_file}")
    print("[INFO] Available tables:")
    print("  - sample_data: Basic product data")
    print("  - sample_time_data: Time-formatted data")
    print("  - sales: Sales transactions")
    print("  - customers: Customer information")
    print("  - products: Product catalog")
    print("  - sensor_data: Time series sensor data")
    
    return db_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert CSV files to SQLite database')
    parser.add_argument('--csv', help='CSV file to convert')
    parser.add_argument('--db', help='Output SQLite database file')
    parser.add_argument('--table', help='Table name (default: derived from CSV filename)')
    parser.add_argument('--create-sample', action='store_true', help='Create comprehensive sample database')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_database()
    elif args.csv and args.db:
        csv_to_sqlite(args.csv, args.db, args.table)
    else:
        print("[INFO] Creating sample database...")
        create_sample_database()

if __name__ == "__main__":
    import numpy as np
    main() 