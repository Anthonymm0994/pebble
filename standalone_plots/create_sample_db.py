#!/usr/bin/env python3
"""
Sample Database Creator
Creates a sample SQLite database with test data for the plotting scripts.
"""

import sqlite3
import pandas as pd
import numpy as np

def create_sample_database():
    """Create a sample database with test data."""
    
    # Create database
    conn = sqlite3.connect('data.sqlite')
    
    # Create sample data
    np.random.seed(42)  # For reproducible results
    
    # Sample data for histogram
    sales_data = pd.DataFrame({
        'amount': np.random.normal(1000, 300, 1000),  # Normal distribution
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'], 1000),
        'date': pd.date_range('2023-01-01', periods=1000, freq='D')
    })
    
    # Sample data for polar plot
    angles = np.linspace(0, 360, 50)
    radii = 10 + 5 * np.sin(np.radians(angles * 3)) + np.random.normal(0, 1, 50)
    polar_data = pd.DataFrame({
        'angle': angles,
        'radius': radii
    })
    
    # Sample data for bar chart
    category_totals = sales_data.groupby('category')['amount'].sum().reset_index()
    category_totals.columns = ['category', 'total_sales']
    
    # Write to database
    sales_data.to_sql('sales', conn, if_exists='replace', index=False)
    polar_data.to_sql('measurements', conn, if_exists='replace', index=False)
    category_totals.to_sql('category_summary', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Sample database 'data.sqlite' created with tables:")
    print("- sales (amount, category, date)")
    print("- measurements (angle, radius)")
    print("- category_summary (category, total_sales)")
    print("\nYou can now test the plotting scripts!")

if __name__ == "__main__":
    create_sample_database() 