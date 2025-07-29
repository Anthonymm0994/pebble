#!/usr/bin/env python3
"""
Histogram Plot Generator
Creates histogram plots from CSV files or SQLite databases.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_data(data_source):
    """Load data from CSV file or database."""
    print(f"[DATA] Loading data from: {data_source}")
    
    try:
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            print(f"[OK] Loaded CSV dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            # Assume it's a database file
            conn = sqlite3.connect(data_source)
            # Get first table
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                first_table = tables[0][0]
                df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                print(f"[INFO] Loaded table: {first_table}")
            else:
                raise ValueError("No tables found in database")
            conn.close()
            
            print(f"[OK] Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
            
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None

def create_histogram(data_source, column_name=None, bin_count=20, output_file="histogram_output.png"):
    """Create histogram plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # If no column specified, use first numeric column
        if column_name is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                column_name = numeric_cols[0]
            else:
                print("[ERROR] No numeric columns found for histogram")
                return
        
        # Check if column exists
        if column_name not in df.columns:
            print(f"[ERROR] Column '{column_name}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(df[column_name].dropna(), bins=bin_count, color='steelblue', alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        
        # Customize plot
        plt.title(f"Histogram of {column_name}", fontsize=14, fontweight='bold')
        plt.xlabel(column_name, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        data = df[column_name].dropna()
        mean_val = data.mean()
        std_val = data.std()
        plt.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nCount: {len(data)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Histogram saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create histogram plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--column', help='Column name to plot (default: first numeric column)')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins (default: 20)')
    parser.add_argument('--output', default='histogram_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_histogram(args.data_source, args.column, args.bins, args.output)

if __name__ == "__main__":
    main() 