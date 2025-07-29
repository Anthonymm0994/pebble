#!/usr/bin/env python3
"""
Scatter Plot Generator
Creates scatter plots from CSV files or SQLite databases.
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

def create_scatter_plot(data_source, x_column=None, y_column=None, output_file="scatter_output.png"):
    """Create scatter plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # If no columns specified, use first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if x_column is None and len(numeric_cols) >= 2:
            x_column = numeric_cols[0]
            y_column = numeric_cols[1]
        elif x_column is None:
            print("[ERROR] Need at least 2 numeric columns for scatter plot")
            print(f"[INFO] Available numeric columns: {list(numeric_cols)}")
            return
        
        # Check if columns exist
        if x_column not in df.columns:
            print(f"[ERROR] Column '{x_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        if y_column not in df.columns:
            print(f"[ERROR] Column '{y_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(df[x_column], df[y_column], alpha=0.6, s=50, color='steelblue')
        
        # Customize plot
        plt.title(f"Scatter Plot: {x_column} vs {y_column}", fontsize=14, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = df[x_column].corr(df[y_column])
        plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Scatter plot saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create scatter plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--x', help='X-axis column name (default: first numeric column)')
    parser.add_argument('--y', help='Y-axis column name (default: second numeric column)')
    parser.add_argument('--output', default='scatter_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_scatter_plot(args.data_source, args.x, args.y, args.output)

if __name__ == "__main__":
    main() 