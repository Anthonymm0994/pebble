#!/usr/bin/env python3
"""
Heatmap Plot Generator
Creates heatmap plots from CSV files or SQLite databases.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns
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

def create_heatmap_plot(data_source, output_file="heatmap_output.png"):
    """Create heatmap plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # Select numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            print("[ERROR] Need at least 2 numeric columns for heatmap")
            print(f"[INFO] Available numeric columns: {list(numeric_df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        
        # Customize plot
        plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Heatmap saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create heatmap plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--output', default='heatmap_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_heatmap_plot(args.data_source, args.output)

if __name__ == "__main__":
    main() 