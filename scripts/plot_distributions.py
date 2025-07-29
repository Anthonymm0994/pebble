#!/usr/bin/env python3
"""
Distribution Plot Generator
Creates distribution plots from CSV files or SQLite databases.
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

def create_distribution_plot(data_source, column_name=None, output_file="distribution_output.png"):
    """Create distribution plot from data."""
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
                print("[ERROR] No numeric columns found for distribution plot")
                print(f"[INFO] Available columns: {list(df.columns)}")
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        data = df[column_name].dropna()
        
        # Histogram
        ax1.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title(f'Histogram of {column_name}', fontweight='bold')
        ax1.set_xlabel(column_name)
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax2.set_title(f'Box Plot of {column_name}', fontweight='bold')
        ax2.set_ylabel(column_name)
        ax2.grid(True, alpha=0.3)
        
        # KDE plot
        ax3.hist(data, bins=20, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.plot(data.sort_values(), np.linspace(0, 1, len(data)), 'r-', linewidth=2, label='CDF')
        ax3.set_title(f'Distribution of {column_name}', fontweight='bold')
        ax3.set_xlabel(column_name)
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title(f'Q-Q Plot of {column_name}', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        fig.suptitle(f'Distribution Analysis: {column_name}\nMean: {mean_val:.2f}, Std: {std_val:.2f}, Median: {median_val:.2f}', 
                    fontsize=14, fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Distribution plot saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create distribution plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--column', help='Column name to plot (default: first numeric column)')
    parser.add_argument('--output', default='distribution_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_distribution_plot(args.data_source, args.column, args.output)

if __name__ == "__main__":
    main() 