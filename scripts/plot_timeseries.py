#!/usr/bin/env python3
"""
Time Series Plot Generator
Creates time series plots from CSV files or SQLite databases.
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

def create_timeseries_plot(data_source, time_column=None, value_column=None, output_file="timeseries_output.png"):
    """Create time series plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # If no columns specified, try to find suitable columns
        if time_column is None:
            # Look for time-related columns
            time_cols = [col for col in df.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp'])]
            if len(time_cols) > 0:
                time_column = time_cols[0]
            else:
                # Use index as time
                time_column = 'index'
                df = df.reset_index()
        
        if value_column is None:
            # Look for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
            else:
                print("[ERROR] No numeric columns found for time series plot")
                print(f"[INFO] Available columns: {list(df.columns)}")
                return
        
        # Check if columns exist
        if time_column not in df.columns:
            print(f"[ERROR] Column '{time_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        if value_column not in df.columns:
            print(f"[ERROR] Column '{value_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Try to parse time column
        if time_column != 'index':
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except:
                print(f"[WARNING] Could not parse {time_column} as datetime, using as-is")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Create time series plot
        if time_column == 'index':
            plt.plot(df.index, df[value_column], marker='o', linestyle='-', linewidth=2, markersize=4)
            plt.xlabel('Index', fontsize=12)
        else:
            plt.plot(df[time_column], df[value_column], marker='o', linestyle='-', linewidth=2, markersize=4)
            plt.xlabel(time_column, fontsize=12)
        
        # Customize plot
        plt.title(f"Time Series: {value_column}", fontsize=14, fontweight='bold')
        plt.ylabel(value_column, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if time_column != 'index':
            plt.xticks(rotation=45, ha='right')
        
        # Add statistics
        mean_val = df[value_column].mean()
        plt.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Time series plot saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create time series plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--time', help='Time column name (default: auto-detect)')
    parser.add_argument('--value', help='Value column name (default: first numeric column)')
    parser.add_argument('--output', default='timeseries_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_timeseries_plot(args.data_source, args.time, args.value, args.output)

if __name__ == "__main__":
    main() 