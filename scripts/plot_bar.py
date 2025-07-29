#!/usr/bin/env python3
"""
Bar Plot Generator
Creates bar plots from CSV files or SQLite databases.
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

def create_bar_plot(data_source, category_column=None, value_column=None, output_file="bar_output.png"):
    """Create bar plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # If no columns specified, try to find suitable columns
        if category_column is None:
            # Look for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                category_column = categorical_cols[0]
            else:
                print("[ERROR] No categorical columns found for bar plot")
                print(f"[INFO] Available columns: {list(df.columns)}")
                return
        
        if value_column is None:
            # Look for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
            else:
                print("[ERROR] No numeric columns found for bar plot")
                print(f"[INFO] Available columns: {list(df.columns)}")
                return
        
        # Check if columns exist
        if category_column not in df.columns:
            print(f"[ERROR] Column '{category_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        if value_column not in df.columns:
            print(f"[ERROR] Column '{value_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Aggregate data by category
        grouped_data = df.groupby(category_column)[value_column].agg(['mean', 'count']).reset_index()
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean values bar plot
        bars1 = ax1.bar(range(len(grouped_data)), grouped_data['mean'], color='steelblue', alpha=0.7)
        ax1.set_title(f"Mean {value_column} by {category_column}", fontsize=14, fontweight='bold')
        ax1.set_xlabel(category_column, fontsize=12)
        ax1.set_ylabel(f"Mean {value_column}", fontsize=12)
        ax1.set_xticks(range(len(grouped_data)))
        ax1.set_xticklabels(grouped_data[category_column], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars1, grouped_data['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(grouped_data['mean']),
                    f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Count bar plot
        bars2 = ax2.bar(range(len(grouped_data)), grouped_data['count'], color='orange', alpha=0.7)
        ax2.set_title(f"Count by {category_column}", fontsize=14, fontweight='bold')
        ax2.set_xlabel(category_column, fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xticks(range(len(grouped_data)))
        ax2.set_xticklabels(grouped_data[category_column], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count_val in zip(bars2, grouped_data['count']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(grouped_data['count']),
                    f'{count_val}', ha='center', va='bottom', fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Bar plot saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create bar plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--category', help='Category column name (default: first categorical column)')
    parser.add_argument('--value', help='Value column name (default: first numeric column)')
    parser.add_argument('--output', default='bar_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_bar_plot(args.data_source, args.category, args.value, args.output)

if __name__ == "__main__":
    main() 