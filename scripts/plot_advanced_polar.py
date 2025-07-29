#!/usr/bin/env python3
"""
Advanced Polar Plot Generator
Creates advanced polar plots from CSV files or SQLite databases.
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

def create_advanced_polar_plot(data_source, angle_column=None, radius_column=None, output_file="advanced_polar_output.png"):
    """Create advanced polar plot from data."""
    try:
        # Load data
        df = load_data(data_source)
        if df is None:
            return
        
        # If no columns specified, use first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if angle_column is None and len(numeric_cols) >= 2:
            angle_column = numeric_cols[0]
            radius_column = numeric_cols[1]
        elif angle_column is None:
            print("[ERROR] Need at least 2 numeric columns for polar plot")
            print(f"[INFO] Available numeric columns: {list(numeric_cols)}")
            return
        
        # Check if columns exist
        if angle_column not in df.columns:
            print(f"[ERROR] Column '{angle_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        if radius_column not in df.columns:
            print(f"[ERROR] Column '{radius_column}' not found in data.")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), 
                                                      subplot_kw={'projection': 'polar'})
        
        angles = df[angle_column].values
        radii = df[radius_column].values
        
        # Normalize angles to 0-2π range
        angles = np.mod(angles, 2 * np.pi)
        
        # Basic polar scatter
        ax1.scatter(angles, radii, alpha=0.6, s=50, color='steelblue')
        ax1.set_title(f'Basic Polar Plot: {angle_column} vs {radius_column}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Polar histogram
        ax2.hist2d(angles, radii, bins=20, cmap='viridis')
        ax2.set_title(f'Polar Histogram: {angle_column} vs {radius_column}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Polar line plot
        sorted_indices = np.argsort(angles)
        ax3.plot(angles[sorted_indices], radii[sorted_indices], 'r-', linewidth=2, alpha=0.7)
        ax3.scatter(angles, radii, alpha=0.6, s=30, color='blue')
        ax3.set_title(f'Polar Line Plot: {angle_column} vs {radius_column}', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Polar area plot
        ax4.fill(angles[sorted_indices], radii[sorted_indices], alpha=0.3, color='green')
        ax4.plot(angles[sorted_indices], radii[sorted_indices], 'g-', linewidth=2)
        ax4.set_title(f'Polar Area Plot: {angle_column} vs {radius_column}', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_radius = np.mean(radii)
        max_radius = np.max(radii)
        
        fig.suptitle(f'Advanced Polar Analysis: {angle_column} vs {radius_column}\nMean Radius: {mean_radius:.2f}, Max Radius: {max_radius:.2f}', 
                    fontsize=14, fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Advanced polar plot saved as: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create advanced polar plot from data')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--angle', help='Angle column name (default: first numeric column)')
    parser.add_argument('--radius', help='Radius column name (default: second numeric column)')
    parser.add_argument('--output', default='advanced_polar_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    create_advanced_polar_plot(args.data_source, args.angle, args.radius, args.output)

if __name__ == "__main__":
    main() 