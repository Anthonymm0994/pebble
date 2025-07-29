#!/usr/bin/env python3
"""
Plot Permutation Builder
========================

An intuitive system for building various plots with permutations of WHERE clause predicates.
Focuses on histograms and polar plots with shared axis labels and clear legends.

Features:
- Easy plot building with intuitive controls
- WHERE clause predicate permutations
- Comparative histograms with shared axes
- Polar plots with clear legends
- Professional visualizations
- SQLite database support
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, permutations
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
from pathlib import Path
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class PlotPermutationBuilder:
    """
    Build various plots with permutations of WHERE clause predicates.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize the plot permutation builder.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self.conn = None
        self.df = None
        self.table_name = None
        self.plot_results = {}
        
    def connect_database(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.database_path)
            print(f"[CONNECT] Connected to database: {self.database_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            return False
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables in the database."""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    
    def load_table_data(self, table_name: str = None):
        """Load data from specified table or first available table."""
        if not table_name:
            tables = self.get_available_tables()
            if tables:
                table_name = tables[0]
            else:
                print("[ERROR] No tables found in database")
                return False
        
        try:
            self.df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
            self.table_name = table_name
            print(f"[OK] Loaded table '{table_name}': {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load table: {e}")
            return False
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns for plotting."""
        if self.df is None:
            return []
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns for filtering."""
        if self.df is None:
            return []
        return self.df.select_dtypes(include=['object']).columns.tolist()
    
    def generate_where_permutations(self, column: str, values: List[str], max_combinations: int = 5) -> List[Dict]:
        """Generate permutations of WHERE clauses for a column."""
        permutations_list = []
        
        # Generate combinations of values
        for r in range(1, min(len(values) + 1, max_combinations + 1)):
            for combo in combinations(values, r):
                where_conditions = [f"{column} = '{val}'" for val in combo]
                permutations_list.append({
                    'conditions': where_conditions,
                    'description': f"{column} in {list(combo)}",
                    'values': list(combo)
                })
        
        return permutations_list
    
    def execute_query_with_conditions(self, conditions: List[str]) -> pd.DataFrame:
        """Execute query with WHERE conditions."""
        if not conditions:
            return self.df
        
        where_clause = ' AND '.join(conditions)
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        
        try:
            result = pd.read_sql_query(query, self.conn)
            return result
        except Exception as e:
            print(f"[WARNING] Query failed: {e}")
            return pd.DataFrame()
    
    def create_comparative_histograms(self, column: str, permutations: List[Dict], 
                                    output_file: str = "comparative_histograms.png"):
        """Create comparative histograms with shared axis labels."""
        print(f"[HISTOGRAM] Creating comparative histograms for {column}")
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Filter out empty results
        valid_permutations = []
        for perm in permutations:
            result_df = self.execute_query_with_conditions(perm['conditions'])
            if not result_df.empty and column in result_df.columns:
                valid_permutations.append({
                    **perm,
                    'data': result_df[column].dropna()
                })
        
        if not valid_permutations:
            print(f"[WARNING] No valid data found for comparative histograms")
            return
        
        # Create comparative plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overlapping histograms
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_permutations)))
        
        for i, perm in enumerate(valid_permutations):
            data = perm['data']
            if len(data) > 0:
                ax1.hist(data, bins=20, alpha=0.7, density=True, 
                        color=colors[i], label=perm['description'], edgecolor='black')
        
        ax1.set_title(f'Comparative Histograms: {column}', fontweight='bold', fontsize=14)
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Box plots side by side
        box_data = [perm['data'] for perm in valid_permutations if len(perm['data']) > 0]
        box_labels = [perm['description'] for perm in valid_permutations if len(perm['data']) > 0]
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title(f'Box Plot Comparison: {column}', fontweight='bold', fontsize=14)
            ax2.set_ylabel(column, fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Comparative histograms saved as: {output_path}")
        plt.close()
    
    def create_comparative_polar_plots(self, angle_col: str, radius_col: str, permutations: List[Dict],
                                     output_file: str = "comparative_polar_plots.png"):
        """Create comparative polar plots with clear legends."""
        print(f"[POLAR] Creating comparative polar plots for {angle_col} vs {radius_col}")
        
        # Create output directory
        os.makedirs('../outputs/plot_outputs', exist_ok=True)
        output_path = f'../outputs/plot_outputs/{output_file}'
        
        # Filter out empty results
        valid_permutations = []
        for perm in permutations:
            result_df = self.execute_query_with_conditions(perm['conditions'])
            if not result_df.empty and angle_col in result_df.columns and radius_col in result_df.columns:
                valid_permutations.append({
                    **perm,
                    'data': result_df[[angle_col, radius_col]].dropna()
                })
        
        if not valid_permutations:
            print(f"[WARNING] No valid data found for comparative polar plots")
            return
        
        # Create comparative plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_permutations)))
        
        # Individual polar plots
        for i, perm in enumerate(valid_permutations):
            data = perm['data']
            if len(data) > 0:
                angles = data[angle_col].values
                radii = data[radius_col].values
                
                # Create polar subplot
                if i < 4:  # Show first 4 permutations
                    ax = [ax1, ax2, ax3, ax4][i]
                    ax.scatter(angles, radii, alpha=0.7, s=50, color=colors[i], 
                             label=perm['description'])
                    ax.set_title(f'Polar Plot {i+1}: {perm["description"]}', 
                               fontweight='bold', fontsize=12)
                    ax.set_xlabel(angle_col, fontsize=10)
                    ax.set_ylabel(radius_col, fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
        
        # Combined polar plot
        ax_combined = plt.subplot(2, 2, 4, projection='polar')
        for i, perm in enumerate(valid_permutations):
            data = perm['data']
            if len(data) > 0:
                angles = data[angle_col].values
                radii = data[radius_col].values
                ax_combined.scatter(angles, radii, alpha=0.7, s=50, color=colors[i],
                                 label=perm['description'])
        
        ax_combined.set_title('Combined Polar Plot', fontweight='bold', fontsize=12)
        ax_combined.legend(fontsize=8, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Comparative polar plots saved as: {output_path}")
        plt.close()
    
    def create_histogram_permutations(self, column: str, filter_column: str = None, 
                                    max_permutations: int = 5):
        """Create histogram permutations with different WHERE clause predicates."""
        print(f"[PERMUTATION] Creating histogram permutations for {column}")
        
        if filter_column is None:
            # Use first categorical column
            categorical_cols = self.get_categorical_columns()
            if categorical_cols:
                filter_column = categorical_cols[0]
            else:
                print(f"[WARNING] No categorical columns found for filtering")
                return
        
        # Get unique values for filtering
        unique_values = self.df[filter_column].dropna().unique()
        if len(unique_values) == 0:
            print(f"[WARNING] No values found in {filter_column}")
            return
        
        # Generate permutations
        permutations = self.generate_where_permutations(filter_column, unique_values, max_permutations)
        
        print(f"[INFO] Generated {len(permutations)} permutations for {filter_column}")
        
        # Create comparative histograms
        self.create_comparative_histograms(column, permutations, 
                                         f"histogram_permutations_{column}_{filter_column}.png")
        
        return permutations
    
    def create_polar_permutations(self, angle_col: str, radius_col: str, filter_column: str = None,
                                max_permutations: int = 5):
        """Create polar plot permutations with different WHERE clause predicates."""
        print(f"[PERMUTATION] Creating polar plot permutations for {angle_col} vs {radius_col}")
        
        if filter_column is None:
            # Use first categorical column
            categorical_cols = self.get_categorical_columns()
            if categorical_cols:
                filter_column = categorical_cols[0]
            else:
                print(f"[WARNING] No categorical columns found for filtering")
                return
        
        # Get unique values for filtering
        unique_values = self.df[filter_column].dropna().unique()
        if len(unique_values) == 0:
            print(f"[WARNING] No values found in {filter_column}")
            return
        
        # Generate permutations
        permutations = self.generate_where_permutations(filter_column, unique_values, max_permutations)
        
        print(f"[INFO] Generated {len(permutations)} permutations for {filter_column}")
        
        # Create comparative polar plots
        self.create_comparative_polar_plots(angle_col, radius_col, permutations,
                                          f"polar_permutations_{angle_col}_{radius_col}_{filter_column}.png")
        
        return permutations
    
    def create_comprehensive_analysis(self, numeric_columns: List[str] = None, 
                                   categorical_columns: List[str] = None):
        """Create comprehensive analysis with multiple plot types."""
        print(f"[ANALYSIS] Creating comprehensive plot analysis")
        
        if numeric_columns is None:
            numeric_columns = self.get_numeric_columns()
        
        if categorical_columns is None:
            categorical_columns = self.get_categorical_columns()
        
        print(f"[INFO] Using numeric columns: {numeric_columns}")
        print(f"[INFO] Using categorical columns: {categorical_columns}")
        
        # Create histogram permutations for each numeric column
        for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
            for cat_col in categorical_columns[:2]:  # Limit to first 2 categorical columns
                print(f"\n[PROCESSING] Creating histograms for {num_col} filtered by {cat_col}")
                self.create_histogram_permutations(num_col, cat_col, max_permutations=3)
        
        # Create polar plot permutations if we have at least 2 numeric columns
        if len(numeric_columns) >= 2:
            angle_col = numeric_columns[0]
            radius_col = numeric_columns[1]
            
            for cat_col in categorical_columns[:2]:
                print(f"\n[PROCESSING] Creating polar plots for {angle_col} vs {radius_col} filtered by {cat_col}")
                self.create_polar_permutations(angle_col, radius_col, cat_col, max_permutations=3)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run plot permutation builder."""
    parser = argparse.ArgumentParser(description='Build plots with WHERE clause permutations')
    parser.add_argument('database_path', help='Path to SQLite database file')
    parser.add_argument('--table', help='Table name to use (default: first available)')
    parser.add_argument('--histogram', help='Column for histogram analysis')
    parser.add_argument('--filter', help='Column to use for filtering permutations')
    parser.add_argument('--polar-angle', help='Angle column for polar plots')
    parser.add_argument('--polar-radius', help='Radius column for polar plots')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive analysis')
    parser.add_argument('--max-permutations', type=int, default=5, help='Maximum number of permutations')
    
    args = parser.parse_args()
    
    # Create builder
    builder = PlotPermutationBuilder(args.database_path)
    
    try:
        # Connect to database
        if not builder.connect_database():
            return
        
        # Load table data
        if not builder.load_table_data(args.table):
            return
        
        # Show available columns
        numeric_cols = builder.get_numeric_columns()
        categorical_cols = builder.get_categorical_columns()
        
        print(f"\n[INFO] Available numeric columns: {numeric_cols}")
        print(f"[INFO] Available categorical columns: {categorical_cols}")
        
        if args.comprehensive:
            # Run comprehensive analysis
            builder.create_comprehensive_analysis()
        
        elif args.histogram:
            # Create histogram permutations
            builder.create_histogram_permutations(args.histogram, args.filter, args.max_permutations)
        
        elif args.polar_angle and args.polar_radius:
            # Create polar plot permutations
            builder.create_polar_permutations(args.polar_angle, args.polar_radius, args.filter, args.max_permutations)
        
        else:
            # Default: comprehensive analysis
            print("[INFO] Running comprehensive analysis (use --comprehensive for explicit mode)")
            builder.create_comprehensive_analysis()
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        builder.close()


if __name__ == "__main__":
    main() 