#!/usr/bin/env python3
"""
Comprehensive Histogram Analysis
===============================

A comprehensive tool for creating various types of histograms
to analyze data distributions and patterns.

This script provides multiple histogram types:
- Basic histograms
- Overlay histograms (comparing two datasets)
- Cumulative histograms
- Log-scale histograms
- Binned histograms
- Statistical summary histograms
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HistogramAnalyzer:
    """
    Comprehensive histogram analysis tool.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the histogram analyzer.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.df_source = None
        self.df_derived = None
        
    def connect(self) -> bool:
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"[OK] Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error connecting to database: {e}")
            return False
    
    def load_datasets(self, source_table: str, derived_table: str):
        """Load source and derived datasets."""
        print(f"\n[DATA] Loading datasets: {source_table} -> {derived_table}")
        
        self.df_source = pd.read_sql_query(f"SELECT * FROM {source_table}", self.conn)
        self.df_derived = pd.read_sql_query(f"SELECT * FROM {derived_table}", self.conn)
        
        print(f"[OK] Loaded source: {len(self.df_source)} rows, {len(self.df_source.columns)} columns")
        print(f"[OK] Loaded derived: {len(self.df_derived)} rows, {len(self.df_derived.columns)} columns")
        
        self.source_table = source_table
        self.derived_table = derived_table
    
    def create_basic_histograms(self, column_name: str, bins: int = 30):
        """Create basic histograms for a column."""
        print(f"\n[CHART] Creating basic histograms for: {column_name}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Basic Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        # Check if column exists in both datasets
        source_exists = column_name in self.df_source.columns
        derived_exists = column_name in self.df_derived.columns
        
        if source_exists:
            source_data = self.df_source[column_name].dropna()
            if len(source_data) > 0:
                # Source histogram
                axes[0, 0].hist(source_data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 0].set_title(f'Source Dataset: {column_name}')
                axes[0, 0].set_xlabel(column_name)
                axes[0, 0].set_ylabel('Frequency')
                
                # Source statistics
                axes[0, 1].text(0.1, 0.9, f'Count: {len(source_data)}', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.8, f'Mean: {source_data.mean():.3f}', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.7, f'Std: {source_data.std():.3f}', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.6, f'Min: {source_data.min():.3f}', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.5, f'Max: {source_data.max():.3f}', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Source Statistics')
                axes[0, 1].axis('off')
        
        if derived_exists:
            derived_data = self.df_derived[column_name].dropna()
            if len(derived_data) > 0:
                # Derived histogram
                axes[1, 0].hist(derived_data, bins=bins, alpha=0.7, color='orange', edgecolor='black')
                axes[1, 0].set_title(f'Derived Dataset: {column_name}')
                axes[1, 0].set_xlabel(column_name)
                axes[1, 0].set_ylabel('Frequency')
                
                # Derived statistics
                axes[1, 1].text(0.1, 0.9, f'Count: {len(derived_data)}', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].text(0.1, 0.8, f'Mean: {derived_data.mean():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].text(0.1, 0.7, f'Std: {derived_data.std():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].text(0.1, 0.6, f'Min: {derived_data.min():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].text(0.1, 0.5, f'Max: {derived_data.max():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Derived Statistics')
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'basic_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Basic histograms saved as 'basic_histogram_{column_name}.png'")
    
    def create_overlay_histograms(self, column_name: str, bins: int = 30):
        """Create overlay histograms comparing source and derived datasets."""
        print(f"\n[CHART] Creating overlay histograms for: {column_name}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Overlay Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_exists = column_name in self.df_source.columns
        derived_exists = column_name in self.df_derived.columns
        
        if source_exists and derived_exists:
            source_data = self.df_source[column_name].dropna()
            derived_data = self.df_derived[column_name].dropna()
            
            if len(source_data) > 0 and len(derived_data) > 0:
                # Overlay histogram
                axes[0].hist(source_data, bins=bins, alpha=0.7, label='Source', color='blue')
                axes[0].hist(derived_data, bins=bins, alpha=0.7, label='Derived', color='orange')
                axes[0].set_title(f'Overlay Histogram: {column_name}')
                axes[0].set_xlabel(column_name)
                axes[0].set_ylabel('Frequency')
                axes[0].legend()
                
                # Distribution comparison
                axes[1].hist(source_data, bins=bins, alpha=0.7, density=True, label='Source', color='blue')
                axes[1].hist(derived_data, bins=bins, alpha=0.7, density=True, label='Derived', color='orange')
                axes[1].set_title(f'Normalized Distribution: {column_name}')
                axes[1].set_xlabel(column_name)
                axes[1].set_ylabel('Density')
                axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'overlay_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Overlay histograms saved as 'overlay_histogram_{column_name}.png'")
    
    def create_cumulative_histograms(self, column_name: str, bins: int = 30):
        """Create cumulative histograms."""
        print(f"\n[CHART] Creating cumulative histograms for: {column_name}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Cumulative Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_exists = column_name in self.df_source.columns
        derived_exists = column_name in self.df_derived.columns
        
        if source_exists:
            source_data = self.df_source[column_name].dropna()
            if len(source_data) > 0:
                axes[0].hist(source_data, bins=bins, cumulative=True, alpha=0.7, color='blue', edgecolor='black')
                axes[0].set_title(f'Cumulative Histogram - Source: {column_name}')
                axes[0].set_xlabel(column_name)
                axes[0].set_ylabel('Cumulative Frequency')
        
        if derived_exists:
            derived_data = self.df_derived[column_name].dropna()
            if len(derived_data) > 0:
                axes[1].hist(derived_data, bins=bins, cumulative=True, alpha=0.7, color='orange', edgecolor='black')
                axes[1].set_title(f'Cumulative Histogram - Derived: {column_name}')
                axes[1].set_xlabel(column_name)
                axes[1].set_ylabel('Cumulative Frequency')
        
        plt.tight_layout()
        plt.savefig(f'cumulative_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Cumulative histograms saved as 'cumulative_histogram_{column_name}.png'")
    
    def create_log_scale_histograms(self, column_name: str, bins: int = 30):
        """Create log-scale histograms for wide-ranging data."""
        print(f"\n[CHART] Creating log-scale histograms for: {column_name}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Log-Scale Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_exists = column_name in self.df_source.columns
        derived_exists = column_name in self.df_derived.columns
        
        if source_exists:
            source_data = self.df_source[column_name].dropna()
            if len(source_data) > 0 and source_data.min() > 0:
                axes[0].hist(source_data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
                axes[0].set_yscale('log')
                axes[0].set_title(f'Log-Scale Histogram - Source: {column_name}')
                axes[0].set_xlabel(column_name)
                axes[0].set_ylabel('Frequency (log scale)')
        
        if derived_exists:
            derived_data = self.df_derived[column_name].dropna()
            if len(derived_data) > 0 and derived_data.min() > 0:
                axes[1].hist(derived_data, bins=bins, alpha=0.7, color='orange', edgecolor='black')
                axes[1].set_yscale('log')
                axes[1].set_title(f'Log-Scale Histogram - Derived: {column_name}')
                axes[1].set_xlabel(column_name)
                axes[1].set_ylabel('Frequency (log scale)')
        
        plt.tight_layout()
        plt.savefig(f'log_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Log-scale histograms saved as 'log_histogram_{column_name}.png'")
    
    def create_binned_histograms(self, column_name: str, num_bins: int = 10):
        """Create histograms with custom binning."""
        print(f"\n[CHART] Creating binned histograms for: {column_name}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Binned Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_exists = column_name in self.df_source.columns
        derived_exists = column_name in self.df_derived.columns
        
        if source_exists:
            source_data = self.df_source[column_name].dropna()
            if len(source_data) > 0:
                # Create custom bins
                bins = np.linspace(source_data.min(), source_data.max(), num_bins + 1)
                axes[0].hist(source_data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
                axes[0].set_title(f'Binned Histogram - Source: {column_name}')
                axes[0].set_xlabel(column_name)
                axes[0].set_ylabel('Frequency')
        
        if derived_exists:
            derived_data = self.df_derived[column_name].dropna()
            if len(derived_data) > 0:
                # Create custom bins
                bins = np.linspace(derived_data.min(), derived_data.max(), num_bins + 1)
                axes[1].hist(derived_data, bins=bins, alpha=0.7, color='orange', edgecolor='black')
                axes[1].set_title(f'Binned Histogram - Derived: {column_name}')
                axes[1].set_xlabel(column_name)
                axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'binned_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Binned histograms saved as 'binned_histogram_{column_name}.png'")
    
    def create_statistical_summary_histograms(self):
        """Create statistical summary histograms for all numeric columns."""
        print(f"\n[CHART] Creating statistical summary histograms")
        
        # Get numeric columns from both datasets
        source_numeric = self.df_source.select_dtypes(include=[np.number]).columns
        derived_numeric = self.df_derived.select_dtypes(include=[np.number]).columns
        
        all_numeric = list(set(source_numeric) | set(derived_numeric))
        
        if len(all_numeric) > 0:
            # Create subplots
            cols = min(3, len(all_numeric))
            rows = (len(all_numeric) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            fig.suptitle('Statistical Summary Histograms', fontsize=16, fontweight='bold')
            
            # Flatten axes if needed
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, col in enumerate(all_numeric):
                row = i // cols
                col_idx = i % cols
                
                if source_numeric is not None and col in source_numeric:
                    source_data = self.df_source[col].dropna()
                    if len(source_data) > 0:
                        axes[row, col_idx].hist(source_data, alpha=0.7, color='blue', label='Source')
                
                if derived_numeric is not None and col in derived_numeric:
                    derived_data = self.df_derived[col].dropna()
                    if len(derived_data) > 0:
                        axes[row, col_idx].hist(derived_data, alpha=0.7, color='orange', label='Derived')
                
                axes[row, col_idx].set_title(col)
                axes[row, col_idx].legend()
            
            # Hide empty subplots
            for i in range(len(all_numeric), rows * cols):
                row = i // cols
                col_idx = i % cols
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('statistical_summary_histograms.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Statistical summary histograms saved as 'statistical_summary_histograms.png'")
    
    def create_timestamp_histograms(self):
        """Create histograms for timestamp columns."""
        print(f"\n[CHART] Creating timestamp histograms")
        
        # Find timestamp columns
        timestamp_patterns = ['time', 'date', 'timestamp', 'created', 'updated', 'message']
        
        source_timestamps = []
        derived_timestamps = []
        
        for col in self.df_source.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in timestamp_patterns):
                source_timestamps.append(col)
        
        for col in self.df_derived.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in timestamp_patterns):
                derived_timestamps.append(col)
        
        all_timestamps = list(set(source_timestamps) | set(derived_timestamps))
        
        if len(all_timestamps) > 0:
            fig, axes = plt.subplots(len(all_timestamps), 1, figsize=(15, 5 * len(all_timestamps)))
            fig.suptitle('Timestamp Distribution Analysis', fontsize=16, fontweight='bold')
            
            if len(all_timestamps) == 1:
                axes = [axes]
            
            for i, col in enumerate(all_timestamps):
                if col in source_timestamps:
                    source_data = self.df_source[col].dropna()
                    if len(source_data) > 0:
                        # Try to parse timestamps
                        try:
                            source_times = pd.to_datetime(source_data)
                            axes[i].hist(source_times, bins=30, alpha=0.7, color='blue', label='Source')
                        except:
                            axes[i].hist(source_data, bins=30, alpha=0.7, color='blue', label='Source')
                
                if col in derived_timestamps:
                    derived_data = self.df_derived[col].dropna()
                    if len(derived_data) > 0:
                        # Try to parse timestamps
                        try:
                            derived_times = pd.to_datetime(derived_data)
                            axes[i].hist(derived_times, bins=30, alpha=0.7, color='orange', label='Derived')
                        except:
                            axes[i].hist(derived_data, bins=30, alpha=0.7, color='orange', label='Derived')
                
                axes[i].set_title(col)
                axes[i].legend()
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('timestamp_histograms.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Timestamp histograms saved as 'timestamp_histograms.png'")
    
    def create_comprehensive_histogram_analysis(self, columns: List[str] = None):
        """Create comprehensive histogram analysis for specified columns."""
        print(f"\n[START] Starting comprehensive histogram analysis")
        
        if columns is None:
            # Get all numeric columns
            source_numeric = self.df_source.select_dtypes(include=[np.number]).columns
            derived_numeric = self.df_derived.select_dtypes(include=[np.number]).columns
            columns = list(set(source_numeric) | set(derived_numeric))
        
        print(f"[DATA] Analyzing {len(columns)} columns: {columns}")
        
        for column in columns:
            try:
                # Create all types of histograms for each column
                self.create_basic_histograms(column)
                self.create_overlay_histograms(column)
                self.create_cumulative_histograms(column)
                self.create_log_scale_histograms(column)
                self.create_binned_histograms(column)
                
            except Exception as e:
                print(f"[ERROR] Error creating histograms for {column}: {e}")
        
        # Create summary histograms
        self.create_statistical_summary_histograms()
        self.create_timestamp_histograms()
        
        print(f"\n[OK] Comprehensive histogram analysis complete!")
        print(f"[DATA] Generated histograms for {len(columns)} columns")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run histogram analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive histogram analysis')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--source', help='Name of the source table')
    parser.add_argument('--derived', help='Name of the derived table')
    parser.add_argument('--columns', nargs='+', help='Specific columns to analyze')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HistogramAnalyzer(args.db_path)
    
    try:
        # Connect to database
        if not analyzer.connect():
            return
        
        # Load datasets
        analyzer.load_datasets(args.source, args.derived)
        
        # Run comprehensive analysis
        analyzer.create_comprehensive_histogram_analysis(args.columns)
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 