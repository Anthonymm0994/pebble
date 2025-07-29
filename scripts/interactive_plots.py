#!/usr/bin/env python3
"""
Interactive Plots Generator
==========================

A comprehensive tool for creating interactive-like visualizations using matplotlib
and seaborn, including polar plots and timeline plots with 24-hour format support.

Features:
- Interactive-style polar plots
- Timeline visualizations with 24-hour format
- Real-time data exploration capabilities
- Professional styling
- Export capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.collections import PatchCollection
import sqlite3
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime, timedelta
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class InteractivePlots:
    """
    Generate interactive-style visualizations including polar plots and timelines.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the interactive plots generator.
        
        Args:
            data_source: Path to CSV file or database file
        """
        self.data_source = data_source
        self.df = None
        self.plots = {}
        
    def load_data(self):
        """Load data from CSV file or database."""
        print(f"[DATA] Loading data from: {self.data_source}")
        
        try:
            if self.data_source.endswith('.csv'):
                self.df = pd.read_csv(self.data_source)
            else:
                # Assume it's a database file
                conn = sqlite3.connect(self.data_source)
                # Get first table
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if tables:
                    first_table = tables[0][0]
                    self.df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                    print(f"[INFO] Loaded table: {first_table}")
                else:
                    raise ValueError("No tables found in database")
                conn.close()
            
            print(f"[OK] Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
            return self.df
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return None
    
    def parse_24h_time(self, time_str: str) -> float:
        """Parse 24-hour time format (HH:MM:SS.mmm) to decimal hours."""
        try:
            # Handle the format like "16:07:34.053"
            time_parts = time_str.strip().split(':')
            
            # Check if this is a time format (has at least 2 parts and first part is < 24)
            if len(time_parts) >= 2:
                hours = int(time_parts[0])
                if hours < 24:  # This looks like a time format
                    minutes = int(time_parts[1])
                    if len(time_parts) >= 3:
                        seconds = float(time_parts[2])
                    else:
                        seconds = 0
                    
                    # Convert to decimal hours
                    decimal_hours = hours + minutes/60 + seconds/3600
                    return decimal_hours
                else:
                    # This might be a date format, try to extract time component
                    print(f"[INFO] Detected date format: {time_str}, skipping time parsing")
                    return 0.0
            else:
                print(f"[WARNING] Could not parse time format: {time_str}")
                return 0.0
                
        except Exception as e:
            print(f"[WARNING] Could not parse time: {time_str}, error: {e}")
            return 0.0
    
    def create_polar_plot(self, column: str, title: str = None):
        """Create an interactive-style polar plot for a numeric column."""
        print(f"\n[POLAR] Creating polar plot for: {column}")
        
        if column not in self.df.columns:
            print(f"[ERROR] Column {column} not found in dataset")
            return None
        
        data = self.df[column].dropna()
        if len(data) == 0:
            print(f"[WARNING] No data available for column: {column}")
            return None
        
        # Create polar coordinates
        angles = np.linspace(0, 2*np.pi, len(data), endpoint=False)
        radii = data.values
        
        # Create the plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        
        # Plot the data points
        scatter = ax.scatter(angles, radii, c=radii, s=100, cmap='viridis', alpha=0.7, edgecolors='white', linewidth=1)
        
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(f'{column} Values', rotation=270, labelpad=20)
        
        # Customize the plot
        ax.set_title(title or f"Polar Plot: {column}", fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add angle labels
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
        
        # Add radius labels
        max_radius = max(radii)
        ax.set_yticks(np.linspace(0, max_radius, 5))
        ax.set_yticklabels([f'{r:.1f}' for r in np.linspace(0, max_radius, 5)])
        
        plt.tight_layout()
        return fig
    
    def create_timeline_plot(self, time_column: str, value_column: str, title: str = None):
        """Create an interactive-style timeline plot with 24-hour format."""
        print(f"\n[TIMELINE] Creating timeline plot: {time_column} vs {value_column}")
        
        if time_column not in self.df.columns or value_column not in self.df.columns:
            print(f"[ERROR] Columns {time_column} or {value_column} not found")
            return None
        
        # Parse time column
        time_data = []
        value_data = []
        
        for idx, row in self.df.iterrows():
            time_str = str(row[time_column])
            value = row[value_column]
            
            if pd.notna(value) and time_str.strip():
                decimal_hours = self.parse_24h_time(time_str)
                if decimal_hours > 0:
                    time_data.append(decimal_hours)
                    value_data.append(value)
        
        if len(time_data) == 0:
            print(f"[WARNING] No valid time data found")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the timeline
        ax.plot(time_data, value_data, 'o-', linewidth=2, markersize=6, 
               color='steelblue', alpha=0.8, label=f'{value_column} over time')
        
        # Add trend line
        z = np.polyfit(time_data, value_data, 1)
        p = np.poly1d(z)
        ax.plot(time_data, p(time_data), "--", color='red', alpha=0.7, 
               linewidth=2, label='Trend line')
        
        # Customize the plot
        ax.set_title(title or f"Timeline: {value_column} over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (24-hour format)", fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis to show hours
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
        
        plt.tight_layout()
        return fig
    
    def create_24h_heatmap(self, time_column: str, category_column: str, value_column: str):
        """Create a 24-hour heatmap showing activity patterns."""
        print(f"\n[HEATMAP] Creating 24-hour heatmap")
        
        if not all(col in self.df.columns for col in [time_column, category_column, value_column]):
            print(f"[ERROR] Required columns not found")
            return None
        
        # Parse times and create heatmap data
        heatmap_data = {}
        
        for idx, row in self.df.iterrows():
            time_str = str(row[time_column])
            category = str(row[category_column])
            value = row[value_column]
            
            if pd.notna(value) and time_str.strip():
                hour = int(self.parse_24h_time(time_str))
                if hour < 24:
                    if category not in heatmap_data:
                        heatmap_data[category] = {}
                    if hour not in heatmap_data[category]:
                        heatmap_data[category][hour] = []
                    heatmap_data[category][hour].append(value)
        
        if not heatmap_data:
            print(f"[WARNING] No valid data for heatmap")
            return None
        
        # Prepare data for plotting
        categories = list(heatmap_data.keys())
        hours = list(range(24))
        
        # Calculate average values for each hour-category combination
        heatmap_matrix = []
        for category in categories:
            row = []
            for hour in hours:
                if hour in heatmap_data[category]:
                    avg_value = np.mean(heatmap_data[category][hour])
                else:
                    avg_value = 0
                row.append(avg_value)
            heatmap_matrix.append(row)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap using seaborn
        sns.heatmap(heatmap_matrix, 
                   xticklabels=[f'{h:02d}:00' for h in hours],
                   yticklabels=categories,
                   annot=True, fmt='.1f', cmap='viridis',
                   cbar_kws={'label': f'Average {value_column}'})
        
        ax.set_title("24-Hour Activity Heatmap", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
        ax.set_ylabel("Category", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_radar_plot(self, columns: List[str], title: str = "Radar Plot Analysis"):
        """Create a radar/spider plot for multiple numeric columns."""
        print(f"\n[RADAR] Creating radar plot for multiple columns")
        
        if len(columns) < 3:
            print(f"[WARNING] Need at least 3 columns for radar plot")
            return None
        
        # Calculate statistics for each column
        stats = {}
        for col in columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats[col] = {
                        'mean': data.mean(),
                        'std': data.std(),
                        'max': data.max(),
                        'min': data.min()
                    }
        
        if len(stats) < 3:
            print(f"[WARNING] Not enough valid columns for radar plot")
            return None
        
        # Prepare data for radar plot
        categories = list(stats.keys())
        values = [stats[cat]['mean'] for cat in categories]
        
        # Normalize values to 0-1 range
        min_val = min(values)
        max_val = max(values)
        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Create the radar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        
        # Calculate angles
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        normalized_values += normalized_values[:1]
        
        # Plot the radar
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, normalized_values, alpha=0.25, color='steelblue')
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set the y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self):
        """Create a comprehensive interactive-style dashboard."""
        print(f"\n[DASHBOARD] Creating interactive dashboard")
        
        # Create output directory
        os.makedirs('../outputs/interactive_outputs', exist_ok=True)
        
        plots = []
        
        # Find numeric columns for polar plots
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Create polar plot for first numeric column
            polar_plot = self.create_polar_plot(numeric_cols[0])
            if polar_plot:
                polar_plot.savefig('../outputs/interactive_outputs/polar_plot.png', dpi=300, bbox_inches='tight')
                plt.close(polar_plot)
                plots.append(f"Polar plot saved")
        
        # Find time-related columns
        time_patterns = ['time', 'timestamp', 'date', 'created', 'updated']
        time_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in time_patterns):
                time_cols.append(col)
        
        # Create timeline plots
        if len(time_cols) > 0 and len(numeric_cols) > 0:
            time_col = time_cols[0]
            value_col = numeric_cols[0]
            
            timeline_plot = self.create_timeline_plot(time_col, value_col)
            if timeline_plot:
                timeline_plot.savefig('../outputs/interactive_outputs/timeline_plot.png', dpi=300, bbox_inches='tight')
                plt.close(timeline_plot)
                plots.append(f"Timeline plot saved")
            
            # Create heatmap if we have categorical data
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                heatmap = self.create_24h_heatmap(time_col, categorical_cols[0], value_col)
                if heatmap:
                    heatmap.savefig('../outputs/interactive_outputs/24h_heatmap.png', dpi=300, bbox_inches='tight')
                    plt.close(heatmap)
                    plots.append(f"24-hour heatmap saved")
        
        # Create radar plot if we have multiple numeric columns
        if len(numeric_cols) >= 3:
            radar_plot = self.create_radar_plot(numeric_cols[:5])  # Use up to 5 columns
            if radar_plot:
                radar_plot.savefig('../outputs/interactive_outputs/radar_plot.png', dpi=300, bbox_inches='tight')
                plt.close(radar_plot)
                plots.append(f"Radar plot saved")
        
        print(f"[OK] Interactive dashboard created with {len(plots)} plots")
        return plots
    
    def create_advanced_polar_analysis(self):
        """Create advanced polar analysis with multiple variables."""
        print(f"\n[ADVANCED] Creating advanced polar analysis")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
        
        if len(numeric_cols) < 2:
            print(f"[WARNING] Need at least 2 numeric columns for advanced analysis")
            return None
        
        plots = []
        
        for i, col in enumerate(numeric_cols):
            polar_plot = self.create_polar_plot(col, f"Polar Analysis: {col}")
            if polar_plot:
                polar_plot.savefig(f'../outputs/interactive_outputs/polar_analysis_{col}.png', dpi=300, bbox_inches='tight')
                plt.close(polar_plot)
                plots.append(f"Polar analysis for {col} saved")
        
        print(f"[OK] Advanced polar analysis created with {len(plots)} plots")
        return plots
    
    def create_timeline_analysis(self):
        """Create comprehensive timeline analysis."""
        print(f"\n[TIMELINE] Creating comprehensive timeline analysis")
        
        # Find time columns
        time_patterns = ['time', 'timestamp', 'date', 'created', 'updated']
        time_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in time_patterns):
                time_cols.append(col)
        
        if len(time_cols) == 0:
            print(f"[WARNING] No time columns found")
            return None
        
        time_col = time_cols[0]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(f"[WARNING] No numeric columns found for timeline analysis")
            return None
        
        plots = []
        
        # Create timeline plots for each numeric column
        for value_col in numeric_cols[:3]:  # Limit to 3 columns
            timeline_plot = self.create_timeline_plot(time_col, value_col, f"Timeline: {value_col}")
            if timeline_plot:
                timeline_plot.savefig(f'../outputs/interactive_outputs/timeline_{value_col}.png', dpi=300, bbox_inches='tight')
                plt.close(timeline_plot)
                plots.append(f"Timeline for {value_col} saved")
        
        print(f"[OK] Timeline analysis created with {len(plots)} plots")
        return plots
    
    def create_24h_pattern_analysis(self):
        """Create 24-hour pattern analysis."""
        print(f"\n[PATTERN] Creating 24-hour pattern analysis")
        
        # Find time and categorical columns
        time_patterns = ['time', 'timestamp', 'date', 'created', 'updated']
        time_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in time_patterns):
                time_cols.append(col)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(time_cols) == 0 or len(categorical_cols) == 0 or len(numeric_cols) == 0:
            print(f"[WARNING] Need time, categorical, and numeric columns for pattern analysis")
            return None
        
        time_col = time_cols[0]
        category_col = categorical_cols[0]
        value_col = numeric_cols[0]
        
        # Create heatmap
        heatmap = self.create_24h_heatmap(time_col, category_col, value_col)
        
        if heatmap:
            heatmap.savefig('../outputs/interactive_outputs/24h_pattern_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(heatmap)
            print(f"[OK] 24-hour pattern analysis saved")
        
        return heatmap
    
    def run_comprehensive_analysis(self):
        """Run comprehensive interactive analysis."""
        print(f"\n[START] Starting comprehensive interactive analysis")
        
        # Load data
        data = self.load_data()
        if data is None:
            return
        
        # Create all types of visualizations
        print(f"\n[PLOTS] Creating interactive-style visualizations...")
        
        # 1. Interactive Dashboard
        self.create_interactive_dashboard()
        
        # 2. Advanced Polar Analysis
        self.create_advanced_polar_analysis()
        
        # 3. Timeline Analysis
        self.create_timeline_analysis()
        
        # 4. 24-Hour Pattern Analysis
        self.create_24h_pattern_analysis()
        
        print(f"\n[OK] Comprehensive interactive analysis complete!")
        print(f"[DATA] Check '../outputs/interactive_outputs/' directory for visualization files")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Interactive plots generator closed")


def main():
    """Main function to run interactive analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate interactive-style visualizations')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    
    args = parser.parse_args()
    
    # Create interactive generator
    generator = InteractivePlots(args.data_source)
    
    try:
        # Run comprehensive analysis
        generator.run_comprehensive_analysis()
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.close()


if __name__ == "__main__":
    main() 