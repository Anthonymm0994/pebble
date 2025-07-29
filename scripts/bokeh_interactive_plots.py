#!/usr/bin/env python3
"""
Bokeh Interactive Plots Generator
================================

A comprehensive tool for creating interactive visualizations using Bokeh,
including polar plots and timeline plots with 24-hour format support.

Features:
- Interactive polar plots
- Timeline visualizations with 24-hour format
- Real-time data exploration
- Professional styling
- Export capabilities
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, HoverTool, PanTool, WheelZoomTool, ResetTool
from bokeh.models import LinearColorMapper, ColorBar, Legend, LegendItem
from bokeh.models import Range1d, Circle, Line, Text
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Spectral6, Viridis256, Category10
from bokeh.io import curdoc
from bokeh.server.server import Server
import sqlite3
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime, timedelta
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BokehInteractivePlots:
    """
    Generate interactive Bokeh visualizations including polar plots and timelines.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the Bokeh interactive plots generator.
        
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
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])
            
            # Convert to decimal hours
            decimal_hours = hours + minutes/60 + seconds/3600
            return decimal_hours
        except Exception as e:
            print(f"[WARNING] Could not parse time: {time_str}, error: {e}")
            return 0.0
    
    def create_polar_plot(self, column: str, title: str = None):
        """Create an interactive polar plot for a numeric column."""
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
        
        # Create data source
        source = ColumnDataSource(data=dict(
            angle=angles,
            radius=radii,
            value=data.values,
            index=range(len(data))
        ))
        
        # Create polar plot
        p = figure(
            width=600, height=600,
            title=title or f"Polar Plot: {column}",
            x_range=(-max(radii)*1.2, max(radii)*1.2),
            y_range=(-max(radii)*1.2, max(radii)*1.2),
            tools="pan,wheel_zoom,reset,save"
        )
        
        # Add polar grid
        max_radius = max(radii)
        for r in np.linspace(0, max_radius, 5):
            p.circle(0, 0, radius=r, fill_color=None, line_color='gray', line_alpha=0.3)
        
        # Add angle lines
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = max_radius * 1.1 * np.cos(angle)
            y = max_radius * 1.1 * np.sin(angle)
            p.line([0, x], [0, y], line_color='gray', line_alpha=0.3)
        
        # Plot the data points
        p.circle(
            'radius*cos(angle)', 'radius*sin(angle)',
            size=8, fill_color='steelblue', line_color='white',
            source=source, legend_label=f"{column} values"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Index", "@index"),
            ("Value", "@value"),
            ("Angle", "@angle{0.00}"),
            ("Radius", "@radius{0.00}")
        ])
        p.add_tools(hover)
        
        # Style the plot
        p.axis.axis_label = None
        p.axis.major_tick_line_color = None
        p.axis.minor_tick_line_color = None
        p.grid.grid_line_color = None
        p.legend.location = "top_right"
        
        return p
    
    def create_timeline_plot(self, time_column: str, value_column: str, title: str = None):
        """Create an interactive timeline plot with 24-hour format."""
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
        
        # Create data source
        source = ColumnDataSource(data=dict(
            time=time_data,
            value=value_data,
            index=range(len(time_data))
        ))
        
        # Create timeline plot
        p = figure(
            width=800, height=400,
            title=title or f"Timeline: {value_column} over Time",
            x_axis_label="Time (24-hour format)",
            y_axis_label=value_column,
            tools="pan,wheel_zoom,reset,save"
        )
        
        # Plot the timeline
        p.line('time', 'value', line_width=2, line_color='steelblue', source=source)
        p.circle('time', 'value', size=6, fill_color='steelblue', line_color='white', source=source)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Index", "@index"),
            ("Time", "@time{0.00} hours"),
            ("Value", "@value{0.00}")
        ])
        p.add_tools(hover)
        
        # Style the plot
        p.grid.grid_line_alpha = 0.3
        p.legend.location = "top_left"
        
        return p
    
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
        values = []
        for category in categories:
            for hour in hours:
                if hour in heatmap_data[category]:
                    avg_value = np.mean(heatmap_data[category][hour])
                else:
                    avg_value = 0
                values.append(avg_value)
        
        # Create data source
        source = ColumnDataSource(data=dict(
            category=categories * 24,
            hour=hours * len(categories),
            value=values
        ))
        
        # Create heatmap
        p = figure(
            width=800, height=400,
            title="24-Hour Activity Heatmap",
            x_axis_label="Hour of Day",
            y_axis_label="Category",
            tools="pan,wheel_zoom,reset,save"
        )
        
        # Create color mapper
        mapper = linear_cmap('value', Viridis256, 0, max(values))
        
        # Plot heatmap
        p.rect('hour', 'category', width=0.9, height=0.9, source=source,
               fill_color=mapper, line_color='white', line_width=1)
        
        # Add color bar
        color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
        p.add_layout(color_bar, 'right')
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Category", "@category"),
            ("Hour", "@hour"),
            ("Value", "@value{0.00}")
        ])
        p.add_tools(hover)
        
        # Style the plot
        p.xaxis.ticker = list(range(0, 24, 2))
        p.grid.grid_line_alpha = 0.3
        
        return p
    
    def create_interactive_dashboard(self):
        """Create a comprehensive interactive dashboard."""
        print(f"\n[DASHBOARD] Creating interactive dashboard")
        
        # Create output directory
        os.makedirs('../outputs/bokeh_outputs', exist_ok=True)
        
        plots = []
        
        # Find numeric columns for polar plots
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Create polar plot for first numeric column
            polar_plot = self.create_polar_plot(numeric_cols[0])
            if polar_plot:
                plots.append(polar_plot)
        
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
                plots.append(timeline_plot)
            
            # Create heatmap if we have categorical data
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                heatmap = self.create_24h_heatmap(time_col, categorical_cols[0], value_col)
                if heatmap:
                    plots.append(heatmap)
        
        # Save the dashboard
        if plots:
            output_file('../outputs/bokeh_outputs/interactive_dashboard.html')
            
            # Arrange plots in a grid
            if len(plots) == 1:
                layout = plots[0]
            elif len(plots) == 2:
                layout = column(plots[0], plots[1])
            else:
                layout = gridplot(plots, ncols=2)
            
            save(layout)
            print(f"[OK] Interactive dashboard saved to '../outputs/bokeh_outputs/interactive_dashboard.html'")
        
        return plots
    
    def create_advanced_polar_analysis(self):
        """Create advanced polar analysis with multiple variables."""
        print(f"\n[ADVANCED] Creating advanced polar analysis")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
        
        if len(numeric_cols) < 2:
            print(f"[WARNING] Need at least 2 numeric columns for advanced analysis")
            return None
        
        plots = []
        
        for col in numeric_cols:
            polar_plot = self.create_polar_plot(col, f"Polar Analysis: {col}")
            if polar_plot:
                plots.append(polar_plot)
        
        if plots:
            output_file('../outputs/bokeh_outputs/advanced_polar_analysis.html')
            layout = gridplot(plots, ncols=2)
            save(layout)
            print(f"[OK] Advanced polar analysis saved to '../outputs/bokeh_outputs/advanced_polar_analysis.html'")
        
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
                plots.append(timeline_plot)
        
        if plots:
            output_file('../outputs/bokeh_outputs/timeline_analysis.html')
            layout = column(plots)
            save(layout)
            print(f"[OK] Timeline analysis saved to '../outputs/bokeh_outputs/timeline_analysis.html'")
        
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
            output_file('../outputs/bokeh_outputs/24h_pattern_analysis.html')
            save(heatmap)
            print(f"[OK] 24-hour pattern analysis saved to '../outputs/bokeh_outputs/24h_pattern_analysis.html'")
        
        return heatmap
    
    def run_comprehensive_analysis(self):
        """Run comprehensive Bokeh interactive analysis."""
        print(f"\n[START] Starting comprehensive Bokeh interactive analysis")
        
        # Load data
        data = self.load_data()
        if data is None:
            return
        
        # Create all types of visualizations
        print(f"\n[PLOTS] Creating interactive visualizations...")
        
        # 1. Interactive Dashboard
        self.create_interactive_dashboard()
        
        # 2. Advanced Polar Analysis
        self.create_advanced_polar_analysis()
        
        # 3. Timeline Analysis
        self.create_timeline_analysis()
        
        # 4. 24-Hour Pattern Analysis
        self.create_24h_pattern_analysis()
        
        print(f"\n[OK] Comprehensive Bokeh analysis complete!")
        print(f"[DATA] Check '../outputs/bokeh_outputs/' directory for interactive HTML files")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Bokeh interactive plots generator closed")


def main():
    """Main function to run Bokeh interactive analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate interactive Bokeh visualizations')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    
    args = parser.parse_args()
    
    # Create Bokeh generator
    generator = BokehInteractivePlots(args.data_source)
    
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