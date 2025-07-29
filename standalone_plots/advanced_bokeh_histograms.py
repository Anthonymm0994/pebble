#!/usr/bin/env python3
"""
Advanced Bokeh Histogram Generator
==================================

Professional histogram generator using Bokeh with interactive features,
multiple plot types, and sophisticated styling.

Features:
- Interactive bokeh histograms with hover tools
- Multiple histogram types (regular, cumulative, density)
- Distribution fitting with interactive overlays
- Statistical analysis panels
- Professional styling and themes
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma, weibull_min
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime

# Bokeh imports
try:
    from bokeh.plotting import figure, show, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import (
        HoverTool, ColumnDataSource, Div, Panel, Tabs,
        Legend, LegendItem, RangeSlider, Select, Button,
        TextInput, Spinner, CheckboxGroup, RadioGroup
    )
    from bokeh.io import output_notebook
    from bokeh.palettes import Category10, Spectral6, Viridis256
    from bokeh.themes import built_in_themes
    from bokeh.transform import linear_cmap
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    print("Bokeh not available. Install with: pip install bokeh")

# Configuration
DATABASE_PATH = "sample_data.sqlite"
OUTPUT_DIR = "bokeh_histogram_outputs"
DPI = 300

class AdvancedBokehHistograms:
    """Advanced histogram generator using Bokeh with interactive features."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        
        if not BOKEH_AVAILABLE:
            raise ImportError("Bokeh is required for this script. Install with: pip install bokeh")
    
    def get_database_info(self) -> Dict[str, List[str]]:
        """Get database structure information."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_info = {}
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = [row[1] for row in cursor.fetchall()]
                    table_info[table] = columns
                
                return table_info
        except Exception as e:
            print(f"Error accessing database: {e}")
            return {}
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def calculate_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive statistics for data."""
        if data.empty:
            return {}
        
        data_clean = data.dropna()
        
        stats_dict = {
            'count': len(data_clean),
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'q25': data_clean.quantile(0.25),
            'q75': data_clean.quantile(0.75),
            'iqr': data_clean.quantile(0.75) - data_clean.quantile(0.25),
            'skewness': data_clean.skew(),
            'kurtosis': data_clean.kurtosis(),
            'coefficient_of_variation': data_clean.std() / data_clean.mean() if data_clean.mean() != 0 else 0
        }
        
        # Normality test
        try:
            stat, p_value = stats.shapiro(data_clean)
            stats_dict['shapiro_statistic'] = stat
            stats_dict['shapiro_p_value'] = p_value
            stats_dict['is_normal'] = p_value > 0.05
        except:
            stats_dict['shapiro_statistic'] = np.nan
            stats_dict['shapiro_p_value'] = np.nan
            stats_dict['is_normal'] = False
        
        return stats_dict
    
    def fit_distributions(self, data: pd.Series) -> Dict[str, Dict]:
        """Fit various theoretical distributions to the data."""
        if data.empty:
            return {}
        
        fits = {}
        data_clean = data.dropna()
        
        distributions = {
            'normal': norm,
            'lognormal': lognorm,
            'exponential': expon,
            'gamma': gamma,
            'weibull': weibull_min
        }
        
        for dist_name, dist_func in distributions.items():
            try:
                params = dist_func.fit(data_clean)
                log_likelihood = np.sum(dist_func.logpdf(data_clean, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                fits[dist_name] = {
                    'params': params,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                }
            except Exception as e:
                print(f"Error fitting {dist_name} distribution: {e}")
        
        return fits
    
    def create_basic_histogram(self, data: pd.Series, title: str = "Histogram") -> figure:
        """Create a basic interactive histogram."""
        data_clean = data.dropna()
        
        # Create histogram data
        hist, edges = np.histogram(data_clean, bins=50, density=True)
        
        # Create data source
        source = ColumnDataSource(data=dict(
            left=edges[:-1],
            right=edges[1:],
            top=hist,
            bottom=np.zeros_like(hist),
            center=(edges[:-1] + edges[1:]) / 2,
            width=edges[1:] - edges[:-1]
        ))
        
        # Create figure
        p = figure(
            title=title,
            width=800, height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#fafafa"
        )
        
        # Add histogram
        p.quad(
            top='top', bottom='bottom', left='left', right='right',
            fill_color='steelblue', line_color='white', alpha=0.7,
            source=source, legend_label="Histogram"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Range', '@left{0.2f} - @right{0.2f}'),
            ('Density', '@top{0.4f}'),
            ('Width', '@width{0.2f}'),
        ])
        p.add_tools(hover)
        
        # Styling
        p.xaxis.axis_label = 'Value'
        p.yaxis.axis_label = 'Density'
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.legend.location = 'top_right'
        
        return p
    
    def create_distribution_histogram(self, data: pd.Series, title: str = "Histogram with Distribution Fits") -> figure:
        """Create histogram with fitted distribution overlays."""
        data_clean = data.dropna()
        
        # Create histogram data
        hist, edges = np.histogram(data_clean, bins=50, density=True)
        
        # Create data source
        source = ColumnDataSource(data=dict(
            left=edges[:-1],
            right=edges[1:],
            top=hist,
            bottom=np.zeros_like(hist)
        ))
        
        # Create figure
        p = figure(
            title=title,
            width=800, height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#fafafa"
        )
        
        # Add histogram
        p.quad(
            top='top', bottom='bottom', left='left', right='right',
            fill_color='steelblue', line_color='white', alpha=0.7,
            source=source, legend_label="Histogram"
        )
        
        # Fit and plot distributions
        fits = self.fit_distributions(data)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        x = np.linspace(data_clean.min(), data_clean.max(), 1000)
        
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, *fit_info['params'])
                elif dist_name == "lognormal":
                    y = lognorm.pdf(x, *fit_info['params'])
                elif dist_name == "exponential":
                    y = expon.pdf(x, *fit_info['params'])
                elif dist_name == "gamma":
                    y = gamma.pdf(x, *fit_info['params'])
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, *fit_info['params'])
                
                p.line(x, y, line_width=2, color=colors[i % len(colors)],
                      legend_label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Range', '@left{0.2f} - @right{0.2f}'),
            ('Density', '@top{0.4f}'),
        ])
        p.add_tools(hover)
        
        # Styling
        p.xaxis.axis_label = 'Value'
        p.yaxis.axis_label = 'Density'
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.legend.location = 'top_right'
        
        return p
    
    def create_cumulative_histogram(self, data: pd.Series, title: str = "Cumulative Distribution") -> figure:
        """Create cumulative distribution histogram."""
        data_clean = data.dropna()
        
        # Sort data for cumulative plot
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Create figure
        p = figure(
            title=title,
            width=800, height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#fafafa"
        )
        
        # Add cumulative line
        p.line(sorted_data, y, line_width=3, color='steelblue', legend_label="Cumulative Distribution")
        
        # Add theoretical normal CDF for comparison
        mean, std = data_clean.mean(), data_clean.std()
        x_norm = np.linspace(data_clean.min(), data_clean.max(), 1000)
        y_norm = stats.norm.cdf(x_norm, mean, std)
        p.line(x_norm, y_norm, line_width=2, color='red', line_dash='dashed',
               legend_label="Normal CDF")
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Value', '@x{0.2f}'),
            ('Cumulative Probability', '@y{0.4f}'),
        ])
        p.add_tools(hover)
        
        # Styling
        p.xaxis.axis_label = 'Value'
        p.yaxis.axis_label = 'Cumulative Probability'
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.legend.location = 'top_left'
        
        return p
    
    def create_statistics_panel(self, data: pd.Series) -> Div:
        """Create a statistics information panel."""
        stats_dict = self.calculate_statistics(data)
        
        if not stats_dict:
            return Div(text="<h3>Statistics</h3><p>No data available</p>")
        
        stats_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
            <h3>Statistical Summary</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Count:</strong></td><td>{stats_dict['count']:,.0f}</td></tr>
                <tr><td><strong>Mean:</strong></td><td>{stats_dict['mean']:.4f}</td></tr>
                <tr><td><strong>Median:</strong></td><td>{stats_dict['median']:.4f}</td></tr>
                <tr><td><strong>Std Dev:</strong></td><td>{stats_dict['std']:.4f}</td></tr>
                <tr><td><strong>Min:</strong></td><td>{stats_dict['min']:.4f}</td></tr>
                <tr><td><strong>Max:</strong></td><td>{stats_dict['max']:.4f}</td></tr>
                <tr><td><strong>Q25:</strong></td><td>{stats_dict['q25']:.4f}</td></tr>
                <tr><td><strong>Q75:</strong></td><td>{stats_dict['q75']:.4f}</td></tr>
                <tr><td><strong>IQR:</strong></td><td>{stats_dict['iqr']:.4f}</td></tr>
                <tr><td><strong>Skewness:</strong></td><td>{stats_dict['skewness']:.4f}</td></tr>
                <tr><td><strong>Kurtosis:</strong></td><td>{stats_dict['kurtosis']:.4f}</td></tr>
                <tr><td><strong>Coeff of Variation:</strong></td><td>{stats_dict['coefficient_of_variation']:.4f}</td></tr>
                <tr><td><strong>Shapiro-Wilk p-value:</strong></td><td>{stats_dict['shapiro_p_value']:.4f}</td></tr>
                <tr><td><strong>Is Normal:</strong></td><td>{'Yes' if stats_dict['is_normal'] else 'No'}</td></tr>
            </table>
        </div>
        """
        
        return Div(text=stats_html)
    
    def create_comparison_histogram(self, data_dict: Dict[str, pd.Series], title: str = "Comparison Histogram") -> figure:
        """Create overlapping histograms for comparison."""
        # Create figure
        p = figure(
            title=title,
            width=800, height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#fafafa"
        )
        
        colors = Category10[10]
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
            
            data_clean = data.dropna()
            hist, edges = np.histogram(data_clean, bins=50, density=True)
            
            # Create data source
            source = ColumnDataSource(data=dict(
                left=edges[:-1],
                right=edges[1:],
                top=hist,
                bottom=np.zeros_like(hist)
            ))
            
            # Add histogram
            p.quad(
                top='top', bottom='bottom', left='left', right='right',
                fill_color=colors[i % len(colors)], line_color='white', alpha=0.6,
                source=source, legend_label=label
            )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Range', '@left{0.2f} - @right{0.2f}'),
            ('Density', '@top{0.4f}'),
        ])
        p.add_tools(hover)
        
        # Styling
        p.xaxis.axis_label = 'Value'
        p.yaxis.axis_label = 'Density'
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.legend.location = 'top_right'
        
        return p
    
    def create_interactive_dashboard(self, data: pd.Series, title: str = "Interactive Histogram Dashboard") -> Tabs:
        """Create an interactive dashboard with multiple histogram types."""
        # Create different histogram types
        basic_hist = self.create_basic_histogram(data, f"{title} - Basic")
        dist_hist = self.create_distribution_histogram(data, f"{title} - With Distribution Fits")
        cum_hist = self.create_cumulative_histogram(data, f"{title} - Cumulative")
        stats_panel = self.create_statistics_panel(data)
        
        # Create tabs
        tab1 = Panel(child=basic_hist, title="Basic Histogram")
        tab2 = Panel(child=dist_hist, title="Distribution Fits")
        tab3 = Panel(child=cum_hist, title="Cumulative")
        tab4 = Panel(child=stats_panel, title="Statistics")
        
        return Tabs(tabs=[tab1, tab2, tab3, tab4])
    
    def create_advanced_dashboard(self, queries: List[Dict[str, Any]]) -> Tabs:
        """Create an advanced dashboard with multiple datasets."""
        tabs = []
        
        for i, query_info in enumerate(queries):
            print(f"Processing query {i+1}: {query_info.get('description', 'Custom query')}")
            
            # Execute query
            df = self.execute_query(query_info['query'])
            
            if df.empty:
                continue
            
            # Create dashboard for each numeric column
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                data = df[col]
                
                # Create dashboard
                dashboard = self.create_interactive_dashboard(
                    data, 
                    f"{query_info.get('description', 'Custom query')} - {col}"
                )
                
                tab = Panel(child=dashboard, title=f"{col} - {query_info.get('description', 'Custom')[:20]}")
                tabs.append(tab)
        
        return Tabs(tabs=tabs)
    
    def save_plots(self, plots: List[Any], base_filename: str):
        """Save plots in HTML format."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, plot in enumerate(plots):
            filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.html"
            save(plot, filename)
            print(f"Saved: {filename}")
    
    def generate_bokeh_analysis(self, queries: List[Dict[str, Any]]):
        """Generate comprehensive bokeh histogram analysis."""
        print("Generating Bokeh histogram analysis...")
        
        plots = []
        
        for i, query_info in enumerate(queries):
            print(f"\nProcessing query {i+1}: {query_info.get('description', 'Custom query')}")
            
            # Execute query
            df = self.execute_query(query_info['query'])
            
            if df.empty:
                print(f"No data for query {i+1}")
                continue
            
            # Create plots for each numeric column
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                data = df[col]
                
                # Create different types of plots
                basic_hist = self.create_basic_histogram(
                    data, 
                    f"Basic Histogram: {col}\n{query_info.get('description', 'Custom query')}"
                )
                plots.append(basic_hist)
                
                dist_hist = self.create_distribution_histogram(
                    data,
                    f"Distribution Fit: {col}\n{query_info.get('description', 'Custom query')}"
                )
                plots.append(dist_hist)
                
                cum_hist = self.create_cumulative_histogram(
                    data,
                    f"Cumulative: {col}\n{query_info.get('description', 'Custom query')}"
                )
                plots.append(cum_hist)
                
                # Create dashboard
                dashboard = self.create_interactive_dashboard(
                    data,
                    f"Dashboard: {col}\n{query_info.get('description', 'Custom query')}"
                )
                plots.append(dashboard)
            
            # Create comparison plots if multiple columns
            if len(numeric_columns) > 1:
                data_dict = {col: df[col] for col in numeric_columns}
                comp_hist = self.create_comparison_histogram(
                    data_dict,
                    f"Comparison: {query_info.get('description', 'Custom query')}"
                )
                plots.append(comp_hist)
        
        # Save all plots
        if plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_plots(plots, f"bokeh_histogram_analysis_{timestamp}")
            print(f"\nGenerated {len(plots)} Bokeh histogram plots")
        else:
            print("No plots generated - check your data and queries")

def main():
    """Main function to demonstrate Bokeh histogram capabilities."""
    print("Advanced Bokeh Histogram Generator")
    print("=" * 40)
    
    if not BOKEH_AVAILABLE:
        print("Bokeh is required. Install with: pip install bokeh")
        return
    
    # Initialize the Bokeh histogram generator
    bokeh_gen = AdvancedBokehHistograms()
    
    # Get database information
    db_info = bokeh_gen.get_database_info()
    print(f"Available tables: {list(db_info.keys())}")
    
    if not db_info:
        print("No tables found in database. Please check your database path.")
        return
    
    # Example queries for demonstration
    example_queries = [
        {
            'query': "SELECT amount, quantity, profit_margin FROM sales WHERE region = 'North'",
            'description': "North Region Sales"
        },
        {
            'query': "SELECT amount, quantity, profit_margin FROM sales WHERE category = 'Electronics'",
            'description': "Electronics Category"
        },
        {
            'query': "SELECT amount, quantity, profit_margin FROM sales WHERE amount > 1000",
            'description': "High Value Sales (>$1000)"
        }
    ]
    
    # Generate Bokeh analysis
    print("\nGenerating advanced Bokeh histogram analysis...")
    bokeh_gen.generate_bokeh_analysis(example_queries)
    
    print("\nAnalysis complete! Check the 'bokeh_histogram_outputs' directory for results.")

if __name__ == "__main__":
    main() 