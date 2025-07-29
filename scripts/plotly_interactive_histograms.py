#!/usr/bin/env python3
"""
Advanced Plotly Interactive Histogram Generator
==============================================

Professional histogram generator using Plotly with interactive features,
animations, and sophisticated statistical analysis.

Features:
- Interactive plotly histograms with hover tools
- Multiple histogram types and variations
- Statistical analysis with interactive annotations
- Distribution fitting with interactive overlays
- Animations and transitions
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, lognormal, expon, gamma, weibull_min
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import itertools

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from plotly.offline import plot
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Configuration
DATABASE_PATH = "sample_data.sqlite"
OUTPUT_DIR = "plotly_histogram_outputs"
DPI = 300

class AdvancedPlotlyHistograms:
    """Advanced histogram generator using Plotly with interactive features."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this script. Install with: pip install plotly")
        
        # Set plotly template
        pio.templates.default = "plotly_white"
    
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
            'lognormal': lognormal,
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
    
    def create_basic_histogram(self, data: pd.Series, title: str = "Histogram") -> go.Figure:
        """Create a basic interactive histogram."""
        data_clean = data.dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data_clean,
            nbinsx=50,
            name='Histogram',
            opacity=0.7,
            marker_color='steelblue',
            marker_line_color='white',
            marker_line_width=1,
            hovertemplate='<b>Range:</b> %{x}<br>' +
                        '<b>Count:</b> %{y}<br>' +
                        '<extra></extra>'
        ))
        
        # Add KDE
        kde_x = np.linspace(data_clean.min(), data_clean.max(), 100)
        kde_y = stats.gaussian_kde(data_clean)(kde_x) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2),
            hovertemplate='<b>Value:</b> %{x}<br>' +
                        '<b>Density:</b> %{y:.2f}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Count',
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def create_distribution_histogram(self, data: pd.Series, title: str = "Histogram with Distribution Fits") -> go.Figure:
        """Create histogram with fitted distribution overlays."""
        data_clean = data.dropna()
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data_clean,
            nbinsx=50,
            name='Histogram',
            opacity=0.7,
            marker_color='steelblue',
            marker_line_color='white',
            marker_line_width=1,
            yaxis='y',
            hovertemplate='<b>Range:</b> %{x}<br>' +
                        '<b>Count:</b> %{y}<br>' +
                        '<extra></extra>'
        ))
        
        # Fit and plot distributions
        fits = self.fit_distributions(data)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        x = np.linspace(data_clean.min(), data_clean.max(), 1000)
        
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "lognormal":
                    y = lognormal.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "exponential":
                    y = expon.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "gamma":
                    y = gamma.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f'{dist_name} (AIC: {fit_info["aic"]:.2f})',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    hovertemplate='<b>Value:</b> %{x}<br>' +
                                '<b>Density:</b> %{y:.2f}<br>' +
                                '<extra></extra>'
                ))
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Count',
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    def create_comparison_histogram(self, data_dict: Dict[str, pd.Series], title: str = "Comparison Histogram") -> go.Figure:
        """Create overlapping histograms for comparison."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
            
            data_clean = data.dropna()
            
            fig.add_trace(go.Histogram(
                x=data_clean,
                nbinsx=50,
                name=label,
                opacity=0.6,
                marker_color=colors[i % len(colors)],
                marker_line_color='white',
                marker_line_width=1,
                hovertemplate='<b>Range:</b> %{x}<br>' +
                            '<b>Count:</b> %{y}<br>' +
                            '<b>Dataset:</b> ' + label + '<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Count',
            showlegend=True,
            template='plotly_white',
            hovermode='closest',
            barmode='overlay'
        )
        
        return fig
    
    def create_statistical_summary_plot(self, data: pd.Series, title: str = "Statistical Summary") -> go.Figure:
        """Create a comprehensive statistical summary plot."""
        data_clean = data.dropna()
        stats_dict = self.calculate_statistics(data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram with KDE', 'Box Plot', 'Q-Q Plot', 'Cumulative Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histogram with KDE
        fig.add_trace(go.Histogram(
            x=data_clean,
            nbinsx=50,
            name='Histogram',
            opacity=0.7,
            marker_color='steelblue',
            marker_line_color='white',
            marker_line_width=1
        ), row=1, col=1)
        
        # Add KDE
        kde_x = np.linspace(data_clean.min(), data_clean.max(), 100)
        kde_y = stats.gaussian_kde(data_clean)(kde_x) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ), row=1, col=1)
        
        # 2. Box plot
        fig.add_trace(go.Box(
            y=data_clean,
            name='Box Plot',
            marker_color='lightblue',
            boxpoints='outliers'
        ), row=1, col=2)
        
        # 3. Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_clean)))
        sample_quantiles = np.sort(data_clean)
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='green', size=6)
        ), row=2, col=1)
        
        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash')
        ), row=2, col=1)
        
        # 4. Cumulative distribution
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(go.Scatter(
            x=sorted_data,
            y=y,
            mode='lines',
            name='Cumulative',
            line=dict(color='purple', width=2)
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            template='plotly_white',
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Value", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
        
        return fig
    
    def create_advanced_analysis_plot(self, data: pd.Series, title: str = "Advanced Analysis") -> go.Figure:
        """Create an advanced analysis plot with multiple subplots."""
        data_clean = data.dropna()
        stats_dict = self.calculate_statistics(data)
        fits = self.fit_distributions(data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Histogram with Distribution Fits', 'Statistics', 'Box Plot', 'Q-Q Plot', 'Cumulative Distribution', 'Distribution Comparison'),
            specs=[[{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Main histogram with distribution fits
        fig.add_trace(go.Histogram(
            x=data_clean,
            nbinsx=50,
            name='Histogram',
            opacity=0.7,
            marker_color='steelblue',
            marker_line_color='white',
            marker_line_width=1
        ), row=1, col=1)
        
        # Add distribution fits
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        x = np.linspace(data_clean.min(), data_clean.max(), 1000)
        
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "lognormal":
                    y = lognormal.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "exponential":
                    y = expon.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "gamma":
                    y = gamma.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, *fit_info['params']) * len(data_clean) * (data_clean.max() - data_clean.min()) / 50
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f'{dist_name} (AIC: {fit_info["aic"]:.2f})',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ), row=1, col=1)
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        # 2. Statistics text (as annotation)
        if stats_dict:
            stats_text = f"""
Count: {stats_dict['count']:,.0f}<br>
Mean: {stats_dict['mean']:.2f}<br>
Median: {stats_dict['median']:.2f}<br>
Std Dev: {stats_dict['std']:.2f}<br>
Skewness: {stats_dict['skewness']:.3f}<br>
Kurtosis: {stats_dict['kurtosis']:.3f}
            """
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.5,
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="black",
                borderwidth=1
            )
        
        # 3. Box plot
        fig.add_trace(go.Box(
            y=data_clean,
            name='Box Plot',
            marker_color='lightblue',
            boxpoints='outliers'
        ), row=2, col=1)
        
        # 4. Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_clean)))
        sample_quantiles = np.sort(data_clean)
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='green', size=6)
        ), row=2, col=2)
        
        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash')
        ), row=2, col=2)
        
        # 5. Cumulative distribution
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(go.Scatter(
            x=sorted_data,
            y=y,
            mode='lines',
            name='Cumulative',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        # 6. Distribution comparison (AIC values)
        if fits:
            dist_names = list(fits.keys())
            aic_values = [fits[name]['aic'] for name in dist_names]
            
            fig.add_trace(go.Bar(
                x=dist_names,
                y=aic_values,
                name='AIC Values',
                marker_color='orange'
            ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            template='plotly_white',
            height=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_xaxes(title_text="Value", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=3, col=1)
        fig.update_xaxes(title_text="Distribution", row=3, col=2)
        fig.update_yaxes(title_text="AIC Value", row=3, col=2)
        
        return fig
    
    def save_plots(self, plots: List[go.Figure], base_filename: str):
        """Save plots in HTML format."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, fig in enumerate(plots):
            filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.html"
            fig.write_html(filename, include_plotlyjs=True)
            print(f"Saved: {filename}")
    
    def generate_plotly_analysis(self, queries: List[Dict[str, Any]]):
        """Generate comprehensive plotly histogram analysis."""
        print("Generating Plotly histogram analysis...")
        
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
                    f"Basic Histogram: {col}<br>{query_info.get('description', 'Custom query')}"
                )
                plots.append(basic_hist)
                
                dist_hist = self.create_distribution_histogram(
                    data,
                    f"Distribution Fit: {col}<br>{query_info.get('description', 'Custom query')}"
                )
                plots.append(dist_hist)
                
                stat_summary = self.create_statistical_summary_plot(
                    data,
                    f"Statistical Summary: {col}<br>{query_info.get('description', 'Custom query')}"
                )
                plots.append(stat_summary)
                
                advanced_analysis = self.create_advanced_analysis_plot(
                    data,
                    f"Advanced Analysis: {col}<br>{query_info.get('description', 'Custom query')}"
                )
                plots.append(advanced_analysis)
            
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
            self.save_plots(plots, f"plotly_histogram_analysis_{timestamp}")
            print(f"\nGenerated {len(plots)} Plotly histogram plots")
        else:
            print("No plots generated - check your data and queries")

def main():
    """Main function to demonstrate Plotly histogram capabilities."""
    print("Advanced Plotly Interactive Histogram Generator")
    print("=" * 50)
    
    if not PLOTLY_AVAILABLE:
        print("Plotly is required. Install with: pip install plotly")
        return
    
    # Initialize the Plotly histogram generator
    plotly_gen = AdvancedPlotlyHistograms()
    
    # Get database information
    db_info = plotly_gen.get_database_info()
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
    
    # Generate Plotly analysis
    print("\nGenerating advanced Plotly histogram analysis...")
    plotly_gen.generate_plotly_analysis(example_queries)
    
    print("\nAnalysis complete! Check the 'plotly_histogram_outputs' directory for results.")

if __name__ == "__main__":
    main() 