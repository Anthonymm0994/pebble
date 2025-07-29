#!/usr/bin/env python3
"""
Advanced Histogram Analysis with Bokeh Integration
=================================================

Advanced histogram analysis with interactive features, statistical analysis,
and comprehensive visualization capabilities using both matplotlib and bokeh.

Features:
- Interactive bokeh histograms
- Advanced statistical analysis
- Distribution fitting and comparison
- Query-based data filtering
- Multiple visualization styles
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma, weibull_min, chi2, shapiro, anderson, kstest, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import itertools
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
DATABASE_PATH = "sample_data.sqlite"
OUTPUT_DIR = "advanced_histogram_outputs"
SAVE_FORMATS = ["png", "pdf", "svg", "html"]
DPI = 300
FIGURE_SIZE = (14, 10)

# Statistical Analysis Settings
CONFIDENCE_LEVEL = 0.95
NORMALITY_TESTS = ["shapiro", "anderson", "ks", "jarque_bera"]
DISTRIBUTION_FITS = ["normal", "lognormal", "exponential", "gamma", "weibull"]

class AdvancedHistogramAnalysis:
    """Advanced histogram analysis with comprehensive statistical capabilities."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        self.statistics_cache = {}
        self.setup_styles()
        
    def setup_styles(self):
        """Setup matplotlib and seaborn styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
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
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """Execute a custom SQL query."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive advanced statistics."""
        if data.empty:
            return {}
        
        data_clean = data.dropna()
        
        # Basic statistics
        stats = {
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
            'coefficient_of_variation': data_clean.std() / data_clean.mean() if data_clean.mean() != 0 else 0,
            'mad': data_clean.mad() if hasattr(data_clean, 'mad') else data_clean.abs().mean(),  # Mean Absolute Deviation
            'range': data_clean.max() - data_clean.min(),
            'percentiles': {
                'p10': data_clean.quantile(0.10),
                'p90': data_clean.quantile(0.90),
                'p95': data_clean.quantile(0.95),
                'p99': data_clean.quantile(0.99)
            }
        }
        
        # Advanced normality tests
        for test_name in NORMALITY_TESTS:
            try:
                if test_name == "shapiro":
                    stat, p_value = shapiro(data_clean)
                elif test_name == "anderson":
                    result = anderson(data_clean)
                    stat, p_value = result.statistic, result.significance_level[2]
                elif test_name == "ks":
                    stat, p_value = kstest(data_clean, 'norm', 
                                         args=(data_clean.mean(), data_clean.std()))
                elif test_name == "jarque_bera":
                    stat, p_value = jarque_bera(data_clean)
                
                stats[f'{test_name}_statistic'] = stat
                stats[f'{test_name}_p_value'] = p_value
                stats[f'{test_name}_normal'] = p_value > (1 - CONFIDENCE_LEVEL)
            except Exception as e:
                print(f"Error in {test_name} test: {e}")
        
        # Outlier detection using IQR method
        q1, q3 = data_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
        
        stats['outliers_count'] = len(outliers)
        stats['outliers_percentage'] = len(outliers) / len(data_clean) * 100
        stats['outlier_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
        
        return stats
    
    def fit_advanced_distributions(self, data: pd.Series) -> Dict[str, Dict]:
        """Fit various theoretical distributions with advanced metrics."""
        if data.empty:
            return {}
        
        fits = {}
        data_clean = data.dropna()
        
        for dist_name in DISTRIBUTION_FITS:
            try:
                if dist_name == "normal":
                    params = norm.fit(data_clean)
                    log_likelihood = np.sum(norm.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                    
                    fits[dist_name] = {
                        'params': params,
                        'loc': params[0],
                        'scale': params[1],
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
                elif dist_name == "lognormal":
                    params = lognorm.fit(data_clean)
                    log_likelihood = np.sum(lognorm.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                    
                    fits[dist_name] = {
                        'params': params,
                        'shape': params[0],
                        'loc': params[1],
                        'scale': params[2],
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
                elif dist_name == "exponential":
                    params = expon.fit(data_clean)
                    log_likelihood = np.sum(expon.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                    
                    fits[dist_name] = {
                        'params': params,
                        'loc': params[0],
                        'scale': params[1],
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
                elif dist_name == "gamma":
                    params = gamma.fit(data_clean)
                    log_likelihood = np.sum(gamma.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                    
                    fits[dist_name] = {
                        'params': params,
                        'shape': params[0],
                        'loc': params[1],
                        'scale': params[2],
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
                elif dist_name == "weibull":
                    params = weibull_min.fit(data_clean)
                    log_likelihood = np.sum(weibull_min.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                    
                    fits[dist_name] = {
                        'params': params,
                        'shape': params[0],
                        'loc': params[1],
                        'scale': params[2],
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood
                    }
            except Exception as e:
                print(f"Error fitting {dist_name} distribution: {e}")
        
        return fits
    
    def create_interactive_histogram_bokeh(self, data: pd.Series, title: str = "Interactive Histogram"):
        """Create an interactive histogram using bokeh."""
        try:
            from bokeh.plotting import figure, show, output_file
            from bokeh.layouts import column, row
            from bokeh.models import HoverTool, ColumnDataSource, Div
            from bokeh.io import output_notebook
            
            # Create histogram data
            hist, edges = np.histogram(data.dropna(), bins=50, density=True)
            
            # Create data source
            source = ColumnDataSource(data=dict(
                left=edges[:-1],
                right=edges[1:],
                top=hist,
                bottom=np.zeros_like(hist)
            ))
            
            # Create bokeh figure
            p = figure(title=title, width=800, height=600, tools="pan,wheel_zoom,box_zoom,reset,save")
            
            # Add histogram
            p.quad(top='top', bottom='bottom', left='left', right='right',
                   fill_color='skyblue', line_color='black', alpha=0.7, source=source)
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ('Range', '@left{0.2f} - @right{0.2f}'),
                ('Density', '@top{0.4f}'),
            ])
            p.add_tools(hover)
            
            p.xaxis.axis_label = 'Value'
            p.yaxis.axis_label = 'Density'
            p.grid.grid_line_color = 'gray'
            p.grid.grid_line_alpha = 0.3
            
            return p
            
        except ImportError:
            print("Bokeh not available. Install with: pip install bokeh")
            return None
    
    def create_plotly_histogram(self, data: pd.Series, title: str = "Plotly Histogram"):
        """Create an interactive histogram using plotly."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=data.dropna(),
                nbinsx=50,
                name='Histogram',
                opacity=0.7,
                marker_color='skyblue',
                marker_line_color='black',
                marker_line_width=1
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Value',
                yaxis_title='Frequency',
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return None
    
    def create_comprehensive_analysis_plot(self, data: pd.Series, title: str = "Comprehensive Analysis"):
        """Create a comprehensive analysis plot with multiple subplots."""
        if data.empty:
            raise ValueError("Data is empty")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        data_clean = data.dropna()
        
        # Main histogram with distribution fits
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.hist(data_clean, bins=50, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Add distribution fits
        fits = self.fit_advanced_distributions(data)
        x = np.linspace(data_clean.min(), data_clean.max(), 1000)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, fit_info['loc'], fit_info['scale'])
                elif dist_name == "lognormal":
                    y = lognorm.pdf(x, fit_info['shape'], fit_info['loc'], fit_info['scale'])
                elif dist_name == "exponential":
                    y = expon.pdf(x, fit_info['loc'], fit_info['scale'])
                elif dist_name == "gamma":
                    y = gamma.pdf(x, fit_info['shape'], fit_info['loc'], fit_info['scale'])
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, fit_info['shape'], fit_info['loc'], fit_info['scale'])
                
                ax_main.plot(x, y, '--', linewidth=2, color=colors[i % len(colors)],
                           label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Value', fontsize=12)
        ax_main.set_ylabel('Density', fontsize=12)
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Statistics text
        stats = self.calculate_advanced_statistics(data)
        stats_text = f"""
Advanced Statistics:
Count: {stats.get('count', 'N/A'):,.0f}
Mean: {stats.get('mean', 'N/A'):.2f}
Median: {stats.get('median', 'N/A'):.2f}
Std Dev: {stats.get('std', 'N/A'):.2f}
Skewness: {stats.get('skewness', 'N/A'):.3f}
Kurtosis: {stats.get('kurtosis', 'N/A'):.3f}
Outliers: {stats.get('outliers_count', 'N/A')} ({stats.get('outliers_percentage', 'N/A'):.1f}%)
        """
        
        ax_stats = fig.add_subplot(gs[1, 0])
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        # Normality tests
        normality_text = "Normality Tests:\n"
        for test_name in NORMALITY_TESTS:
            stat_key = f'{test_name}_statistic'
            p_key = f'{test_name}_p_value'
            normal_key = f'{test_name}_normal'
            
            if stat_key in stats:
                normality_text += f"{test_name.title()}: "
                normality_text += f"p={stats[p_key]:.3f}, "
                normality_text += f"normal={stats[normal_key]}\n"
        
        ax_norm = fig.add_subplot(gs[1, 1])
        ax_norm.text(0.05, 0.95, normality_text, transform=ax_norm.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax_norm.set_xlim(0, 1)
        ax_norm.set_ylim(0, 1)
        ax_norm.axis('off')
        
        # Distribution comparison
        if fits:
            aic_values = {name: info['aic'] for name, info in fits.items()}
            best_dist = min(aic_values, key=aic_values.get)
            
            comparison_text = f"Distribution Comparison:\n"
            for name, aic in sorted(aic_values.items(), key=lambda x: x[1]):
                marker = "★" if name == best_dist else "  "
                comparison_text += f"{marker} {name}: AIC={aic:.2f}\n"
            
            ax_comp = fig.add_subplot(gs[1, 2])
            ax_comp.text(0.05, 0.95, comparison_text, transform=ax_comp.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax_comp.set_xlim(0, 1)
            ax_comp.set_ylim(0, 1)
            ax_comp.axis('off')
        
        # Box plot
        ax_box = fig.add_subplot(gs[2, 0])
        ax_box.boxplot(data_clean, vert=False)
        ax_box.set_title('Box Plot', fontsize=12)
        ax_box.set_xlabel('Value', fontsize=10)
        
        # Q-Q plot
        ax_qq = fig.add_subplot(gs[2, 1])
        from scipy.stats import probplot
        probplot(data_clean, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot (Normal)', fontsize=12)
        
        # Cumulative distribution
        ax_cdf = fig.add_subplot(gs[2, 2])
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax_cdf.plot(sorted_data, y, 'b-', linewidth=2)
        ax_cdf.set_title('Cumulative Distribution', fontsize=12)
        ax_cdf.set_xlabel('Value', fontsize=10)
        ax_cdf.set_ylabel('Cumulative Probability', fontsize=10)
        ax_cdf.grid(True, alpha=0.3)
        
        # Histogram variations
        ax_hist1 = fig.add_subplot(gs[3, 0])
        ax_hist1.hist(data_clean, bins=20, density=True, alpha=0.7, color='lightcoral')
        ax_hist1.set_title('Histogram (20 bins)', fontsize=10)
        ax_hist1.set_xlabel('Value', fontsize=8)
        
        ax_hist2 = fig.add_subplot(gs[3, 1])
        ax_hist2.hist(data_clean, bins=100, density=True, alpha=0.7, color='lightgreen')
        ax_hist2.set_title('Histogram (100 bins)', fontsize=10)
        ax_hist2.set_xlabel('Value', fontsize=8)
        
        ax_hist3 = fig.add_subplot(gs[3, 2])
        ax_hist3.hist(data_clean, bins='auto', density=True, alpha=0.7, color='lightblue')
        ax_hist3.set_title('Histogram (auto bins)', fontsize=10)
        ax_hist3.set_xlabel('Value', fontsize=8)
        
        return fig
    
    def create_overlapping_analysis(self, data_dict: Dict[str, pd.Series], 
                                  title: str = "Overlapping Analysis") -> plt.Figure:
        """Create comprehensive overlapping histogram analysis."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Main overlapping histogram
        ax_main = fig.add_subplot(gs[0, :])
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
            ax_main.hist(data.dropna(), bins=50, alpha=0.6, density=True,
                        label=label, color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Value', fontsize=12)
        ax_main.set_ylabel('Density', fontsize=12)
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Statistical comparison
        comparison_data = []
        for label, data in data_dict.items():
            if not data.empty:
                stats = self.calculate_advanced_statistics(data)
                comparison_data.append({
                    'label': label,
                    'mean': stats.get('mean', 0),
                    'std': stats.get('std', 0),
                    'skewness': stats.get('skewness', 0),
                    'count': stats.get('count', 0)
                })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            
            # Mean comparison
            ax_mean = fig.add_subplot(gs[1, 0])
            ax_mean.bar(df_comp['label'], df_comp['mean'], color='skyblue', alpha=0.7)
            ax_mean.set_title('Mean Comparison', fontsize=12)
            ax_mean.set_ylabel('Mean', fontsize=10)
            ax_mean.tick_params(axis='x', rotation=45)
            
            # Standard deviation comparison
            ax_std = fig.add_subplot(gs[1, 1])
            ax_std.bar(df_comp['label'], df_comp['std'], color='lightcoral', alpha=0.7)
            ax_std.set_title('Standard Deviation Comparison', fontsize=12)
            ax_std.set_ylabel('Standard Deviation', fontsize=10)
            ax_std.tick_params(axis='x', rotation=45)
            
            # Skewness comparison
            ax_skew = fig.add_subplot(gs[1, 2])
            ax_skew.bar(df_comp['label'], df_comp['skewness'], color='lightgreen', alpha=0.7)
            ax_skew.set_title('Skewness Comparison', fontsize=12)
            ax_skew.set_ylabel('Skewness', fontsize=10)
            ax_skew.tick_params(axis='x', rotation=45)
        
        return fig
    
    def save_plots(self, figures: List[plt.Figure], base_filename: str):
        """Save plots in multiple formats."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, fig in enumerate(figures):
            for fmt in SAVE_FORMATS:
                if fmt == "html":
                    # For plotly figures, save as HTML
                    continue
                filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.{fmt}"
                fig.savefig(filename, dpi=DPI, bbox_inches='tight')
                print(f"Saved: {filename}")
    
    def generate_advanced_analysis(self, queries: List[Dict[str, Any]]):
        """Generate advanced histogram analysis for multiple queries."""
        figures = []
        
        for i, query_info in enumerate(queries):
            print(f"\nProcessing query {i+1}: {query_info.get('description', 'Custom query')}")
            
            # Execute query
            if 'query' in query_info:
                df = self.execute_custom_query(query_info['query'])
            else:
                df = pd.DataFrame()
            
            if df.empty:
                print(f"No data for query {i+1}")
                continue
            
            # Create comprehensive analysis for each numeric column
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                data = df[col]
                
                # Create comprehensive analysis plot
                fig = self.create_comprehensive_analysis_plot(
                    data, 
                    title=f"Advanced Analysis: {col}\n{query_info.get('description', 'Custom query')}"
                )
                figures.append(fig)
                
                # Create plotly interactive plot
                plotly_fig = self.create_plotly_histogram(
                    data,
                    title=f"Interactive: {col}"
                )
                if plotly_fig:
                    Path(OUTPUT_DIR).mkdir(exist_ok=True)
                    plotly_filename = f"{OUTPUT_DIR}/interactive_{col}_{i+1}.html"
                    plotly_fig.write_html(plotly_filename)
                    print(f"Saved interactive plot: {plotly_filename}")
            
            # Create overlapping analysis if multiple columns
            if len(numeric_columns) > 1:
                data_dict = {col: df[col] for col in numeric_columns}
                fig_overlap = self.create_overlapping_analysis(
                    data_dict,
                    title=f"Overlapping Analysis\n{query_info.get('description', 'Custom query')}"
                )
                figures.append(fig_overlap)
        
        # Save all figures
        if figures:
            self.save_plots(figures, "advanced_histogram_analysis")
            print(f"\nGenerated {len(figures)} advanced histogram plots")
        else:
            print("No figures generated - check your data and queries")

def main():
    """Main function to demonstrate advanced histogram capabilities."""
    print("Advanced Histogram Analysis")
    print("=" * 40)
    
    # Initialize the analysis
    analyzer = AdvancedHistogramAnalysis()
    
    # Get database information
    db_info = analyzer.get_database_info()
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
    
    # Generate advanced analysis
    print("\nGenerating advanced histogram analysis...")
    analyzer.generate_advanced_analysis(example_queries)
    
    print("\nAnalysis complete! Check the 'advanced_histogram_outputs' directory for results.")

if __name__ == "__main__":
    main() 