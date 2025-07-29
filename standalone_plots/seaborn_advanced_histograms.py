#!/usr/bin/env python3
"""
Advanced Seaborn Histogram Generator
====================================

Professional histogram generator using Seaborn with advanced styling,
statistical analysis, and sophisticated visualizations.

Features:
- Advanced seaborn styling and themes
- Multiple histogram types and variations
- Statistical analysis with annotations
- Distribution fitting and comparison
- Professional color schemes
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognormal, expon, gamma, weibull_min
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import itertools

# Configuration
DATABASE_PATH = "sample_data.sqlite"
OUTPUT_DIR = "seaborn_histogram_outputs"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300
FIGURE_SIZE = (12, 8)

# Seaborn styling
sns.set_style("whitegrid")
sns.set_palette("husl")

class AdvancedSeabornHistograms:
    """Advanced histogram generator using Seaborn with professional styling."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        self.setup_styles()
    
    def setup_styles(self):
        """Setup seaborn and matplotlib styles."""
        # Set seaborn style
        sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})
        sns.set_palette("husl")
        
        # Set matplotlib parameters
        plt.rcParams['figure.figsize'] = FIGURE_SIZE
        plt.rcParams['figure.dpi'] = DPI
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
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
    
    def calculate_advanced_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive advanced statistics."""
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
            'coefficient_of_variation': data_clean.std() / data_clean.mean() if data_clean.mean() != 0 else 0,
            'mad': data_clean.abs().mean(),  # Mean Absolute Deviation
            'range': data_clean.max() - data_clean.min(),
            'percentiles': {
                'p10': data_clean.quantile(0.10),
                'p90': data_clean.quantile(0.90),
                'p95': data_clean.quantile(0.95),
                'p99': data_clean.quantile(0.99)
            }
        }
        
        # Advanced normality tests
        normality_tests = {}
        try:
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(data_clean)
            normality_tests['shapiro'] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
        except:
            normality_tests['shapiro'] = {'statistic': np.nan, 'p_value': np.nan, 'normal': False}
        
        try:
            # Anderson-Darling test
            result = stats.anderson(data_clean)
            normality_tests['anderson'] = {'statistic': result.statistic, 'p_value': result.significance_level[2], 'normal': result.statistic < result.critical_values[2]}
        except:
            normality_tests['anderson'] = {'statistic': np.nan, 'p_value': np.nan, 'normal': False}
        
        try:
            # Kolmogorov-Smirnov test
            stat, p_value = stats.kstest(data_clean, 'norm', args=(data_clean.mean(), data_clean.std()))
            normality_tests['ks'] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
        except:
            normality_tests['ks'] = {'statistic': np.nan, 'p_value': np.nan, 'normal': False}
        
        stats_dict['normality_tests'] = normality_tests
        
        # Outlier detection using IQR method
        q1, q3 = data_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
        
        stats_dict['outliers_count'] = len(outliers)
        stats_dict['outliers_percentage'] = len(outliers) / len(data_clean) * 100
        stats_dict['outlier_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
        
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
                bic = len(params) * np.log(len(data_clean)) - 2 * log_likelihood
                
                fits[dist_name] = {
                    'params': params,
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood
                }
            except Exception as e:
                print(f"Error fitting {dist_name} distribution: {e}")
        
        return fits
    
    def create_basic_histogram(self, data: pd.Series, title: str = "Histogram") -> plt.Figure:
        """Create a basic seaborn histogram."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Create histogram
        sns.histplot(data=data.dropna(), kde=True, ax=ax, bins=50, alpha=0.7, color='steelblue')
        
        # Add statistics text
        stats_dict = self.calculate_advanced_statistics(data)
        if stats_dict:
            stats_text = f"n={stats_dict['count']:,.0f}\nμ={stats_dict['mean']:.2f}\nσ={stats_dict['std']:.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        
        return fig
    
    def create_distribution_histogram(self, data: pd.Series, title: str = "Histogram with Distribution Fits") -> plt.Figure:
        """Create histogram with fitted distribution overlays."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Create histogram
        sns.histplot(data=data.dropna(), kde=True, ax=ax, bins=50, alpha=0.7, color='steelblue', stat='density')
        
        # Fit and plot distributions
        fits = self.fit_distributions(data)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        x = np.linspace(data.dropna().min(), data.dropna().max(), 1000)
        
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, *fit_info['params'])
                elif dist_name == "lognormal":
                    y = lognormal.pdf(x, *fit_info['params'])
                elif dist_name == "exponential":
                    y = expon.pdf(x, *fit_info['params'])
                elif dist_name == "gamma":
                    y = gamma.pdf(x, *fit_info['params'])
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, *fit_info['params'])
                
                ax.plot(x, y, '--', linewidth=2, color=colors[i % len(colors)],
                       label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
    
    def create_comparison_histogram(self, data_dict: Dict[str, pd.Series], title: str = "Comparison Histogram") -> plt.Figure:
        """Create overlapping histograms for comparison."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        colors = sns.color_palette("husl", len(data_dict))
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
            
            sns.histplot(data=data.dropna(), kde=True, ax=ax, bins=50, alpha=0.6, 
                        color=colors[i], label=label, stat='density')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
    
    def create_faceted_histogram(self, data: pd.Series, facet_column: pd.Series, title: str = "Faceted Histogram") -> plt.Figure:
        """Create faceted histograms."""
        # Create DataFrame for faceting
        df_facet = pd.DataFrame({
            'value': data,
            'facet': facet_column
        }).dropna()
        
        if df_facet.empty:
            return plt.figure()
        
        # Create faceted plot
        g = sns.FacetGrid(df_facet, col='facet', col_wrap=3, height=4, aspect=1.5)
        g.map_dataframe(sns.histplot, x='value', kde=True, bins=30)
        g.fig.suptitle(title, fontweight='bold', y=1.02)
        g.fig.tight_layout()
        
        return g.fig
    
    def create_statistical_summary_plot(self, data: pd.Series, title: str = "Statistical Summary") -> plt.Figure:
        """Create a comprehensive statistical summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontweight='bold', fontsize=16)
        
        data_clean = data.dropna()
        stats_dict = self.calculate_advanced_statistics(data)
        
        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        sns.histplot(data=data_clean, kde=True, ax=ax1, bins=50, alpha=0.7, color='steelblue')
        ax1.set_title('Histogram with KDE')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        
        # Add statistics text
        if stats_dict:
            stats_text = f"n={stats_dict['count']:,.0f}\nμ={stats_dict['mean']:.2f}\nσ={stats_dict['std']:.2f}"
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Box plot
        ax2 = axes[0, 1]
        sns.boxplot(data=data_clean, ax=ax2, color='lightblue')
        ax2.set_title('Box Plot')
        ax2.set_ylabel('Value')
        
        # 3. Q-Q plot
        ax3 = axes[1, 0]
        stats.probplot(data_clean, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal)')
        
        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, y, 'b-', linewidth=2)
        ax4.set_title('Cumulative Distribution')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_advanced_analysis_plot(self, data: pd.Series, title: str = "Advanced Analysis") -> plt.Figure:
        """Create an advanced analysis plot with multiple subplots."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        data_clean = data.dropna()
        stats_dict = self.calculate_advanced_statistics(data)
        fits = self.fit_distributions(data)
        
        # Main histogram with distribution fits
        ax_main = fig.add_subplot(gs[0, :])
        sns.histplot(data=data_clean, kde=True, ax=ax_main, bins=50, alpha=0.7, color='steelblue', stat='density')
        
        # Add distribution fits
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        x = np.linspace(data_clean.min(), data_clean.max(), 1000)
        
        for i, (dist_name, fit_info) in enumerate(fits.items()):
            try:
                if dist_name == "normal":
                    y = norm.pdf(x, *fit_info['params'])
                elif dist_name == "lognormal":
                    y = lognormal.pdf(x, *fit_info['params'])
                elif dist_name == "exponential":
                    y = expon.pdf(x, *fit_info['params'])
                elif dist_name == "gamma":
                    y = gamma.pdf(x, *fit_info['params'])
                elif dist_name == "weibull":
                    y = weibull_min.pdf(x, *fit_info['params'])
                
                ax_main.plot(x, y, '--', linewidth=2, color=colors[i % len(colors)],
                           label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
            except Exception as e:
                print(f"Error plotting {dist_name} fit: {e}")
        
        ax_main.set_title(title, fontweight='bold')
        ax_main.set_xlabel('Value')
        ax_main.set_ylabel('Density')
        ax_main.legend()
        
        # Statistics text
        if stats_dict:
            stats_text = f"""
Advanced Statistics:
Count: {stats_dict['count']:,.0f}
Mean: {stats_dict['mean']:.2f}
Median: {stats_dict['median']:.2f}
Std Dev: {stats_dict['std']:.2f}
Skewness: {stats_dict['skewness']:.3f}
Kurtosis: {stats_dict['kurtosis']:.3f}
Outliers: {stats_dict['outliers_count']} ({stats_dict['outliers_percentage']:.1f}%)
            """
            
            ax_stats = fig.add_subplot(gs[1, 0])
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax_stats.set_xlim(0, 1)
            ax_stats.set_ylim(0, 1)
            ax_stats.axis('off')
        
        # Normality tests
        if stats_dict and 'normality_tests' in stats_dict:
            normality_text = "Normality Tests:\n"
            for test_name, test_result in stats_dict['normality_tests'].items():
                normality_text += f"{test_name.title()}: "
                normality_text += f"p={test_result['p_value']:.3f}, "
                normality_text += f"normal={test_result['normal']}\n"
            
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
        sns.boxplot(data=data_clean, ax=ax_box, color='lightblue')
        ax_box.set_title('Box Plot')
        ax_box.set_ylabel('Value')
        
        # Q-Q plot
        ax_qq = fig.add_subplot(gs[2, 1])
        stats.probplot(data_clean, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot (Normal)')
        
        # Cumulative distribution
        ax_cdf = fig.add_subplot(gs[2, 2])
        sorted_data = np.sort(data_clean)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax_cdf.plot(sorted_data, y, 'b-', linewidth=2)
        ax_cdf.set_title('Cumulative Distribution')
        ax_cdf.set_xlabel('Value')
        ax_cdf.set_ylabel('Cumulative Probability')
        ax_cdf.grid(True, alpha=0.3)
        
        # Histogram variations
        ax_hist1 = fig.add_subplot(gs[3, 0])
        sns.histplot(data=data_clean, bins=20, ax=ax_hist1, alpha=0.7, color='lightcoral')
        ax_hist1.set_title('Histogram (20 bins)')
        ax_hist1.set_xlabel('Value')
        
        ax_hist2 = fig.add_subplot(gs[3, 1])
        sns.histplot(data=data_clean, bins=100, ax=ax_hist2, alpha=0.7, color='lightgreen')
        ax_hist2.set_title('Histogram (100 bins)')
        ax_hist2.set_xlabel('Value')
        
        ax_hist3 = fig.add_subplot(gs[3, 2])
        sns.histplot(data=data_clean, bins='auto', ax=ax_hist3, alpha=0.7, color='lightblue')
        ax_hist3.set_title('Histogram (auto bins)')
        ax_hist3.set_xlabel('Value')
        
        return fig
    
    def save_plots(self, figures: List[plt.Figure], base_filename: str):
        """Save plots in multiple formats."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, fig in enumerate(figures):
            for fmt in SAVE_FORMATS:
                filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.{fmt}"
                fig.savefig(filename, dpi=DPI, bbox_inches='tight')
                print(f"Saved: {filename}")
    
    def generate_seaborn_analysis(self, queries: List[Dict[str, Any]]):
        """Generate comprehensive seaborn histogram analysis."""
        print("Generating Seaborn histogram analysis...")
        
        figures = []
        
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
                figures.append(basic_hist)
                
                dist_hist = self.create_distribution_histogram(
                    data,
                    f"Distribution Fit: {col}\n{query_info.get('description', 'Custom query')}"
                )
                figures.append(dist_hist)
                
                stat_summary = self.create_statistical_summary_plot(
                    data,
                    f"Statistical Summary: {col}\n{query_info.get('description', 'Custom query')}"
                )
                figures.append(stat_summary)
                
                advanced_analysis = self.create_advanced_analysis_plot(
                    data,
                    f"Advanced Analysis: {col}\n{query_info.get('description', 'Custom query')}"
                )
                figures.append(advanced_analysis)
            
            # Create comparison plots if multiple columns
            if len(numeric_columns) > 1:
                data_dict = {col: df[col] for col in numeric_columns}
                comp_hist = self.create_comparison_histogram(
                    data_dict,
                    f"Comparison: {query_info.get('description', 'Custom query')}"
                )
                figures.append(comp_hist)
        
        # Save all figures
        if figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_plots(figures, f"seaborn_histogram_analysis_{timestamp}")
            print(f"\nGenerated {len(figures)} Seaborn histogram plots")
        else:
            print("No plots generated - check your data and queries")

def main():
    """Main function to demonstrate Seaborn histogram capabilities."""
    print("Advanced Seaborn Histogram Generator")
    print("=" * 40)
    
    # Initialize the Seaborn histogram generator
    seaborn_gen = AdvancedSeabornHistograms()
    
    # Get database information
    db_info = seaborn_gen.get_database_info()
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
    
    # Generate Seaborn analysis
    print("\nGenerating advanced Seaborn histogram analysis...")
    seaborn_gen.generate_seaborn_analysis(example_queries)
    
    print("\nAnalysis complete! Check the 'seaborn_histogram_outputs' directory for results.")

if __name__ == "__main__":
    main() 