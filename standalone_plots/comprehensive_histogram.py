#!/usr/bin/env python3
"""
Comprehensive Histogram Generator
================================

Advanced histogram plotting with overlapping distributions, query permutations,
statistical analysis, and multiple visualization options.

Features:
- Overlapping histograms with transparency
- Query permutation generation
- Statistical analysis (normality tests, distribution fitting)
- Multiple visualization styles (matplotlib, bokeh)
- Interactive plots
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma, shapiro, anderson, kstest, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
import warnings
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path

# Configuration
DATABASE_PATH = "sample_data.sqlite"  # Change to your database
OUTPUT_DIR = "histogram_outputs"
SAVE_FORMATS = ["png", "pdf", "svg"]  # Multiple formats
DPI = 300
FIGURE_SIZE = (12, 8)

# Query Configuration
BASE_QUERY = "SELECT {columns} FROM {table} WHERE {conditions}"
DEFAULT_TABLE = "sales"
DEFAULT_COLUMNS = ["amount", "quantity", "profit_margin"]

# Statistical Analysis Settings
CONFIDENCE_LEVEL = 0.95
NORMALITY_TESTS = ["shapiro", "anderson", "ks"]
DISTRIBUTION_FITS = ["normal", "lognormal", "exponential", "gamma"]

class ComprehensiveHistogram:
    """Advanced histogram generator with comprehensive analysis capabilities."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        self.statistics_cache = {}
        self.setup_styles()
        
    def setup_styles(self):
        """Setup matplotlib and seaborn styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def get_database_tables(self) -> List[str]:
        """Get all available tables in the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error accessing database: {e}")
            return []
    
    def get_table_columns(self, table: str) -> List[str]:
        """Get all columns for a specific table."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table});")
                return [row[1] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting columns for table {table}: {e}")
            return []
    
    def generate_query_permutations(self, 
                                  table: str,
                                  columns: List[str],
                                  filters: Dict[str, List] = None,
                                  max_combinations: int = 10) -> List[Dict]:
        """Generate query permutations based on filters."""
        permutations = []
        
        if not filters:
            # Simple column combinations
            for i in range(1, min(len(columns) + 1, 4)):
                for combo in itertools.combinations(columns, i):
                    permutations.append({
                        'table': table,
                        'columns': list(combo),
                        'conditions': '1=1',
                        'description': f"Columns: {', '.join(combo)}"
                    })
        else:
            # Complex filter combinations
            filter_keys = list(filters.keys())
            filter_values = list(filters.values())
            
            for i in range(1, min(len(filter_keys) + 1, 4)):
                for key_combo in itertools.combinations(filter_keys, i):
                    for value_combo in itertools.product(*[filters[k] for k in key_combo]):
                        conditions = []
                        for key, value in zip(key_combo, value_combo):
                            if isinstance(value, str):
                                conditions.append(f"{key} = '{value}'")
                            else:
                                conditions.append(f"{key} = {value}")
                        
                        permutations.append({
                            'table': table,
                            'columns': columns,
                            'conditions': ' AND '.join(conditions),
                            'description': f"Filters: {dict(zip(key_combo, value_combo))}"
                        })
                        
                        if len(permutations) >= max_combinations:
                            break
                    if len(permutations) >= max_combinations:
                        break
        
        return permutations[:max_combinations]
    
    def execute_query(self, query_config: Dict) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        cache_key = f"{query_config['table']}_{query_config['conditions']}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = f"""
                SELECT {', '.join(query_config['columns'])} 
                FROM {query_config['table']} 
                WHERE {query_config['conditions']}
                """
                df = pd.read_sql_query(query, conn)
                self.data_cache[cache_key] = df
                return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def calculate_statistics(self, data: pd.Series) -> Dict:
        """Calculate comprehensive statistics for a data series."""
        if data.empty:
            return {}
        
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'coefficient_of_variation': data.std() / data.mean() if data.mean() != 0 else 0
        }
        
        # Normality tests
        for test_name in NORMALITY_TESTS:
            try:
                if test_name == "shapiro":
                    stat, p_value = shapiro(data.dropna())
                elif test_name == "anderson":
                    result = anderson(data.dropna())
                    stat, p_value = result.statistic, result.significance_level[2]
                elif test_name == "ks":
                    stat, p_value = kstest(data.dropna(), 'norm', 
                                         args=(data.mean(), data.std()))
                
                stats[f'{test_name}_statistic'] = stat
                stats[f'{test_name}_p_value'] = p_value
                stats[f'{test_name}_normal'] = p_value > (1 - CONFIDENCE_LEVEL)
            except Exception as e:
                print(f"Error in {test_name} test: {e}")
        
        return stats
    
    def fit_distributions(self, data: pd.Series) -> Dict:
        """Fit various theoretical distributions to the data."""
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
                    fits[dist_name] = {
                        'params': params,
                        'loc': params[0],
                        'scale': params[1],
                        'aic': aic
                    }
                elif dist_name == "lognormal":
                    params = lognorm.fit(data_clean)
                    log_likelihood = np.sum(lognorm.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    fits[dist_name] = {
                        'params': params,
                        'shape': params[0],
                        'loc': params[1],
                        'scale': params[2],
                        'aic': aic
                    }
                elif dist_name == "exponential":
                    params = expon.fit(data_clean)
                    log_likelihood = np.sum(expon.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    fits[dist_name] = {
                        'params': params,
                        'loc': params[0],
                        'scale': params[1],
                        'aic': aic
                    }
                elif dist_name == "gamma":
                    params = gamma.fit(data_clean)
                    log_likelihood = np.sum(gamma.logpdf(data_clean, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    fits[dist_name] = {
                        'params': params,
                        'shape': params[0],
                        'loc': params[1],
                        'scale': params[2],
                        'aic': aic
                    }
            except Exception as e:
                print(f"Error fitting {dist_name} distribution: {e}")
        
        return fits
    
    def create_overlapping_histograms(self, 
                                    data_dict: Dict[str, pd.Series],
                                    title: str = "Overlapping Histograms",
                                    bins: Union[int, str] = 'auto',
                                    alpha: float = 0.6,
                                    density: bool = True) -> plt.Figure:
        """Create overlapping histograms for multiple datasets."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
                
            ax.hist(data.dropna(), bins=bins, alpha=alpha, density=density,
                   label=label, color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density' if density else 'Frequency', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_comprehensive_histogram(self,
                                     data: pd.Series,
                                     title: str = "Comprehensive Histogram Analysis",
                                     bins: Union[int, str] = 'auto',
                                     show_stats: bool = True,
                                     show_fits: bool = True) -> plt.Figure:
        """Create a comprehensive histogram with statistics and distribution fits."""
        if data.empty:
            raise ValueError("Data is empty")
        
        # Create subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main histogram
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.hist(data.dropna(), bins=bins, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Add distribution fits
        if show_fits:
            fits = self.fit_distributions(data)
            x = np.linspace(data.min(), data.max(), 1000)
            
            for dist_name, fit_info in fits.items():
                try:
                    if dist_name == "normal":
                        y = norm.pdf(x, fit_info['loc'], fit_info['scale'])
                        ax_main.plot(x, y, '--', linewidth=2, 
                                   label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
                    elif dist_name == "lognormal":
                        y = lognorm.pdf(x, fit_info['shape'], fit_info['loc'], fit_info['scale'])
                        ax_main.plot(x, y, '--', linewidth=2,
                                   label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
                    elif dist_name == "exponential":
                        y = expon.pdf(x, fit_info['loc'], fit_info['scale'])
                        ax_main.plot(x, y, '--', linewidth=2,
                                   label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
                    elif dist_name == "gamma":
                        y = gamma.pdf(x, fit_info['shape'], fit_info['loc'], fit_info['scale'])
                        ax_main.plot(x, y, '--', linewidth=2,
                                   label=f'{dist_name} (AIC: {fit_info["aic"]:.2f})')
                except Exception as e:
                    print(f"Error plotting {dist_name} fit: {e}")
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Value', fontsize=12)
        ax_main.set_ylabel('Density', fontsize=12)
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Statistics text
        if show_stats:
            stats = self.calculate_statistics(data)
            stats_text = f"""
Statistics:
Count: {stats.get('count', 'N/A'):,.0f}
Mean: {stats.get('mean', 'N/A'):.2f}
Median: {stats.get('median', 'N/A'):.2f}
Std Dev: {stats.get('std', 'N/A'):.2f}
Min: {stats.get('min', 'N/A'):.2f}
Max: {stats.get('max', 'N/A'):.2f}
Skewness: {stats.get('skewness', 'N/A'):.3f}
Kurtosis: {stats.get('kurtosis', 'N/A'):.3f}
            """
            
            ax_stats = fig.add_subplot(gs[1, 0])
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax_stats.set_xlim(0, 1)
            ax_stats.set_ylim(0, 1)
            ax_stats.axis('off')
        
        # Normality tests
        if show_stats:
            normality_text = "Normality Tests:\n"
            for test_name in NORMALITY_TESTS:
                stat_key = f'{test_name}_statistic'
                p_key = f'{test_name}_p_value'
                normal_key = f'{test_name}_normal'
                
                if stat_key in stats:
                    normality_text += f"{test_name.title()}: "
                    normality_text += f"stat={stats[stat_key]:.3f}, "
                    normality_text += f"p={stats[p_key]:.3f}, "
                    normality_text += f"normal={stats[normal_key]}\n"
            
            ax_norm = fig.add_subplot(gs[1, 1])
            ax_norm.text(0.05, 0.95, normality_text, transform=ax_norm.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax_norm.set_xlim(0, 1)
            ax_norm.set_ylim(0, 1)
            ax_norm.axis('off')
        
        # Box plot
        ax_box = fig.add_subplot(gs[2, 0])
        ax_box.boxplot(data.dropna(), vert=False)
        ax_box.set_title('Box Plot', fontsize=12)
        ax_box.set_xlabel('Value', fontsize=10)
        
        # Q-Q plot
        ax_qq = fig.add_subplot(gs[2, 1])
        probplot(data.dropna(), dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot (Normal)', fontsize=12)
        
        return fig
    
    def create_interactive_histogram(self, data: pd.Series, title: str = "Interactive Histogram"):
        """Create an interactive histogram using bokeh (if available)."""
        try:
            from bokeh.plotting import figure, show, output_file
            from bokeh.layouts import column
            from bokeh.models import HoverTool, ColumnDataSource
            
            # Create histogram data
            hist, edges = np.histogram(data.dropna(), bins=50, density=True)
            
            # Create bokeh figure
            p = figure(title=title, width=800, height=600)
            
            # Add histogram
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   fill_color='skyblue', line_color='black', alpha=0.7)
            
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
    
    def save_plots(self, figures: List[plt.Figure], base_filename: str):
        """Save plots in multiple formats."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, fig in enumerate(figures):
            for fmt in SAVE_FORMATS:
                filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.{fmt}"
                fig.savefig(filename, dpi=DPI, bbox_inches='tight')
                print(f"Saved: {filename}")
    
    def generate_comprehensive_analysis(self,
                                      table: str = DEFAULT_TABLE,
                                      columns: List[str] = None,
                                      filters: Dict[str, List] = None,
                                      max_permutations: int = 5):
        """Generate comprehensive histogram analysis with permutations."""
        if columns is None:
            columns = DEFAULT_COLUMNS
        
        # Get query permutations
        permutations = self.generate_query_permutations(
            table, columns, filters, max_permutations
        )
        
        figures = []
        
        for i, perm in enumerate(permutations):
            print(f"\nProcessing permutation {i+1}: {perm['description']}")
            
            # Execute query
            df = self.execute_query(perm)
            
            if df.empty:
                print(f"No data for permutation {i+1}")
                continue
            
            # Create comprehensive histogram for each column
            for col in perm['columns']:
                if col in df.columns:
                    data = df[col]
                    
                    # Create comprehensive histogram
                    fig = self.create_comprehensive_histogram(
                        data, 
                        title=f"Comprehensive Analysis: {col}\n{perm['description']}"
                    )
                    figures.append(fig)
                    
                    # Create overlapping histogram if multiple columns
                    if len(perm['columns']) > 1:
                        data_dict = {col: df[col] for col in perm['columns'] if col in df.columns}
                        fig_overlap = self.create_overlapping_histograms(
                            data_dict,
                            title=f"Overlapping Histograms\n{perm['description']}"
                        )
                        figures.append(fig_overlap)
        
        # Save all figures
        if figures:
            self.save_plots(figures, "comprehensive_histogram_analysis")
            print(f"\nGenerated {len(figures)} comprehensive histogram plots")
        else:
            print("No figures generated - check your data and queries")

def main():
    """Main function to demonstrate comprehensive histogram capabilities."""
    print("Comprehensive Histogram Generator")
    print("=" * 40)
    
    # Initialize the histogram generator
    hist_gen = ComprehensiveHistogram()
    
    # Get available tables
    tables = hist_gen.get_database_tables()
    print(f"Available tables: {tables}")
    
    if not tables:
        print("No tables found in database. Please check your database path.")
        return
    
    # Example filters for demonstration
    example_filters = {
        'region': ['North', 'South', 'East', 'West'],
        'category': ['Electronics', 'Clothing', 'Books']
    }
    
    # Generate comprehensive analysis
    print("\nGenerating comprehensive histogram analysis...")
    hist_gen.generate_comprehensive_analysis(
        table='sales',
        columns=['amount', 'quantity', 'profit_margin'],
        filters=example_filters,
        max_permutations=3
    )
    
    print("\nAnalysis complete! Check the 'histogram_outputs' directory for results.")

if __name__ == "__main__":
    main() 