#!/usr/bin/env python3
"""
Histogram Permutations and Filtering Generator
=============================================

Advanced histogram generation with query permutations, filtering combinations,
and comprehensive analysis capabilities.

Features:
- Query permutation generation
- Advanced filtering combinations
- Multiple visualization styles
- Statistical comparison
- Export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma
import itertools
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from pathlib import Path
from datetime import datetime

# Configuration
DATABASE_PATH = "sample_data.sqlite"
OUTPUT_DIR = "histogram_permutations"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300
FIGURE_SIZE = (12, 8)

class HistogramPermutations:
    """Advanced histogram generator with permutation and filtering capabilities."""
    
    def __init__(self, database_path: str = DATABASE_PATH):
        self.database_path = database_path
        self.data_cache = {}
        self.setup_styles()
        
    def setup_styles(self):
        """Setup matplotlib and seaborn styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
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
    
    def get_column_values(self, table: str, column: str, limit: int = 100) -> List[Any]:
        """Get unique values for a specific column."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}"
                df = pd.read_sql_query(query, conn)
                return df[column].tolist()
        except Exception as e:
            print(f"Error getting values for column {column}: {e}")
            return []
    
    def generate_filter_combinations(self, 
                                   filters: Dict[str, List],
                                   max_combinations: int = 20) -> List[Dict[str, Any]]:
        """Generate filter combinations for histogram analysis."""
        combinations = []
        
        # Generate all possible combinations of filter keys
        for i in range(1, min(len(filters) + 1, 4)):
            for key_combo in itertools.combinations(filters.keys(), i):
                # Generate value combinations for these keys
                value_lists = [filters[k] for k in key_combo]
                
                for value_combo in itertools.product(*value_lists):
                    combination = dict(zip(key_combo, value_combo))
                    combinations.append({
                        'filters': combination,
                        'description': f"Filters: {combination}"
                    })
                    
                    if len(combinations) >= max_combinations:
                        break
                if len(combinations) >= max_combinations:
                    break
        
        return combinations[:max_combinations]
    
    def generate_query_permutations(self,
                                  table: str,
                                  columns: List[str],
                                  filters: Dict[str, List] = None,
                                  max_permutations: int = 15) -> List[Dict[str, Any]]:
        """Generate query permutations with various filter combinations."""
        permutations = []
        
        if not filters:
            # Simple column combinations
            for i in range(1, min(len(columns) + 1, 4)):
                for combo in itertools.combinations(columns, i):
                    permutations.append({
                        'table': table,
                        'columns': list(combo),
                        'conditions': '1=1',
                        'description': f"Columns: {', '.join(combo)}",
                        'filters': {}
                    })
        else:
            # Generate filter combinations
            filter_combinations = self.generate_filter_combinations(filters, max_permutations)
            
            for combo in filter_combinations:
                conditions = []
                for key, value in combo['filters'].items():
                    if isinstance(value, str):
                        conditions.append(f"{key} = '{value}'")
                    else:
                        conditions.append(f"{key} = {value}")
                
                permutations.append({
                    'table': table,
                    'columns': columns,
                    'conditions': ' AND '.join(conditions),
                    'description': combo['description'],
                    'filters': combo['filters']
                })
        
        return permutations[:max_permutations]
    
    def execute_query(self, query_config: Dict[str, Any]) -> pd.DataFrame:
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
    
    def calculate_comparison_statistics(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate comparison statistics for multiple datasets."""
        stats_list = []
        
        for label, data in data_dict.items():
            if data.empty:
                continue
            
            data_clean = data.dropna()
            
            stats = {
                'label': label,
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
                stats['shapiro_statistic'] = stat
                stats['shapiro_p_value'] = p_value
                stats['is_normal'] = p_value > 0.05
            except:
                stats['shapiro_statistic'] = np.nan
                stats['shapiro_p_value'] = np.nan
                stats['is_normal'] = False
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def create_permutation_histograms(self,
                                    permutations: List[Dict[str, Any]],
                                    column: str,
                                    bins: Union[int, str] = 'auto',
                                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create histograms for multiple permutations."""
        # Filter permutations that have the target column
        valid_permutations = [p for p in permutations if column in p['columns']]
        
        if not valid_permutations:
            raise ValueError(f"Column '{column}' not found in any permutation")
        
        # Calculate grid dimensions
        n_plots = len(valid_permutations)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, perm in enumerate(valid_permutations):
            ax = axes_flat[i]
            
            # Execute query
            df = self.execute_query(perm)
            
            if df.empty or column not in df.columns:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Permutation {i+1}\n{perm['description'][:50]}...", fontsize=10)
                continue
            
            data = df[column].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Permutation {i+1}\n{perm['description'][:50]}...", fontsize=10)
                continue
            
            # Create histogram
            ax.hist(data, bins=bins, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            stats_text = f"n={len(data):.0f}\nμ={data.mean():.2f}\nσ={data.std():.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"Permutation {i+1}\n{perm['description'][:50]}...", fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(valid_permutations), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        fig.suptitle(f'Histogram Permutations for Column: {column}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_comparison_analysis(self,
                                 permutations: List[Dict[str, Any]],
                                 column: str) -> plt.Figure:
        """Create comprehensive comparison analysis for permutations."""
        # Collect data from all permutations
        data_dict = {}
        
        for i, perm in enumerate(permutations):
            if column not in perm['columns']:
                continue
            
            df = self.execute_query(perm)
            if not df.empty and column in df.columns:
                label = f"Perm {i+1}: {perm['description'][:30]}..."
                data_dict[label] = df[column]
        
        if not data_dict:
            raise ValueError(f"No data found for column '{column}' in any permutation")
        
        # Create comparison figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Overlapping histograms
        ax_main = fig.add_subplot(gs[0, :])
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))
        
        for i, (label, data) in enumerate(data_dict.items()):
            if data.empty:
                continue
            ax_main.hist(data.dropna(), bins=50, alpha=0.6, density=True,
                        label=label, color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax_main.set_title(f'Overlapping Histograms: {column}', fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Value', fontsize=12)
        ax_main.set_ylabel('Density', fontsize=12)
        ax_main.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Statistical comparison
        stats_df = self.calculate_comparison_statistics(data_dict)
        
        if not stats_df.empty:
            # Mean comparison
            ax_mean = fig.add_subplot(gs[1, 0])
            bars = ax_mean.bar(range(len(stats_df)), stats_df['mean'], 
                             color='skyblue', alpha=0.7)
            ax_mean.set_title('Mean Comparison', fontsize=12)
            ax_mean.set_ylabel('Mean', fontsize=10)
            ax_mean.set_xticks(range(len(stats_df)))
            ax_mean.set_xticklabels([f"P{i+1}" for i in range(len(stats_df))], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_df['mean']):
                ax_mean.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2f}', ha='center', va='bottom')
            
            # Standard deviation comparison
            ax_std = fig.add_subplot(gs[1, 1])
            bars = ax_std.bar(range(len(stats_df)), stats_df['std'], 
                            color='lightcoral', alpha=0.7)
            ax_std.set_title('Standard Deviation Comparison', fontsize=12)
            ax_std.set_ylabel('Standard Deviation', fontsize=10)
            ax_std.set_xticks(range(len(stats_df)))
            ax_std.set_xticklabels([f"P{i+1}" for i in range(len(stats_df))], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_df['std']):
                ax_std.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2f}', ha='center', va='bottom')
            
            # Sample size comparison
            ax_count = fig.add_subplot(gs[1, 2])
            bars = ax_count.bar(range(len(stats_df)), stats_df['count'], 
                              color='lightgreen', alpha=0.7)
            ax_count.set_title('Sample Size Comparison', fontsize=12)
            ax_count.set_ylabel('Count', fontsize=10)
            ax_count.set_xticks(range(len(stats_df)))
            ax_count.set_xticklabels([f"P{i+1}" for i in range(len(stats_df))], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_df['count']):
                ax_count.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{value:.0f}', ha='center', va='bottom')
        
        return fig
    
    def create_statistical_summary(self,
                                 permutations: List[Dict[str, Any]],
                                 column: str) -> pd.DataFrame:
        """Create a comprehensive statistical summary for all permutations."""
        summary_data = []
        
        for i, perm in enumerate(permutations):
            if column not in perm['columns']:
                continue
            
            df = self.execute_query(perm)
            if df.empty or column not in df.columns:
                continue
            
            data = df[column].dropna()
            
            if len(data) == 0:
                continue
            
            # Calculate comprehensive statistics
            stats = {
                'permutation_id': i + 1,
                'description': perm['description'],
                'filters': str(perm.get('filters', {})),
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
            
            # Normality test
            try:
                stat, p_value = stats.shapiro(data)
                stats['shapiro_statistic'] = stat
                stats['shapiro_p_value'] = p_value
                stats['is_normal'] = p_value > 0.05
            except:
                stats['shapiro_statistic'] = np.nan
                stats['shapiro_p_value'] = np.nan
                stats['is_normal'] = False
            
            summary_data.append(stats)
        
        return pd.DataFrame(summary_data)
    
    def save_plots(self, figures: List[plt.Figure], base_filename: str):
        """Save plots in multiple formats."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        for i, fig in enumerate(figures):
            for fmt in SAVE_FORMATS:
                filename = f"{OUTPUT_DIR}/{base_filename}_{i+1}.{fmt}"
                fig.savefig(filename, dpi=DPI, bbox_inches='tight')
                print(f"Saved: {filename}")
    
    def generate_permutation_analysis(self,
                                    table: str,
                                    columns: List[str],
                                    filters: Dict[str, List] = None,
                                    max_permutations: int = 10):
        """Generate comprehensive permutation analysis."""
        print(f"Generating permutation analysis for table: {table}")
        print(f"Columns: {columns}")
        print(f"Filters: {filters}")
        
        # Generate permutations
        permutations = self.generate_query_permutations(
            table, columns, filters, max_permutations
        )
        
        print(f"Generated {len(permutations)} permutations")
        
        figures = []
        summaries = []
        
        # Create analysis for each column
        for col in columns:
            print(f"\nAnalyzing column: {col}")
            
            # Create permutation histograms
            try:
                fig_hist = self.create_permutation_histograms(permutations, col)
                figures.append(fig_hist)
                print(f"Created permutation histograms for {col}")
            except Exception as e:
                print(f"Error creating permutation histograms for {col}: {e}")
            
            # Create comparison analysis
            try:
                fig_comp = self.create_comparison_analysis(permutations, col)
                figures.append(fig_comp)
                print(f"Created comparison analysis for {col}")
            except Exception as e:
                print(f"Error creating comparison analysis for {col}: {e}")
            
            # Create statistical summary
            try:
                summary = self.create_statistical_summary(permutations, col)
                summaries.append(summary)
                print(f"Created statistical summary for {col}")
            except Exception as e:
                print(f"Error creating statistical summary for {col}: {e}")
        
        # Save all figures
        if figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_plots(figures, f"histogram_permutations_{timestamp}")
            print(f"\nSaved {len(figures)} permutation analysis plots")
        
        # Save summaries
        if summaries:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"{OUTPUT_DIR}/statistical_summary_{timestamp}.csv"
            
            # Combine all summaries
            combined_summary = pd.concat(summaries, ignore_index=True)
            combined_summary.to_csv(summary_filename, index=False)
            print(f"Saved statistical summary: {summary_filename}")
        
        return figures, summaries

def main():
    """Main function to demonstrate histogram permutation capabilities."""
    print("Histogram Permutations and Filtering Generator")
    print("=" * 50)
    
    # Initialize the permutation generator
    perm_gen = HistogramPermutations()
    
    # Example configuration
    table = "sales"
    columns = ["amount", "quantity", "profit_margin"]
    
    # Example filters
    filters = {
        'region': ['North', 'South', 'East', 'West'],
        'category': ['Electronics', 'Clothing', 'Books'],
        'amount': [100, 500, 1000]  # Threshold values
    }
    
    # Generate permutation analysis
    print("\nGenerating comprehensive permutation analysis...")
    figures, summaries = perm_gen.generate_permutation_analysis(
        table=table,
        columns=columns,
        filters=filters,
        max_permutations=8
    )
    
    print("\nPermutation analysis complete!")
    print(f"Generated {len(figures)} figures and {len(summaries)} summaries")
    print("Check the 'histogram_permutations' directory for results.")

if __name__ == "__main__":
    main() 