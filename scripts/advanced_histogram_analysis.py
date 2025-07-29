#!/usr/bin/env python3
"""
Advanced Histogram Analysis
==========================

Advanced histogram analysis with comprehensive statistical capabilities,
professional visualizations, and support for both CSV and database files.

Features:
- Interactive histogram analysis
- Advanced statistical analysis
- Distribution fitting and comparison
- Professional visualizations with great legends and labels
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
import argparse
import os
from pathlib import Path
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
import json

# Configuration
OUTPUT_DIR = "../outputs/advanced_histogram_outputs"
SAVE_FORMATS = ["png", "pdf", "svg"]
DPI = 300
FIGURE_SIZE = (14, 10)

# Statistical Analysis Settings
CONFIDENCE_LEVEL = 0.95
NORMALITY_TESTS = ["shapiro", "anderson", "ks", "jarque_bera"]
DISTRIBUTION_FITS = ["normal", "lognormal", "exponential", "gamma", "weibull"]

def load_data(data_source):
    """Load data from CSV file or database."""
    print(f"[DATA] Loading data from: {data_source}")
    
    try:
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            print(f"[OK] Loaded CSV dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            # Assume it's a database file
            conn = sqlite3.connect(data_source)
            # Get first table
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                first_table = tables[0][0]
                df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                print(f"[INFO] Loaded table: {first_table}")
            else:
                raise ValueError("No tables found in database")
            conn.close()
            
            print(f"[OK] Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            return df
            
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None

class AdvancedHistogramAnalysis:
    """Advanced histogram analysis with comprehensive statistical capabilities."""
    
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.df = load_data(data_source)
        self.setup_styles()
        
    def setup_styles(self):
        """Setup matplotlib and seaborn styles for professional appearance."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        
    def calculate_advanced_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive advanced statistics."""
        if data.empty:
            return {}
        
        data_clean = data.dropna()
        
        # Basic statistics
        stats_dict = {
            'count': len(data_clean),
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'range': data_clean.max() - data_clean.min(),
            'q25': data_clean.quantile(0.25),
            'q75': data_clean.quantile(0.75),
            'iqr': data_clean.quantile(0.75) - data_clean.quantile(0.25),
            'skewness': stats.skew(data_clean),
            'kurtosis': stats.kurtosis(data_clean),
            'cv': data_clean.std() / data_clean.mean(),  # Coefficient of variation
        }
        
        # Normality tests
        normality_results = {}
        if len(data_clean) >= 3:
            try:
                normality_results['shapiro'] = shapiro(data_clean)
            except:
                normality_results['shapiro'] = (None, None)
            
            try:
                normality_results['jarque_bera'] = jarque_bera(data_clean)
            except:
                normality_results['jarque_bera'] = (None, None)
        
        stats_dict['normality_tests'] = normality_results
        
        return stats_dict
    
    def fit_distributions(self, data: pd.Series) -> Dict[str, Dict]:
        """Fit multiple distributions to the data."""
        data_clean = data.dropna()
        if len(data_clean) < 10:
            return {}
        
        distributions = {}
        
        # Normal distribution
        try:
            mu, sigma = norm.fit(data_clean)
            distributions['normal'] = {
                'params': (mu, sigma),
                'name': 'Normal',
                'color': 'red'
            }
        except:
            pass
        
        # Log-normal distribution
        try:
            shape, loc, scale = lognorm.fit(data_clean)
            distributions['lognormal'] = {
                'params': (shape, loc, scale),
                'name': 'Log-Normal',
                'color': 'blue'
            }
        except:
            pass
        
        # Exponential distribution
        try:
            loc, scale = expon.fit(data_clean)
            distributions['exponential'] = {
                'params': (loc, scale),
                'name': 'Exponential',
                'color': 'green'
            }
        except:
            pass
        
        # Gamma distribution
        try:
            a, loc, scale = gamma.fit(data_clean)
            distributions['gamma'] = {
                'params': (a, loc, scale),
                'name': 'Gamma',
                'color': 'orange'
            }
        except:
            pass
        
        return distributions
    
    def create_comprehensive_histogram(self, column_name: str, output_file: str = "comprehensive_histogram.png"):
        """Create a comprehensive histogram analysis with professional styling."""
        if self.df is None or column_name not in self.df.columns:
            print(f"[ERROR] Column '{column_name}' not found in data.")
            return
        
        data = self.df[column_name].dropna()
        if len(data) == 0:
            print(f"[ERROR] No valid data for column '{column_name}'")
            return
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f"{OUTPUT_DIR}/{output_file}"
        
        # Calculate statistics
        stats_dict = self.calculate_advanced_statistics(data)
        distributions = self.fit_distributions(data)
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main histogram with distribution fits
        n, bins, patches = ax1.hist(data, bins=20, alpha=0.7, color='steelblue', 
                                   edgecolor='black', density=True, label='Data')
        
        # Add fitted distributions
        x = np.linspace(data.min(), data.max(), 100)
        for dist_name, dist_info in distributions.items():
            if dist_name == 'normal':
                mu, sigma = dist_info['params']
                y = norm.pdf(x, mu, sigma)
                ax1.plot(x, y, color=dist_info['color'], linewidth=2, 
                        label=f"{dist_info['name']} (μ={mu:.2f}, σ={sigma:.2f})")
            elif dist_name == 'lognormal':
                shape, loc, scale = dist_info['params']
                y = lognorm.pdf(x, shape, loc, scale)
                ax1.plot(x, y, color=dist_info['color'], linewidth=2, 
                        label=f"{dist_info['name']}")
            elif dist_name == 'exponential':
                loc, scale = dist_info['params']
                y = expon.pdf(x, loc, scale)
                ax1.plot(x, y, color=dist_info['color'], linewidth=2, 
                        label=f"{dist_info['name']}")
            elif dist_name == 'gamma':
                a, loc, scale = dist_info['params']
                y = gamma.pdf(x, a, loc, scale)
                ax1.plot(x, y, color=dist_info['color'], linewidth=2, 
                        label=f"{dist_info['name']}")
        
        ax1.set_title(f'Histogram with Distribution Fits: {column_name}', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel(column_name, fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title(f'Box Plot: {column_name}', fontweight='bold', fontsize=14)
        ax2.set_ylabel(column_name, fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=ax3)
        ax3.set_title(f'Q-Q Plot: {column_name}', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, y, 'b-', linewidth=2, label='Empirical CDF')
        
        # Add theoretical normal CDF
        mu, sigma = data.mean(), data.std()
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = norm.cdf(x_norm, mu, sigma)
        ax4.plot(x_norm, y_norm, 'r--', linewidth=2, label=f'Normal CDF (μ={mu:.2f}, σ={sigma:.2f})')
        
        ax4.set_title(f'Cumulative Distribution: {column_name}', fontweight='bold', fontsize=14)
        ax4.set_xlabel(column_name, fontsize=12)
        ax4.set_ylabel('Cumulative Probability', fontsize=12)
        ax4.legend(fontsize=10, framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Add comprehensive statistics
        stats_text = f"""Statistics Summary:
Count: {stats_dict['count']:,}
Mean: {stats_dict['mean']:.2f}
Median: {stats_dict['median']:.2f}
Std: {stats_dict['std']:.2f}
Skewness: {stats_dict['skewness']:.3f}
Kurtosis: {stats_dict['kurtosis']:.3f}
Range: {stats_dict['range']:.2f}
IQR: {stats_dict['iqr']:.2f}"""
        
        fig.suptitle(f'Advanced Histogram Analysis: {column_name}\n{stats_text}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"[OK] Comprehensive histogram saved as: {output_path}")
        plt.close()
        
        return stats_dict
    
    def create_comparative_analysis(self, column1: str, column2: str, 
                                  output_file: str = "comparative_analysis.png"):
        """Create comparative analysis between two columns."""
        if self.df is None:
            return
        
        if column1 not in self.df.columns or column2 not in self.df.columns:
            print(f"[ERROR] One or both columns not found: {column1}, {column2}")
            return
        
        data1 = self.df[column1].dropna()
        data2 = self.df[column2].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            print(f"[ERROR] No valid data for comparison")
            return
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f"{OUTPUT_DIR}/{output_file}"
        
        # Create comparative plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overlapping histograms
        ax1.hist(data1, bins=20, alpha=0.7, color='blue', label=column1, density=True)
        ax1.hist(data2, bins=20, alpha=0.7, color='red', label=column2, density=True)
        ax1.set_title(f'Comparative Histograms: {column1} vs {column2}', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Values', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Box plots side by side
        bp = ax2.boxplot([data1, data2], labels=[column1, column2], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_title(f'Box Plot Comparison: {column1} vs {column2}', 
                     fontweight='bold', fontsize=14)
        ax2.set_ylabel('Values', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot
        min_len = min(len(data1), len(data2))
        ax3.scatter(data1[:min_len], data2[:min_len], alpha=0.6, s=50)
        ax3.set_title(f'Scatter Plot: {column1} vs {column2}', 
                     fontweight='bold', fontsize=14)
        ax3.set_xlabel(column1, fontsize=12)
        ax3.set_ylabel(column2, fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Statistics comparison
        stats1 = self.calculate_advanced_statistics(data1)
        stats2 = self.calculate_advanced_statistics(data2)
        
        comparison_text = f"""Statistical Comparison:
{column1}:
  Mean: {stats1['mean']:.2f}
  Std: {stats1['std']:.2f}
  Count: {stats1['count']:,}

{column2}:
  Mean: {stats2['mean']:.2f}
  Std: {stats2['std']:.2f}
  Count: {stats2['count']:,}

Correlation: {data1.corr(data2):.3f}"""
        
        ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax4.set_title('Statistical Comparison', fontweight='bold', fontsize=14)
        ax4.axis('off')
        
        fig.suptitle(f'Comparative Analysis: {column1} vs {column2}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"[OK] Comparative analysis saved as: {output_path}")
        plt.close()
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis on all numeric columns."""
        if self.df is None:
            print("[ERROR] No data loaded")
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("[ERROR] No numeric columns found for analysis")
            return
        
        print(f"[ANALYSIS] Running comprehensive analysis on {len(numeric_cols)} numeric columns")
        
        for i, col in enumerate(numeric_cols):
            print(f"[PROCESSING] Analyzing column {i+1}/{len(numeric_cols)}: {col}")
            self.create_comprehensive_histogram(col, f"comprehensive_histogram_{col}.png")
        
        # Create comparative analysis if we have at least 2 numeric columns
        if len(numeric_cols) >= 2:
            print(f"[COMPARISON] Creating comparative analysis between {numeric_cols[0]} and {numeric_cols[1]}")
            self.create_comparative_analysis(numeric_cols[0], numeric_cols[1], 
                                          "comparative_analysis.png")
        
        print(f"[OK] Comprehensive analysis complete! Check {OUTPUT_DIR} for results")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Advanced histogram analysis with professional visualizations')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--column', help='Specific column to analyze (default: all numeric columns)')
    parser.add_argument('--compare', nargs=2, help='Two columns to compare')
    
    args = parser.parse_args()
    
    analyzer = AdvancedHistogramAnalysis(args.data_source)
    
    if args.column:
        analyzer.create_comprehensive_histogram(args.column)
    elif args.compare:
        analyzer.create_comparative_analysis(args.compare[0], args.compare[1])
    else:
        analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main() 