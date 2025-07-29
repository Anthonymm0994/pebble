#!/usr/bin/env python3
"""
Advanced Histogram Explorer
==========================

An advanced tool for exploring data distributions with interactive features
and comprehensive statistical analysis.

Features:
- Interactive histogram exploration
- Advanced statistical analysis
- Distribution comparison tools
- Outlier detection
- Trend analysis
- Custom binning strategies
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, expon, gamma, beta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AdvancedHistogramExplorer:
    """
    Advanced histogram exploration tool with statistical analysis.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the advanced histogram explorer.
        
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
    
    def analyze_distribution_fit(self, data: pd.Series, column_name: str) -> Dict:
        """Analyze how well different distributions fit the data."""
        print(f"\n[STATS] Analyzing distribution fit for: {column_name}")
        
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return {}
        
        # Test different distributions
        distributions = {
            'normal': norm,
            'exponential': expon,
            'gamma': gamma,
            'beta': beta
        }
        
        results = {}
        
        for dist_name, dist_func in distributions.items():
            try:
                # Fit the distribution
                params = dist_func.fit(clean_data)
                
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.kstest(clean_data, dist_name, params)
                
                # Calculate AIC and BIC
                log_likelihood = dist_func.logpdf(clean_data, *params).sum()
                aic = 2 * len(params) - 2 * log_likelihood
                bic = len(params) * np.log(len(clean_data)) - 2 * log_likelihood
                
                results[dist_name] = {
                    'params': params,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood
                }
                
            except Exception as e:
                print(f"[WARNING] Could not fit {dist_name} distribution: {e}")
        
        return results
    
    def detect_outliers(self, data: pd.Series, method: str = 'iqr') -> Dict:
        """Detect outliers using various methods."""
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return {}
        
        outliers = {}
        
        if method == 'iqr':
            # IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = (clean_data < lower_bound) | (clean_data > upper_bound)
            outliers['iqr'] = {
                'indices': outlier_indices,
                'count': outlier_indices.sum(),
                'percentage': (outlier_indices.sum() / len(clean_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(clean_data))
            outlier_indices = z_scores > 3
            
            outliers['zscore'] = {
                'indices': outlier_indices,
                'count': outlier_indices.sum(),
                'percentage': (outlier_indices.sum() / len(clean_data)) * 100,
                'threshold': 3
            }
        
        elif method == 'both':
            # Both methods
            outliers.update(self.detect_outliers(data, 'iqr'))
            outliers.update(self.detect_outliers(data, 'zscore'))
        
        return outliers
    
    def create_advanced_histogram(self, column_name: str, bins: int = 30):
        """Create advanced histogram with statistical analysis."""
        print(f"\n[CHART] Creating advanced histogram for: {column_name}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Advanced Histogram Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        # Get data from both datasets
        source_data = self.df_source[column_name].dropna() if column_name in self.df_source.columns else pd.Series()
        derived_data = self.df_derived[column_name].dropna() if column_name in self.df_derived.columns else pd.Series()
        
        # 1. Basic histogram with statistics
        if len(source_data) > 0:
            axes[0, 0].hist(source_data, bins=bins, alpha=0.7, color='blue', edgecolor='black', density=True)
            axes[0, 0].set_title(f'Source Distribution: {column_name}')
            axes[0, 0].set_xlabel(column_name)
            axes[0, 0].set_ylabel('Density')
            
            # Add statistics text
            stats_text = f'Mean: {source_data.mean():.3f}\nStd: {source_data.std():.3f}\nSkew: {source_data.skew():.3f}\nKurt: {source_data.kurtosis():.3f}'
            axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if len(derived_data) > 0:
            axes[0, 1].hist(derived_data, bins=bins, alpha=0.7, color='orange', edgecolor='black', density=True)
            axes[0, 1].set_title(f'Derived Distribution: {column_name}')
            axes[0, 1].set_xlabel(column_name)
            axes[0, 1].set_ylabel('Density')
            
            # Add statistics text
            stats_text = f'Mean: {derived_data.mean():.3f}\nStd: {derived_data.std():.3f}\nSkew: {derived_data.skew():.3f}\nKurt: {derived_data.kurtosis():.3f}'
            axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Box plots for comparison
        if len(source_data) > 0 or len(derived_data) > 0:
            data_to_plot = []
            labels = []
            
            if len(source_data) > 0:
                data_to_plot.append(source_data)
                labels.append('Source')
            
            if len(derived_data) > 0:
                data_to_plot.append(derived_data)
                labels.append('Derived')
            
            axes[0, 2].boxplot(data_to_plot, labels=labels)
            axes[0, 2].set_title(f'Box Plot Comparison: {column_name}')
            axes[0, 2].set_ylabel(column_name)
        
        # 3. Q-Q plots for normality testing
        if len(source_data) > 0:
            stats.probplot(source_data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title(f'Q-Q Plot - Source: {column_name}')
        
        if len(derived_data) > 0:
            stats.probplot(derived_data, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f'Q-Q Plot - Derived: {column_name}')
        
        # 4. Distribution comparison
        if len(source_data) > 0 and len(derived_data) > 0:
            # Overlay histograms
            axes[1, 2].hist(source_data, bins=bins, alpha=0.7, label='Source', color='blue', density=True)
            axes[1, 2].hist(derived_data, bins=bins, alpha=0.7, label='Derived', color='orange', density=True)
            axes[1, 2].set_title(f'Distribution Comparison: {column_name}')
            axes[1, 2].set_xlabel(column_name)
            axes[1, 2].set_ylabel('Density')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'advanced_histogram_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Advanced histogram saved as 'advanced_histogram_{column_name}.png'")
    
    def create_statistical_analysis_plot(self, column_name: str):
        """Create comprehensive statistical analysis plot."""
        print(f"\n[STATS] Creating statistical analysis for: {column_name}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Statistical Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_data = self.df_source[column_name].dropna() if column_name in self.df_source.columns else pd.Series()
        derived_data = self.df_derived[column_name].dropna() if column_name in self.df_derived.columns else pd.Series()
        
        # 1. Distribution fit analysis
        if len(source_data) > 0:
            # Fit distributions
            dist_results = self.analyze_distribution_fit(source_data, column_name)
            
            # Plot histogram with fitted distributions
            axes[0, 0].hist(source_data, bins=30, alpha=0.7, density=True, color='blue', label='Data')
            
            x = np.linspace(source_data.min(), source_data.max(), 100)
            
            for dist_name, result in dist_results.items():
                if dist_name == 'normal':
                    y = norm.pdf(x, *result['params'])
                    axes[0, 0].plot(x, y, label=f'{dist_name} (p={result["p_value"]:.3f})')
            
            axes[0, 0].set_title(f'Distribution Fit - Source: {column_name}')
            axes[0, 0].set_xlabel(column_name)
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
        
        # 2. Outlier analysis
        if len(source_data) > 0:
            outliers = self.detect_outliers(source_data, 'both')
            
            axes[0, 1].hist(source_data, bins=30, alpha=0.7, color='blue', label='Data')
            
            if 'iqr' in outliers:
                iqr_outliers = source_data[outliers['iqr']['indices']]
                axes[0, 1].scatter(iqr_outliers, np.zeros_like(iqr_outliers), 
                                  color='red', s=50, label=f'IQR Outliers ({outliers["iqr"]["count"]})')
            
            axes[0, 1].set_title(f'Outlier Detection - Source: {column_name}')
            axes[0, 1].set_xlabel(column_name)
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # 3. Statistical summary
        if len(source_data) > 0 or len(derived_data) > 0:
            summary_data = []
            labels = []
            
            if len(source_data) > 0:
                summary_data.append([
                    source_data.mean(), source_data.std(), source_data.skew(), 
                    source_data.kurtosis(), source_data.median(), source_data.var()
                ])
                labels.append('Source')
            
            if len(derived_data) > 0:
                summary_data.append([
                    derived_data.mean(), derived_data.std(), derived_data.skew(), 
                    derived_data.kurtosis(), derived_data.median(), derived_data.var()
                ])
                labels.append('Derived')
            
            summary_df = pd.DataFrame(summary_data, 
                                    columns=['Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Variance'],
                                    index=labels)
            
            # Create heatmap
            im = axes[1, 0].imshow(summary_df.values, cmap='viridis', aspect='auto')
            axes[1, 0].set_xticks(range(len(summary_df.columns)))
            axes[1, 0].set_yticks(range(len(summary_df.index)))
            axes[1, 0].set_xticklabels(summary_df.columns, rotation=45)
            axes[1, 0].set_yticklabels(summary_df.index)
            axes[1, 0].set_title(f'Statistical Summary: {column_name}')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Transformation analysis
        if len(source_data) > 0 and len(derived_data) > 0:
            # Scatter plot of source vs derived
            min_len = min(len(source_data), len(derived_data))
            axes[1, 1].scatter(source_data[:min_len], derived_data[:min_len], alpha=0.6)
            axes[1, 1].set_xlabel(f'Source {column_name}')
            axes[1, 1].set_ylabel(f'Derived {column_name}')
            axes[1, 1].set_title(f'Transformation Analysis: {column_name}')
            
            # Add correlation line
            correlation = np.corrcoef(source_data[:min_len], derived_data[:min_len])[0, 1]
            axes[1, 1].text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=axes[1, 1].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'statistical_analysis_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Statistical analysis saved as 'statistical_analysis_{column_name}.png'")
    
    def create_trend_analysis_plot(self, column_name: str):
        """Create trend analysis plot for time-series data."""
        print(f"\n[TREND] Creating trend analysis for: {column_name}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Trend Analysis: {column_name}', fontsize=16, fontweight='bold')
        
        source_data = self.df_source[column_name].dropna() if column_name in self.df_source.columns else pd.Series()
        derived_data = self.df_derived[column_name].dropna() if column_name in self.df_derived.columns else pd.Series()
        
        # 1. Time series plot (if we have timestamps)
        timestamp_cols = self._find_timestamp_columns()
        
        if timestamp_cols['source'] and len(source_data) > 0:
            # Try to find a timestamp column for the source
            for ts_col in timestamp_cols['source']:
                try:
                    ts_data = pd.to_datetime(self.df_source[ts_col].dropna())
                    if len(ts_data) == len(source_data):
                        axes[0, 0].scatter(ts_data, source_data, alpha=0.6, s=20)
                        axes[0, 0].set_xlabel('Time')
                        axes[0, 0].set_ylabel(column_name)
                        axes[0, 0].set_title(f'Time Series - Source: {column_name}')
                        break
                except:
                    continue
        
        if timestamp_cols['derived'] and len(derived_data) > 0:
            # Try to find a timestamp column for the derived
            for ts_col in timestamp_cols['derived']:
                try:
                    ts_data = pd.to_datetime(self.df_derived[ts_col].dropna())
                    if len(ts_data) == len(derived_data):
                        axes[0, 1].scatter(ts_data, derived_data, alpha=0.6, s=20, color='orange')
                        axes[0, 1].set_xlabel('Time')
                        axes[0, 1].set_ylabel(column_name)
                        axes[0, 1].set_title(f'Time Series - Derived: {column_name}')
                        break
                except:
                    continue
        
        # 2. Rolling statistics
        if len(source_data) > 10:
            rolling_mean = source_data.rolling(window=5).mean()
            rolling_std = source_data.rolling(window=5).std()
            
            axes[1, 0].plot(source_data.index, source_data, alpha=0.6, label='Data')
            axes[1, 0].plot(source_data.index, rolling_mean, label='Rolling Mean', linewidth=2)
            axes[1, 0].plot(source_data.index, rolling_std, label='Rolling Std', linewidth=2)
            axes[1, 0].set_xlabel('Index')
            axes[1, 0].set_ylabel(column_name)
            axes[1, 0].set_title(f'Rolling Statistics - Source: {column_name}')
            axes[1, 0].legend()
        
        if len(derived_data) > 10:
            rolling_mean = derived_data.rolling(window=5).mean()
            rolling_std = derived_data.rolling(window=5).std()
            
            axes[1, 1].plot(derived_data.index, derived_data, alpha=0.6, label='Data', color='orange')
            axes[1, 1].plot(derived_data.index, rolling_mean, label='Rolling Mean', linewidth=2, color='red')
            axes[1, 1].plot(derived_data.index, rolling_std, label='Rolling Std', linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel(column_name)
            axes[1, 1].set_title(f'Rolling Statistics - Derived: {column_name}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'trend_analysis_{column_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Trend analysis saved as 'trend_analysis_{column_name}.png'")
    
    def _find_timestamp_columns(self) -> Dict[str, List[str]]:
        """Find timestamp columns in both datasets."""
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
        
        return {
            'source': source_timestamps,
            'derived': derived_timestamps
        }
    
    def create_comprehensive_analysis(self, columns: List[str] = None):
        """Create comprehensive analysis for specified columns."""
        print(f"\n[START] Starting comprehensive advanced analysis")
        
        if columns is None:
            # Get all numeric columns
            source_numeric = self.df_source.select_dtypes(include=[np.number]).columns
            derived_numeric = self.df_derived.select_dtypes(include=[np.number]).columns
            columns = list(set(source_numeric) | set(derived_numeric))
        
        print(f"[DATA] Analyzing {len(columns)} columns: {columns}")
        
        for column in columns:
            try:
                # Create all types of analysis for each column
                self.create_advanced_histogram(column)
                self.create_statistical_analysis_plot(column)
                self.create_trend_analysis_plot(column)
                
            except Exception as e:
                print(f"[ERROR] Error creating analysis for {column}: {e}")
        
        print(f"\n[OK] Comprehensive advanced analysis complete!")
        print(f"[DATA] Generated analysis for {len(columns)} columns")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run advanced histogram exploration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create advanced histogram exploration and statistical analysis')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--source', help='Name of the source table')
    parser.add_argument('--derived', help='Name of the derived table')
    parser.add_argument('--columns', nargs='+', help='Specific columns to analyze')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = AdvancedHistogramExplorer(args.db_path)
    
    try:
        # Connect to database
        if not explorer.connect():
            return
        
        # Load datasets
        explorer.load_datasets(args.source, args.derived)
        
        # Run comprehensive analysis
        explorer.create_comprehensive_analysis(args.columns)
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        explorer.close()


if __name__ == "__main__":
    main() 