#!/usr/bin/env python3
"""
Data Profiler
============

A comprehensive tool for profiling and analyzing datasets with automatic
data type detection, statistical analysis, and quality assessment.

Features:
- Automatic data type detection
- Statistical analysis
- Data quality assessment
- Pattern detection
- Correlation analysis
- Missing data analysis
- Distribution analysis
- Professional visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class DataProfiler:
    """
    Comprehensive data profiling and analysis tool.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the data profiler.
        
        Args:
            data_source: Path to CSV file or database file
        """
        self.data_source = data_source
        self.df = None
        self.profile_data = {}
        
    def load_data(self):
        """Load data from CSV file or database."""
        print(f"[DATA] Loading data from: {self.data_source}")
        
        try:
            if self.data_source.endswith('.csv'):
                self.df = pd.read_csv(self.data_source)
            else:
                # Assume it's a database file
                import sqlite3
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
    
    def profile_data_types(self):
        """Analyze and profile data types."""
        print(f"\n[ANALYSIS] Profiling data types...")
        
        type_profile = {}
        
        for column in self.df.columns:
            dtype = self.df[column].dtype
            unique_count = self.df[column].nunique()
            null_count = self.df[column].isnull().sum()
            null_percentage = (null_count / len(self.df)) * 100
            
            # Detect if column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            
            # Detect if column is datetime
            is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
            
            # Detect if column is categorical
            is_categorical = self.df[column].dtype == 'object' and unique_count < len(self.df) * 0.5
            
            type_profile[column] = {
                'dtype': str(dtype),
                'unique_count': unique_count,
                'null_count': null_count,
                'null_percentage': null_percentage,
                'is_numeric': is_numeric,
                'is_datetime': is_datetime,
                'is_categorical': is_categorical,
                'cardinality': 'high' if unique_count > 100 else 'medium' if unique_count > 10 else 'low'
            }
        
        self.profile_data['type_profile'] = type_profile
        return type_profile
    
    def analyze_missing_data(self):
        """Analyze missing data patterns."""
        print(f"\n[ANALYSIS] Analyzing missing data...")
        
        missing_analysis = {}
        
        # Overall missing data
        total_missing = self.df.isnull().sum().sum()
        total_cells = len(self.df) * len(self.df.columns)
        overall_missing_percentage = (total_missing / total_cells) * 100
        
        missing_analysis['overall'] = {
            'total_missing': total_missing,
            'total_cells': total_cells,
            'missing_percentage': overall_missing_percentage
        }
        
        # Missing data by column
        missing_by_column = {}
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            missing_by_column[column] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'completeness': 100 - missing_percentage
            }
        
        missing_analysis['by_column'] = missing_by_column
        
        self.profile_data['missing_analysis'] = missing_analysis
        return missing_analysis
    
    def analyze_numeric_columns(self):
        """Analyze numeric columns with statistical measures."""
        print(f"\n[ANALYSIS] Analyzing numeric columns...")
        
        numeric_analysis = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            data = self.df[column].dropna()
            if len(data) == 0:
                continue
            
            # Basic statistics
            stats = {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            }
            
            # Detect outliers using IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(data)) * 100
            
            # Distribution type detection
            if abs(stats['skewness']) < 0.5:
                stats['distribution_type'] = 'normal'
            elif stats['skewness'] > 0.5:
                stats['distribution_type'] = 'right_skewed'
            else:
                stats['distribution_type'] = 'left_skewed'
            
            numeric_analysis[column] = stats
        
        self.profile_data['numeric_analysis'] = numeric_analysis
        return numeric_analysis
    
    def analyze_categorical_columns(self):
        """Analyze categorical columns."""
        print(f"\n[ANALYSIS] Analyzing categorical columns...")
        
        categorical_analysis = {}
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for column in categorical_cols:
            data = self.df[column].dropna()
            if len(data) == 0:
                continue
            
            # Value counts
            value_counts = data.value_counts()
            
            analysis = {
                'count': len(data),
                'unique_count': data.nunique(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'entropy': self._calculate_entropy(data),
                'top_5_values': value_counts.head(5).to_dict()
            }
            
            categorical_analysis[column] = analysis
        
        self.profile_data['categorical_analysis'] = categorical_analysis
        return categorical_analysis
    
    def _calculate_entropy(self, data):
        """Calculate entropy for categorical data."""
        value_counts = data.value_counts()
        probabilities = value_counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def analyze_correlations(self):
        """Analyze correlations between numeric columns."""
        print(f"\n[ANALYSIS] Analyzing correlations...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print(f"[INFO] Not enough numeric columns for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        correlation_analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations
        }
        
        self.profile_data['correlation_analysis'] = correlation_analysis
        return correlation_analysis
    
    def detect_patterns(self):
        """Detect patterns in the data."""
        print(f"\n[ANALYSIS] Detecting patterns...")
        
        patterns = {}
        
        # Check for duplicate rows
        duplicate_count = len(self.df) - len(self.df.drop_duplicates())
        duplicate_percentage = (duplicate_count / len(self.df)) * 100
        
        patterns['duplicates'] = {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage
        }
        
        # Check for constant columns
        constant_columns = []
        for column in self.df.columns:
            if self.df[column].nunique() == 1:
                constant_columns.append(column)
        
        patterns['constant_columns'] = constant_columns
        
        # Check for ID-like columns
        id_columns = []
        for column in self.df.columns:
            if self.df[column].nunique() == len(self.df):
                id_columns.append(column)
        
        patterns['id_columns'] = id_columns
        
        self.profile_data['patterns'] = patterns
        return patterns
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n[CHART] Creating visualizations...")
        
        # Create output directory
        os.makedirs('../outputs/profiler_outputs', exist_ok=True)
        
        # 1. Data Types Overview
        self._create_data_types_overview()
        
        # 2. Missing Data Summary
        self._create_missing_data_summary()
        
        # 3. Numeric Distributions
        self._create_numeric_distributions()
        
        # 4. Categorical Distributions
        self._create_categorical_distributions()
        
        # 5. Correlation Heatmap
        self._create_correlation_heatmap()
        
        # 6. Data Quality Summary
        self._create_data_quality_summary()
        
        print(f"[OK] All visualizations saved in '../outputs/profiler_outputs/' directory")
    
    def _create_data_types_overview(self):
        """Create data types overview visualization."""
        type_profile = self.profile_data.get('type_profile', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Types Overview', fontsize=16, fontweight='bold')
        
        # Data types pie chart
        dtype_counts = {}
        for col_info in type_profile.values():
            dtype = col_info['dtype']
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        axes[0, 0].pie(dtype_counts.values(), labels=dtype_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Data Type Distribution')
        
        # Null percentage by column
        columns = list(type_profile.keys())
        null_percentages = [type_profile[col]['null_percentage'] for col in columns]
        
        bars = axes[0, 1].bar(range(len(columns)), null_percentages, color='lightcoral')
        axes[0, 1].set_title('Missing Data by Column')
        axes[0, 1].set_ylabel('Missing Percentage (%)')
        axes[0, 1].set_xticks(range(len(columns)))
        axes[0, 1].set_xticklabels(columns, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, null_percentages):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Cardinality analysis
        cardinality_counts = {}
        for col_info in type_profile.values():
            cardinality = col_info['cardinality']
            cardinality_counts[cardinality] = cardinality_counts.get(cardinality, 0) + 1
        
        axes[1, 0].bar(cardinality_counts.keys(), cardinality_counts.values(), color='lightblue')
        axes[1, 0].set_title('Column Cardinality')
        axes[1, 0].set_ylabel('Number of Columns')
        
        # Unique values by column
        unique_counts = [type_profile[col]['unique_count'] for col in columns]
        axes[1, 1].bar(range(len(columns)), unique_counts, color='lightgreen')
        axes[1, 1].set_title('Unique Values by Column')
        axes[1, 1].set_ylabel('Unique Count')
        axes[1, 1].set_xticks(range(len(columns)))
        axes[1, 1].set_xticklabels(columns, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/data_types_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_missing_data_summary(self):
        """Create missing data summary visualization."""
        missing_analysis = self.profile_data.get('missing_analysis', {})
        
        if not missing_analysis:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Overall missing data
        overall = missing_analysis.get('overall', {})
        missing_pct = overall.get('missing_percentage', 0)
        complete_pct = 100 - missing_pct
        
        axes[0].pie([complete_pct, missing_pct], labels=['Complete', 'Missing'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0].set_title('Overall Data Completeness')
        
        # Missing data by column
        by_column = missing_analysis.get('by_column', {})
        if by_column:
            columns = list(by_column.keys())
            completeness = [by_column[col]['completeness'] for col in columns]
            
            bars = axes[1].bar(range(len(columns)), completeness, color='steelblue')
            axes[1].set_title('Data Completeness by Column')
            axes[1].set_ylabel('Completeness (%)')
            axes[1].set_ylim(0, 100)
            axes[1].set_xticks(range(len(columns)))
            axes[1].set_xticklabels(columns, rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, completeness):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/data_quality_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_numeric_distributions(self):
        """Create numeric distributions visualization."""
        numeric_analysis = self.profile_data.get('numeric_analysis', {})
        
        if not numeric_analysis:
            return
        
        n_cols = len(numeric_analysis)
        if n_cols == 0:
            return
        
        cols_per_row = 3
        rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
        fig.suptitle('Numeric Column Distributions', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (column, analysis) in enumerate(numeric_analysis.items()):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            data = self.df[column].dropna()
            
            # Create histogram with improved styling
            n, bins, patches = axes[row, col_idx].hist(data, bins=30, alpha=0.7, 
                                                      color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Add mean and median lines
            mean_val = analysis['mean']
            median_val = analysis['median']
            
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                      label=f'Mean: {mean_val:.2f}')
            axes[row, col_idx].axvline(median_val, color='orange', linestyle='-', linewidth=2, 
                                      label=f'Median: {median_val:.2f}')
            
            axes[row, col_idx].set_title(f'{column}\n(n={analysis["count"]})', fontweight='bold')
            axes[row, col_idx].set_xlabel(column)
            axes[row, col_idx].set_ylabel('Frequency')
            axes[row, col_idx].legend(loc='upper right')
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_categorical_distributions(self):
        """Create categorical distributions visualization."""
        categorical_analysis = self.profile_data.get('categorical_analysis', {})
        
        if not categorical_analysis:
            return
        
        n_cols = len(categorical_analysis)
        if n_cols == 0:
            return
        
        cols_per_row = 2
        rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 6 * rows))
        fig.suptitle('Categorical Column Distributions', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (column, analysis) in enumerate(categorical_analysis.items()):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            data = self.df[column].dropna()
            value_counts = data.value_counts().head(10)  # Top 10 values
            
            bars = axes[row, col_idx].bar(range(len(value_counts)), value_counts.values, 
                                        color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[row, col_idx].set_title(f'{column}\n(n={analysis["count"]})', fontweight='bold')
            axes[row, col_idx].set_xlabel('Values')
            axes[row, col_idx].set_ylabel('Count')
            axes[row, col_idx].set_xticks(range(len(value_counts)))
            axes[row, col_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, value_counts.values):
                height = bar.get_height()
                axes[row, col_idx].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{value}', ha='center', va='bottom', fontweight='bold')
            
            axes[row, col_idx].grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for i in range(n_cols, rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap."""
        correlation_analysis = self.profile_data.get('correlation_analysis', {})
        
        if not correlation_analysis or 'correlation_matrix' not in correlation_analysis:
            return
        
        correlation_matrix = pd.DataFrame(correlation_analysis['correlation_matrix'])
        
        if correlation_matrix.empty:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with improved styling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_data_quality_summary(self):
        """Create data quality summary visualization."""
        type_profile = self.profile_data.get('type_profile', {})
        patterns = self.profile_data.get('patterns', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Summary', fontsize=16, fontweight='bold')
        
        # Data completeness
        completeness_scores = []
        column_names = []
        for column, info in type_profile.items():
            completeness = 100 - info['null_percentage']
            completeness_scores.append(completeness)
            column_names.append(column)
        
        bars = axes[0, 0].bar(range(len(column_names)), completeness_scores, 
                              color=['green' if score > 90 else 'orange' if score > 70 else 'red' for score in completeness_scores])
        axes[0, 0].set_title('Data Completeness by Column')
        axes[0, 0].set_ylabel('Completeness (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].set_xticks(range(len(column_names)))
        axes[0, 0].set_xticklabels(column_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, completeness_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Duplicate analysis
        duplicate_info = patterns.get('duplicates', {})
        duplicate_pct = duplicate_info.get('duplicate_percentage', 0)
        unique_pct = 100 - duplicate_pct
        
        axes[0, 1].pie([unique_pct, duplicate_pct], labels=['Unique', 'Duplicate'], 
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Data Uniqueness')
        
        # Column types
        type_counts = {}
        for info in type_profile.values():
            col_type = 'Numeric' if info['is_numeric'] else 'Categorical' if info['is_categorical'] else 'Other'
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        axes[1, 0].bar(type_counts.keys(), type_counts.values(), color='lightblue')
        axes[1, 0].set_title('Column Types')
        axes[1, 0].set_ylabel('Number of Columns')
        
        # Constant columns
        constant_cols = patterns.get('constant_columns', [])
        non_constant_cols = len(type_profile) - len(constant_cols)
        
        axes[1, 1].pie([non_constant_cols, len(constant_cols)], 
                      labels=['Variable', 'Constant'], autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Column Variability')
        
        plt.tight_layout()
        plt.savefig('../outputs/profiler_outputs/data_quality_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, format_type: str = 'text'):
        """Generate comprehensive profiling report."""
        print(f"\n[REPORT] Generating {format_type.upper()} report...")
        
        if format_type.lower() == 'html':
            self._generate_html_report()
        elif format_type.lower() == 'json':
            self._generate_json_report()
        else:
            self._generate_text_report()
        
        print(f"[OK] {format_type.upper()} report saved")
    
    def _generate_text_report(self):
        """Generate text report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open('../outputs/profiler_outputs/data_profile_report.txt', 'w') as f:
            f.write("COMPREHENSIVE DATA PROFILE REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Data Source: {self.data_source}\n")
            f.write(f"Dataset Shape: {len(self.df)} rows x {len(self.df.columns)} columns\n\n")
            
            # Data types summary
            f.write("DATA TYPES SUMMARY\n")
            f.write("-" * 20 + "\n")
            type_profile = self.profile_data.get('type_profile', {})
            for column, info in type_profile.items():
                f.write(f"{column}: {info['dtype']} ({info['unique_count']} unique values)\n")
            f.write("\n")
            
            # Missing data summary
            f.write("MISSING DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            missing_analysis = self.profile_data.get('missing_analysis', {})
            overall = missing_analysis.get('overall', {})
            f.write(f"Overall missing data: {overall.get('missing_percentage', 0):.2f}%\n")
            f.write("Missing data by column:\n")
            by_column = missing_analysis.get('by_column', {})
            for column, info in by_column.items():
                f.write(f"  {column}: {info['missing_percentage']:.2f}% missing\n")
            f.write("\n")
            
            # Numeric analysis
            f.write("NUMERIC COLUMN ANALYSIS\n")
            f.write("-" * 20 + "\n")
            numeric_analysis = self.profile_data.get('numeric_analysis', {})
            for column, stats in numeric_analysis.items():
                f.write(f"{column}:\n")
                f.write(f"  Mean: {stats['mean']:.2f}\n")
                f.write(f"  Median: {stats['median']:.2f}\n")
                f.write(f"  Std: {stats['std']:.2f}\n")
                f.write(f"  Outliers: {stats['outlier_count']} ({stats['outlier_percentage']:.2f}%)\n")
                f.write(f"  Distribution: {stats['distribution_type']}\n\n")
            
            # Patterns
            f.write("DATA PATTERNS\n")
            f.write("-" * 20 + "\n")
            patterns = self.profile_data.get('patterns', {})
            f.write(f"Duplicate rows: {patterns.get('duplicates', {}).get('duplicate_count', 0)}\n")
            f.write(f"Constant columns: {len(patterns.get('constant_columns', []))}\n")
            f.write(f"ID columns: {len(patterns.get('id_columns', []))}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            
            # Missing data recommendations
            high_missing_cols = [col for col, info in by_column.items() if info['missing_percentage'] > 50]
            if high_missing_cols:
                f.write(f"• Consider removing or imputing columns with high missing data: {', '.join(high_missing_cols)}\n")
            
            # Outlier recommendations
            high_outlier_cols = [col for col, stats in numeric_analysis.items() if stats['outlier_percentage'] > 20]
            if high_outlier_cols:
                f.write(f"• Investigate outliers in columns: {', '.join(high_outlier_cols)}\n")
            
            # Duplicate recommendations
            if patterns.get('duplicates', {}).get('duplicate_percentage', 0) > 10:
                f.write("• Consider removing duplicate rows\n")
            
            f.write("• Review data quality before analysis\n")
            f.write("• Consider data preprocessing steps\n")
    
    def _generate_json_report(self):
        """Generate JSON report."""
        with open('../outputs/profiler_outputs/data_profile_report.json', 'w') as f:
            json.dump(self.profile_data, f, indent=2, default=str)
    
    def _generate_html_report(self):
        """Generate HTML report."""
        # This would be a more complex HTML generation
        # For now, we'll create a simple HTML version
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <html>
        <head>
            <title>Data Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Data Profile Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {self.data_source}</p>
            <p><strong>Dataset Shape:</strong> {len(self.df)} rows x {len(self.df.columns)} columns</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This report provides a comprehensive analysis of the dataset structure, 
                data quality, and statistical properties.</p>
            </div>
            
            <div class="section">
                <h2>Data Types</h2>
                <table>
                    <tr><th>Column</th><th>Data Type</th><th>Unique Values</th><th>Missing %</th></tr>
        """
        
        type_profile = self.profile_data.get('type_profile', {})
        for column, info in type_profile.items():
            html_content += f"""
                    <tr>
                        <td>{column}</td>
                        <td>{info['dtype']}</td>
                        <td>{info['unique_count']}</td>
                        <td>{info['null_percentage']:.2f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open('../outputs/profiler_outputs/data_profile_report.html', 'w') as f:
            f.write(html_content)
    
    def run_comprehensive_profiling(self, format_type: str = 'text'):
        """Run comprehensive data profiling."""
        print(f"\n[START] Starting comprehensive data profiling")
        
        # Load data
        data = self.load_data()
        if data is None:
            return
        
        # Run all analyses
        self.profile_data_types()
        self.analyze_missing_data()
        self.analyze_numeric_columns()
        self.analyze_categorical_columns()
        self.analyze_correlations()
        self.detect_patterns()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report(format_type)
        
        print(f"\n[OK] Comprehensive profiling complete!")
        print(f"[DATA] Check '../outputs/profiler_outputs/' directory for results")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Data profiler closed")


def main():
    """Main function to run data profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive data profiling tool')
    parser.add_argument('data_source', help='Path to CSV file or database file')
    parser.add_argument('--format', choices=['text', 'html', 'json'], default='text',
                       help='Output format for the report')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = DataProfiler(args.data_source)
    
    try:
        # Run comprehensive profiling
        profiler.run_comprehensive_profiling(args.format)
        
    except Exception as e:
        print(f"[ERROR] Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        profiler.close()


if __name__ == "__main__":
    main() 