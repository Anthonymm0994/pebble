#!/usr/bin/env python3
"""
Comprehensive Data Explorer
==========================

A powerful tool for exploring any dataset with interactive visualizations,
statistical analysis, and automated insights.

Features:
- Automated data exploration
- Interactive visualizations
- Statistical analysis
- Pattern detection
- Anomaly detection
- Correlation analysis
- Data quality assessment
- Automated insights generation
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataExplorer:
    """
    Comprehensive data exploration tool.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the data explorer.
        
        Args:
            data_source: Path to database file or CSV file
        """
        self.data_source = data_source
        self.df = None
        self.exploration_results = {}
        
    def load_data(self, table_name: str = None):
        """Load data from SQLite database or CSV file."""
        print(f"\n[DATA] Loading data from: {self.data_source}")
        
        if self.data_source.endswith('.db') or self.data_source.endswith('.sqlite'):
            # Load from SQLite database
            conn = sqlite3.connect(self.data_source)
            
            if table_name:
                self.df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
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
        else:
            # Load from CSV file
            self.df = pd.read_csv(self.data_source)
        
        print(f"[OK] Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def explore_data_structure(self) -> Dict:
        """Explore the basic structure of the data."""
        print(f"\n[EXPLORE] Exploring data structure...")
        
        structure = {
            'shape': self.df.shape,
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'columns': list(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns),
            'boolean_columns': list(self.df.select_dtypes(include=['bool']).columns)
        }
        
        # Add column information
        structure['column_info'] = {}
        for col in self.df.columns:
            structure['column_info'][col] = {
                'dtype': str(self.df[col].dtype),
                'unique_count': self.df[col].nunique(),
                'missing_count': self.df[col].isnull().sum(),
                'missing_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        
        self.exploration_results['structure'] = structure
        return structure
    
    def analyze_numeric_columns(self) -> Dict:
        """Analyze numeric columns with detailed statistics."""
        print(f"\n[ANALYSIS] Analyzing numeric columns...")
        
        numeric_analysis = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                analysis = {
                    'count': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'unique_count': data.nunique(),
                    'unique_ratio': data.nunique() / len(data)
                }
                
                # Normality tests
                if len(data) >= 3:
                    try:
                        shapiro_stat, shapiro_p = shapiro(data)
                        analysis['shapiro_test'] = {
                            'statistic': shapiro_stat,
                            'p_value': shapiro_p,
                            'is_normal': shapiro_p > 0.05
                        }
                    except:
                        analysis['shapiro_test'] = None
                    
                    try:
                        d_agostino_stat, d_agostino_p = normaltest(data)
                        analysis['d_agostino_test'] = {
                            'statistic': d_agostino_stat,
                            'p_value': d_agostino_p,
                            'is_normal': d_agostino_p > 0.05
                        }
                    except:
                        analysis['d_agostino_test'] = None
                
                # Outlier analysis
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                analysis['outliers'] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'indices': outliers.index.tolist()
                }
                
                numeric_analysis[col] = analysis
        
        self.exploration_results['numeric_analysis'] = numeric_analysis
        return numeric_analysis
    
    def analyze_categorical_columns(self) -> Dict:
        """Analyze categorical columns."""
        print(f"\n[ANALYSIS] Analyzing categorical columns...")
        
        categorical_analysis = {}
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                value_counts = data.value_counts()
                
                analysis = {
                    'count': len(data),
                    'unique_count': data.nunique(),
                    'unique_ratio': data.nunique() / len(data),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict(),
                    'value_distribution': value_counts.to_dict()
                }
                
                # Check for patterns
                if len(data) > 0:
                    # Check for leading/trailing whitespace
                    has_whitespace = any(str(x).strip() != str(x) for x in data if pd.notna(x))
                    analysis['has_whitespace_issues'] = has_whitespace
                    
                    # Check for mixed case
                    string_data = data.astype(str)
                    has_mixed_case = any(x != x.lower() and x != x.upper() for x in string_data if x not in ['nan', 'None'])
                    analysis['has_mixed_case'] = has_mixed_case
                    
                    # Check for potential date patterns
                    date_patterns = ['date', 'time', 'created', 'updated', 'timestamp']
                    is_potential_date = any(pattern in col.lower() for pattern in date_patterns)
                    analysis['is_potential_date'] = is_potential_date
                
                categorical_analysis[col] = analysis
        
        self.exploration_results['categorical_analysis'] = categorical_analysis
        return categorical_analysis
    
    def detect_anomalies(self) -> Dict:
        """Detect anomalies in the data."""
        print(f"\n[ANOMALY] Detecting anomalies...")
        
        anomalies = {
            'outliers': {},
            'missing_patterns': {},
            'duplicates': {},
            'inconsistencies': {}
        }
        
        # Detect outliers in numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                
                if len(outliers) > 0:
                    anomalies['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(data)) * 100,
                        'values': outliers.tolist(),
                        'indices': outliers.index.tolist()
                    }
        
        # Detect missing data patterns
        missing_matrix = self.df.isnull()
        if missing_matrix.sum().sum() > 0:
            # Find columns with high missing rates
            missing_rates = missing_matrix.sum() / len(self.df)
            high_missing = missing_rates[missing_rates > 0.1]  # More than 10% missing
            
            if len(high_missing) > 0:
                anomalies['missing_patterns']['high_missing_columns'] = high_missing.to_dict()
            
            # Find rows with many missing values
            row_missing_counts = missing_matrix.sum(axis=1)
            high_missing_rows = row_missing_counts[row_missing_counts > len(self.df.columns) * 0.5]
            
            if len(high_missing_rows) > 0:
                anomalies['missing_patterns']['high_missing_rows'] = {
                    'count': len(high_missing_rows),
                    'indices': high_missing_rows.index.tolist()
                }
        
        # Detect duplicates
        duplicate_rows = self.df.duplicated()
        if duplicate_rows.sum() > 0:
            anomalies['duplicates']['duplicate_rows'] = {
                'count': duplicate_rows.sum(),
                'percentage': (duplicate_rows.sum() / len(self.df)) * 100,
                'indices': duplicate_rows[duplicate_rows].index.tolist()
            }
        
        # Detect inconsistencies
        inconsistencies = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for mixed data types in object columns
                data_types = self.df[col].apply(type).value_counts()
                if len(data_types) > 1:
                    inconsistencies[col] = {
                        'mixed_types': data_types.to_dict(),
                        'description': 'Column contains mixed data types'
                    }
        
        if inconsistencies:
            anomalies['inconsistencies'] = inconsistencies
        
        self.exploration_results['anomalies'] = anomalies
        return anomalies
    
    def analyze_correlations(self) -> Dict:
        """Analyze correlations between variables."""
        print(f"\n[CORRELATION] Analyzing correlations...")
        
        correlation_analysis = {}
        
        # Numeric correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = self.df[numeric_cols].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_val,
                            'strength': 'strong' if abs(corr_val) > 0.8 else 'moderate'
                        })
            
            correlation_analysis['numeric_correlations'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'strongest_correlation': max(high_correlations, key=lambda x: abs(x['correlation'])) if high_correlations else None
            }
        
        # Categorical correlations (using chi-square test)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 1:
            categorical_correlations = []
            
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        # Create contingency table
                        contingency = pd.crosstab(self.df[col1], self.df[col2])
                        
                        # Chi-square test
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        if p_value < 0.05:  # Significant relationship
                            categorical_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'chi2_statistic': chi2,
                                'p_value': p_value,
                                'significant': True
                            })
                    except:
                        continue
            
            correlation_analysis['categorical_correlations'] = categorical_correlations
        
        self.exploration_results['correlations'] = correlation_analysis
        return correlation_analysis
    
    def generate_insights(self) -> List[str]:
        """Generate automated insights from the data."""
        print(f"\n[INSIGHTS] Generating insights...")
        
        insights = []
        
        # Data quality insights
        structure = self.exploration_results.get('structure', {})
        total_missing = sum(info['missing_count'] for info in structure.get('column_info', {}).values())
        total_cells = len(self.df) * len(self.df.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        if missing_percentage > 10:
            insights.append(f"[WARNING]  Data quality concern: {missing_percentage:.1f}% of data is missing")
        elif missing_percentage > 0:
            insights.append(f"[CORRELATION] Data completeness: {100-missing_percentage:.1f}% of data is complete")
        else:
            insights.append("[OK] Excellent data quality: No missing values detected")
        
        # Numeric insights
        numeric_analysis = self.exploration_results.get('numeric_analysis', {})
        for col, analysis in numeric_analysis.items():
            # Outlier insights
            outlier_pct = analysis['outliers']['percentage']
            if outlier_pct > 10:
                insights.append(f"[ANOMALY] {col}: {outlier_pct:.1f}% outliers detected - consider investigation")
            
            # Distribution insights
            skewness = analysis['skewness']
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(f"[TREND] {col}: Distribution is {direction}-skewed (skewness: {skewness:.2f})")
            
            # Normality insights
            if analysis.get('shapiro_test'):
                if analysis['shapiro_test']['is_normal']:
                    insights.append(f"[CORRELATION] {col}: Data appears to follow normal distribution")
                else:
                    insights.append(f"[CORRELATION] {col}: Data does not follow normal distribution")
        
        # Categorical insights
        categorical_analysis = self.exploration_results.get('categorical_analysis', {})
        for col, analysis in categorical_analysis.items():
            unique_ratio = analysis['unique_ratio']
            if unique_ratio > 0.9:
                insights.append(f"[KEY] {col}: High uniqueness ({unique_ratio:.1%}) - potential identifier")
            elif unique_ratio < 0.1:
                insights.append(f"[METRICS] {col}: Low diversity ({unique_ratio:.1%}) - mostly uniform values")
        
        # Correlation insights
        correlations = self.exploration_results.get('correlations', {})
        high_correlations = correlations.get('numeric_correlations', {}).get('high_correlations', [])
        for corr in high_correlations:
            strength = corr['strength']
            insights.append(f"[COMPARE] Strong {strength} correlation between {corr['column1']} and {corr['column2']} ({corr['correlation']:.3f})")
        
        # Anomaly insights
        anomalies = self.exploration_results.get('anomalies', {})
        if anomalies.get('outliers'):
            total_outliers = sum(info['count'] for info in anomalies['outliers'].values())
            insights.append(f"[WARNING]  {total_outliers} outliers detected across numeric columns")
        
        if anomalies.get('duplicates', {}).get('duplicate_rows'):
            duplicate_pct = anomalies['duplicates']['duplicate_rows']['percentage']
            insights.append(f"[TRANSFORM] {duplicate_pct:.1f}% of rows are duplicates")
        
        self.exploration_results['insights'] = insights
        return insights
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n[CHART] Creating comprehensive visualizations...")
        
        # Create output directory
        os.makedirs('explorer_outputs', exist_ok=True)
        
        # 1. Data Overview
        self._create_data_overview()
        
        # 2. Numeric Analysis
        self._create_numeric_analysis()
        
        # 3. Categorical Analysis
        self._create_categorical_analysis()
        
        # 4. Correlation Analysis
        self._create_correlation_analysis()
        
        # 5. Anomaly Detection
        self._create_anomaly_analysis()
        
        # 6. Distribution Analysis
        self._create_distribution_analysis()
        
        print(f"[OK] All visualizations saved in 'explorer_outputs/' directory")
    
    def _create_data_overview(self):
        """Create data overview visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Overview', fontsize=16, fontweight='bold')
        
        structure = self.exploration_results.get('structure', {})
        
        # 1. Data types distribution
        data_types = structure.get('data_types', {})
        type_counts = {}
        for dtype in data_types.values():
            type_name = str(dtype)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        if type_counts:
            axes[0, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Data Types Distribution')
        
        # 2. Missing data heatmap
        missing_data = self.df.isnull()
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data, cbar=True, ax=axes[0, 1])
            axes[0, 1].set_title('Missing Data Pattern')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Missing Data Pattern')
        
        # 3. Column information
        column_info = structure.get('column_info', {})
        if column_info:
            missing_percentages = [info['missing_percentage'] for info in column_info.values()]
            column_names = list(column_info.keys())
            
            axes[1, 0].barh(column_names, missing_percentages, color='red' if any(p > 10 for p in missing_percentages) else 'green')
            axes[1, 0].set_xlabel('Missing Percentage')
            axes[1, 0].set_title('Missing Data by Column')
        
        # 4. Data shape info
        shape = structure.get('shape', (0, 0))
        axes[1, 1].text(0.5, 0.7, f'Rows: {shape[0]:,}', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.5, 0.5, f'Columns: {shape[1]:,}', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.5, 0.3, f'Memory: {structure.get("memory_usage", 0):,.0f} bytes', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Dataset Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('explorer_outputs/data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_numeric_analysis(self):
        """Create numeric analysis visualizations."""
        numeric_analysis = self.exploration_results.get('numeric_analysis', {})
        
        if numeric_analysis:
            n_cols = len(numeric_analysis)
            cols_per_row = 3
            n_rows = (n_cols + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
            fig.suptitle('Numeric Analysis', fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (col, analysis) in enumerate(numeric_analysis.items()):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                
                data = self.df[col].dropna()
                if len(data) > 0:
                    # Histogram with statistics
                    axes[row, col_idx].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    axes[row, col_idx].set_title(f'{col}\n(mean={data.mean():.2f}, std={data.std():.2f})')
                    axes[row, col_idx].set_xlabel(col)
                    axes[row, col_idx].set_ylabel('Frequency')
                    
                    # Add outlier markers
                    outliers = analysis['outliers']
                    if outliers['count'] > 0:
                        outlier_data = data[outliers['indices']]
                        axes[row, col_idx].scatter(outlier_data, np.zeros_like(outlier_data), 
                                                  color='red', s=50, alpha=0.7, label=f'Outliers ({outliers["count"]})')
                        axes[row, col_idx].legend()
            
            # Hide empty subplots
            for i in range(n_cols, n_rows * cols_per_row):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('explorer_outputs/numeric_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_categorical_analysis(self):
        """Create categorical analysis visualizations."""
        categorical_analysis = self.exploration_results.get('categorical_analysis', {})
        
        if categorical_analysis:
            n_cols = len(categorical_analysis)
            cols_per_row = 2
            n_rows = (n_cols + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 6 * n_rows))
            fig.suptitle('Categorical Analysis', fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (col, analysis) in enumerate(categorical_analysis.items()):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                
                data = self.df[col].dropna()
                if len(data) > 0:
                    value_counts = data.value_counts().head(10)  # Top 10 values
                    
                    axes[row, col_idx].bar(range(len(value_counts)), value_counts.values, color='orange')
                    axes[row, col_idx].set_title(f'{col}\n({len(data)} values, {data.nunique()} unique)')
                    axes[row, col_idx].set_xlabel('Values')
                    axes[row, col_idx].set_ylabel('Count')
                    axes[row, col_idx].set_xticks(range(len(value_counts)))
                    axes[row, col_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Hide empty subplots
            for i in range(n_cols, n_rows * cols_per_row):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('explorer_outputs/categorical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_correlation_analysis(self):
        """Create correlation analysis visualizations."""
        correlations = self.exploration_results.get('correlations', {})
        
        if correlations.get('numeric_correlations'):
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
                
                # Correlation heatmap
                correlation_matrix = self.df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', ax=axes[0])
                axes[0].set_title('Correlation Heatmap')
                
                # High correlations bar chart
                high_correlations = correlations['numeric_correlations']['high_correlations']
                if high_correlations:
                    corr_values = [abs(corr['correlation']) for corr in high_correlations]
                    corr_labels = [f"{corr['column1']} vs {corr['column2']}" for corr in high_correlations]
                    
                    axes[1].barh(corr_labels, corr_values, color='red')
                    axes[1].set_xlabel('Absolute Correlation')
                    axes[1].set_title('High Correlations')
                else:
                    axes[1].text(0.5, 0.5, 'No high correlations found', ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('High Correlations')
                
                plt.tight_layout()
                plt.savefig('explorer_outputs/correlation_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_anomaly_analysis(self):
        """Create anomaly analysis visualizations."""
        anomalies = self.exploration_results.get('anomalies', {})
        
        if anomalies:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anomaly Detection', fontsize=16, fontweight='bold')
            
            # 1. Outliers summary
            outliers = anomalies.get('outliers', {})
            if outliers:
                outlier_counts = [info['count'] for info in outliers.values()]
                outlier_columns = list(outliers.keys())
                
                axes[0, 0].bar(outlier_columns, outlier_counts, color='red')
                axes[0, 0].set_title('Outlier Count by Column')
                axes[0, 0].tick_params(axis='x', rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, 'No outliers detected', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Outliers Summary')
            
            # 2. Missing data summary
            missing_patterns = anomalies.get('missing_patterns', {})
            if missing_patterns.get('high_missing_columns'):
                missing_rates = list(missing_patterns['high_missing_columns'].values())
                missing_columns = list(missing_patterns['high_missing_columns'].keys())
                
                axes[0, 1].bar(missing_columns, missing_rates, color='orange')
                axes[0, 1].set_title('High Missing Data Columns')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No high missing data columns', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Missing Data Summary')
            
            # 3. Duplicates summary
            duplicates = anomalies.get('duplicates', {})
            if duplicates.get('duplicate_rows'):
                duplicate_count = duplicates['duplicate_rows']['count']
                total_rows = len(self.df)
                duplicate_pct = (duplicate_count / total_rows) * 100
                
                axes[1, 0].pie([duplicate_pct, 100-duplicate_pct], 
                               labels=['Duplicates', 'Unique'], 
                               autopct='%1.1f%%', 
                               colors=['red', 'green'])
                axes[1, 0].set_title('Duplicate Rows')
            else:
                axes[1, 0].text(0.5, 0.5, 'No duplicates found', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Duplicate Analysis')
            
            # 4. Inconsistencies summary
            inconsistencies = anomalies.get('inconsistencies', {})
            if inconsistencies:
                inconsistency_count = len(inconsistencies)
                axes[1, 1].bar(['Inconsistent Columns'], [inconsistency_count], color='purple')
                axes[1, 1].set_title('Data Inconsistencies')
            else:
                axes[1, 1].text(0.5, 0.5, 'No inconsistencies found', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Data Inconsistencies')
            
            plt.tight_layout()
            plt.savefig('explorer_outputs/anomaly_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_distribution_analysis(self):
        """Create distribution analysis visualizations."""
        numeric_analysis = self.exploration_results.get('numeric_analysis', {})
        
        if numeric_analysis:
            n_cols = len(numeric_analysis)
            cols_per_row = 2
            n_rows = (n_cols + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 6 * n_rows))
            fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (col, analysis) in enumerate(numeric_analysis.items()):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                
                data = self.df[col].dropna()
                if len(data) > 0:
                    # Q-Q plot for normality
                    stats.probplot(data, dist="norm", plot=axes[row, col_idx])
                    axes[row, col_idx].set_title(f'{col} Q-Q Plot')
            
            # Hide empty subplots
            for i in range(n_cols, n_rows * cols_per_row):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('explorer_outputs/distribution_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, output_format: str = 'html') -> str:
        """Generate comprehensive exploration report."""
        print(f"\n[REPORT] Generating {output_format.upper()} report...")
        
        if output_format.lower() == 'html':
            return self._generate_html_report()
        elif output_format.lower() == 'json':
            return self._generate_json_report()
        else:
            return self._generate_text_report()
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        insights = self.exploration_results.get('insights', [])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Exploration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .insight {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Exploration Report</h1>
                <p><strong>Dataset:</strong> {self.data_source}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Shape:</strong> {len(self.df)} rows × {len(self.df.columns)} columns</p>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
        """
        
        for insight in insights:
            html_content += f'<div class="insight">{insight}</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('explorer_outputs/data_exploration_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"[OK] HTML report saved as 'explorer_outputs/data_exploration_report.html'")
        return html_content
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        report = {
            'metadata': {
                'dataset': self.data_source,
                'generated_at': datetime.now().isoformat(),
                'shape': {'rows': len(self.df), 'columns': len(self.df.columns)}
            },
            'exploration_results': self.exploration_results
        }
        
        json_content = json.dumps(report, indent=2, default=str)
        
        # Save JSON file
        with open('explorer_outputs/data_exploration_report.json', 'w') as f:
            f.write(json_content)
        
        print(f"[OK] JSON report saved as 'explorer_outputs/data_exploration_report.json'")
        return json_content
    
    def _generate_text_report(self) -> str:
        """Generate text report."""
        insights = self.exploration_results.get('insights', [])
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA EXPLORATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Dataset: {self.data_source}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Shape: {len(self.df)} rows × {len(self.df.columns)} columns")
        report_lines.append("")
        
        report_lines.append("KEY INSIGHTS:")
        report_lines.append("-" * 20)
        for insight in insights:
            report_lines.append(f"• {insight}")
        
        report_text = "\n".join(report_lines)
        
        # Save text file
        with open('explorer_outputs/data_exploration_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Text report saved as 'explorer_outputs/data_exploration_report.txt'")
        return report_text
    
    def run_comprehensive_exploration(self, table_name: str = None, output_format: str = 'html'):
        """Run comprehensive data exploration."""
        print(f"\n[START] Starting comprehensive data exploration")
        
        # Load data
        self.load_data(table_name)
        
        # Run all analyses
        self.explore_data_structure()
        self.analyze_numeric_columns()
        self.analyze_categorical_columns()
        self.detect_anomalies()
        self.analyze_correlations()
        self.generate_insights()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Generate report
        self.generate_report(output_format)
        
        print(f"\n[OK] Comprehensive exploration complete!")
        print(f"[DATA] Check 'explorer_outputs/' directory for results")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Data explorer closed")


def main():
    """Main function to run data exploration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive data exploration')
    parser.add_argument('data_source', help='Path to database file or CSV file')
    parser.add_argument('--table', help='Table name (for SQLite databases)')
    parser.add_argument('--format', choices=['html', 'json', 'text'], default='html', 
                       help='Output format for report')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = DataExplorer(args.data_source)
    
    try:
        # Run comprehensive exploration
        explorer.run_comprehensive_exploration(args.table, args.format)
        
    except Exception as e:
        print(f"[ERROR] Error during exploration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        explorer.close()


if __name__ == "__main__":
    main() 