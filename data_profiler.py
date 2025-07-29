#!/usr/bin/env python3
"""
Comprehensive Data Profiler
==========================

A powerful tool for profiling any dataset and generating detailed reports
with insights, statistics, and visualizations.

Features:
- Automatic data type detection
- Statistical analysis
- Data quality assessment
- Pattern detection
- Correlation analysis
- Missing data analysis
- Outlier detection
- Distribution analysis
- Report generation (HTML, PDF, Excel)
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataProfiler:
    """
    Comprehensive data profiling tool.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the data profiler.
        
        Args:
            data_source: Path to database file or CSV file
        """
        self.data_source = data_source
        self.df = None
        self.profile_results = {}
        
    def load_data(self, table_name: str = None, csv_sep: str = ','):
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
            self.df = pd.read_csv(self.data_source, sep=csv_sep)
        
        print(f"[OK] Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def profile_data_types(self) -> Dict:
        """Analyze data types and provide insights."""
        print(f"\n[ANALYSIS] Profiling data types...")
        
        type_analysis = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'boolean_columns': [],
            'mixed_columns': []
        }
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                type_analysis['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_analysis['datetime_columns'].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                type_analysis['boolean_columns'].append(col)
            elif pd.api.types.is_object_dtype(dtype):
                # Check if it's categorical or text
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.5:  # Likely categorical
                    type_analysis['categorical_columns'].append(col)
                else:
                    type_analysis['text_columns'].append(col)
            else:
                type_analysis['mixed_columns'].append(col)
        
        self.profile_results['data_types'] = type_analysis
        return type_analysis
    
    def analyze_missing_data(self) -> Dict:
        """Analyze missing data patterns."""
        print(f"\n[ANALYSIS] Analyzing missing data...")
        
        missing_analysis = {
            'total_missing': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'columns_with_missing': {},
            'missing_patterns': {}
        }
        
        # Analyze each column
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                missing_analysis['columns_with_missing'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        # Find missing data patterns
        missing_matrix = self.df.isnull()
        if missing_matrix.sum().sum() > 0:
            # Find columns that have missing data together
            missing_correlations = missing_matrix.corr()
            high_corr_pairs = []
            
            for i in range(len(missing_correlations.columns)):
                for j in range(i+1, len(missing_correlations.columns)):
                    corr_val = missing_correlations.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation
                        col1 = missing_correlations.columns[i]
                        col2 = missing_correlations.columns[j]
                        high_corr_pairs.append((col1, col2, corr_val))
            
            missing_analysis['missing_patterns']['high_correlation_pairs'] = high_corr_pairs
        
        self.profile_results['missing_data'] = missing_analysis
        return missing_analysis
    
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
                
                # Detect outliers using IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                analysis['outlier_count'] = len(outliers)
                analysis['outlier_percentage'] = (len(outliers) / len(data)) * 100
                
                numeric_analysis[col] = analysis
        
        self.profile_results['numeric_analysis'] = numeric_analysis
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
                
                categorical_analysis[col] = analysis
        
        self.profile_results['categorical_analysis'] = categorical_analysis
        return categorical_analysis
    
    def analyze_correlations(self) -> Dict:
        """Analyze correlations between numeric columns."""
        print(f"\n[ANALYSIS] Analyzing correlations...")
        
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
                        high_correlations.append((col1, col2, corr_val))
            
            correlation_analysis = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'strongest_correlation': max(high_correlations, key=lambda x: abs(x[2])) if high_correlations else None
            }
        else:
            correlation_analysis = {
                'correlation_matrix': {},
                'high_correlations': [],
                'strongest_correlation': None
            }
        
        self.profile_results['correlation_analysis'] = correlation_analysis
        return correlation_analysis
    
    def detect_patterns(self) -> Dict:
        """Detect patterns in the data."""
        print(f"\n[ANALYSIS] Detecting patterns...")
        
        patterns = {
            'duplicate_rows': len(self.df) - len(self.df.drop_duplicates()),
            'duplicate_percentage': ((len(self.df) - len(self.df.drop_duplicates())) / len(self.df)) * 100,
            'constant_columns': [],
            'near_constant_columns': [],
            'potential_keys': [],
            'date_patterns': []
        }
        
        # Check for constant columns
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio == 0:
                patterns['constant_columns'].append(col)
            elif unique_ratio < 0.01:  # Near constant
                patterns['near_constant_columns'].append(col)
        
        # Check for potential primary keys
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df) and not self.df[col].isnull().any():
                patterns['potential_keys'].append(col)
        
        # Check for date patterns
        for col in self.df.columns:
            if pd.api.types.is_object_dtype(self.df[col].dtype):
                # Try to detect date patterns
                sample_values = self.df[col].dropna().head(10)
                date_patterns = []
                
                for val in sample_values:
                    try:
                        pd.to_datetime(val)
                        date_patterns.append(True)
                    except:
                        date_patterns.append(False)
                
                if any(date_patterns):
                    patterns['date_patterns'].append(col)
        
        self.profile_results['patterns'] = patterns
        return patterns
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n[CHART] Creating visualizations...")
        
        # Create output directory
        os.makedirs('profiler_outputs', exist_ok=True)
        
        # 1. Data types overview
        self._create_data_types_chart()
        
        # 2. Missing data visualization
        self._create_missing_data_chart()
        
        # 3. Numeric distributions
        self._create_numeric_distributions()
        
        # 4. Categorical distributions
        self._create_categorical_distributions()
        
        # 5. Correlation heatmap
        self._create_correlation_heatmap()
        
        # 6. Data quality summary
        self._create_data_quality_summary()
        
        print(f"[OK] All visualizations saved in 'profiler_outputs/' directory")
    
    def _create_data_types_chart(self):
        """Create data types overview chart."""
        type_counts = {
            'Numeric': len(self.profile_results.get('data_types', {}).get('numeric_columns', [])),
            'Categorical': len(self.profile_results.get('data_types', {}).get('categorical_columns', [])),
            'Datetime': len(self.profile_results.get('data_types', {}).get('datetime_columns', [])),
            'Text': len(self.profile_results.get('data_types', {}).get('text_columns', [])),
            'Boolean': len(self.profile_results.get('data_types', {}).get('boolean_columns', [])),
            'Mixed': len(self.profile_results.get('data_types', {}).get('mixed_columns', []))
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Data Types Overview', fontsize=16, fontweight='bold')
        
        # Pie chart
        labels = [k for k, v in type_counts.items() if v > 0]
        sizes = [v for v in type_counts.values() if v > 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Column Types Distribution')
        
        # Bar chart
        ax2.bar(type_counts.keys(), type_counts.values(), color='skyblue')
        ax2.set_title('Column Count by Type')
        ax2.set_ylabel('Number of Columns')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('profiler_outputs/data_types_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_missing_data_chart(self):
        """Create missing data visualization."""
        missing_data = self.profile_results.get('missing_data', {})
        
        if missing_data.get('total_missing', 0) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
            
            # Missing data by column
            columns_with_missing = missing_data.get('columns_with_missing', {})
            if columns_with_missing:
                cols = list(columns_with_missing.keys())
                percentages = [columns_with_missing[col]['percentage'] for col in cols]
                
                ax1.barh(cols, percentages, color='red', alpha=0.7)
                ax1.set_xlabel('Missing Percentage')
                ax1.set_title('Missing Data by Column')
            
            # Overall missing data
            total_missing = missing_data.get('total_missing', 0)
            total_cells = len(self.df) * len(self.df.columns)
            missing_pct = missing_data.get('missing_percentage', 0)
            
            ax2.pie([missing_pct, 100-missing_pct], 
                   labels=['Missing', 'Present'], 
                   autopct='%1.1f%%', 
                   colors=['red', 'green'])
            ax2.set_title('Overall Data Completeness')
            
            plt.tight_layout()
            plt.savefig('profiler_outputs/missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_numeric_distributions(self):
        """Create numeric distributions charts."""
        numeric_analysis = self.profile_results.get('numeric_analysis', {})
        
        if numeric_analysis:
            # Create subplots for each numeric column
            n_cols = len(numeric_analysis)
            if n_cols > 0:
                cols_per_row = 3
                n_rows = (n_cols + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
                fig.suptitle('Numeric Distributions', fontsize=16, fontweight='bold')
                
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, (col, analysis) in enumerate(numeric_analysis.items()):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        axes[row, col_idx].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                        axes[row, col_idx].set_title(f'{col}\n(mean={data.mean():.2f}, std={data.std():.2f})')
                        axes[row, col_idx].set_xlabel(col)
                        axes[row, col_idx].set_ylabel('Frequency')
                
                # Hide empty subplots
                for i in range(n_cols, n_rows * cols_per_row):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    axes[row, col_idx].axis('off')
                
                plt.tight_layout()
                plt.savefig('profiler_outputs/numeric_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_categorical_distributions(self):
        """Create categorical distributions charts."""
        categorical_analysis = self.profile_results.get('categorical_analysis', {})
        
        if categorical_analysis:
            # Create subplots for top categorical columns
            top_cols = list(categorical_analysis.keys())[:6]  # Limit to 6 columns
            
            if top_cols:
                n_cols = len(top_cols)
                cols_per_row = 2
                n_rows = (n_cols + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 6 * n_rows))
                fig.suptitle('Categorical Distributions', fontsize=16, fontweight='bold')
                
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(top_cols):
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
                plt.savefig('profiler_outputs/categorical_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap."""
        correlation_analysis = self.profile_results.get('correlation_analysis', {})
        
        if correlation_analysis.get('correlation_matrix'):
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                correlation_matrix = self.df[numeric_cols].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('profiler_outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_data_quality_summary(self):
        """Create data quality summary chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Summary', fontsize=16, fontweight='bold')
        
        # 1. Data completeness
        missing_data = self.profile_results.get('missing_data', {})
        completeness = 100 - missing_data.get('missing_percentage', 0)
        
        axes[0, 0].pie([completeness, 100-completeness], 
                       labels=['Complete', 'Missing'], 
                       autopct='%1.1f%%', 
                       colors=['green', 'red'])
        axes[0, 0].set_title('Data Completeness')
        
        # 2. Data types distribution
        data_types = self.profile_results.get('data_types', {})
        type_counts = {k: len(v) for k, v in data_types.items()}
        
        axes[0, 1].bar(type_counts.keys(), type_counts.values(), color='skyblue')
        axes[0, 1].set_title('Data Types Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Outlier analysis
        numeric_analysis = self.profile_results.get('numeric_analysis', {})
        if numeric_analysis:
            outlier_percentages = [analysis.get('outlier_percentage', 0) for analysis in numeric_analysis.values()]
            columns = list(numeric_analysis.keys())
            
            axes[1, 0].bar(columns, outlier_percentages, color='orange')
            axes[1, 0].set_title('Outlier Percentage by Column')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Duplicate analysis
        patterns = self.profile_results.get('patterns', {})
        duplicate_pct = patterns.get('duplicate_percentage', 0)
        
        axes[1, 1].pie([100-duplicate_pct, duplicate_pct], 
                       labels=['Unique', 'Duplicate'], 
                       autopct='%1.1f%%', 
                       colors=['blue', 'red'])
        axes[1, 1].set_title('Data Uniqueness')
        
        plt.tight_layout()
        plt.savefig('profiler_outputs/data_quality_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_format: str = 'html') -> str:
        """Generate comprehensive report."""
        print(f"\n[REPORT] Generating {output_format.upper()} report...")
        
        if output_format.lower() == 'html':
            return self._generate_html_report()
        elif output_format.lower() == 'json':
            return self._generate_json_report()
        else:
            return self._generate_text_report()
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Profile Report</h1>
                <p><strong>Dataset:</strong> {self.data_source}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Shape:</strong> {len(self.df)} rows × {len(self.df.columns)} columns</p>
            </div>
        """
        
        # Add sections
        html_content += self._generate_html_section("Data Types", self.profile_results.get('data_types', {}))
        html_content += self._generate_html_section("Missing Data", self.profile_results.get('missing_data', {}))
        html_content += self._generate_html_section("Numeric Analysis", self.profile_results.get('numeric_analysis', {}))
        html_content += self._generate_html_section("Categorical Analysis", self.profile_results.get('categorical_analysis', {}))
        html_content += self._generate_html_section("Patterns", self.profile_results.get('patterns', {}))
        html_content += self._generate_html_section("Correlations", self.profile_results.get('correlation_analysis', {}))
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        with open('profiler_outputs/data_profile_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"[OK] HTML report saved as 'profiler_outputs/data_profile_report.html'")
        return html_content
    
    def _generate_html_section(self, title: str, data: Dict) -> str:
        """Generate HTML section."""
        html = f'<div class="section"><h2>{title}</h2>'
        
        if title == "Data Types":
            for data_type, columns in data.items():
                if columns:
                    html += f'<div class="metric"><strong>{data_type}:</strong> {len(columns)} columns</div>'
        
        elif title == "Missing Data":
            missing_pct = data.get('missing_percentage', 0)
            if missing_pct > 10:
                html += f'<div class="warning"><strong>Warning:</strong> {missing_pct:.1f}% of data is missing</div>'
            else:
                html += f'<div class="success"><strong>Good:</strong> Only {missing_pct:.1f}% of data is missing</div>'
            
            if data.get('columns_with_missing'):
                html += '<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>'
                for col, info in data['columns_with_missing'].items():
                    html += f'<tr><td>{col}</td><td>{info["count"]}</td><td>{info["percentage"]:.1f}%</td></tr>'
                html += '</table>'
        
        elif title == "Numeric Analysis":
            if data:
                html += '<table><tr><th>Column</th><th>Mean</th><th>Std</th><th>Outliers %</th></tr>'
                for col, info in data.items():
                    html += f'<tr><td>{col}</td><td>{info["mean"]:.2f}</td><td>{info["std"]:.2f}</td><td>{info["outlier_percentage"]:.1f}%</td></tr>'
                html += '</table>'
        
        elif title == "Patterns":
            duplicate_pct = data.get('duplicate_percentage', 0)
            if duplicate_pct > 5:
                html += f'<div class="warning"><strong>Warning:</strong> {duplicate_pct:.1f}% of rows are duplicates</div>'
            else:
                html += f'<div class="success"><strong>Good:</strong> Only {duplicate_pct:.1f}% of rows are duplicates</div>'
            
            if data.get('constant_columns'):
                html += f'<div class="warning"><strong>Warning:</strong> {len(data["constant_columns"])} constant columns found</div>'
            
            if data.get('potential_keys'):
                html += f'<div class="success"><strong>Found:</strong> {len(data["potential_keys"])} potential primary keys</div>'
        
        html += '</div>'
        return html
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        report = {
            'metadata': {
                'dataset': self.data_source,
                'generated_at': datetime.now().isoformat(),
                'shape': {'rows': len(self.df), 'columns': len(self.df.columns)}
            },
            'profile_results': self.profile_results
        }
        
        json_content = json.dumps(report, indent=2, default=str)
        
        # Save JSON file
        with open('profiler_outputs/data_profile_report.json', 'w') as f:
            f.write(json_content)
        
        print(f"[OK] JSON report saved as 'profiler_outputs/data_profile_report.json'")
        return json_content
    
    def _generate_text_report(self) -> str:
        """Generate text report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA PROFILE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Dataset: {self.data_source}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Shape: {len(self.df)} rows × {len(self.df.columns)} columns")
        report_lines.append("")
        
        # Add sections
        report_lines.extend(self._generate_text_section("DATA TYPES", self.profile_results.get('data_types', {})))
        report_lines.extend(self._generate_text_section("MISSING DATA", self.profile_results.get('missing_data', {})))
        report_lines.extend(self._generate_text_section("NUMERIC ANALYSIS", self.profile_results.get('numeric_analysis', {})))
        report_lines.extend(self._generate_text_section("CATEGORICAL ANALYSIS", self.profile_results.get('categorical_analysis', {})))
        report_lines.extend(self._generate_text_section("PATTERNS", self.profile_results.get('patterns', {})))
        report_lines.extend(self._generate_text_section("CORRELATIONS", self.profile_results.get('correlation_analysis', {})))
        
        report_text = "\n".join(report_lines)
        
        # Save text file
        with open('profiler_outputs/data_profile_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Text report saved as 'profiler_outputs/data_profile_report.txt'")
        return report_text
    
    def _generate_text_section(self, title: str, data: Dict) -> List[str]:
        """Generate text section."""
        lines = []
        lines.append(f"{title}")
        lines.append("-" * len(title))
        
        if title == "DATA TYPES":
            for data_type, columns in data.items():
                if columns:
                    lines.append(f"{data_type}: {len(columns)} columns")
        
        elif title == "MISSING DATA":
            missing_pct = data.get('missing_percentage', 0)
            lines.append(f"Total missing: {data.get('total_missing', 0)} cells ({missing_pct:.1f}%)")
            
            if data.get('columns_with_missing'):
                lines.append("Columns with missing data:")
                for col, info in data['columns_with_missing'].items():
                    lines.append(f"  {col}: {info['count']} missing ({info['percentage']:.1f}%)")
        
        elif title == "NUMERIC ANALYSIS":
            if data:
                lines.append("Numeric column statistics:")
                for col, info in data.items():
                    lines.append(f"  {col}:")
                    lines.append(f"    Mean: {info['mean']:.2f}")
                    lines.append(f"    Std: {info['std']:.2f}")
                    lines.append(f"    Outliers: {info['outlier_count']} ({info['outlier_percentage']:.1f}%)")
        
        elif title == "PATTERNS":
            duplicate_pct = data.get('duplicate_percentage', 0)
            lines.append(f"Duplicate rows: {data.get('duplicate_rows', 0)} ({duplicate_pct:.1f}%)")
            
            if data.get('constant_columns'):
                lines.append(f"Constant columns: {data['constant_columns']}")
            
            if data.get('potential_keys'):
                lines.append(f"Potential primary keys: {data['potential_keys']}")
        
        lines.append("")
        return lines
    
    def run_comprehensive_profiling(self, table_name: str = None, output_format: str = 'html'):
        """Run comprehensive data profiling."""
        print(f"\n[START] Starting comprehensive data profiling")
        
        # Load data
        self.load_data(table_name)
        
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
        self.generate_report(output_format)
        
        print(f"\n[OK] Comprehensive profiling complete!")
        print(f"[DATA] Check 'profiler_outputs/' directory for results")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Data profiler closed")


def main():
    """Main function to run data profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive data profile')
    parser.add_argument('data_source', help='Path to database file or CSV file')
    parser.add_argument('--table', help='Table name (for SQLite databases)')
    parser.add_argument('--format', choices=['html', 'json', 'text'], default='html', 
                       help='Output format for report')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = DataProfiler(args.data_source)
    
    try:
        # Run comprehensive profiling
        profiler.run_comprehensive_profiling(args.table, args.format)
        
    except Exception as e:
        print(f"[ERROR] Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        profiler.close()


if __name__ == "__main__":
    main() 