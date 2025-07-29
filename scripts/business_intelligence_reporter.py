#!/usr/bin/env python3
"""
Business Intelligence Reporter
============================

A comprehensive tool for generating business intelligence reports
with KPIs, trends, and actionable insights.

Features:
- KPI calculation and tracking
- Trend analysis
- Performance metrics
- Comparative analysis
- Forecasting capabilities
- Executive summaries
- Interactive dashboards
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BusinessIntelligenceReporter:
    """
    Business Intelligence reporting tool.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the BI reporter.
        
        Args:
            data_source: Path to database file or CSV file
        """
        self.data_source = data_source
        self.df = None
        self.report_data = {}
        
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
    
    def calculate_kpis(self) -> Dict:
        """Calculate key performance indicators."""
        print(f"\n[KPI] Calculating key performance indicators...")
        
        kpis = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'data_completeness': (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'unique_values_ratio': self.df.nunique().sum() / (len(self.df) * len(self.df.columns))
        }
        
        # Financial KPIs (if applicable)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if 'amount' in col.lower() or 'price' in col.lower() or 'cost' in col.lower():
                    kpis[f'total_{col}'] = self.df[col].sum()
                    kpis[f'average_{col}'] = self.df[col].mean()
                    kpis[f'median_{col}'] = self.df[col].median()
                    kpis[f'max_{col}'] = self.df[col].max()
                    kpis[f'min_{col}'] = self.df[col].min()
        
        # Categorical KPIs
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].nunique() < 20:  # Not too many categories
                value_counts = self.df[col].value_counts()
                kpis[f'top_{col}'] = value_counts.index[0] if len(value_counts) > 0 else None
                kpis[f'top_{col}_count'] = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        self.report_data['kpis'] = kpis
        return kpis
    
    def analyze_trends(self, date_column: str = None) -> Dict:
        """Analyze trends over time."""
        print(f"\n[TREND] Analyzing trends...")
        
        trends = {}
        
        # Find date column if not specified
        if not date_column:
            date_patterns = ['date', 'time', 'created', 'updated', 'timestamp']
            for col in self.df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in date_patterns):
                    date_column = col
                    break
        
        if date_column and date_column in self.df.columns:
            try:
                # Convert to datetime
                self.df[date_column] = pd.to_datetime(self.df[date_column])
                
                # Sort by date
                self.df_sorted = self.df.sort_values(date_column)
                
                # Calculate trends for numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != date_column:
                        # Calculate moving averages
                        self.df_sorted[f'{col}_ma_7'] = self.df_sorted[col].rolling(window=7).mean()
                        self.df_sorted[f'{col}_ma_30'] = self.df_sorted[col].rolling(window=30).mean()
                        
                        # Calculate trend direction
                        recent_data = self.df_sorted[col].tail(10)
                        if len(recent_data) > 1:
                            trend_slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                            trends[f'{col}_trend'] = 'increasing' if trend_slope > 0 else 'decreasing'
                            trends[f'{col}_trend_strength'] = abs(trend_slope)
                
                trends['date_range'] = {
                    'start': self.df_sorted[date_column].min(),
                    'end': self.df_sorted[date_column].max(),
                    'duration_days': (self.df_sorted[date_column].max() - self.df_sorted[date_column].min()).days
                }
                
            except Exception as e:
                print(f"[WARNING] Could not analyze trends: {e}")
        
        self.report_data['trends'] = trends
        return trends
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        print(f"\n[METRICS] Calculating performance metrics...")
        
        metrics = {}
        
        # Data quality metrics
        metrics['data_quality'] = {
            'completeness': (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_ratio': (len(self.df) - len(self.df.drop_duplicates())) / len(self.df) * 100,
            'consistency_score': self._calculate_consistency_score()
        }
        
        # Efficiency metrics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metrics['efficiency'] = {}
            for col in numeric_cols:
                if 'amount' in col.lower() or 'price' in col.lower():
                    metrics['efficiency'][f'{col}_efficiency'] = {
                        'total': self.df[col].sum(),
                        'average': self.df[col].mean(),
                        'std_dev': self.df[col].std(),
                        'coefficient_of_variation': self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else 0
                    }
        
        # Distribution metrics
        metrics['distribution'] = {}
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                metrics['distribution'][col] = {
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'percentiles': {
                        '25th': data.quantile(0.25),
                        '50th': data.quantile(0.50),
                        '75th': data.quantile(0.75),
                        '90th': data.quantile(0.90),
                        '95th': data.quantile(0.95)
                    }
                }
        
        self.report_data['performance_metrics'] = metrics
        return metrics
    
    def _calculate_consistency_score(self) -> float:
        """Calculate data consistency score."""
        score = 100.0
        
        # Penalize for missing data
        missing_ratio = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))
        score -= missing_ratio * 50
        
        # Penalize for duplicates
        duplicate_ratio = (len(self.df) - len(self.df.drop_duplicates())) / len(self.df)
        score -= duplicate_ratio * 30
        
        # Penalize for outliers in numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                outlier_ratio = len(outliers) / len(data)
                score -= outlier_ratio * 20
        
        return max(0, score)
    
    def perform_comparative_analysis(self, comparison_column: str = None) -> Dict:
        """Perform comparative analysis."""
        print(f"\n[COMPARE] Performing comparative analysis...")
        
        comparison = {}
        
        # Find good comparison column
        if not comparison_column:
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if 2 <= self.df[col].nunique() <= 10:  # Good for comparison
                    comparison_column = col
                    break
        
        if comparison_column and comparison_column in self.df.columns:
            comparison['comparison_column'] = comparison_column
            comparison['groups'] = {}
            
            for group in self.df[comparison_column].unique():
                group_data = self.df[self.df[comparison_column] == group]
                comparison['groups'][group] = {
                    'count': len(group_data),
                    'percentage': len(group_data) / len(self.df) * 100
                }
                
                # Add numeric metrics for each group
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != comparison_column:
                        group_numeric = group_data[col].dropna()
                        if len(group_numeric) > 0:
                            comparison['groups'][group][col] = {
                                'mean': group_numeric.mean(),
                                'median': group_numeric.median(),
                                'std': group_numeric.std(),
                                'count': len(group_numeric)
                            }
        
        self.report_data['comparative_analysis'] = comparison
        return comparison
    
    def generate_forecasts(self, target_column: str = None, periods: int = 5) -> Dict:
        """Generate simple forecasts."""
        print(f"\n[FORECAST] Generating forecasts...")
        
        forecasts = {}
        
        # Find target column if not specified
        if not target_column:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[0]
        
        if target_column and target_column in self.df.columns:
            data = self.df[target_column].dropna()
            if len(data) > periods:
                # Simple linear trend forecast
                x = np.arange(len(data))
                y = data.values
                
                # Fit linear trend
                coeffs = np.polyfit(x, y, 1)
                trend_line = np.poly1d(coeffs)
                
                # Generate forecast
                future_x = np.arange(len(data), len(data) + periods)
                forecast_values = trend_line(future_x)
                
                forecasts[target_column] = {
                    'trend_coefficient': coeffs[0],
                    'intercept': coeffs[1],
                    'forecast_periods': periods,
                    'forecast_values': forecast_values.tolist(),
                    'last_actual_value': data.iloc[-1],
                    'forecast_accuracy': self._calculate_forecast_accuracy(data, trend_line)
                }
        
        self.report_data['forecasts'] = forecasts
        return forecasts
    
    def _calculate_forecast_accuracy(self, data: pd.Series, trend_line) -> float:
        """Calculate forecast accuracy using historical data."""
        x = np.arange(len(data))
        predicted = trend_line(x)
        actual = data.values
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return max(0, 100 - mape)  # Return accuracy percentage
    
    def create_executive_summary(self) -> str:
        """Create executive summary."""
        print(f"\n[SUMMARY] Creating executive summary...")
        
        summary = []
        summary.append("EXECUTIVE SUMMARY")
        summary.append("=" * 50)
        summary.append("")
        
        # Key findings
        kpis = self.report_data.get('kpis', {})
        trends = self.report_data.get('trends', {})
        metrics = self.report_data.get('performance_metrics', {})
        
        summary.append("KEY FINDINGS:")
        summary.append(f"- Total records analyzed: {kpis.get('total_records', 0):,}")
        summary.append(f"- Data completeness: {kpis.get('data_completeness', 0):.1f}%")
        
        # Trend insights
        if trends:
            summary.append("")
            summary.append("TREND INSIGHTS:")
            for key, value in trends.items():
                if 'trend' in key and 'strength' not in key:
                    summary.append(f"- {key}: {value}")
        
        # Performance insights
        if metrics.get('data_quality'):
            quality = metrics['data_quality']
            summary.append("")
            summary.append("DATA QUALITY:")
            summary.append(f"- Consistency score: {quality.get('consistency_score', 0):.1f}%")
            summary.append(f"- Duplicate ratio: {quality.get('duplicate_ratio', 0):.1f}%")
        
        # Recommendations
        summary.append("")
        summary.append("RECOMMENDATIONS:")
        
        if kpis.get('data_completeness', 100) < 90:
            summary.append("- Address missing data issues to improve data quality")
        
        if metrics.get('data_quality', {}).get('duplicate_ratio', 0) > 5:
            summary.append("- Implement duplicate detection and removal processes")
        
        if trends:
            summary.append("- Monitor trend changes and adjust strategies accordingly")
        
        summary.append("- Regular data quality audits recommended")
        
        summary_text = "\n".join(summary)
        
        # Save summary
        with open('../outputs/bi_outputs/executive_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(f"[OK] Executive summary saved as '../outputs/bi_outputs/executive_summary.txt'")
        return summary_text
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n[CHART] Creating visualizations...")
        
        # Create output directory
        os.makedirs('../outputs/bi_outputs', exist_ok=True)
        
        # 1. KPI Dashboard
        self._create_kpi_dashboard()
        
        # 2. Trend Analysis
        self._create_trend_analysis()
        
        # 3. Performance Metrics
        self._create_performance_metrics()
        
        # 4. Comparative Analysis
        self._create_comparative_analysis()
        
        # 5. Forecast Charts
        self._create_forecast_charts()
        
        print(f"[OK] All visualizations saved in '../outputs/bi_outputs/' directory")
    
    def _create_kpi_dashboard(self):
        """Create KPI dashboard."""
        kpis = self.report_data.get('kpis', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KPI Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Data Overview
        axes[0, 0].pie([kpis.get('total_records', 0), kpis.get('total_columns', 0)], 
                       labels=['Records', 'Columns'], 
                       autopct='%1.1f%%', 
                       colors=['lightblue', 'lightgreen'])
        axes[0, 0].set_title('Data Overview')
        
        # 2. Data Quality
        completeness = kpis.get('data_completeness', 0)
        axes[0, 1].bar(['Completeness'], [completeness], color='green' if completeness > 90 else 'orange')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].set_title('Data Completeness (%)')
        axes[0, 1].set_ylabel('Percentage')
        
        # 3. Numeric KPIs
        numeric_kpis = {k: v for k, v in kpis.items() if isinstance(v, (int, float)) and 'total' in k.lower()}
        if numeric_kpis:
            axes[1, 0].bar(numeric_kpis.keys(), numeric_kpis.values(), color='skyblue')
            axes[1, 0].set_title('Total Values by Metric')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Unique Values Ratio
        unique_ratio = kpis.get('unique_values_ratio', 0)
        axes[1, 1].bar(['Uniqueness'], [unique_ratio * 100], color='purple')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_title('Data Uniqueness (%)')
        axes[1, 1].set_ylabel('Percentage')
        
        plt.tight_layout()
        plt.savefig('../outputs/bi_outputs/kpi_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trend_analysis(self):
        """Create trend analysis charts."""
        trends = self.report_data.get('trends', {})
        
        if trends and hasattr(self, 'df_sorted'):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Trend Analysis', fontsize=16, fontweight='bold')
            
            # Get numeric columns for trend analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
            
            for i, col in enumerate(numeric_cols):
                row = i // 2
                col_idx = i % 2
                
                if f'{col}_ma_7' in self.df_sorted.columns:
                    axes[row, col_idx].plot(self.df_sorted.index, self.df_sorted[col], alpha=0.6, label='Actual')
                    axes[row, col_idx].plot(self.df_sorted.index, self.df_sorted[f'{col}_ma_7'], label='7-period MA', linewidth=2)
                    axes[row, col_idx].set_title(f'{col} Trend')
                    axes[row, col_idx].legend()
            
            plt.tight_layout()
            plt.savefig('../outputs/bi_outputs/trend_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_performance_metrics(self):
        """Create performance metrics charts."""
        metrics = self.report_data.get('performance_metrics', {})
        
        if metrics.get('distribution'):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
            
            # Distribution metrics
            dist_metrics = metrics['distribution']
            cols = list(dist_metrics.keys())[:4]  # Limit to 4 columns
            
            for i, col in enumerate(cols):
                row = i // 2
                col_idx = i % 2
                
                data = self.df[col].dropna()
                if len(data) > 0:
                    axes[row, col_idx].hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    axes[row, col_idx].set_title(f'{col} Distribution')
                    axes[row, col_idx].set_xlabel(col)
                    axes[row, col_idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('../outputs/bi_outputs/performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_comparative_analysis(self):
        """Create comparative analysis charts."""
        comparison = self.report_data.get('comparative_analysis', {})
        
        if comparison.get('groups'):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Comparative Analysis', fontsize=16, fontweight='bold')
            
            groups = list(comparison['groups'].keys())
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 columns
            
            for i, col in enumerate(numeric_cols):
                row = i // 2
                col_idx = i % 2
                
                group_means = []
                group_labels = []
                
                for group in groups:
                    if col in comparison['groups'][group]:
                        group_means.append(comparison['groups'][group][col]['mean'])
                        group_labels.append(group)
                
                if group_means:
                    axes[row, col_idx].bar(group_labels, group_means, color='orange')
                    axes[row, col_idx].set_title(f'{col} by Group')
                    axes[row, col_idx].set_ylabel('Mean')
                    axes[row, col_idx].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('../outputs/bi_outputs/comparative_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_forecast_charts(self):
        """Create forecast charts."""
        forecasts = self.report_data.get('forecasts', {})
        
        if forecasts:
            fig, axes = plt.subplots(1, len(forecasts), figsize=(15, 6))
            fig.suptitle('Forecast Analysis', fontsize=16, fontweight='bold')
            
            if len(forecasts) == 1:
                axes = [axes]
            
            for i, (col, forecast) in enumerate(forecasts.items()):
                # Plot historical data
                data = self.df[col].dropna()
                axes[i].plot(range(len(data)), data, label='Historical', color='blue')
                
                # Plot forecast
                forecast_values = forecast['forecast_values']
                forecast_x = range(len(data), len(data) + len(forecast_values))
                axes[i].plot(forecast_x, forecast_values, label='Forecast', color='red', linestyle='--')
                
                axes[i].set_title(f'{col} Forecast')
                axes[i].set_xlabel('Time Period')
                axes[i].set_ylabel(col)
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig('../outputs/bi_outputs/forecast_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self, table_name: str = None):
        """Generate comprehensive BI report."""
        print(f"\n[START] Generating comprehensive BI report")
        
        # Load data
        self.load_data(table_name)
        
        # Run all analyses
        self.calculate_kpis()
        self.analyze_trends()
        self.calculate_performance_metrics()
        self.perform_comparative_analysis()
        self.generate_forecasts()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate executive summary
        self.create_executive_summary()
        
        print(f"\n[OK] Comprehensive BI report complete!")
        print(f"[DATA] Check '../outputs/bi_outputs/' directory for results")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] BI reporter closed")


def main():
    """Main function to run BI reporting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive business intelligence report')
    parser.add_argument('data_source', help='Path to database file or CSV file')
    parser.add_argument('--table', help='Table name (for SQLite databases)')
    
    args = parser.parse_args()
    
    # Create BI reporter
    reporter = BusinessIntelligenceReporter(args.data_source)
    
    try:
        # Generate comprehensive report
        reporter.generate_comprehensive_report(args.table)
        
    except Exception as e:
        print(f"[ERROR] Error during reporting: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        reporter.close()


if __name__ == "__main__":
    main() 