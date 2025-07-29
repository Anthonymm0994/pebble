#!/usr/bin/env python3
"""
Table Relationship Analyzer for SQLite
=====================================

A comprehensive tool for analyzing relationships between two SQLite tables,
finding connections, detecting derivations, and understanding data transformations.

Features:
- SQLite-specific table analysis
- Timestamp correlation with variable delays
- Column similarity detection
- Transformation pattern detection
- Join suggestion algorithms
- Professional visualizations
- Comparative analysis
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class TableRelationshipAnalyzer:
    """
    Analyze relationships between two SQLite tables to understand derivations and connections.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize the table relationship analyzer.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self.conn = None
        self.table1_data = None
        self.table2_data = None
        self.table1_name = None
        self.table2_name = None
        self.analysis_results = {}
        
    def connect_database(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.database_path)
            print(f"[CONNECT] Connected to database: {self.database_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            return False
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables in the database."""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    
    def load_table_data(self, table1_name: str, table2_name: str):
        """Load data from two tables for analysis."""
        print(f"[LOAD] Loading tables: {table1_name} and {table2_name}")
        
        try:
            # Load first table
            self.table1_data = pd.read_sql_query(f"SELECT * FROM {table1_name}", self.conn)
            self.table1_name = table1_name
            print(f"[OK] Loaded {table1_name}: {len(self.table1_data)} rows, {len(self.table1_data.columns)} columns")
            
            # Load second table
            self.table2_data = pd.read_sql_query(f"SELECT * FROM {table2_name}", self.conn)
            self.table2_name = table2_name
            print(f"[OK] Loaded {table2_name}: {len(self.table2_data)} rows, {len(self.table2_data.columns)} columns")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load tables: {e}")
            return False
    
    def parse_24h_time(self, time_str: str) -> Optional[float]:
        """Parse 24-hour time format (HH:MM:SS.mmm) to decimal hours."""
        try:
            # Handle the format like "16:07:34.053"
            time_parts = time_str.strip().split(':')
            
            # Check if this is a time format (has at least 2 parts and first part is < 24)
            if len(time_parts) >= 2:
                hours = int(time_parts[0])
                if hours < 24:  # This looks like a time format
                    minutes = int(time_parts[1])
                    if len(time_parts) >= 3:
                        seconds = float(time_parts[2])
                    else:
                        seconds = 0
                    
                    # Convert to decimal hours
                    decimal_hours = hours + minutes/60 + seconds/3600
                    return decimal_hours
                else:
                    # This might be a date format, try to extract time component
                    return None
            else:
                return None

        except Exception as e:
            return None
    
    def find_timestamp_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that might contain timestamps."""
        timestamp_patterns = ['time', 'timestamp', 'date', 'created', 'updated', 'message']
        timestamp_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in timestamp_patterns):
                timestamp_cols.append(col)
        
        return timestamp_cols
    
    def analyze_timestamp_correlations(self) -> Dict[str, Any]:
        """Analyze timestamp correlations between tables."""
        print(f"\n[TIMESTAMP] Analyzing timestamp correlations...")
        
        results = {
            'timestamp_cols_table1': [],
            'timestamp_cols_table2': [],
            'correlations': [],
            'delays': [],
            'patterns': []
        }
        
        # Find timestamp columns in both tables
        table1_time_cols = self.find_timestamp_columns(self.table1_data)
        table2_time_cols = self.find_timestamp_columns(self.table2_data)
        
        results['timestamp_cols_table1'] = table1_time_cols
        results['timestamp_cols_table2'] = table2_time_cols
        
        print(f"[INFO] Found timestamp columns in {self.table1_name}: {table1_time_cols}")
        print(f"[INFO] Found timestamp columns in {self.table2_name}: {table2_time_cols}")
        
        # Analyze correlations between timestamp columns
        for col1 in table1_time_cols:
            for col2 in table2_time_cols:
                print(f"[ANALYZE] Comparing {col1} vs {col2}")
                
                # Parse timestamps
                times1 = []
                times2 = []
                
                for val in self.table1_data[col1].dropna():
                    parsed = self.parse_24h_time(str(val))
                    if parsed is not None:
                        times1.append(parsed)
                
                for val in self.table2_data[col2].dropna():
                    parsed = self.parse_24h_time(str(val))
                    if parsed is not None:
                        times2.append(parsed)
                
                if times1 and times2:
                    # Calculate correlation
                    min_len = min(len(times1), len(times2))
                    correlation = np.corrcoef(times1[:min_len], times2[:min_len])[0, 1]
                    
                    # Calculate average delay
                    delays = [t2 - t1 for t1, t2 in zip(times1[:min_len], times2[:min_len])]
                    avg_delay = np.mean(delays) if delays else 0
                    
                    results['correlations'].append({
                        'col1': col1,
                        'col2': col2,
                        'correlation': correlation,
                        'avg_delay': avg_delay,
                        'sample_size': min_len
                    })
                    
                    print(f"[CORRELATION] {col1} vs {col2}: {correlation:.3f}, avg delay: {avg_delay:.2f} hours")
        
        return results
    
    def analyze_column_similarities(self) -> Dict[str, Any]:
        """Analyze similarities between columns in both tables."""
        print(f"\n[COLUMNS] Analyzing column similarities...")
        
        results = {
            'exact_matches': [],
            'similar_names': [],
            'similar_data_types': [],
            'value_overlaps': []
        }
        
        # Find exact column name matches
        common_cols = set(self.table1_data.columns) & set(self.table2_data.columns)
        results['exact_matches'] = list(common_cols)
        
        print(f"[MATCHES] Exact column matches: {common_cols}")
        
        # Find similar column names
        for col1 in self.table1_data.columns:
            for col2 in self.table2_data.columns:
                if col1 != col2:
                    # Simple similarity check
                    col1_lower = col1.lower().replace('_', '').replace(' ', '')
                    col2_lower = col2.lower().replace('_', '').replace(' ', '')
                    
                    if col1_lower in col2_lower or col2_lower in col1_lower:
                        results['similar_names'].append((col1, col2))
        
        # Analyze data type similarities
        for col1 in self.table1_data.columns:
            for col2 in self.table2_data.columns:
                if col1 != col2:
                    dtype1 = self.table1_data[col1].dtype
                    dtype2 = self.table2_data[col2].dtype
                    
                    if dtype1 == dtype2:
                        results['similar_data_types'].append((col1, col2, str(dtype1)))
        
        # Analyze value overlaps for categorical columns
        for col1 in self.table1_data.select_dtypes(include=['object']).columns:
            for col2 in self.table2_data.select_dtypes(include=['object']).columns:
                if col1 != col2:
                    values1 = set(self.table1_data[col1].dropna().unique())
                    values2 = set(self.table2_data[col2].dropna().unique())
                    
                    if values1 and values2:
                        overlap = len(values1 & values2)
                        total = len(values1 | values2)
                        overlap_ratio = overlap / total if total > 0 else 0
                        
                        if overlap_ratio > 0.1:  # At least 10% overlap
                            results['value_overlaps'].append({
                                'col1': col1,
                                'col2': col2,
                                'overlap_ratio': overlap_ratio,
                                'overlap_count': overlap,
                                'total_unique': total
                            })
        
        return results
    
    def detect_transformations(self) -> Dict[str, Any]:
        """Detect potential transformations between tables."""
        print(f"\n[TRANSFORM] Detecting transformations...")
        
        results = {
            'filters': [],
            'aggregations': [],
            'column_mappings': [],
            'value_transformations': []
        }
        
        # Detect filters (missing values in table2 that exist in table1)
        for col in set(self.table1_data.columns) & set(self.table2_data.columns):
            values1 = set(self.table1_data[col].dropna().unique())
            values2 = set(self.table2_data[col].dropna().unique())
            
            missing_in_table2 = values1 - values2
            if missing_in_table2:
                results['filters'].append({
                    'column': col,
                    'filtered_values': list(missing_in_table2),
                    'filter_type': 'exclusion'
                })
        
        # Detect aggregations (counts, sums, etc.)
        numeric_cols1 = self.table1_data.select_dtypes(include=[np.number]).columns
        numeric_cols2 = self.table2_data.select_dtypes(include=[np.number]).columns
        
        for col1 in numeric_cols1:
            for col2 in numeric_cols2:
                if col1 != col2:
                    # Check if col2 might be an aggregation of col1
                    if len(self.table2_data) < len(self.table1_data):
                        # Potential aggregation
                        results['aggregations'].append({
                            'source_column': col1,
                            'target_column': col2,
                            'aggregation_type': 'potential'
                        })
        
        # Detect column mappings
        for col1 in self.table1_data.columns:
            for col2 in self.table2_data.columns:
                if col1 != col2:
                    # Check for potential mapping
                    if len(self.table1_data[col1].dropna()) == len(self.table2_data[col2].dropna()):
                        results['column_mappings'].append({
                            'source_column': col1,
                            'target_column': col2,
                            'mapping_type': 'potential'
                        })
        
        return results
    
    def suggest_joins(self) -> Dict[str, Any]:
        """Suggest potential join strategies between tables."""
        print(f"\n[JOIN] Suggesting join strategies...")
        
        results = {
            'exact_joins': [],
            'fuzzy_joins': [],
            'timestamp_joins': [],
            'recommendations': []
        }
        
        # Exact value joins
        for col1 in self.table1_data.columns:
            for col2 in self.table2_data.columns:
                if col1 != col2:
                    values1 = set(self.table1_data[col1].dropna().unique())
                    values2 = set(self.table2_data[col2].dropna().unique())
                    
                    if values1 and values2:
                        overlap = len(values1 & values2)
                        min_size = min(len(values1), len(values2))
                        
                        if overlap / min_size > 0.8:  # 80% overlap
                            results['exact_joins'].append({
                                'table1_column': col1,
                                'table2_column': col2,
                                'overlap_ratio': overlap / min_size,
                                'join_type': 'exact'
                            })
        
        # Timestamp-based joins
        timestamp_results = self.analyze_timestamp_correlations()
        for corr in timestamp_results['correlations']:
            if corr['correlation'] > 0.7:  # High correlation
                results['timestamp_joins'].append({
                    'table1_column': corr['col1'],
                    'table2_column': corr['col2'],
                    'correlation': corr['correlation'],
                    'avg_delay': corr['avg_delay'],
                    'join_type': 'timestamp'
                })
        
        # Generate recommendations
        if results['exact_joins']:
            results['recommendations'].append("Use exact value joins for high-confidence relationships")
        
        if results['timestamp_joins']:
            results['recommendations'].append("Use timestamp-based joins with delay compensation")
        
        return results
    
    def create_comparative_visualizations(self):
        """Create comparative visualizations between tables."""
        print(f"\n[VISUALIZE] Creating comparative visualizations...")
        
        # Create output directory
        os.makedirs('../outputs/relationship_outputs', exist_ok=True)
        
        # 1. Row count comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Row counts
        tables = [self.table1_name, self.table2_name]
        counts = [len(self.table1_data), len(self.table2_data)]
        colors = ['steelblue', 'lightcoral']
        
        bars = ax1.bar(tables, counts, color=colors, alpha=0.7)
        ax1.set_title('Row Count Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Number of Rows', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Column count comparison
        col_counts = [len(self.table1_data.columns), len(self.table2_data.columns)]
        bars = ax2.bar(tables, col_counts, color=colors, alpha=0.7)
        ax2.set_title('Column Count Comparison', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Number of Columns', fontsize=12)
        
        for bar, count in zip(bars, col_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Data type comparison
        dtype1 = self.table1_data.dtypes.value_counts()
        dtype2 = self.table2_data.dtypes.value_counts()
        
        all_dtypes = set(dtype1.index) | set(dtype2.index)
        x = np.arange(len(all_dtypes))
        width = 0.35
        
        ax3.bar(x - width/2, [dtype1.get(dtype, 0) for dtype in all_dtypes], 
                width, label=self.table1_name, color='steelblue', alpha=0.7)
        ax3.bar(x + width/2, [dtype2.get(dtype, 0) for dtype in all_dtypes], 
                width, label=self.table2_name, color='lightcoral', alpha=0.7)
        
        ax3.set_title('Data Type Distribution', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Number of Columns', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_dtypes, rotation=45)
        ax3.legend()
        
        # Missing data comparison
        missing1 = self.table1_data.isnull().sum().sum()
        missing2 = self.table2_data.isnull().sum().sum()
        total1 = self.table1_data.size
        total2 = self.table2_data.size
        
        missing_rates = [missing1/total1*100, missing2/total2*100]
        bars = ax4.bar(tables, missing_rates, color=colors, alpha=0.7)
        ax4.set_title('Missing Data Rate (%)', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Missing Data (%)', fontsize=12)
        
        for bar, rate in zip(bars, missing_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../outputs/relationship_outputs/table_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Table comparison visualization saved")
        
        # 2. Column similarity heatmap
        if len(self.table1_data.columns) > 0 and len(self.table2_data.columns) > 0:
            similarity_matrix = np.zeros((len(self.table1_data.columns), len(self.table2_data.columns)))
            
            for i, col1 in enumerate(self.table1_data.columns):
                for j, col2 in enumerate(self.table2_data.columns):
                    # Calculate similarity score
                    score = 0
                    
                    # Name similarity
                    if col1.lower() == col2.lower():
                        score += 0.5
                    elif col1.lower() in col2.lower() or col2.lower() in col1.lower():
                        score += 0.3
                    
                    # Data type similarity
                    if self.table1_data[col1].dtype == self.table2_data[col2].dtype:
                        score += 0.3
                    
                    # Value overlap for categorical
                    if (self.table1_data[col1].dtype == 'object' and 
                        self.table2_data[col2].dtype == 'object'):
                        values1 = set(self.table1_data[col1].dropna().unique())
                        values2 = set(self.table2_data[col2].dropna().unique())
                        if values1 and values2:
                            overlap = len(values1 & values2) / len(values1 | values2)
                            score += overlap * 0.2
                    
                    similarity_matrix[i, j] = score
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(similarity_matrix, 
                       xticklabels=self.table2_data.columns,
                       yticklabels=self.table1_data.columns,
                       annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Column Similarity Matrix', fontweight='bold', fontsize=14)
            plt.xlabel(f'{self.table2_name} Columns', fontsize=12)
            plt.ylabel(f'{self.table1_name} Columns', fontsize=12)
            plt.tight_layout()
            plt.savefig('../outputs/relationship_outputs/column_similarity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Column similarity heatmap saved")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive relationship analysis report."""
        print(f"\n[REPORT] Generating comprehensive relationship analysis...")
        
        # Run all analyses
        timestamp_analysis = self.analyze_timestamp_correlations()
        column_analysis = self.analyze_column_similarities()
        transformation_analysis = self.detect_transformations()
        join_analysis = self.suggest_joins()
        
        # Create visualizations
        self.create_comparative_visualizations()
        
        # Compile comprehensive report
        report = {
            'table_info': {
                'table1_name': self.table1_name,
                'table2_name': self.table2_name,
                'table1_rows': len(self.table1_data),
                'table2_rows': len(self.table2_data),
                'table1_columns': len(self.table1_data.columns),
                'table2_columns': len(self.table2_data.columns)
            },
            'timestamp_analysis': timestamp_analysis,
            'column_analysis': column_analysis,
            'transformation_analysis': transformation_analysis,
            'join_analysis': join_analysis,
            'summary': {
                'high_correlation_pairs': len([c for c in timestamp_analysis['correlations'] if c['correlation'] > 0.7]),
                'exact_column_matches': len(column_analysis['exact_matches']),
                'potential_transformations': len(transformation_analysis['filters']) + len(transformation_analysis['aggregations']),
                'suggested_joins': len(join_analysis['exact_joins']) + len(join_analysis['timestamp_joins'])
            }
        }
        
        # Save report
        import json
        with open('../outputs/relationship_outputs/relationship_analysis.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text summary
        with open('../outputs/relationship_outputs/relationship_summary.txt', 'w') as f:
            f.write("TABLE RELATIONSHIP ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Tables Analyzed:\n")
            f.write(f"  - {self.table1_name}: {len(self.table1_data)} rows, {len(self.table1_data.columns)} columns\n")
            f.write(f"  - {self.table2_name}: {len(self.table2_data)} rows, {len(self.table2_data.columns)} columns\n\n")
            
            f.write("Key Findings:\n")
            f.write(f"  - High correlation timestamp pairs: {report['summary']['high_correlation_pairs']}\n")
            f.write(f"  - Exact column matches: {report['summary']['exact_column_matches']}\n")
            f.write(f"  - Potential transformations: {report['summary']['potential_transformations']}\n")
            f.write(f"  - Suggested joins: {report['summary']['suggested_joins']}\n\n")
            
            if timestamp_analysis['correlations']:
                f.write("Timestamp Correlations:\n")
                for corr in timestamp_analysis['correlations']:
                    f.write(f"  - {corr['col1']} vs {corr['col2']}: {corr['correlation']:.3f}\n")
                f.write("\n")
            
            if column_analysis['exact_matches']:
                f.write("Exact Column Matches:\n")
                for col in column_analysis['exact_matches']:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            if join_analysis['recommendations']:
                f.write("Join Recommendations:\n")
                for rec in join_analysis['recommendations']:
                    f.write(f"  - {rec}\n")
        
        print(f"[OK] Comprehensive relationship analysis complete!")
        print(f"[DATA] Check '../outputs/relationship_outputs/' directory for results")
        
        return report
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run table relationship analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze relationships between two SQLite tables')
    parser.add_argument('database_path', help='Path to SQLite database file')
    parser.add_argument('table1', help='First table name')
    parser.add_argument('table2', help='Second table name')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TableRelationshipAnalyzer(args.database_path)
    
    try:
        # Connect to database
        if not analyzer.connect_database():
            return
        
        # Check available tables
        tables = analyzer.get_available_tables()
        print(f"[INFO] Available tables: {tables}")
        
        if args.table1 not in tables or args.table2 not in tables:
            print(f"[ERROR] One or both tables not found. Available: {tables}")
            return
        
        # Load table data
        if not analyzer.load_table_data(args.table1, args.table2):
            return
        
        # Generate comprehensive analysis
        analyzer.generate_comprehensive_report()
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 