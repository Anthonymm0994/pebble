#!/usr/bin/env python3
"""
Table Relationship Analyzer
===========================

A comprehensive tool to analyze relationships between two SQLite tables,
with special focus on timestamp-based connections and pattern detection.

This script helps uncover how one table might be derived from another
through various transformations like filtering, aggregation, or data processing.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import defaultdict, Counter
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TableRelationshipAnalyzer:
    """
    Comprehensive analyzer for finding relationships between two SQLite tables.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the analyzer with a SQLite database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.table_names = []
        self.table_info = {}
        self.df1 = None
        self.df2 = None
        self.timestamp_cols = []
        
    def connect(self):
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
        return True
    
    def get_table_names(self) -> List[str]:
        """Get all table names from the database."""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        self.table_names = tables
        print(f"📋 Found tables: {tables}")
        return tables
    
    def analyze_table_structure(self, table_name: str) -> Dict:
        """Analyze the structure of a table."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        # Analyze column types and patterns
        column_info = {}
        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col
            column_info[col_name] = {
                'type': col_type,
                'not_null': not_null,
                'primary_key': pk,
                'default': default_val
            }
        
        return {
            'columns': column_info,
            'sample_data': sample_data,
            'row_count': self.get_row_count(table_name)
        }
    
    def get_row_count(self, table_name: str) -> int:
        """Get the row count for a table."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    
    def load_tables(self, table1_name: str, table2_name: str):
        """Load both tables into pandas DataFrames."""
        print(f"\n📊 Loading tables: {table1_name} and {table2_name}")
        
        self.df1 = pd.read_sql_query(f"SELECT * FROM {table1_name}", self.conn)
        self.df2 = pd.read_sql_query(f"SELECT * FROM {table2_name}", self.conn)
        
        print(f"✅ Loaded {table1_name}: {len(self.df1)} rows, {len(self.df1.columns)} columns")
        print(f"✅ Loaded {table2_name}: {len(self.df2)} rows, {len(self.df2.columns)} columns")
        
        # Store table info
        self.table_info[table1_name] = self.analyze_table_structure(table1_name)
        self.table_info[table2_name] = self.analyze_table_structure(table2_name)
    
    def find_timestamp_columns(self) -> List[str]:
        """Find columns that might contain timestamps."""
        timestamp_patterns = [
            r'time', r'date', r'timestamp', r'created', r'updated', r'message',
            r'log', r'event', r'when', r'at', r'ts'
        ]
        
        all_columns = list(self.df1.columns) + list(self.df2.columns)
        timestamp_cols = []
        
        for col in all_columns:
            col_lower = col.lower()
            for pattern in timestamp_patterns:
                if re.search(pattern, col_lower):
                    timestamp_cols.append(col)
                    break
        
        self.timestamp_cols = list(set(timestamp_cols))
        print(f"🕒 Found potential timestamp columns: {self.timestamp_cols}")
        return self.timestamp_cols
    
    def parse_timestamps(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """Parse timestamps from a column."""
        if col_name not in df.columns:
            return pd.Series(dtype='datetime64[ns]')
        
        # Try different timestamp formats
        formats = [
            '%H:%M:%S.%f',  # 16:07:34.053
            '%H:%M:%S',     # 16:07:34
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(df[col_name], format=fmt, errors='coerce')
            except:
                continue
        
        # Try pandas automatic parsing
        return pd.to_datetime(df[col_name], errors='coerce')
    
    def analyze_timestamp_relationships(self) -> Dict:
        """Analyze relationships between timestamp columns."""
        print("\n🔍 Analyzing timestamp relationships...")
        
        timestamp_analysis = {}
        
        for col1 in self.df1.columns:
            for col2 in self.df2.columns:
                # Check if both columns might be timestamps
                if any(pattern in col1.lower() for pattern in ['time', 'date', 'message']) and \
                   any(pattern in col2.lower() for pattern in ['time', 'date', 'message']):
                    
                    print(f"  Analyzing: {col1} ↔ {col2}")
                    
                    # Parse timestamps
                    ts1 = self.parse_timestamps(self.df1, col1)
                    ts2 = self.parse_timestamps(self.df2, col2)
                    
                    if ts1.notna().any() and ts2.notna().any():
                        # Calculate time differences
                        valid_ts1 = ts1.dropna()
                        valid_ts2 = ts2.dropna()
                        
                        if len(valid_ts1) > 0 and len(valid_ts2) > 0:
                            # Find potential matches within reasonable time windows
                            time_diffs = []
                            matches = []
                            
                            for i, t1 in enumerate(valid_ts1):
                                for j, t2 in enumerate(valid_ts2):
                                    diff = abs((t1 - t2).total_seconds())
                                    if diff <= 3600:  # Within 1 hour
                                        time_diffs.append(diff)
                                        matches.append((i, j, diff))
                            
                            if time_diffs:
                                analysis = {
                                    'mean_delay': np.mean(time_diffs),
                                    'std_delay': np.std(time_diffs),
                                    'min_delay': np.min(time_diffs),
                                    'max_delay': np.max(time_diffs),
                                    'match_count': len(matches),
                                    'potential_matches': matches[:10]  # First 10 matches
                                }
                                
                                timestamp_analysis[f"{col1} ↔ {col2}"] = analysis
                                print(f"    ✅ Found {len(matches)} potential matches")
                                print(f"    ⏱️  Average delay: {analysis['mean_delay']:.2f}s")
        
        return timestamp_analysis
    
    def find_column_similarities(self) -> Dict:
        """Find similarities between columns across tables."""
        print("\n🔍 Analyzing column similarities...")
        
        similarities = {}
        
        for col1 in self.df1.columns:
            for col2 in self.df2.columns:
                similarity_score = self.calculate_column_similarity(col1, col2)
                if similarity_score > 0.3:  # Threshold for potential relationship
                    similarities[f"{col1} ↔ {col2}"] = similarity_score
        
        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
    
    def calculate_column_similarity(self, col1: str, col2: str) -> float:
        """Calculate similarity between two columns."""
        # Name similarity
        name_similarity = self.calculate_name_similarity(col1, col2)
        
        # Data type similarity
        type_similarity = self.calculate_type_similarity(col1, col2)
        
        # Value distribution similarity
        value_similarity = self.calculate_value_similarity(col1, col2)
        
        # Weighted average
        return 0.4 * name_similarity + 0.3 * type_similarity + 0.3 * value_similarity
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between column names."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8
        
        # Word overlap
        words1 = set(name1_lower.split('_'))
        words2 = set(name2_lower.split('_'))
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union if union > 0 else 0.0
        
        return 0.0
    
    def calculate_type_similarity(self, col1: str, col2: str) -> float:
        """Calculate similarity based on data types."""
        try:
            dtype1 = self.df1[col1].dtype
            dtype2 = self.df2[col2].dtype
            
            # Exact match
            if dtype1 == dtype2:
                return 1.0
            
            # Numeric types
            numeric_types = [np.number]
            if np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number):
                return 0.9
            
            # String types
            if dtype1 == 'object' and dtype2 == 'object':
                return 0.8
            
            # Datetime types
            if 'datetime' in str(dtype1) and 'datetime' in str(dtype2):
                return 0.9
            
            return 0.0
        except:
            return 0.0
    
    def calculate_value_similarity(self, col1: str, col2: str) -> float:
        """Calculate similarity based on value distributions."""
        try:
            # Get unique values
            unique1 = set(self.df1[col1].dropna().astype(str))
            unique2 = set(self.df2[col2].dropna().astype(str))
            
            if not unique1 or not unique2:
                return 0.0
            
            # Calculate overlap
            overlap = len(unique1.intersection(unique2))
            union = len(unique1.union(unique2))
            
            return overlap / union if union > 0 else 0.0
        except:
            return 0.0
    
    def detect_transformations(self) -> Dict:
        """Detect potential transformations between tables."""
        print("\n🔍 Detecting transformations...")
        
        transformations = {
            'filtering': self.detect_filtering(),
            'aggregation': self.detect_aggregation(),
            'column_mapping': self.detect_column_mapping(),
            'value_transformations': self.detect_value_transformations()
        }
        
        return transformations
    
    def detect_filtering(self) -> Dict:
        """Detect if one table is a filtered version of the other."""
        filtering_info = {}
        
        # Compare row counts
        row_ratio = len(self.df2) / len(self.df1) if len(self.df1) > 0 else 0
        
        if row_ratio < 1.0:
            filtering_info['row_count_ratio'] = row_ratio
            filtering_info['filtering_likelihood'] = 'high' if row_ratio < 0.8 else 'medium'
        
        # Check for common columns with different value distributions
        for col in self.df1.columns:
            if col in self.df2.columns:
                unique1 = self.df1[col].nunique()
                unique2 = self.df2[col].nunique()
                
                if unique2 < unique1:
                    filtering_info[f'column_{col}_unique_ratio'] = unique2 / unique1
        
        return filtering_info
    
    def detect_aggregation(self) -> Dict:
        """Detect if one table contains aggregated data."""
        aggregation_info = {}
        
        # Check for numeric columns that might be aggregated
        numeric_cols1 = self.df1.select_dtypes(include=[np.number]).columns
        numeric_cols2 = self.df2.select_dtypes(include=[np.number]).columns
        
        for col1 in numeric_cols1:
            for col2 in numeric_cols2:
                if col1.lower() in col2.lower() or col2.lower() in col1.lower():
                    # Check if values in df2 are larger (sum) or different (avg)
                    if len(self.df2) < len(self.df1):
                        mean1 = self.df1[col1].mean()
                        mean2 = self.df2[col2].mean()
                        
                        if abs(mean2 - mean1) > 0.01:
                            aggregation_info[f'{col1} → {col2}'] = {
                                'type': 'aggregation',
                                'mean_difference': mean2 - mean1
                            }
        
        return aggregation_info
    
    def detect_column_mapping(self) -> Dict:
        """Detect how columns map between tables."""
        mapping = {}
        
        for col1 in self.df1.columns:
            best_match = None
            best_score = 0
            
            for col2 in self.df2.columns:
                score = self.calculate_column_similarity(col1, col2)
                if score > best_score:
                    best_score = score
                    best_match = col2
            
            if best_score > 0.5:
                mapping[col1] = {
                    'mapped_to': best_match,
                    'confidence': best_score
                }
        
        return mapping
    
    def detect_value_transformations(self) -> Dict:
        """Detect value transformations between tables."""
        transformations = {}
        
        for col1 in self.df1.columns:
            for col2 in self.df2.columns:
                if col1.lower() in col2.lower() or col2.lower() in col1.lower():
                    # Check for mathematical transformations
                    if self.df1[col1].dtype in [np.number] and self.df2[col2].dtype in [np.number]:
                        try:
                            # Check for scaling
                            ratio = self.df2[col2].mean() / self.df1[col1].mean()
                            if 0.1 < ratio < 10 and ratio != 1:
                                transformations[f'{col1} → {col2}'] = {
                                    'type': 'scaling',
                                    'ratio': ratio
                                }
                            
                            # Check for offset
                            diff = self.df2[col2].mean() - self.df1[col1].mean()
                            if abs(diff) > 0.01:
                                transformations[f'{col1} → {col2}'] = {
                                    'type': 'offset',
                                    'difference': diff
                                }
                        except:
                            pass
        
        return transformations
    
    def suggest_joins(self) -> List[Dict]:
        """Suggest potential joins between the tables."""
        print("\n🔗 Suggesting potential joins...")
        
        join_suggestions = []
        
        # Find timestamp-based joins
        timestamp_analysis = self.analyze_timestamp_relationships()
        for join_key, analysis in timestamp_analysis.items():
            if analysis['match_count'] > 0:
                col1, col2 = join_key.split(' ↔ ')
                join_suggestions.append({
                    'type': 'timestamp_join',
                    'columns': (col1, col2),
                    'confidence': min(analysis['match_count'] / min(len(self.df1), len(self.df2)), 1.0),
                    'details': analysis
                })
        
        # Find exact value matches
        for col1 in self.df1.columns:
            for col2 in self.df2.columns:
                if col1 != col2:
                    # Check for exact value matches
                    common_values = set(self.df1[col1].dropna()) & set(self.df2[col2].dropna())
                    if len(common_values) > 0:
                        match_ratio = len(common_values) / min(len(self.df1), len(self.df2))
                        if match_ratio > 0.1:  # At least 10% match
                            join_suggestions.append({
                                'type': 'exact_match',
                                'columns': (col1, col2),
                                'confidence': match_ratio,
                                'details': {'common_values': len(common_values)}
                            })
        
        return sorted(join_suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def generate_visualizations(self):
        """Generate visualizations to help understand the relationships."""
        print("\n📊 Generating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Table Relationship Analysis', fontsize=16, fontweight='bold')
        
        # 1. Row count comparison
        axes[0, 0].bar(['Table 1', 'Table 2'], [len(self.df1), len(self.df2)])
        axes[0, 0].set_title('Row Count Comparison')
        axes[0, 0].set_ylabel('Number of Rows')
        
        # 2. Column count comparison
        axes[0, 1].bar(['Table 1', 'Table 2'], [len(self.df1.columns), len(self.df2.columns)])
        axes[0, 1].set_title('Column Count Comparison')
        axes[0, 1].set_ylabel('Number of Columns')
        
        # 3. Data type distribution
        dtype1 = self.df1.dtypes.value_counts()
        dtype2 = self.df2.dtypes.value_counts()
        
        x = np.arange(len(dtype1))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, dtype1.values, width, label='Table 1')
        axes[1, 0].bar(x + width/2, dtype2.values, width, label='Table 2')
        axes[1, 0].set_title('Data Type Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dtype1.index, rotation=45)
        axes[1, 0].legend()
        
        # 4. Missing data comparison
        missing1 = self.df1.isnull().sum().sum()
        missing2 = self.df2.isnull().sum().sum()
        
        axes[1, 1].bar(['Table 1', 'Table 2'], [missing1, missing2])
        axes[1, 1].set_title('Missing Data Comparison')
        axes[1, 1].set_ylabel('Number of Missing Values')
        
        plt.tight_layout()
        plt.savefig('table_relationship_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'table_relationship_analysis.png'")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        print("\n📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 80)
        report.append("TABLE RELATIONSHIP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic table information
        report.append("📋 TABLE OVERVIEW")
        report.append("-" * 40)
        report.append(f"Table 1: {len(self.df1)} rows, {len(self.df1.columns)} columns")
        report.append(f"Table 2: {len(self.df2)} rows, {len(self.df2.columns)} columns")
        report.append("")
        
        # Column similarities
        similarities = self.find_column_similarities()
        if similarities:
            report.append("🔍 COLUMN SIMILARITIES")
            report.append("-" * 40)
            for pair, score in list(similarities.items())[:10]:
                report.append(f"{pair}: {score:.3f}")
            report.append("")
        
        # Timestamp analysis
        timestamp_analysis = self.analyze_timestamp_relationships()
        if timestamp_analysis:
            report.append("🕒 TIMESTAMP RELATIONSHIPS")
            report.append("-" * 40)
            for pair, analysis in timestamp_analysis.items():
                report.append(f"{pair}:")
                report.append(f"  - Average delay: {analysis['mean_delay']:.2f}s")
                report.append(f"  - Match count: {analysis['match_count']}")
                report.append(f"  - Delay range: {analysis['min_delay']:.2f}s - {analysis['max_delay']:.2f}s")
            report.append("")
        
        # Transformations
        transformations = self.detect_transformations()
        if any(transformations.values()):
            report.append("🔄 DETECTED TRANSFORMATIONS")
            report.append("-" * 40)
            
            if transformations['filtering']:
                report.append("Filtering:")
                for key, value in transformations['filtering'].items():
                    report.append(f"  - {key}: {value}")
            
            if transformations['aggregation']:
                report.append("Aggregation:")
                for key, value in transformations['aggregation'].items():
                    report.append(f"  - {key}: {value}")
            
            if transformations['column_mapping']:
                report.append("Column Mapping:")
                for col1, mapping in transformations['column_mapping'].items():
                    report.append(f"  - {col1} → {mapping['mapped_to']} (confidence: {mapping['confidence']:.3f})")
            
            report.append("")
        
        # Join suggestions
        join_suggestions = self.suggest_joins()
        if join_suggestions:
            report.append("🔗 JOIN SUGGESTIONS")
            report.append("-" * 40)
            for suggestion in join_suggestions[:5]:
                report.append(f"Type: {suggestion['type']}")
                report.append(f"Columns: {suggestion['columns'][0]} ↔ {suggestion['columns'][1]}")
                report.append(f"Confidence: {suggestion['confidence']:.3f}")
                report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open('table_relationship_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Report saved as 'table_relationship_report.txt'")
        return report_text
    
    def run_comprehensive_analysis(self, table1_name: str, table2_name: str):
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive table relationship analysis...")
        print("=" * 80)
        
        # Connect to database
        if not self.connect():
            return
        
        # Get table names if not provided
        if not table1_name or not table2_name:
            tables = self.get_table_names()
            if len(tables) < 2:
                print("❌ Need at least 2 tables in the database")
                return
            table1_name, table2_name = tables[0], tables[1]
        
        # Load tables
        self.load_tables(table1_name, table2_name)
        
        # Find timestamp columns
        self.find_timestamp_columns()
        
        # Run all analyses
        similarities = self.find_column_similarities()
        timestamp_analysis = self.analyze_timestamp_relationships()
        transformations = self.detect_transformations()
        join_suggestions = self.suggest_joins()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete! Check the generated files:")
        print("  - table_relationship_report.txt (detailed report)")
        print("  - table_relationship_analysis.png (visualizations)")
        print("=" * 80)
        
        return {
            'similarities': similarities,
            'timestamp_analysis': timestamp_analysis,
            'transformations': transformations,
            'join_suggestions': join_suggestions,
            'report': report
        }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze relationships between SQLite tables')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--table1', help='Name of the first table')
    parser.add_argument('--table2', help='Name of the second table')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = TableRelationshipAnalyzer(args.db_path)
    
    try:
        results = analyzer.run_comprehensive_analysis(args.table1, args.table2)
        
        # Print summary
        if results:
            print("\n📊 SUMMARY:")
            print(f"  - Column similarities found: {len(results['similarities'])}")
            print(f"  - Timestamp relationships: {len(results['timestamp_analysis'])}")
            print(f"  - Join suggestions: {len(results['join_suggestions'])}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 