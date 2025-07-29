#!/usr/bin/env python3
"""
Dataset Relationship Detector
============================

A comprehensive tool to reverse-engineer how one dataset was derived from another.
Focuses on timestamp relationships, column mapping, and pattern detection.

This script helps uncover:
- How timestamps relate with processing delays
- Which columns map to which across tables
- What transformations were applied
- Filtering and aggregation patterns
- High-confidence join suggestions
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
from difflib import SequenceMatcher
from scipy import stats
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DatasetRelationshipDetector:
    """
    Comprehensive detector for finding relationships between two datasets.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the detector with a SQLite database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.df_source = None
        self.df_derived = None
        self.analysis_results = {}
        
    def connect(self) -> bool:
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"[OK] Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error connecting to database: {e}")
            return False
    
    def load_datasets(self, source_table: str, derived_table: str, sample_size: int = None):
        """Load source and derived datasets."""
        print(f"\n[DATA] Loading datasets: {source_table} -> {derived_table}")
        
        if sample_size:
            self.df_source = pd.read_sql_query(
                f"SELECT * FROM {source_table} ORDER BY RANDOM() LIMIT {sample_size}", 
                self.conn
            )
            self.df_derived = pd.read_sql_query(
                f"SELECT * FROM {derived_table} ORDER BY RANDOM() LIMIT {sample_size}", 
                self.conn
            )
        else:
            self.df_source = pd.read_sql_query(f"SELECT * FROM {source_table}", self.conn)
            self.df_derived = pd.read_sql_query(f"SELECT * FROM {derived_table}", self.conn)
        
        print(f"[OK] Loaded source: {len(self.df_source)} rows, {len(self.df_source.columns)} columns")
        print(f"[OK] Loaded derived: {len(self.df_derived)} rows, {len(self.df_derived.columns)} columns")
        
        self.source_table = source_table
        self.derived_table = derived_table
    
    def find_timestamp_columns(self) -> Dict[str, List[str]]:
        """Find timestamp columns in both datasets."""
        timestamp_patterns = [
            'time', 'date', 'timestamp', 'created', 'updated', 'message',
            'log', 'event', 'when', 'at', 'ts', 'received', 'sent'
        ]
        
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
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string in various formats."""
        if pd.isna(timestamp_str):
            return None
        
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
                return datetime.strptime(str(timestamp_str), fmt)
            except:
                continue
        
        # Try pandas automatic parsing
        try:
            return pd.to_datetime(timestamp_str)
        except:
            return None
    
    def analyze_timestamp_relationships(self) -> Dict:
        """Analyze relationships between timestamp columns."""
        print("\n[TIME] Analyzing timestamp relationships...")
        
        timestamp_cols = self.find_timestamp_columns()
        relationships = {
            'timestamp_columns': timestamp_cols,
            'time_delays': [],
            'potential_matches': []
        }
        
        # Analyze each pair of timestamp columns
        for source_col in timestamp_cols['source']:
            for derived_col in timestamp_cols['derived']:
                print(f"  Analyzing: {source_col} ↔ {derived_col}")
                
                # Parse timestamps
                source_times = []
                derived_times = []
                
                for val in self.df_source[source_col].dropna():
                    parsed = self.parse_timestamp(val)
                    if parsed:
                        source_times.append(parsed)
                
                for val in self.df_derived[derived_col].dropna():
                    parsed = self.parse_timestamp(val)
                    if parsed:
                        derived_times.append(parsed)
                
                if source_times and derived_times:
                    # Find time delays
                    delays = []
                    matches = []
                    
                    for i, source_time in enumerate(source_times):
                        for j, derived_time in enumerate(derived_times):
                            delay = abs((source_time - derived_time).total_seconds())
                            if delay <= 3600:  # Within 1 hour
                                delays.append(delay)
                                matches.append((i, j, delay))
                    
                    if delays:
                        analysis = {
                            'source_column': source_col,
                            'derived_column': derived_col,
                            'mean_delay': np.mean(delays),
                            'std_delay': np.std(delays),
                            'min_delay': np.min(delays),
                            'max_delay': np.max(delays),
                            'match_count': len(matches),
                            'confidence': min(len(matches) / min(len(source_times), len(derived_times)), 1.0)
                        }
                        
                        relationships['time_delays'].append(analysis)
                        relationships['potential_matches'].append({
                            'source_col': source_col,
                            'derived_col': derived_col,
                            'confidence': analysis['confidence'],
                            'mean_delay': analysis['mean_delay']
                        })
                        
                        print(f"    [OK] Found {len(matches)} matches, avg delay: {analysis['mean_delay']:.2f}s")
        
        return relationships
    
    def find_column_similarities(self) -> Dict:
        """Find similarities between columns across datasets."""
        print("\n[SEARCH] Finding column similarities...")
        
        similarities = {
            'exact_matches': [],
            'similar_columns': [],
            'value_overlaps': [],
            'new_columns': [],
            'dropped_columns': []
        }
        
        # Find exact matches
        common_columns = set(self.df_source.columns) & set(self.df_derived.columns)
        similarities['exact_matches'] = list(common_columns)
        
        # Find similar column names
        for source_col in self.df_source.columns:
            for derived_col in self.df_derived.columns:
                if source_col != derived_col:
                    similarity = self._calculate_name_similarity(source_col, derived_col)
                    if similarity > 0.6:  # High similarity threshold
                        similarities['similar_columns'].append({
                            'source': source_col,
                            'derived': derived_col,
                            'similarity': similarity
                        })
        
        # Find value overlaps
        for source_col in self.df_source.columns:
            for derived_col in self.df_derived.columns:
                if source_col != derived_col:
                    overlap = self._calculate_value_overlap(source_col, derived_col)
                    if overlap > 0.1:  # At least 10% overlap
                        similarities['value_overlaps'].append({
                            'source': source_col,
                            'derived': derived_col,
                            'overlap_ratio': overlap
                        })
        
        # Find new and dropped columns
        similarities['new_columns'] = list(set(self.df_derived.columns) - set(self.df_source.columns))
        similarities['dropped_columns'] = list(set(self.df_source.columns) - set(self.df_derived.columns))
        
        return similarities
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between column names."""
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def _calculate_value_overlap(self, col1: str, col2: str) -> float:
        """Calculate overlap between column values."""
        try:
            values1 = set(self.df_source[col1].dropna().astype(str))
            values2 = set(self.df_derived[col2].dropna().astype(str))
            
            if not values1 or not values2:
                return 0.0
            
            overlap = len(values1.intersection(values2))
            union = len(values1.union(values2))
            
            return overlap / union if union > 0 else 0.0
        except:
            return 0.0
    
    def detect_transformations(self) -> Dict:
        """Detect transformations between datasets."""
        print("\n[TRANSFORM] Detecting transformations...")
        
        transformations = {
            'filtering': self._detect_filtering(),
            'aggregation': self._detect_aggregation(),
            'value_transformations': self._detect_value_transformations(),
            'column_transformations': self._detect_column_transformations()
        }
        
        return transformations
    
    def _detect_filtering(self) -> Dict:
        """Detect if one dataset is a filtered version of the other."""
        filtering_info = {
            'row_count_ratio': len(self.df_derived) / len(self.df_source) if len(self.df_source) > 0 else 0,
            'filtering_detected': False,
            'filtered_columns': []
        }
        
        if filtering_info['row_count_ratio'] < 1.0:
            filtering_info['filtering_detected'] = True
            
            # Check for columns with reduced unique values
            for col in self.df_source.columns:
                if col in self.df_derived.columns:
                    unique_ratio = self.df_derived[col].nunique() / self.df_source[col].nunique()
                    if unique_ratio < 0.8:
                        filtering_info['filtered_columns'].append({
                            'column': col,
                            'unique_ratio': unique_ratio
                        })
        
        return filtering_info
    
    def _detect_aggregation(self) -> Dict:
        """Detect aggregation patterns."""
        aggregation_info = {
            'aggregated_columns': [],
            'grouping_indicators': []
        }
        
        # Check for numeric columns that might be aggregated
        numeric_source = self.df_source.select_dtypes(include=[np.number]).columns
        numeric_derived = self.df_derived.select_dtypes(include=[np.number]).columns
        
        for source_col in numeric_source:
            for derived_col in numeric_derived:
                if self._are_columns_related(source_col, derived_col):
                    # Check if derived values are different (aggregated)
                    if len(self.df_derived) < len(self.df_source):
                        mean1 = self.df_source[source_col].mean()
                        mean2 = self.df_derived[derived_col].mean()
                        
                        if abs(mean2 - mean1) > 0.01:
                            aggregation_info['aggregated_columns'].append({
                                'source_column': source_col,
                                'derived_column': derived_col,
                                'mean_difference': mean2 - mean1,
                                'aggregation_type': 'mean' if abs(mean2 - mean1) < abs(mean2) else 'sum'
                            })
        
        return aggregation_info
    
    def _detect_value_transformations(self) -> Dict:
        """Detect value transformations."""
        transformations = {
            'mathematical_transforms': [],
            'categorical_transforms': [],
            'encoding_changes': []
        }
        
        # Check for mathematical transformations
        numeric_source = self.df_source.select_dtypes(include=[np.number]).columns
        numeric_derived = self.df_derived.select_dtypes(include=[np.number]).columns
        
        for source_col in numeric_source:
            for derived_col in numeric_derived:
                if self._are_columns_related(source_col, derived_col):
                    transform = self._detect_mathematical_transform(source_col, derived_col)
                    if transform:
                        transformations['mathematical_transforms'].append(transform)
        
        # Check for categorical transformations
        categorical_source = self.df_source.select_dtypes(include=['object']).columns
        categorical_derived = self.df_derived.select_dtypes(include=['object']).columns
        
        for source_col in categorical_source:
            for derived_col in categorical_derived:
                if self._are_columns_related(source_col, derived_col):
                    transform = self._detect_categorical_transform(source_col, derived_col)
                    if transform:
                        transformations['categorical_transforms'].append(transform)
        
        return transformations
    
    def _detect_column_transformations(self) -> Dict:
        """Detect column-level transformations."""
        return {
            'renamed_columns': self._find_renamed_columns(),
            'split_columns': self._find_split_columns(),
            'merged_columns': self._find_merged_columns()
        }
    
    def _are_columns_related(self, col1: str, col2: str) -> bool:
        """Check if two columns are related."""
        name_similarity = self._calculate_name_similarity(col1, col2)
        value_overlap = self._calculate_value_overlap(col1, col2)
        
        return name_similarity > 0.5 or value_overlap > 0.1
    
    def _detect_mathematical_transform(self, source_col: str, derived_col: str) -> Optional[Dict]:
        """Detect mathematical transformations between columns."""
        try:
            values1 = self.df_source[source_col].dropna()
            values2 = self.df_derived[derived_col].dropna()
            
            if len(values1) == 0 or len(values2) == 0:
                return None
            
            # Check for scaling
            ratio = values2.mean() / values1.mean()
            if 0.1 < ratio < 10 and ratio != 1:
                return {
                    'type': 'scaling',
                    'source_column': source_col,
                    'derived_column': derived_col,
                    'ratio': ratio
                }
            
            # Check for offset
            diff = values2.mean() - values1.mean()
            if abs(diff) > 0.01:
                return {
                    'type': 'offset',
                    'source_column': source_col,
                    'derived_column': derived_col,
                    'difference': diff
                }
            
            return None
        except:
            return None
    
    def _detect_categorical_transform(self, source_col: str, derived_col: str) -> Optional[Dict]:
        """Detect categorical transformations between columns."""
        try:
            values1 = set(self.df_source[source_col].dropna().astype(str))
            values2 = set(self.df_derived[derived_col].dropna().astype(str))
            
            if not values1 or not values2:
                return None
            
            # Check for value mapping
            overlap = len(values1.intersection(values2))
            if overlap > 0:
                return {
                    'type': 'value_mapping',
                    'source_column': source_col,
                    'derived_column': derived_col,
                    'overlap_ratio': overlap / len(values1.union(values2))
                }
            
            return None
        except:
            return None
    
    def _find_renamed_columns(self) -> List[Dict]:
        """Find columns that might have been renamed."""
        renamed = []
        
        for source_col in self.df_source.columns:
            for derived_col in self.df_derived.columns:
                if source_col != derived_col:
                    similarity = self._calculate_name_similarity(source_col, derived_col)
                    overlap = self._calculate_value_overlap(source_col, derived_col)
                    
                    if similarity > 0.7 or overlap > 0.5:
                        renamed.append({
                            'source_column': source_col,
                            'derived_column': derived_col,
                            'name_similarity': similarity,
                            'value_overlap': overlap
                        })
        
        return renamed
    
    def _find_split_columns(self) -> List[Dict]:
        """Find columns that might have been split."""
        # This is a simplified implementation
        return []
    
    def _find_merged_columns(self) -> List[Dict]:
        """Find columns that might have been merged."""
        # This is a simplified implementation
        return []
    
    def suggest_joins(self) -> List[Dict]:
        """Suggest potential joins between the datasets."""
        print("\n[JOIN] Suggesting potential joins...")
        
        join_suggestions = []
        
        # Timestamp-based joins
        timestamp_analysis = self.analyze_timestamp_relationships()
        for match in timestamp_analysis['potential_matches']:
            join_suggestions.append({
                'type': 'timestamp_join',
                'source_column': match['source_col'],
                'derived_column': match['derived_col'],
                'confidence': match['confidence'],
                'details': {
                    'mean_delay': match['mean_delay'],
                    'join_condition': f"ABS({match['source_col']} - {match['derived_col']}) <= 3600"
                }
            })
        
        # Exact value matches
        similarities = self.find_column_similarities()
        for overlap in similarities['value_overlaps']:
            if overlap['overlap_ratio'] > 0.3:  # High overlap
                join_suggestions.append({
                    'type': 'exact_match',
                    'source_column': overlap['source'],
                    'derived_column': overlap['derived'],
                    'confidence': overlap['overlap_ratio'],
                    'details': {
                        'overlap_ratio': overlap['overlap_ratio'],
                        'join_condition': f"{overlap['source']} = {overlap['derived']}"
                    }
                })
        
        # Similar column names
        for similar in similarities['similar_columns']:
            if similar['similarity'] > 0.8:  # Very similar names
                join_suggestions.append({
                    'type': 'name_similarity',
                    'source_column': similar['source'],
                    'derived_column': similar['derived'],
                    'confidence': similar['similarity'],
                    'details': {
                        'name_similarity': similar['similarity'],
                        'join_condition': f"{similar['source']} = {similar['derived']}"
                    }
                })
        
        return sorted(join_suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n[DATA] Generating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create multiple visualization types
        self._create_basic_comparison_plots()
        self._create_timestamp_analysis_plots()
        self._create_correlation_plots()
        self._create_transformation_plots()
        
        print("[OK] All visualizations generated and saved")
    
    def _create_basic_comparison_plots(self):
        """Create basic comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Row and column counts
        axes[0, 0].bar(['Source', 'Derived'], [len(self.df_source), len(self.df_derived)])
        axes[0, 0].set_title('Row Counts')
        axes[0, 0].set_ylabel('Number of Rows')
        
        axes[0, 1].bar(['Source', 'Derived'], [len(self.df_source.columns), len(self.df_derived.columns)])
        axes[0, 1].set_title('Column Counts')
        axes[0, 1].set_ylabel('Number of Columns')
        
        # Data type distribution
        dtype_source = self.df_source.dtypes.value_counts()
        dtype_derived = self.df_derived.dtypes.value_counts()
        
        x = np.arange(len(dtype_source))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, dtype_source.values, width, label='Source')
        axes[1, 0].bar(x + width/2, dtype_derived.values, width, label='Derived')
        axes[1, 0].set_title('Data Type Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dtype_source.index, rotation=45)
        axes[1, 0].legend()
        
        # Missing data comparison
        missing_source = self.df_source.isnull().sum().sum()
        missing_derived = self.df_derived.isnull().sum().sum()
        
        axes[1, 1].bar(['Source', 'Derived'], [missing_source, missing_derived])
        axes[1, 1].set_title('Missing Data Comparison')
        axes[1, 1].set_ylabel('Number of Missing Values')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_timestamp_analysis_plots(self):
        """Create timestamp analysis plots."""
        timestamp_analysis = self.analyze_timestamp_relationships()
        time_delays = timestamp_analysis['time_delays']
        
        if time_delays:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Timestamp Analysis', fontsize=16, fontweight='bold')
            
            # Time delays
            delays = [delay['mean_delay'] for delay in time_delays]
            labels = [f"{delay['source_column']} -> {delay['derived_column']}" for delay in time_delays]
            
            axes[0].bar(range(len(delays)), delays)
            axes[0].set_xlabel('Column Pairs')
            axes[0].set_ylabel('Mean Delay (seconds)')
            axes[0].set_title('Processing Delays')
            axes[0].set_xticks(range(len(delays)))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            
            # Confidence scores
            confidences = [delay['confidence'] for delay in time_delays]
            axes[1].bar(range(len(confidences)), confidences)
            axes[1].set_xlabel('Column Pairs')
            axes[1].set_ylabel('Confidence Score')
            axes[1].set_title('Match Confidence')
            axes[1].set_xticks(range(len(confidences)))
            axes[1].set_xticklabels(labels, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('timestamp_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_correlation_plots(self):
        """Create correlation analysis plots."""
        # Find common numeric columns
        common_numeric = set(self.df_source.select_dtypes(include=[np.number]).columns) & \
                        set(self.df_derived.select_dtypes(include=[np.number]).columns)
        
        if len(common_numeric) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
            
            # Plot correlations for first few common columns
            for i, col in enumerate(list(common_numeric)[:4]):
                row = i // 2
                col_idx = i % 2
                
                try:
                    values1 = self.df_source[col].dropna()
                    values2 = self.df_derived[col].dropna()
                    
                    min_len = min(len(values1), len(values2))
                    if min_len > 0:
                        axes[row, col_idx].scatter(values1[:min_len], values2[:min_len], alpha=0.6)
                        axes[row, col_idx].set_xlabel(f'Source {col}')
                        axes[row, col_idx].set_ylabel(f'Derived {col}')
                        axes[row, col_idx].set_title(f'Correlation: {col}')
                        
                        # Add correlation line
                        correlation = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                        axes[row, col_idx].text(0.05, 0.95, f'r = {correlation:.3f}', 
                                              transform=axes[row, col_idx].transAxes, 
                                              bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                except:
                    axes[row, col_idx].text(0.5, 0.5, f'Error plotting {col}', 
                                          ha='center', va='center', transform=axes[row, col_idx].transAxes)
            
            plt.tight_layout()
            plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_transformation_plots(self):
        """Create transformation analysis plots."""
        transformations = self.detect_transformations()
        
        # Create transformation summary plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Transformation Analysis', fontsize=16, fontweight='bold')
        
        # Filtering analysis
        filtering = transformations['filtering']
        if filtering['filtering_detected']:
            axes[0].bar(['Source', 'Derived'], [1.0, filtering['row_count_ratio']])
            axes[0].set_title('Row Count Ratio (Filtering)')
            axes[0].set_ylabel('Ratio')
        
        # Value transformations
        value_transforms = transformations['value_transformations']
        transform_counts = {
            'Mathematical': len(value_transforms['mathematical_transforms']),
            'Categorical': len(value_transforms['categorical_transforms']),
            'Encoding': len(value_transforms['encoding_changes'])
        }
        
        axes[1].bar(transform_counts.keys(), transform_counts.values())
        axes[1].set_title('Transformation Types')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('transformation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def explore_field(self, field_name: str, max_samples: int = 50):
        """Explore a specific field in detail."""
        print(f"\n[SEARCH] Exploring field: {field_name}")
        
        if field_name in self.df_source.columns and field_name in self.df_derived.columns:
            print(f"Field found in both datasets")
            
            # Sample values for comparison
            source_sample = self.df_source[field_name].dropna().head(max_samples)
            derived_sample = self.df_derived[field_name].dropna().head(max_samples)
            
            print(f"\nSource dataset sample values:")
            print(source_sample.value_counts().head(10))
            
            print(f"\nDerived dataset sample values:")
            print(derived_sample.value_counts().head(10))
            
            # Statistical comparison for numeric fields
            if self.df_source[field_name].dtype in ['int64', 'float64'] and \
               self.df_derived[field_name].dtype in ['int64', 'float64']:
                
                print(f"\nStatistical comparison:")
                print(f"Source - Mean: {self.df_source[field_name].mean():.3f}, "
                      f"Std: {self.df_source[field_name].std():.3f}")
                print(f"Derived - Mean: {self.df_derived[field_name].mean():.3f}, "
                      f"Std: {self.df_derived[field_name].std():.3f}")
                
                # Calculate correlation
                min_len = min(len(source_sample), len(derived_sample))
                if min_len > 1:
                    correlation = np.corrcoef(source_sample[:min_len], derived_sample[:min_len])[0, 1]
                    print(f"Correlation: {correlation:.3f}")
        
        elif field_name in self.df_source.columns:
            print(f"Field only found in source dataset")
            print(f"Sample values: {self.df_source[field_name].dropna().head(10).tolist()}")
        
        elif field_name in self.df_derived.columns:
            print(f"Field only found in derived dataset")
            print(f"Sample values: {self.df_derived[field_name].dropna().head(10).tolist()}")
        
        else:
            print(f"Field '{field_name}' not found in either dataset")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        print("\n[REPORT] Generating comprehensive report...")
        
        report = []
        report.append("=" * 100)
        report.append("DATASET RELATIONSHIP ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("[SUMMARY] EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Source Dataset: {len(self.df_source)} rows, {len(self.df_source.columns)} columns")
        report.append(f"Derived Dataset: {len(self.df_derived)} rows, {len(self.df_derived.columns)} columns")
        report.append(f"Row Count Ratio: {len(self.df_derived) / len(self.df_source):.3f}")
        report.append("")
        
        # Timestamp Analysis
        timestamp_analysis = self.analyze_timestamp_relationships()
        if timestamp_analysis['time_delays']:
            report.append("[TIME] TIMESTAMP ANALYSIS")
            report.append("-" * 50)
            for delay in timestamp_analysis['time_delays']:
                report.append(f"{delay['source_column']} -> {delay['derived_column']}:")
                report.append(f"  - Mean delay: {delay['mean_delay']:.2f}s")
                report.append(f"  - Match count: {delay['match_count']}")
                report.append(f"  - Confidence: {delay['confidence']:.3f}")
            report.append("")
        
        # Column Similarities
        similarities = self.find_column_similarities()
        report.append("[SEARCH] COLUMN SIMILARITIES")
        report.append("-" * 50)
        report.append(f"Exact matches: {len(similarities['exact_matches'])}")
        report.append(f"Similar columns: {len(similarities['similar_columns'])}")
        report.append(f"Value overlaps: {len(similarities['value_overlaps'])}")
        report.append(f"New columns: {len(similarities['new_columns'])}")
        report.append(f"Dropped columns: {len(similarities['dropped_columns'])}")
        report.append("")
        
        # Transformations
        transformations = self.detect_transformations()
        report.append("[TRANSFORM] DETECTED TRANSFORMATIONS")
        report.append("-" * 50)
        
        if transformations['filtering']['filtering_detected']:
            report.append(f"[OK] FILTERING DETECTED: {transformations['filtering']['row_count_ratio']:.1%} of source rows retained")
        
        if transformations['aggregation']['aggregated_columns']:
            report.append(f"[OK] AGGREGATION DETECTED: {len(transformations['aggregation']['aggregated_columns'])} columns aggregated")
        
        if transformations['value_transformations']['mathematical_transforms']:
            report.append(f"[OK] MATHEMATICAL TRANSFORMATIONS: {len(transformations['value_transformations']['mathematical_transforms'])} detected")
        
        if transformations['value_transformations']['categorical_transforms']:
            report.append(f"[OK] CATEGORICAL TRANSFORMATIONS: {len(transformations['value_transformations']['categorical_transforms'])} detected")
        
        report.append("")
        
        # Join Suggestions
        join_suggestions = self.suggest_joins()
        if join_suggestions:
            report.append("[JOIN] JOIN SUGGESTIONS")
            report.append("-" * 50)
            for i, suggestion in enumerate(join_suggestions[:5], 1):
                report.append(f"{i}. {suggestion['type'].upper()}:")
                report.append(f"   Source: {suggestion['source_column']}")
                report.append(f"   Derived: {suggestion['derived_column']}")
                report.append(f"   Confidence: {suggestion['confidence']:.3f}")
                if 'details' in suggestion:
                    report.append(f"   Details: {suggestion['details']}")
                report.append("")
        
        report.append("=" * 100)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open('dataset_relationship_report.txt', 'w') as f:
            f.write(report_text)
        
        print("[OK] Report saved as 'dataset_relationship_report.txt'")
        return report_text
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline."""
        print("[START] Starting comprehensive dataset relationship analysis...")
        print("=" * 100)
        
        # Run all analyses
        timestamp_analysis = self.analyze_timestamp_relationships()
        similarities = self.find_column_similarities()
        transformations = self.detect_transformations()
        join_suggestions = self.suggest_joins()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        report = self.generate_report()
        
        # Store results
        self.analysis_results = {
            'timestamp_analysis': timestamp_analysis,
            'similarities': similarities,
            'transformations': transformations,
            'join_suggestions': join_suggestions,
            'report': report
        }
        
        print("\n" + "=" * 100)
        print("[OK] Analysis complete! Check the generated files:")
        print("  - dataset_relationship_report.txt (detailed report)")
        print("  - dataset_comparison.png (basic comparisons)")
        print("  - timestamp_analysis.png (temporal analysis)")
        print("  - correlation_analysis.png (correlation plots)")
        print("  - transformation_analysis.png (transformation analysis)")
        print("=" * 100)
        
        return self.analysis_results
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze relationships between SQLite datasets')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--source', help='Name of the source table')
    parser.add_argument('--derived', help='Name of the derived table')
    parser.add_argument('--sample-size', type=int, help='Sample size for large datasets')
    parser.add_argument('--explore-field', help='Explore a specific field in detail')
    
    args = parser.parse_args()
    
    # Create detector and run analysis
    detector = DatasetRelationshipDetector(args.db_path)
    
    try:
        # Connect to database
        if not detector.connect():
            return
        
        # Load datasets
        detector.load_datasets(args.source, args.derived, args.sample_size)
        
        # Run comprehensive analysis
        results = detector.run_comprehensive_analysis()
        
        # Explore specific field if requested
        if args.explore_field:
            detector.explore_field(args.explore_field)
        
        # Print summary
        if results:
            print("\n[DATA] SUMMARY:")
            print(f"  - Timestamp relationships: {len(results['timestamp_analysis']['time_delays'])}")
            print(f"  - Column similarities: {len(results['similarities']['similar_columns'])}")
            print(f"  - Join suggestions: {len(results['join_suggestions'])}")
            
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.close()


if __name__ == "__main__":
    main() 