#!/usr/bin/env python3
"""
Advanced Dataset Relationship Analyzer
=====================================

A comprehensive tool to analyze how one dataset was derived from another,
using visualizations, comparisons, correlation analysis, and heuristics.

This script provides deep insights into data transformations, filtering,
aggregation patterns, and temporal relationships.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from collections import defaultdict, Counter
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AdvancedDatasetAnalyzer:
    """
    Advanced analyzer for uncovering how one dataset was derived from another.
    """
    
    def __init__(self, db_path: str, config: Dict = None):
        """
        Initialize the analyzer with a SQLite database path and optional config.
        
        Args:
            db_path: Path to the SQLite database file
            config: Optional configuration dictionary
        """
        self.db_path = db_path
        self.conn = None
        self.df_source = None
        self.df_derived = None
        self.config = config or self._default_config()
        self.analysis_results = {}
        self.visualizations = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for the analyzer."""
        return {
            'timestamp_patterns': ['time', 'date', 'timestamp', 'created', 'updated', 'message', 'log', 'event'],
            'correlation_threshold': 0.7,
            'similarity_threshold': 0.5,
            'time_window_seconds': 3600,
            'max_sample_size': 10000,
            'pca_components': 3,
            'plot_style': 'seaborn-v0_8',
            'output_dir': 'analysis_output'
        }
    
    def connect(self) -> bool:
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def load_datasets(self, source_table: str, derived_table: str, sample_size: int = None):
        """Load source and derived datasets."""
        print(f"\n📊 Loading datasets: {source_table} → {derived_table}")
        
        # Load with optional sampling
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
        
        print(f"✅ Loaded source: {len(self.df_source)} rows, {len(self.df_source.columns)} columns")
        print(f"✅ Loaded derived: {len(self.df_derived)} rows, {len(self.df_derived.columns)} columns")
        
        # Store table names
        self.source_table = source_table
        self.derived_table = derived_table
    
    def comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis pipeline."""
        print("\n🚀 Starting comprehensive dataset analysis...")
        print("=" * 80)
        
        results = {}
        
        # 1. Basic dataset comparison
        results['basic_comparison'] = self._analyze_basic_comparison()
        
        # 2. Column relationship analysis
        results['column_relationships'] = self._analyze_column_relationships()
        
        # 3. Temporal analysis
        results['temporal_analysis'] = self._analyze_temporal_relationships()
        
        # 4. Value transformation analysis
        results['value_transformations'] = self._analyze_value_transformations()
        
        # 5. Statistical correlation analysis
        results['correlation_analysis'] = self._analyze_correlations()
        
        # 6. Pattern detection
        results['pattern_detection'] = self._detect_patterns()
        
        # 7. Heuristic analysis
        results['heuristic_analysis'] = self._apply_heuristics()
        
        # 8. Generate visualizations
        self._generate_comprehensive_visualizations()
        
        self.analysis_results = results
        return results
    
    def _analyze_basic_comparison(self) -> Dict:
        """Analyze basic differences between datasets."""
        print("📋 Analyzing basic dataset comparison...")
        
        comparison = {
            'row_counts': {
                'source': len(self.df_source),
                'derived': len(self.df_derived),
                'ratio': len(self.df_derived) / len(self.df_source) if len(self.df_source) > 0 else 0
            },
            'column_counts': {
                'source': len(self.df_source.columns),
                'derived': len(self.df_derived.columns),
                'ratio': len(self.df_derived.columns) / len(self.df_source.columns) if len(self.df_source.columns) > 0 else 0
            },
            'data_types': {
                'source': self.df_source.dtypes.value_counts().to_dict(),
                'derived': self.df_derived.dtypes.value_counts().to_dict()
            },
            'missing_data': {
                'source': self.df_source.isnull().sum().sum(),
                'derived': self.df_derived.isnull().sum().sum()
            },
            'memory_usage': {
                'source': self.df_source.memory_usage(deep=True).sum(),
                'derived': self.df_derived.memory_usage(deep=True).sum()
            }
        }
        
        # Detect filtering patterns
        if comparison['row_counts']['ratio'] < 1.0:
            comparison['filtering_detected'] = True
            comparison['filtering_ratio'] = comparison['row_counts']['ratio']
        else:
            comparison['filtering_detected'] = False
        
        return comparison
    
    def _analyze_column_relationships(self) -> Dict:
        """Analyze relationships between columns across datasets."""
        print("🔗 Analyzing column relationships...")
        
        relationships = {
            'exact_matches': [],
            'similar_columns': [],
            'transformed_columns': [],
            'new_columns': [],
            'dropped_columns': []
        }
        
        # Find exact column matches
        common_columns = set(self.df_source.columns) & set(self.df_derived.columns)
        relationships['exact_matches'] = list(common_columns)
        
        # Find similar columns (name similarity)
        for col1 in self.df_source.columns:
            for col2 in self.df_derived.columns:
                if col1 != col2:
                    similarity = self._calculate_name_similarity(col1, col2)
                    if similarity > self.config['similarity_threshold']:
                        relationships['similar_columns'].append({
                            'source': col1,
                            'derived': col2,
                            'similarity': similarity
                        })
        
        # Find new columns in derived dataset
        relationships['new_columns'] = list(set(self.df_derived.columns) - set(self.df_source.columns))
        
        # Find dropped columns from source dataset
        relationships['dropped_columns'] = list(set(self.df_source.columns) - set(self.df_derived.columns))
        
        return relationships
    
    def _analyze_temporal_relationships(self) -> Dict:
        """Analyze temporal relationships between datasets."""
        print("🕒 Analyzing temporal relationships...")
        
        temporal = {
            'timestamp_columns': [],
            'time_delays': [],
            'temporal_patterns': []
        }
        
        # Find timestamp columns
        timestamp_cols_source = self._find_timestamp_columns(self.df_source)
        timestamp_cols_derived = self._find_timestamp_columns(self.df_derived)
        
        temporal['timestamp_columns'] = {
            'source': timestamp_cols_source,
            'derived': timestamp_cols_derived
        }
        
        # Analyze time delays between corresponding timestamps
        for ts1 in timestamp_cols_source:
            for ts2 in timestamp_cols_derived:
                if self._are_columns_related(ts1, ts2):
                    delays = self._calculate_time_delays(ts1, ts2)
                    if delays:
                        temporal['time_delays'].append({
                            'source_column': ts1,
                            'derived_column': ts2,
                            'mean_delay': np.mean(delays),
                            'std_delay': np.std(delays),
                            'min_delay': np.min(delays),
                            'max_delay': np.max(delays),
                            'delay_count': len(delays)
                        })
        
        return temporal
    
    def _analyze_value_transformations(self) -> Dict:
        """Analyze how values are transformed between datasets."""
        print("🔄 Analyzing value transformations...")
        
        transformations = {
            'mathematical_transforms': [],
            'categorical_transforms': [],
            'aggregation_patterns': [],
            'scaling_factors': []
        }
        
        # Analyze numeric columns for mathematical transformations
        numeric_source = self.df_source.select_dtypes(include=[np.number]).columns
        numeric_derived = self.df_derived.select_dtypes(include=[np.number]).columns
        
        for col1 in numeric_source:
            for col2 in numeric_derived:
                if self._are_columns_related(col1, col2):
                    transform = self._detect_mathematical_transform(col1, col2)
                    if transform:
                        transformations['mathematical_transforms'].append(transform)
        
        # Analyze categorical transformations
        categorical_source = self.df_source.select_dtypes(include=['object']).columns
        categorical_derived = self.df_derived.select_dtypes(include=['object']).columns
        
        for col1 in categorical_source:
            for col2 in categorical_derived:
                if self._are_columns_related(col1, col2):
                    transform = self._detect_categorical_transform(col1, col2)
                    if transform:
                        transformations['categorical_transforms'].append(transform)
        
        return transformations
    
    def _analyze_correlations(self) -> Dict:
        """Analyze statistical correlations between datasets."""
        print("📊 Analyzing statistical correlations...")
        
        correlations = {
            'column_correlations': [],
            'value_distributions': [],
            'statistical_tests': []
        }
        
        # Find common numeric columns
        common_numeric = set(self.df_source.select_dtypes(include=[np.number]).columns) & \
                        set(self.df_derived.select_dtypes(include=[np.number]).columns)
        
        for col in common_numeric:
            if col in self.df_source.columns and col in self.df_derived.columns:
                # Calculate correlation if we can align the data
                correlation = self._calculate_column_correlation(col)
                if correlation is not None:
                    correlations['column_correlations'].append({
                        'column': col,
                        'correlation': correlation
                    })
                
                # Compare value distributions
                distribution_comparison = self._compare_value_distributions(col)
                correlations['value_distributions'].append(distribution_comparison)
        
        return correlations
    
    def _detect_patterns(self) -> Dict:
        """Detect patterns in the data transformation."""
        print("🔍 Detecting transformation patterns...")
        
        patterns = {
            'filtering_patterns': self._detect_filtering_patterns(),
            'aggregation_patterns': self._detect_aggregation_patterns(),
            'transformation_patterns': self._detect_transformation_patterns(),
            'temporal_patterns': self._detect_temporal_patterns()
        }
        
        return patterns
    
    def _apply_heuristics(self) -> Dict:
        """Apply heuristics to understand the derivation process."""
        print("🧠 Applying heuristics...")
        
        heuristics = {
            'derivation_hypothesis': self._generate_derivation_hypothesis(),
            'confidence_scores': self._calculate_confidence_scores(),
            'recommendations': self._generate_recommendations()
        }
        
        return heuristics
    
    def _find_timestamp_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that might contain timestamps."""
        timestamp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in self.config['timestamp_patterns']):
                timestamp_cols.append(col)
        return timestamp_cols
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
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
    
    def _are_columns_related(self, col1: str, col2: str) -> bool:
        """Check if two columns are related."""
        return self._calculate_name_similarity(col1, col2) > self.config['similarity_threshold']
    
    def _calculate_time_delays(self, col1: str, col2: str) -> List[float]:
        """Calculate time delays between timestamp columns."""
        try:
            ts1 = pd.to_datetime(self.df_source[col1], errors='coerce')
            ts2 = pd.to_datetime(self.df_derived[col2], errors='coerce')
            
            valid_ts1 = ts1.dropna()
            valid_ts2 = ts2.dropna()
            
            if len(valid_ts1) == 0 or len(valid_ts2) == 0:
                return []
            
            delays = []
            for t1 in valid_ts1:
                for t2 in valid_ts2:
                    diff = abs((t1 - t2).total_seconds())
                    if diff <= self.config['time_window_seconds']:
                        delays.append(diff)
            
            return delays
        except:
            return []
    
    def _detect_mathematical_transform(self, col1: str, col2: str) -> Optional[Dict]:
        """Detect mathematical transformations between columns."""
        try:
            values1 = self.df_source[col1].dropna()
            values2 = self.df_derived[col2].dropna()
            
            if len(values1) == 0 or len(values2) == 0:
                return None
            
            # Check for scaling
            ratio = values2.mean() / values1.mean()
            if 0.1 < ratio < 10 and ratio != 1:
                return {
                    'type': 'scaling',
                    'source_column': col1,
                    'derived_column': col2,
                    'ratio': ratio
                }
            
            # Check for offset
            diff = values2.mean() - values1.mean()
            if abs(diff) > 0.01:
                return {
                    'type': 'offset',
                    'source_column': col1,
                    'derived_column': col2,
                    'difference': diff
                }
            
            return None
        except:
            return None
    
    def _detect_categorical_transform(self, col1: str, col2: str) -> Optional[Dict]:
        """Detect categorical transformations between columns."""
        try:
            values1 = set(self.df_source[col1].dropna().astype(str))
            values2 = set(self.df_derived[col2].dropna().astype(str))
            
            if not values1 or not values2:
                return None
            
            # Check for value mapping
            overlap = len(values1.intersection(values2))
            if overlap > 0:
                return {
                    'type': 'value_mapping',
                    'source_column': col1,
                    'derived_column': col2,
                    'overlap_ratio': overlap / len(values1.union(values2))
                }
            
            return None
        except:
            return None
    
    def _calculate_column_correlation(self, col: str) -> Optional[float]:
        """Calculate correlation between same column in both datasets."""
        try:
            values1 = self.df_source[col].dropna()
            values2 = self.df_derived[col].dropna()
            
            if len(values1) < 2 or len(values2) < 2:
                return None
            
            # Use the smaller length for comparison
            min_len = min(len(values1), len(values2))
            correlation = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
            
            return correlation if not np.isnan(correlation) else None
        except:
            return None
    
    def _compare_value_distributions(self, col: str) -> Dict:
        """Compare value distributions between datasets."""
        try:
            values1 = self.df_source[col].dropna()
            values2 = self.df_derived[col].dropna()
            
            return {
                'column': col,
                'source_stats': {
                    'mean': values1.mean(),
                    'std': values1.std(),
                    'min': values1.min(),
                    'max': values1.max(),
                    'count': len(values1)
                },
                'derived_stats': {
                    'mean': values2.mean(),
                    'std': values2.std(),
                    'min': values2.min(),
                    'max': values2.max(),
                    'count': len(values2)
                }
            }
        except:
            return {'column': col, 'error': 'Could not compare distributions'}
    
    def _detect_filtering_patterns(self) -> Dict:
        """Detect filtering patterns between datasets."""
        patterns = {
            'row_count_ratio': len(self.df_derived) / len(self.df_source) if len(self.df_source) > 0 else 0,
            'filtered_columns': []
        }
        
        # Check for columns with reduced unique values
        for col in self.df_source.columns:
            if col in self.df_derived.columns:
                unique_ratio = self.df_derived[col].nunique() / self.df_source[col].nunique()
                if unique_ratio < 0.8:  # Significant reduction in unique values
                    patterns['filtered_columns'].append({
                        'column': col,
                        'unique_ratio': unique_ratio
                    })
        
        return patterns
    
    def _detect_aggregation_patterns(self) -> Dict:
        """Detect aggregation patterns."""
        patterns = {
            'aggregated_columns': [],
            'grouping_indicators': []
        }
        
        # Check for numeric columns that might be aggregated
        numeric_source = self.df_source.select_dtypes(include=[np.number]).columns
        numeric_derived = self.df_derived.select_dtypes(include=[np.number]).columns
        
        for col1 in numeric_source:
            for col2 in numeric_derived:
                if self._are_columns_related(col1, col2):
                    # Check if derived values are larger (sum) or different (avg)
                    if len(self.df_derived) < len(self.df_source):
                        mean1 = self.df_source[col1].mean()
                        mean2 = self.df_derived[col2].mean()
                        
                        if abs(mean2 - mean1) > 0.01:
                            patterns['aggregated_columns'].append({
                                'source_column': col1,
                                'derived_column': col2,
                                'mean_difference': mean2 - mean1
                            })
        
        return patterns
    
    def _detect_transformation_patterns(self) -> Dict:
        """Detect transformation patterns."""
        return {
            'mathematical_transforms': len(self.analysis_results.get('value_transformations', {}).get('mathematical_transforms', [])),
            'categorical_transforms': len(self.analysis_results.get('value_transformations', {}).get('categorical_transforms', [])),
            'new_columns': len(self.analysis_results.get('column_relationships', {}).get('new_columns', []))
        }
    
    def _detect_temporal_patterns(self) -> Dict:
        """Detect temporal patterns."""
        temporal_analysis = self.analysis_results.get('temporal_analysis', {})
        return {
            'timestamp_columns_found': len(temporal_analysis.get('timestamp_columns', {}).get('source', [])) + 
                                     len(temporal_analysis.get('timestamp_columns', {}).get('derived', [])),
            'time_delays_detected': len(temporal_analysis.get('time_delays', [])),
            'processing_delays': [delay['mean_delay'] for delay in temporal_analysis.get('time_delays', [])]
        }
    
    def _generate_derivation_hypothesis(self) -> str:
        """Generate a hypothesis about how the derived dataset was created."""
        basic = self.analysis_results.get('basic_comparison', {})
        patterns = self.analysis_results.get('pattern_detection', {})
        
        hypothesis_parts = []
        
        # Row count analysis
        if basic.get('row_counts', {}).get('ratio', 1) < 1.0:
            hypothesis_parts.append("FILTERING: The derived dataset appears to be a filtered subset of the source dataset.")
        
        # Column analysis
        if patterns.get('aggregation_patterns', {}).get('aggregated_columns'):
            hypothesis_parts.append("AGGREGATION: Some numeric columns appear to be aggregated or summarized.")
        
        # Transformation analysis
        transforms = self.analysis_results.get('value_transformations', {})
        if transforms.get('mathematical_transforms'):
            hypothesis_parts.append("MATHEMATICAL TRANSFORMATIONS: Some columns show mathematical transformations (scaling, offsets).")
        
        if transforms.get('categorical_transforms'):
            hypothesis_parts.append("CATEGORICAL TRANSFORMATIONS: Some categorical columns show value mappings or transformations.")
        
        # Temporal analysis
        temporal = self.analysis_results.get('temporal_analysis', {})
        if temporal.get('time_delays'):
            hypothesis_parts.append("TEMPORAL PROCESSING: There are consistent time delays, suggesting processing or transformation time.")
        
        if not hypothesis_parts:
            hypothesis_parts.append("The relationship between datasets is unclear. More analysis needed.")
        
        return " ".join(hypothesis_parts)
    
    def _calculate_confidence_scores(self) -> Dict:
        """Calculate confidence scores for different aspects of the analysis."""
        scores = {
            'filtering_confidence': 0.0,
            'transformation_confidence': 0.0,
            'temporal_confidence': 0.0,
            'overall_confidence': 0.0
        }
        
        # Filtering confidence
        basic = self.analysis_results.get('basic_comparison', {})
        if basic.get('filtering_detected'):
            ratio = basic.get('row_counts', {}).get('ratio', 1)
            scores['filtering_confidence'] = 1.0 - ratio  # Higher confidence for more filtering
        
        # Transformation confidence
        transforms = self.analysis_results.get('value_transformations', {})
        transform_count = len(transforms.get('mathematical_transforms', [])) + \
                        len(transforms.get('categorical_transforms', []))
        scores['transformation_confidence'] = min(transform_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Temporal confidence
        temporal = self.analysis_results.get('temporal_analysis', {})
        if temporal.get('time_delays'):
            scores['temporal_confidence'] = min(len(temporal['time_delays']) / 3.0, 1.0)
        
        # Overall confidence
        scores['overall_confidence'] = np.mean([
            scores['filtering_confidence'],
            scores['transformation_confidence'],
            scores['temporal_confidence']
        ])
        
        return scores
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for further analysis."""
        recommendations = []
        
        # Based on confidence scores
        confidence = self.analysis_results.get('heuristic_analysis', {}).get('confidence_scores', {})
        
        if confidence.get('filtering_confidence', 0) < 0.5:
            recommendations.append("Investigate potential filtering criteria by examining value distributions in categorical columns.")
        
        if confidence.get('transformation_confidence', 0) < 0.5:
            recommendations.append("Look for mathematical relationships between numeric columns that might indicate transformations.")
        
        if confidence.get('temporal_confidence', 0) < 0.5:
            recommendations.append("Examine timestamp columns more closely to understand temporal processing patterns.")
        
        # General recommendations
        recommendations.extend([
            "Compare value distributions for key columns to understand transformation patterns.",
            "Look for business logic that might explain the filtering or transformation criteria.",
            "Examine the data lineage or ETL processes if available.",
            "Consider sampling specific rows to trace individual record transformations."
        ])
        
        return recommendations
    
    def _generate_comprehensive_visualizations(self):
        """Generate comprehensive visualizations for the analysis."""
        print("\n📊 Generating comprehensive visualizations...")
        
        # Set up plotting style
        plt.style.use(self.config['plot_style'])
        
        # Create multiple visualization types
        self._create_basic_comparison_plots()
        self._create_correlation_plots()
        self._create_transformation_plots()
        self._create_temporal_plots()
        self._create_interactive_plots()
        
        print("✅ All visualizations generated and saved")
    
    def _create_basic_comparison_plots(self):
        """Create basic comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic Dataset Comparison', fontsize=16, fontweight='bold')
        
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
        plt.savefig('basic_comparison.png', dpi=300, bbox_inches='tight')
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
        # This would show how values are transformed between datasets
        # Implementation depends on the specific transformations found
        pass
    
    def _create_temporal_plots(self):
        """Create temporal analysis plots."""
        temporal = self.analysis_results.get('temporal_analysis', {})
        time_delays = temporal.get('time_delays', [])
        
        if time_delays:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            delays = [delay['mean_delay'] for delay in time_delays]
            labels = [f"{delay['source_column']} → {delay['derived_column']}" for delay in time_delays]
            
            ax.bar(range(len(delays)), delays)
            ax.set_xlabel('Column Pairs')
            ax.set_ylabel('Mean Delay (seconds)')
            ax.set_title('Temporal Processing Delays')
            ax.set_xticks(range(len(delays)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_interactive_plots(self):
        """Create interactive plots using Plotly."""
        try:
            # Create an interactive dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Row Counts', 'Column Counts', 'Data Types', 'Missing Data'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Row counts
            fig.add_trace(
                go.Bar(x=['Source', 'Derived'], y=[len(self.df_source), len(self.df_derived)],
                      name='Row Counts', marker_color=['blue', 'orange']),
                row=1, col=1
            )
            
            # Column counts
            fig.add_trace(
                go.Bar(x=['Source', 'Derived'], y=[len(self.df_source.columns), len(self.df_derived.columns)],
                      name='Column Counts', marker_color=['green', 'red']),
                row=1, col=2
            )
            
            fig.update_layout(height=800, title_text="Interactive Dataset Comparison")
            fig.write_html('interactive_comparison.html')
            
        except Exception as e:
            print(f"Warning: Could not create interactive plots: {e}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        print("\n📝 Generating comprehensive report...")
        
        report = []
        report.append("=" * 100)
        report.append("ADVANCED DATASET RELATIONSHIP ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        basic = self.analysis_results.get('basic_comparison', {})
        report.append(f"Source Dataset: {basic.get('row_counts', {}).get('source', 0)} rows, "
                     f"{basic.get('column_counts', {}).get('source', 0)} columns")
        report.append(f"Derived Dataset: {basic.get('row_counts', {}).get('derived', 0)} rows, "
                     f"{basic.get('column_counts', {}).get('derived', 0)} columns")
        report.append(f"Row Count Ratio: {basic.get('row_counts', {}).get('ratio', 0):.3f}")
        report.append("")
        
        # Key Findings
        report.append("🔍 KEY FINDINGS")
        report.append("-" * 50)
        
        # Filtering analysis
        if basic.get('filtering_detected'):
            report.append(f"✅ FILTERING DETECTED: {basic.get('row_counts', {}).get('ratio', 0):.1%} of source rows retained")
        
        # Transformation analysis
        transforms = self.analysis_results.get('value_transformations', {})
        if transforms.get('mathematical_transforms'):
            report.append(f"✅ MATHEMATICAL TRANSFORMATIONS: {len(transforms['mathematical_transforms'])} detected")
        
        if transforms.get('categorical_transforms'):
            report.append(f"✅ CATEGORICAL TRANSFORMATIONS: {len(transforms['categorical_transforms'])} detected")
        
        # Temporal analysis
        temporal = self.analysis_results.get('temporal_analysis', {})
        if temporal.get('time_delays'):
            report.append(f"✅ TEMPORAL PROCESSING: {len(temporal['time_delays'])} timestamp relationships found")
        
        report.append("")
        
        # Detailed Analysis
        report.append("📊 DETAILED ANALYSIS")
        report.append("-" * 50)
        
        # Column relationships
        relationships = self.analysis_results.get('column_relationships', {})
        report.append("Column Relationships:")
        report.append(f"  - Exact matches: {len(relationships.get('exact_matches', []))}")
        report.append(f"  - Similar columns: {len(relationships.get('similar_columns', []))}")
        report.append(f"  - New columns: {len(relationships.get('new_columns', []))}")
        report.append(f"  - Dropped columns: {len(relationships.get('dropped_columns', []))}")
        report.append("")
        
        # Heuristic analysis
        heuristics = self.analysis_results.get('heuristic_analysis', {})
        report.append("Derivation Hypothesis:")
        report.append(heuristics.get('derivation_hypothesis', 'No hypothesis generated'))
        report.append("")
        
        # Confidence scores
        confidence = heuristics.get('confidence_scores', {})
        report.append("Confidence Scores:")
        report.append(f"  - Filtering: {confidence.get('filtering_confidence', 0):.3f}")
        report.append(f"  - Transformations: {confidence.get('transformation_confidence', 0):.3f}")
        report.append(f"  - Temporal: {confidence.get('temporal_confidence', 0):.3f}")
        report.append(f"  - Overall: {confidence.get('overall_confidence', 0):.3f}")
        report.append("")
        
        # Recommendations
        recommendations = heuristics.get('recommendations', [])
        if recommendations:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("=" * 100)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open('advanced_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Advanced analysis report saved as 'advanced_analysis_report.txt'")
        return report_text
    
    def explore_specific_field(self, field_name: str, max_samples: int = 100):
        """Explore a specific field in detail."""
        print(f"\n🔍 Exploring field: {field_name}")
        
        if field_name in self.df_source.columns and field_name in self.df_derived.columns:
            print(f"Field found in both datasets")
            
            # Sample values for comparison
            source_sample = self.df_source[field_name].dropna().head(max_samples)
            derived_sample = self.df_derived[field_name].dropna().head(max_samples)
            
            print(f"\nSource dataset sample values:")
            print(source_sample.value_counts().head(10))
            
            print(f"\nDerived dataset sample values:")
            print(derived_sample.value_counts().head(10))
            
            # Statistical comparison
            if self.df_source[field_name].dtype in ['int64', 'float64'] and \
               self.df_derived[field_name].dtype in ['int64', 'float64']:
                
                print(f"\nStatistical comparison:")
                print(f"Source - Mean: {self.df_source[field_name].mean():.3f}, "
                      f"Std: {self.df_source[field_name].std():.3f}")
                print(f"Derived - Mean: {self.df_derived[field_name].mean():.3f}, "
                      f"Std: {self.df_derived[field_name].std():.3f}")
        
        elif field_name in self.df_source.columns:
            print(f"Field only found in source dataset")
            print(f"Sample values: {self.df_source[field_name].dropna().head(10).tolist()}")
        
        elif field_name in self.df_derived.columns:
            print(f"Field only found in derived dataset")
            print(f"Sample values: {self.df_derived[field_name].dropna().head(10).tolist()}")
        
        else:
            print(f"Field '{field_name}' not found in either dataset")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")


def main():
    """Main function to run the advanced analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced dataset relationship analysis')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--source', help='Name of the source table')
    parser.add_argument('--derived', help='Name of the derived table')
    parser.add_argument('--sample-size', type=int, help='Sample size for large datasets')
    parser.add_argument('--explore-field', help='Explore a specific field in detail')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AdvancedDatasetAnalyzer(args.db_path)
    
    try:
        # Connect to database
        if not analyzer.connect():
            return
        
        # Load datasets
        analyzer.load_datasets(args.source, args.derived, args.sample_size)
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis()
        
        # Generate report
        report = analyzer.generate_report()
        
        # Explore specific field if requested
        if args.explore_field:
            analyzer.explore_specific_field(args.explore_field)
        
        print("\n" + "=" * 80)
        print("✅ Advanced analysis complete! Check the generated files:")
        print("  - advanced_analysis_report.txt (detailed report)")
        print("  - basic_comparison.png (basic visualizations)")
        print("  - correlation_analysis.png (correlation plots)")
        print("  - temporal_analysis.png (temporal analysis)")
        print("  - interactive_comparison.html (interactive plots)")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 