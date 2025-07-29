#!/usr/bin/env python3
"""
Data Quality Assessor
=====================

A comprehensive tool for assessing data quality and providing
detailed recommendations for improvement.

Features:
- Data completeness analysis
- Data accuracy assessment
- Data consistency evaluation
- Data validity checks
- Data timeliness analysis
- Data uniqueness assessment
- Quality scoring and grading
- Detailed recommendations
- Quality improvement suggestions
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

class DataQualityAssessor:
    """
    Comprehensive data quality assessment tool.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the data quality assessor.
        
        Args:
            data_source: Path to database file or CSV file
        """
        self.data_source = data_source
        self.df = None
        self.quality_results = {}
        
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
    
    def assess_completeness(self) -> Dict:
        """Assess data completeness."""
        print(f"\n[QUALITY] Assessing data completeness...")
        
        completeness = {
            'overall_completeness': 0,
            'column_completeness': {},
            'row_completeness': {},
            'missing_patterns': {},
            'completeness_score': 0
        }
        
        # Calculate overall completeness
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        overall_completeness = ((total_cells - missing_cells) / total_cells) * 100
        completeness['overall_completeness'] = overall_completeness
        
        # Column-wise completeness
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            completeness['column_completeness'][col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'completeness_percentage': 100 - missing_percentage
            }
        
        # Row-wise completeness
        row_missing_counts = self.df.isnull().sum(axis=1)
        completeness['row_completeness'] = {
            'rows_with_missing': (row_missing_counts > 0).sum(),
            'rows_without_missing': (row_missing_counts == 0).sum(),
            'average_missing_per_row': row_missing_counts.mean(),
            'max_missing_per_row': row_missing_counts.max()
        }
        
        # Missing data patterns
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
            
            completeness['missing_patterns']['high_correlation_pairs'] = high_corr_pairs
        
        # Calculate completeness score (0-100)
        completeness_score = overall_completeness
        if completeness_score >= 95:
            completeness['completeness_score'] = completeness_score
            completeness['completeness_grade'] = 'A'
        elif completeness_score >= 85:
            completeness['completeness_score'] = completeness_score
            completeness['completeness_grade'] = 'B'
        elif completeness_score >= 70:
            completeness['completeness_score'] = completeness_score
            completeness['completeness_grade'] = 'C'
        else:
            completeness['completeness_score'] = completeness_score
            completeness['completeness_grade'] = 'D'
        
        self.quality_results['completeness'] = completeness
        return completeness
    
    def assess_accuracy(self) -> Dict:
        """Assess data accuracy."""
        print(f"\n[QUALITY] Assessing data accuracy...")
        
        accuracy = {
            'outlier_analysis': {},
            'data_type_consistency': {},
            'value_range_analysis': {},
            'accuracy_score': 0
        }
        
        # Outlier analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        total_numeric_values = 0
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                # IQR method for outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                
                outlier_count = len(outliers)
                total_outliers += outlier_count
                total_numeric_values += len(data)
                
                accuracy['outlier_analysis'][col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': (outlier_count / len(data)) * 100,
                    'min_value': data.min(),
                    'max_value': data.max(),
                    'mean_value': data.mean(),
                    'std_value': data.std()
                }
        
        # Data type consistency
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for mixed data types
                data_types = self.df[col].apply(type).value_counts()
                accuracy['data_type_consistency'][col] = {
                    'mixed_types': len(data_types) > 1,
                    'type_distribution': data_types.to_dict()
                }
        
        # Value range analysis
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                accuracy['value_range_analysis'][col] = {
                    'range': data.max() - data.min(),
                    'zero_values': (data == 0).sum(),
                    'negative_values': (data < 0).sum(),
                    'positive_values': (data > 0).sum()
                }
        
        # Calculate accuracy score
        if total_numeric_values > 0:
            outlier_percentage = (total_outliers / total_numeric_values) * 100
            accuracy_score = max(0, 100 - outlier_percentage)
        else:
            accuracy_score = 100
        
        accuracy['accuracy_score'] = accuracy_score
        
        if accuracy_score >= 90:
            accuracy['accuracy_grade'] = 'A'
        elif accuracy_score >= 75:
            accuracy['accuracy_grade'] = 'B'
        elif accuracy_score >= 60:
            accuracy['accuracy_grade'] = 'C'
        else:
            accuracy['accuracy_grade'] = 'D'
        
        self.quality_results['accuracy'] = accuracy
        return accuracy
    
    def assess_consistency(self) -> Dict:
        """Assess data consistency."""
        print(f"\n[QUALITY] Assessing data consistency...")
        
        consistency = {
            'duplicate_analysis': {},
            'format_consistency': {},
            'value_consistency': {},
            'consistency_score': 0
        }
        
        # Duplicate analysis
        duplicate_rows = self.df.duplicated()
        duplicate_count = duplicate_rows.sum()
        duplicate_percentage = (duplicate_count / len(self.df)) * 100
        
        consistency['duplicate_analysis'] = {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'unique_rows': len(self.df) - duplicate_count
        }
        
        # Format consistency for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                # Check for whitespace issues
                has_whitespace_issues = any(str(x).strip() != str(x) for x in data if pd.notna(x))
                
                # Check for case consistency
                string_data = data.astype(str)
                has_mixed_case = any(x != x.lower() and x != x.upper() for x in string_data if x not in ['nan', 'None'])
                
                consistency['format_consistency'][col] = {
                    'has_whitespace_issues': has_whitespace_issues,
                    'has_mixed_case': has_mixed_case,
                    'unique_values': data.nunique(),
                    'most_common': data.value_counts().index[0] if len(data.value_counts()) > 0 else None
                }
        
        # Value consistency for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                consistency['value_consistency'][col] = {
                    'unique_ratio': data.nunique() / len(data),
                    'zero_ratio': (data == 0).sum() / len(data),
                    'negative_ratio': (data < 0).sum() / len(data),
                    'positive_ratio': (data > 0).sum() / len(data)
                }
        
        # Calculate consistency score
        consistency_score = 100
        
        # Penalize for duplicates
        consistency_score -= duplicate_percentage * 2
        
        # Penalize for format issues
        format_issues = 0
        for col_info in consistency['format_consistency'].values():
            if col_info['has_whitespace_issues']:
                format_issues += 10
            if col_info['has_mixed_case']:
                format_issues += 5
        
        consistency_score -= format_issues
        
        consistency['consistency_score'] = max(0, consistency_score)
        
        if consistency_score >= 90:
            consistency['consistency_grade'] = 'A'
        elif consistency_score >= 75:
            consistency['consistency_grade'] = 'B'
        elif consistency_score >= 60:
            consistency['consistency_grade'] = 'C'
        else:
            consistency['consistency_grade'] = 'D'
        
        self.quality_results['consistency'] = consistency
        return consistency
    
    def assess_validity(self) -> Dict:
        """Assess data validity."""
        print(f"\n[QUALITY] Assessing data validity...")
        
        validity = {
            'data_type_validity': {},
            'value_validity': {},
            'constraint_violations': {},
            'validity_score': 0
        }
        
        # Data type validity
        for col in self.df.columns:
            dtype = self.df[col].dtype
            validity['data_type_validity'][col] = {
                'expected_type': str(dtype),
                'is_appropriate': True  # Default assumption
            }
            
            # Check for potential type mismatches
            if dtype == 'object':
                # Check if object column might be numeric
                numeric_count = 0
                for val in self.df[col].dropna():
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                
                if numeric_count > len(self.df[col].dropna()) * 0.8:
                    validity['data_type_validity'][col]['is_appropriate'] = False
                    validity['data_type_validity'][col]['suggested_type'] = 'numeric'
        
        # Value validity checks
        for col in self.df.columns:
            data = self.df[col].dropna()
            if len(data) > 0:
                validity['value_validity'][col] = {
                    'null_count': self.df[col].isnull().sum(),
                    'unique_count': data.nunique(),
                    'min_length': data.astype(str).str.len().min() if dtype == 'object' else None,
                    'max_length': data.astype(str).str.len().max() if dtype == 'object' else None
                }
        
        # Constraint violations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                validity['constraint_violations'][col] = {
                    'negative_values': (data < 0).sum(),
                    'zero_values': (data == 0).sum(),
                    'extreme_values': len(data[data > data.mean() + 3 * data.std()])
                }
        
        # Calculate validity score
        validity_score = 100
        
        # Penalize for type mismatches
        type_mismatches = sum(1 for info in validity['data_type_validity'].values() if not info['is_appropriate'])
        validity_score -= type_mismatches * 10
        
        # Penalize for constraint violations
        constraint_violations = 0
        for col_info in validity['constraint_violations'].values():
            if col_info['negative_values'] > 0:
                constraint_violations += 5
            if col_info['extreme_values'] > 0:
                constraint_violations += 3
        
        validity_score -= constraint_violations
        
        validity['validity_score'] = max(0, validity_score)
        
        if validity_score >= 90:
            validity['validity_grade'] = 'A'
        elif validity_score >= 75:
            validity['validity_grade'] = 'B'
        elif validity_score >= 60:
            validity['validity_grade'] = 'C'
        else:
            validity['validity_grade'] = 'D'
        
        self.quality_results['validity'] = validity
        return validity
    
    def assess_timeliness(self) -> Dict:
        """Assess data timeliness."""
        print(f"\n[QUALITY] Assessing data timeliness...")
        
        timeliness = {
            'date_columns': {},
            'freshness_analysis': {},
            'timeliness_score': 0
        }
        
        # Find date columns
        date_patterns = ['date', 'time', 'created', 'updated', 'timestamp']
        date_columns = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                date_columns.append(col)
        
        for col in date_columns:
            try:
                # Try to parse dates
                date_data = pd.to_datetime(self.df[col], errors='coerce')
                valid_dates = date_data.dropna()
                
                if len(valid_dates) > 0:
                    timeliness['date_columns'][col] = {
                        'valid_dates': len(valid_dates),
                        'invalid_dates': len(date_data) - len(valid_dates),
                        'earliest_date': valid_dates.min(),
                        'latest_date': valid_dates.max(),
                        'date_range_days': (valid_dates.max() - valid_dates.min()).days
                    }
                    
                    # Check freshness
                    now = datetime.now()
                    latest_date = valid_dates.max()
                    days_since_update = (now - latest_date).days
                    
                    timeliness['freshness_analysis'][col] = {
                        'days_since_update': days_since_update,
                        'is_fresh': days_since_update <= 30,  # Consider fresh if updated within 30 days
                        'freshness_score': max(0, 100 - days_since_update)
                    }
            except:
                timeliness['date_columns'][col] = {
                    'error': 'Could not parse dates'
                }
        
        # Calculate timeliness score
        if timeliness['freshness_analysis']:
            freshness_scores = [info['freshness_score'] for info in timeliness['freshness_analysis'].values()]
            timeliness_score = sum(freshness_scores) / len(freshness_scores)
        else:
            timeliness_score = 100  # No date columns, assume good timeliness
        
        timeliness['timeliness_score'] = timeliness_score
        
        if timeliness_score >= 90:
            timeliness['timeliness_grade'] = 'A'
        elif timeliness_score >= 75:
            timeliness['timeliness_grade'] = 'B'
        elif timeliness_score >= 60:
            timeliness['timeliness_grade'] = 'C'
        else:
            timeliness['timeliness_grade'] = 'D'
        
        self.quality_results['timeliness'] = timeliness
        return timeliness
    
    def calculate_overall_quality_score(self) -> Dict:
        """Calculate overall data quality score."""
        print(f"\n[QUALITY] Calculating overall quality score...")
        
        # Get individual scores
        completeness_score = self.quality_results.get('completeness', {}).get('completeness_score', 0)
        accuracy_score = self.quality_results.get('accuracy', {}).get('accuracy_score', 0)
        consistency_score = self.quality_results.get('consistency', {}).get('consistency_score', 0)
        validity_score = self.quality_results.get('validity', {}).get('validity_score', 0)
        timeliness_score = self.quality_results.get('timeliness', {}).get('timeliness_score', 0)
        
        # Calculate weighted average
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'validity': 0.20,
            'timeliness': 0.10
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            validity_score * weights['validity'] +
            timeliness_score * weights['timeliness']
        )
        
        # Determine grade
        if overall_score >= 90:
            grade = 'A'
            quality_level = 'Excellent'
        elif overall_score >= 80:
            grade = 'B'
            quality_level = 'Good'
        elif overall_score >= 70:
            grade = 'C'
            quality_level = 'Fair'
        elif overall_score >= 60:
            grade = 'D'
            quality_level = 'Poor'
        else:
            grade = 'F'
            quality_level = 'Very Poor'
        
        overall_quality = {
            'overall_score': overall_score,
            'grade': grade,
            'quality_level': quality_level,
            'component_scores': {
                'completeness': completeness_score,
                'accuracy': accuracy_score,
                'consistency': consistency_score,
                'validity': validity_score,
                'timeliness': timeliness_score
            },
            'weights': weights
        }
        
        self.quality_results['overall_quality'] = overall_quality
        return overall_quality
    
    def generate_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations."""
        print(f"\n[RECOMMENDATIONS] Generating recommendations...")
        
        recommendations = []
        
        # Completeness recommendations
        completeness = self.quality_results.get('completeness', {})
        if completeness.get('overall_completeness', 100) < 95:
            recommendations.append("Address missing data issues to improve completeness")
            
            high_missing_cols = [col for col, info in completeness.get('column_completeness', {}).items() 
                               if info['missing_percentage'] > 10]
            if high_missing_cols:
                recommendations.append(f"Focus on columns with high missing rates: {', '.join(high_missing_cols)}")
        
        # Accuracy recommendations
        accuracy = self.quality_results.get('accuracy', {})
        if accuracy.get('accuracy_score', 100) < 90:
            recommendations.append("Investigate and address outliers to improve accuracy")
            
            high_outlier_cols = [col for col, info in accuracy.get('outlier_analysis', {}).items() 
                               if info['outlier_percentage'] > 10]
            if high_outlier_cols:
                recommendations.append(f"Review outliers in columns: {', '.join(high_outlier_cols)}")
        
        # Consistency recommendations
        consistency = self.quality_results.get('consistency', {})
        if consistency.get('consistency_score', 100) < 90:
            recommendations.append("Standardize data formats to improve consistency")
            
            format_issues = [col for col, info in consistency.get('format_consistency', {}).items() 
                           if info['has_whitespace_issues'] or info['has_mixed_case']]
            if format_issues:
                recommendations.append(f"Clean formatting in columns: {', '.join(format_issues)}")
        
        # Validity recommendations
        validity = self.quality_results.get('validity', {})
        if validity.get('validity_score', 100) < 90:
            recommendations.append("Review data types and constraints for validity")
            
            type_issues = [col for col, info in validity.get('data_type_validity', {}).items() 
                          if not info['is_appropriate']]
            if type_issues:
                recommendations.append(f"Consider data type changes for: {', '.join(type_issues)}")
        
        # Timeliness recommendations
        timeliness = self.quality_results.get('timeliness', {})
        if timeliness.get('timeliness_score', 100) < 90:
            recommendations.append("Update data more frequently to improve timeliness")
        
        # General recommendations
        overall_quality = self.quality_results.get('overall_quality', {})
        if overall_quality.get('overall_score', 100) < 80:
            recommendations.append("Implement regular data quality monitoring")
            recommendations.append("Establish data quality standards and procedures")
            recommendations.append("Train team members on data quality best practices")
        
        self.quality_results['recommendations'] = recommendations
        return recommendations
    
    def create_quality_visualizations(self):
        """Create quality assessment visualizations."""
        print(f"\n[CHART] Creating quality visualizations...")
        
        # Create output directory
        os.makedirs('../outputs/quality_outputs', exist_ok=True)
        
        # 1. Quality Score Dashboard
        self._create_quality_dashboard()
        
        # 2. Component Analysis
        self._create_component_analysis()
        
        # 3. Quality Issues
        self._create_quality_issues()
        
        print(f"[OK] All quality visualizations saved in '../outputs/quality_outputs/' directory")
    
    def _create_quality_dashboard(self):
        """Create quality dashboard."""
        overall_quality = self.quality_results.get('overall_quality', {})
        component_scores = overall_quality.get('component_scores', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall quality score
        overall_score = overall_quality.get('overall_score', 0)
        grade = overall_quality.get('grade', 'N/A')
        
        axes[0, 0].pie([overall_score, 100-overall_score], 
                       labels=[f'Quality ({overall_score:.1f}%)', 'Gap'], 
                       autopct='%1.1f%%', 
                       colors=['green' if overall_score >= 80 else 'orange', 'lightgray'])
        axes[0, 0].set_title(f'Overall Quality Score: {grade}')
        
        # 2. Component scores
        if component_scores:
            components = list(component_scores.keys())
            scores = list(component_scores.values())
            colors = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores]
            
            axes[0, 1].bar(components, scores, color=colors)
            axes[0, 1].set_title('Component Quality Scores')
            axes[0, 1].set_ylabel('Score (%)')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Quality grade distribution
        grades = ['A', 'B', 'C', 'D', 'F']
        grade_counts = [0] * len(grades)
        
        for component in ['completeness', 'accuracy', 'consistency', 'validity', 'timeliness']:
            grade = self.quality_results.get(component, {}).get(f'{component}_grade', 'F')
            if grade in grades:
                grade_counts[grades.index(grade)] += 1
        
        axes[1, 0].bar(grades, grade_counts, color=['green', 'lightgreen', 'orange', 'red', 'darkred'])
        axes[1, 0].set_title('Quality Grade Distribution')
        axes[1, 0].set_ylabel('Number of Components')
        
        # 4. Quality level
        quality_level = overall_quality.get('quality_level', 'Unknown')
        axes[1, 1].text(0.5, 0.5, f'Quality Level:\n{quality_level}', 
                        ha='center', va='center', transform=axes[1, 1].transAxes, 
                        fontsize=16, fontweight='bold')
        axes[1, 1].set_title('Overall Assessment')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('../outputs/quality_outputs/quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_component_analysis(self):
        """Create component analysis visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Component Quality Analysis', fontsize=16, fontweight='bold')
        
        # Completeness
        completeness = self.quality_results.get('completeness', {})
        if completeness.get('column_completeness'):
            cols = list(completeness['column_completeness'].keys())
            completeness_pcts = [info['completeness_percentage'] for info in completeness['column_completeness'].values()]
            
            axes[0, 0].barh(cols, completeness_pcts, color='blue')
            axes[0, 0].set_title('Completeness by Column')
            axes[0, 0].set_xlabel('Completeness (%)')
        
        # Accuracy
        accuracy = self.quality_results.get('accuracy', {})
        if accuracy.get('outlier_analysis'):
            cols = list(accuracy['outlier_analysis'].keys())
            outlier_pcts = [info['outlier_percentage'] for info in accuracy['outlier_analysis'].values()]
            
            axes[0, 1].barh(cols, outlier_pcts, color='red')
            axes[0, 1].set_title('Outlier Percentage by Column')
            axes[0, 1].set_xlabel('Outlier %')
        
        # Consistency
        consistency = self.quality_results.get('consistency', {})
        if consistency.get('format_consistency'):
            cols = list(consistency['format_consistency'].keys())
            format_issues = [1 if info['has_whitespace_issues'] or info['has_mixed_case'] else 0 
                           for info in consistency['format_consistency'].values()]
            
            axes[0, 2].bar(cols, format_issues, color='orange')
            axes[0, 2].set_title('Format Issues by Column')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Validity
        validity = self.quality_results.get('validity', {})
        if validity.get('data_type_validity'):
            cols = list(validity['data_type_validity'].keys())
            type_issues = [0 if info['is_appropriate'] else 1 for info in validity['data_type_validity'].values()]
            
            axes[1, 0].bar(cols, type_issues, color='purple')
            axes[1, 0].set_title('Data Type Issues by Column')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Timeliness
        timeliness = self.quality_results.get('timeliness', {})
        if timeliness.get('freshness_analysis'):
            cols = list(timeliness['freshness_analysis'].keys())
            freshness_scores = [info['freshness_score'] for info in timeliness['freshness_analysis'].values()]
            
            axes[1, 1].bar(cols, freshness_scores, color='green')
            axes[1, 1].set_title('Freshness Scores by Column')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Overall summary
        overall_quality = self.quality_results.get('overall_quality', {})
        component_scores = overall_quality.get('component_scores', {})
        if component_scores:
            components = list(component_scores.keys())
            scores = list(component_scores.values())
            
            axes[1, 2].bar(components, scores, color=['blue', 'red', 'orange', 'purple', 'green'])
            axes[1, 2].set_title('Overall Component Scores')
            axes[1, 2].set_ylabel('Score (%)')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../outputs/quality_outputs/component_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_issues(self):
        """Create quality issues visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quality Issues Analysis', fontsize=16, fontweight='bold')
        
        # Missing data heatmap
        missing_data = self.df.isnull()
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data, cbar=True, ax=axes[0, 0])
            axes[0, 0].set_title('Missing Data Pattern')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Data Pattern')
        
        # Duplicate analysis
        consistency = self.quality_results.get('consistency', {})
        duplicate_info = consistency.get('duplicate_analysis', {})
        if duplicate_info:
            duplicate_pct = duplicate_info.get('duplicate_percentage', 0)
            axes[0, 1].pie([duplicate_pct, 100-duplicate_pct], 
                           labels=['Duplicates', 'Unique'], 
                           autopct='%1.1f%%', 
                           colors=['red', 'green'])
            axes[0, 1].set_title('Duplicate Analysis')
        
        # Outlier summary
        accuracy = self.quality_results.get('accuracy', {})
        outlier_info = accuracy.get('outlier_analysis', {})
        if outlier_info:
            cols = list(outlier_info.keys())
            outlier_counts = [info['outlier_count'] for info in outlier_info.values()]
            
            axes[1, 0].bar(cols, outlier_counts, color='red')
            axes[1, 0].set_title('Outlier Count by Column')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Quality score breakdown
        overall_quality = self.quality_results.get('overall_quality', {})
        component_scores = overall_quality.get('component_scores', {})
        if component_scores:
            labels = list(component_scores.keys())
            sizes = list(component_scores.values())
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
            
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            axes[1, 1].set_title('Quality Score Breakdown')
        
        plt.tight_layout()
        plt.savefig('../outputs/quality_outputs/quality_issues.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_quality_report(self, output_format: str = 'html') -> str:
        """Generate quality assessment report."""
        print(f"\n[REPORT] Generating {output_format.upper()} quality report...")
        
        if output_format.lower() == 'html':
            return self._generate_html_quality_report()
        elif output_format.lower() == 'json':
            return self._generate_json_quality_report()
        else:
            return self._generate_text_quality_report()
    
    def _generate_html_quality_report(self) -> str:
        """Generate HTML quality report."""
        overall_quality = self.quality_results.get('overall_quality', {})
        recommendations = self.quality_results.get('recommendations', [])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .grade-a {{ color: #27ae60; }}
                .grade-b {{ color: #f39c12; }}
                .grade-c {{ color: #e67e22; }}
                .grade-d {{ color: #e74c3c; }}
                .grade-f {{ color: #c0392b; }}
                .recommendation {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Assessment Report</h1>
                <p><strong>Dataset:</strong> {self.data_source}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Shape:</strong> {len(self.df)} rows × {len(self.df.columns)} columns</p>
            </div>
            
            <div class="section">
                <h2>Overall Quality Assessment</h2>
                <p class="score grade-{overall_quality.get('grade', 'f').lower()}">
                    Overall Score: {overall_quality.get('overall_score', 0):.1f}% 
                    (Grade: {overall_quality.get('grade', 'N/A')})
                </p>
                <p><strong>Quality Level:</strong> {overall_quality.get('quality_level', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Component Scores</h2>
                <table>
                    <tr><th>Component</th><th>Score</th><th>Grade</th></tr>
        """
        
        component_scores = overall_quality.get('component_scores', {})
        for component, score in component_scores.items():
            grade = self.quality_results.get(component, {}).get(f'{component}_grade', 'N/A')
            html_content += f'<tr><td>{component.title()}</td><td>{score:.1f}%</td><td>{grade}</td></tr>'
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        for recommendation in recommendations:
            html_content += f'<div class="recommendation">• {recommendation}</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open('../outputs/quality_outputs/quality_assessment_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"[OK] HTML quality report saved as '../outputs/quality_outputs/quality_assessment_report.html'")
        return html_content
    
    def _generate_json_quality_report(self) -> str:
        """Generate JSON quality report."""
        report = {
            'metadata': {
                'dataset': self.data_source,
                'generated_at': datetime.now().isoformat(),
                'shape': {'rows': len(self.df), 'columns': len(self.df.columns)}
            },
            'quality_results': self.quality_results
        }
        
        json_content = json.dumps(report, indent=2, default=str)
        
        # Save JSON file
        with open('../outputs/quality_outputs/quality_assessment_report.json', 'w') as f:
            f.write(json_content)
        
        print(f"[OK] JSON quality report saved as '../outputs/quality_outputs/quality_assessment_report.json'")
        return json_content
    
    def _generate_text_quality_report(self) -> str:
        """Generate text quality report."""
        overall_quality = self.quality_results.get('overall_quality', {})
        recommendations = self.quality_results.get('recommendations', [])
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA QUALITY ASSESSMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Dataset: {self.data_source}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Shape: {len(self.df)} rows × {len(self.df.columns)} columns")
        report_lines.append("")
        
        report_lines.append("OVERALL QUALITY ASSESSMENT:")
        report_lines.append("-" * 30)
        report_lines.append(f"Overall Score: {overall_quality.get('overall_score', 0):.1f}%")
        report_lines.append(f"Grade: {overall_quality.get('grade', 'N/A')}")
        report_lines.append(f"Quality Level: {overall_quality.get('quality_level', 'Unknown')}")
        report_lines.append("")
        
        report_lines.append("COMPONENT SCORES:")
        report_lines.append("-" * 20)
        component_scores = overall_quality.get('component_scores', {})
        for component, score in component_scores.items():
            grade = self.quality_results.get(component, {}).get(f'{component}_grade', 'N/A')
            report_lines.append(f"{component.title()}: {score:.1f}% (Grade: {grade})")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 20)
        for recommendation in recommendations:
            report_lines.append(f"• {recommendation}")
        
        report_text = "\n".join(report_lines)
        
        # Save text file
        with open('../outputs/quality_outputs/quality_assessment_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Text quality report saved as '../outputs/quality_outputs/quality_assessment_report.txt'")
        return report_text
    
    def run_comprehensive_quality_assessment(self, table_name: str = None, output_format: str = 'html'):
        """Run comprehensive quality assessment."""
        print(f"\n[START] Starting comprehensive quality assessment")
        
        # Load data
        self.load_data(table_name)
        
        # Run all assessments
        self.assess_completeness()
        self.assess_accuracy()
        self.assess_consistency()
        self.assess_validity()
        self.assess_timeliness()
        
        # Calculate overall score
        self.calculate_overall_quality_score()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Create visualizations
        self.create_quality_visualizations()
        
        # Generate report
        self.generate_quality_report(output_format)
        
        print(f"\n[OK] Comprehensive quality assessment complete!")
        print(f"[DATA] Check '../outputs/quality_outputs/' directory for results")
    
    def close(self):
        """Clean up resources."""
        print("[CONNECT] Data quality assessor closed")


def main():
    """Main function to run quality assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive data quality assessment')
    parser.add_argument('data_source', help='Path to database file or CSV file')
    parser.add_argument('--table', help='Table name (for SQLite databases)')
    parser.add_argument('--format', choices=['html', 'json', 'text'], default='html', 
                       help='Output format for report')
    
    args = parser.parse_args()
    
    # Create assessor
    assessor = DataQualityAssessor(args.data_source)
    
    try:
        # Run comprehensive assessment
        assessor.run_comprehensive_quality_assessment(args.table, args.format)
        
    except Exception as e:
        print(f"[ERROR] Error during assessment: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        assessor.close()


if __name__ == "__main__":
    main() 