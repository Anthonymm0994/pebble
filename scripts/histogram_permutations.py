#!/usr/bin/env python3
"""
Histogram Permutations Generator
===============================

A comprehensive tool for generating histogram permutations based on SQL queries
with various WHERE clause predicates for comparison and analysis.

Features:
- SQL query parsing and execution
- WHERE clause predicate permutations
- Consistent axis labels across histograms
- Multiple histogram types
- Statistical analysis
- Comparison capabilities
- Automated insights
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, permutations
import re
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class HistogramPermutationsGenerator:
    """
    Generate histogram permutations based on SQL queries with WHERE clause variations.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the histogram permutations generator.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.query_results = {}
        self.histogram_data = {}
        
    def connect(self) -> bool:
        """Establish connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"[OK] Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error connecting to database: {e}")
            return False
    
    def parse_sql_query(self, query: str) -> Dict:
        """Parse SQL query to extract components."""
        print(f"\n[PARSE] Parsing SQL query...")
        
        parsed = {
            'original_query': query,
            'select_columns': [],
            'from_table': '',
            'where_conditions': [],
            'group_by': [],
            'order_by': [],
            'limit': None
        }
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
        if select_match:
            select_clause = select_match.group(1).strip()
            parsed['select_columns'] = [col.strip() for col in select_clause.split(',')]
        
        # Extract FROM table
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            parsed['from_table'] = from_match.group(1)
        
        # Extract WHERE conditions
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR operators
            conditions = re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE)
            parsed['where_conditions'] = [cond.strip() for cond in conditions]
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', query, re.IGNORECASE)
        if group_match:
            group_clause = group_match.group(1).strip()
            parsed['group_by'] = [col.strip() for col in group_clause.split(',')]
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_clause = order_match.group(1).strip()
            parsed['order_by'] = [col.strip() for col in order_clause.split(',')]
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            parsed['limit'] = int(limit_match.group(1))
        
        print(f"[OK] Parsed query components:")
        print(f"  - SELECT: {parsed['select_columns']}")
        print(f"  - FROM: {parsed['from_table']}")
        print(f"  - WHERE: {parsed['where_conditions']}")
        print(f"  - GROUP BY: {parsed['group_by']}")
        print(f"  - ORDER BY: {parsed['order_by']}")
        print(f"  - LIMIT: {parsed['limit']}")
        
        return parsed
    
    def generate_where_permutations(self, where_conditions: List[str]) -> List[List[str]]:
        """Generate all permutations of WHERE conditions."""
        print(f"\n[PERMUTE] Generating WHERE clause permutations...")
        
        if not where_conditions:
            return [[]]
        
        # Generate all combinations of conditions
        all_permutations = []
        
        # Single conditions
        for condition in where_conditions:
            all_permutations.append([condition])
        
        # Multiple condition combinations
        for r in range(2, min(len(where_conditions) + 1, 5)):  # Limit to 4 conditions max
            for combo in combinations(where_conditions, r):
                all_permutations.append(list(combo))
        
        print(f"[OK] Generated {len(all_permutations)} WHERE clause permutations")
        return all_permutations
    
    def build_query_variations(self, parsed_query: Dict) -> List[Dict]:
        """Build query variations with different WHERE clauses."""
        print(f"\n[BUILD] Building query variations...")
        
        variations = []
        where_permutations = self.generate_where_permutations(parsed_query['where_conditions'])
        
        for i, where_combo in enumerate(where_permutations):
            # Build the base query
            select_clause = ', '.join(parsed_query['select_columns'])
            from_clause = f"FROM {parsed_query['from_table']}"
            
            # Build WHERE clause
            if where_combo:
                where_clause = f"WHERE {' AND '.join(where_combo)}"
            else:
                where_clause = ""
            
            # Build GROUP BY clause
            group_clause = ""
            if parsed_query['group_by']:
                group_clause = f"GROUP BY {', '.join(parsed_query['group_by'])}"
            
            # Build ORDER BY clause
            order_clause = ""
            if parsed_query['order_by']:
                order_clause = f"ORDER BY {', '.join(parsed_query['order_by'])}"
            
            # Build LIMIT clause
            limit_clause = ""
            if parsed_query['limit']:
                limit_clause = f"LIMIT {parsed_query['limit']}"
            
            # Combine all clauses
            query_parts = [f"SELECT {select_clause}", from_clause]
            if where_clause:
                query_parts.append(where_clause)
            if group_clause:
                query_parts.append(group_clause)
            if order_clause:
                query_parts.append(order_clause)
            if limit_clause:
                query_parts.append(limit_clause)
            
            full_query = ' '.join(query_parts)
            
            variation = {
                'id': i + 1,
                'query': full_query,
                'where_conditions': where_combo,
                'description': f"Variation {i + 1}: {' AND '.join(where_combo) if where_combo else 'No WHERE clause'}"
            }
            
            variations.append(variation)
        
        print(f"[OK] Built {len(variations)} query variations")
        return variations
    
    def execute_query_variations(self, variations: List[Dict]) -> Dict:
        """Execute all query variations and store results."""
        print(f"\n[EXECUTE] Executing query variations...")
        
        results = {}
        
        for variation in variations:
            try:
                print(f"[QUERY] Executing: {variation['description']}")
                df = pd.read_sql_query(variation['query'], self.conn)
                
                results[variation['id']] = {
                    'query': variation['query'],
                    'description': variation['description'],
                    'data': df,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns)
                }
                
                print(f"[OK] Retrieved {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"[ERROR] Failed to execute variation {variation['id']}: {e}")
                results[variation['id']] = {
                    'query': variation['query'],
                    'description': variation['description'],
                    'error': str(e),
                    'data': None,
                    'row_count': 0,
                    'column_count': 0,
                    'columns': []
                }
        
        self.query_results = results
        return results
    
    def analyze_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns suitable for histogram analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out columns with too few unique values or too many
        suitable_cols = []
        for col in numeric_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 100:  # Reasonable range for histograms
                suitable_cols.append(col)
        
        return suitable_cols
    
    def create_histogram_variations(self, column: str, data_variations: Dict) -> Dict:
        """Create histogram variations for a specific column across all query results."""
        print(f"\n[HISTOGRAM] Creating histogram variations for column: {column}")
        
        histogram_data = {}
        
        for var_id, result in data_variations.items():
            if result.get('error') or result['data'] is None:
                continue
            
            df = result['data']
            if column not in df.columns:
                continue
            
            data = df[column].dropna()
            if len(data) == 0:
                continue
            
            # Calculate statistics
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
            
            # Detect outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(data)) * 100
            
            histogram_data[var_id] = {
                'description': result['description'],
                'data': data,
                'stats': stats,
                'row_count': result['row_count']
            }
        
        return histogram_data
    
    def create_comparative_histograms(self, column: str, histogram_data: Dict):
        """Create comparative histograms with consistent axis labels."""
        print(f"\n[CHART] Creating comparative histograms for: {column}")
        
        # Create output directory
        os.makedirs('../outputs/histogram_outputs', exist_ok=True)
        
        # Determine consistent axis limits
        all_data = []
        for var_data in histogram_data.values():
            all_data.extend(var_data['data'].tolist())
        
        if not all_data:
            print(f"[WARNING] No data available for column: {column}")
            return
        
        global_min = min(all_data)
        global_max = max(all_data)
        global_range = global_max - global_min
        
        # Create subplots
        n_variations = len(histogram_data)
        if n_variations == 0:
            print(f"[WARNING] No valid variations for column: {column}")
            return
        
        cols_per_row = 3
        rows = (n_variations + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 6 * rows))
        fig.suptitle(f'Histogram Permutations Analysis: {column}', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create histograms for each variation
        for i, (var_id, var_data) in enumerate(histogram_data.items()):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            
            data = var_data['data']
            stats = var_data['stats']
            description = var_data['description']
            
            # Create histogram with consistent bins
            bins = np.linspace(global_min - global_range * 0.05, 
                              global_max + global_range * 0.05, 30)
            
            # Create histogram with improved styling
            n, bins, patches = axes[row, col_idx].hist(data, bins=bins, alpha=0.7, 
                                                      color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Add mean line
            mean_val = stats["mean"]
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                      label=f'Mean: {mean_val:.2f}')
            
            # Add median line
            median_val = stats["median"]
            axes[row, col_idx].axvline(median_val, color='orange', linestyle='-', linewidth=2, 
                                      label=f'Median: {median_val:.2f}')
            
            # Set title and labels
            axes[row, col_idx].set_title(f'{description}\n(n={stats["count"]})', fontsize=12, fontweight='bold')
            axes[row, col_idx].set_xlabel(f'{column} Values', fontsize=10, fontweight='bold')
            axes[row, col_idx].set_ylabel('Frequency', fontsize=10, fontweight='bold')
            
            # Add legend
            axes[row, col_idx].legend(loc='upper right', fontsize=8)
            
            # Add statistics text box
            stats_text = f'Mean: {stats["mean"]:.2f}\nStd: {stats["std"]:.2f}\nOutliers: {stats["outlier_count"]}'
            axes[row, col_idx].text(0.02, 0.98, stats_text, transform=axes[row, col_idx].transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                                   fontsize=9, fontweight='bold')
            
            # Set consistent axis limits
            axes[row, col_idx].set_xlim(global_min - global_range * 0.05, 
                                       global_max + global_range * 0.05)
            
            # Add grid for better readability
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_variations, rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            axes[row, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'../outputs/histogram_outputs/histogram_permutations_{column}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved histogram permutations for {column}")
    
    def create_statistical_comparison(self, column: str, histogram_data: Dict):
        """Create statistical comparison chart for all variations."""
        print(f"\n[STATS] Creating statistical comparison for: {column}")
        
        # Prepare data for comparison
        comparison_data = []
        labels = []
        
        for var_id, var_data in histogram_data.items():
            stats = var_data['stats']
            description = var_data['description']
            
            comparison_data.append([
                stats['mean'],
                stats['median'],
                stats['std'],
                stats['skewness'],
                stats['kurtosis'],
                stats['outlier_percentage']
            ])
            labels.append(description[:30] + '...' if len(description) > 30 else description)
        
        if not comparison_data:
            print(f"[WARNING] No data for statistical comparison: {column}")
            return
        
        # Create comparison chart with improved styling
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Statistical Comparison Analysis: {column}', fontsize=16, fontweight='bold')
        
        metrics = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Outlier %']
        colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']
        
        for i, (metric, ax, color) in enumerate(zip(metrics, axes.flat, colors)):
            values = [row[i] for row in comparison_data]
            
            bars = ax.bar(range(len(labels)), values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'../outputs/histogram_outputs/statistical_comparison_{column}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved statistical comparison for {column}")
    
    def generate_insights(self, column: str, histogram_data: Dict) -> List[str]:
        """Generate insights from histogram variations."""
        print(f"\n[INSIGHTS] Generating insights for: {column}")
        
        insights = []
        
        if len(histogram_data) < 2:
            insights.append(f"Only one variation available for {column} - no comparison possible")
            return insights
        
        # Compare statistics across variations
        means = [var_data['stats']['mean'] for var_data in histogram_data.values()]
        stds = [var_data['stats']['std'] for var_data in histogram_data.values()]
        outlier_pcts = [var_data['stats']['outlier_percentage'] for var_data in histogram_data.values()]
        
        # Mean variation analysis
        mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
        if mean_cv > 0.1:
            insights.append(f"High variation in mean values across filters (CV: {mean_cv:.2f})")
        else:
            insights.append(f"Consistent mean values across filters (CV: {mean_cv:.2f})")
        
        # Standard deviation analysis
        std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else 0
        if std_cv > 0.2:
            insights.append(f"High variation in standard deviation across filters (CV: {std_cv:.2f})")
        else:
            insights.append(f"Consistent standard deviation across filters (CV: {std_cv:.2f})")
        
        # Outlier analysis
        max_outlier_pct = max(outlier_pcts)
        min_outlier_pct = min(outlier_pcts)
        if max_outlier_pct - min_outlier_pct > 10:
            insights.append(f"Significant variation in outlier percentage across filters ({min_outlier_pct:.1f}% to {max_outlier_pct:.1f}%)")
        else:
            insights.append(f"Consistent outlier patterns across filters ({min_outlier_pct:.1f}% to {max_outlier_pct:.1f}%)")
        
        # Sample size analysis
        sample_sizes = [var_data['stats']['count'] for var_data in histogram_data.values()]
        max_sample = max(sample_sizes)
        min_sample = min(sample_sizes)
        
        if max_sample / min_sample > 5:
            insights.append(f"Large variation in sample sizes across filters ({min_sample} to {max_sample} records)")
        else:
            insights.append(f"Consistent sample sizes across filters ({min_sample} to {max_sample} records)")
        
        return insights
    
    def create_comprehensive_analysis(self, sql_query: str):
        """Create comprehensive histogram analysis from SQL query."""
        print(f"\n[START] Starting comprehensive histogram analysis")
        
        # Parse the SQL query
        parsed_query = self.parse_sql_query(sql_query)
        
        # Build query variations
        variations = self.build_query_variations(parsed_query)
        
        # Execute all variations
        results = self.execute_query_variations(variations)
        
        # Find numeric columns for analysis
        numeric_columns = []
        for var_id, result in results.items():
            if result.get('error') or result['data'] is None:
                continue
            
            df = result['data']
            cols = self.analyze_numeric_columns(df)
            numeric_columns.extend(cols)
        
        numeric_columns = list(set(numeric_columns))  # Remove duplicates
        
        if not numeric_columns:
            print(f"[WARNING] No suitable numeric columns found for histogram analysis")
            return
        
        print(f"[OK] Found {len(numeric_columns)} suitable numeric columns: {numeric_columns}")
        
        # Create analysis for each column
        all_insights = {}
        
        for column in numeric_columns:
            print(f"\n[ANALYSIS] Analyzing column: {column}")
            
            # Create histogram variations
            histogram_data = self.create_histogram_variations(column, results)
            
            if histogram_data:
                # Create visualizations
                self.create_comparative_histograms(column, histogram_data)
                self.create_statistical_comparison(column, histogram_data)
                
                # Generate insights
                insights = self.generate_insights(column, histogram_data)
                all_insights[column] = insights
                
                print(f"[OK] Completed analysis for {column}")
            else:
                print(f"[WARNING] No valid data for column: {column}")
        
        # Save insights
        self.save_insights(all_insights)
        
        print(f"\n[OK] Comprehensive histogram analysis complete!")
        print(f"[DATA] Check '../outputs/histogram_outputs/' directory for results")
    
    def save_insights(self, insights: Dict[str, List[str]]):
        """Save insights to file."""
        print(f"\n[SAVE] Saving insights...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as text
        with open(f'../outputs/histogram_outputs/insights_{timestamp}.txt', 'w') as f:
            f.write("HISTOGRAM ANALYSIS INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            
            for column, column_insights in insights.items():
                f.write(f"Column: {column}\n")
                f.write("-" * 20 + "\n")
                for insight in column_insights:
                    f.write(f"• {insight}\n")
                f.write("\n")
        
        # Save as JSON
        with open(f'../outputs/histogram_outputs/insights_{timestamp}.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"[OK] Insights saved to ../outputs/histogram_outputs/insights_{timestamp}.txt and .json")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("[CONNECT] Database connection closed")


def main():
    """Main function to run histogram permutations analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate histogram permutations from SQL query')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--query', help='SQL query to analyze')
    parser.add_argument('--file', help='File containing SQL query')
    
    args = parser.parse_args()
    
    # Create generator
    generator = HistogramPermutationsGenerator(args.db_path)
    
    try:
        # Connect to database
        if not generator.connect():
            return
        
        # Get SQL query
        sql_query = None
        if args.query:
            sql_query = args.query
        elif args.file:
            with open(args.file, 'r') as f:
                sql_query = f.read().strip()
        else:
            # Default query for testing
            sql_query = "SELECT * FROM sample_data WHERE price > 0"
            print(f"[INFO] Using default query: {sql_query}")
        
        # Run comprehensive analysis
        generator.create_comprehensive_analysis(sql_query)
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.close()


if __name__ == "__main__":
    main() 