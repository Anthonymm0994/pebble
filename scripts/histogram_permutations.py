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
    Generate histogram permutations based on SQL queries with various WHERE clause predicates.
    """
    
    def __init__(self, data_source: str):
        """
        Initialize the histogram permutations generator.
        
        Args:
            data_source: Path to CSV file or database file
        """
        self.data_source = data_source
        self.df = None
        self.conn = None
        self.query_results = {}
        self.histogram_data = {}
        
    def load_data(self):
        """Load data from CSV file or database."""
        print(f"[DATA] Loading data from: {self.data_source}")
        
        try:
            if self.data_source.endswith('.csv'):
                self.df = pd.read_csv(self.data_source)
                print(f"[OK] Loaded CSV dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
                return self.df
            else:
                # Assume it's a database file
                self.conn = sqlite3.connect(self.data_source)
                # Get first table
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if tables:
                    first_table = tables[0][0]
                    self.df = pd.read_sql_query(f"SELECT * FROM {first_table}", self.conn)
                    print(f"[INFO] Loaded table: {first_table}")
                else:
                    raise ValueError("No tables found in database")
            
            print(f"[OK] Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
            return self.df
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return None
    
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
    
    def build_sql_query(self, variation: Dict) -> str:
        """Build SQL query from variation dictionary."""
        select_clause = ', '.join(variation.get('select_columns', ['*']))
        from_clause = variation.get('from_table', 'sample_data')  # Use actual table name
        
        query = f"SELECT {select_clause} FROM {from_clause}"
        
        where_conditions = variation.get('where_conditions', [])
        if where_conditions:
            where_clause = ' AND '.join(where_conditions)
            query += f" WHERE {where_clause}"
        
        group_by = variation.get('group_by', [])
        if group_by:
            group_clause = ', '.join(group_by)
            query += f" GROUP BY {group_clause}"
        
        order_by = variation.get('order_by', [])
        if order_by:
            order_clause = ', '.join(order_by)
            query += f" ORDER BY {order_clause}"
        
        limit = variation.get('limit')
        if limit:
            query += f" LIMIT {limit}"
        
        return query

    def execute_query_variations(self, variations: List[Dict]) -> Dict:
        """Execute query variations and store results."""
        print(f"\n[EXECUTE] Executing query variations...")
        
        results = {}
        
        for i, variation in enumerate(variations, 1):
            print(f"[QUERY] Executing: Variation {i}: {variation.get('where_conditions', ['No conditions'])}")
            
            try:
                if self.data_source.endswith('.csv'):
                    # For CSV files, filter the DataFrame based on conditions
                    filtered_df = self.df.copy()
                    
                    # Apply WHERE conditions
                    where_conditions = variation.get('where_conditions', [])
                    for condition in where_conditions:
                        # Simple condition parsing for CSV data
                        if '>' in condition:
                            col, val = condition.split('>')
                            col = col.strip()
                            val = float(val.strip())
                            filtered_df = filtered_df[filtered_df[col] > val]
                        elif '<' in condition:
                            col, val = condition.split('<')
                            col = col.strip()
                            val = float(val.strip())
                            filtered_df = filtered_df[filtered_df[col] < val]
                        elif '==' in condition:
                            col, val = condition.split('==')
                            col = col.strip()
                            val = val.strip().strip("'").strip('"')
                            filtered_df = filtered_df[filtered_df[col] == val]
                        elif '!=' in condition:
                            col, val = condition.split('!=')
                            col = col.strip()
                            val = val.strip().strip("'").strip('"')
                            filtered_df = filtered_df[filtered_df[col] != val]
                    
                    results[f"variation_{i}"] = filtered_df
                    print(f"[OK] Filtered to {len(filtered_df)} rows")
                    
                else:
                    # For database files, execute SQL queries
                    query = self.build_sql_query(variation)
                    result_df = pd.read_sql_query(query, self.conn)
                    results[f"variation_{i}"] = result_df
                    print(f"[OK] Retrieved {len(result_df)} rows")
                    
            except Exception as e:
                print(f"[ERROR] Failed to execute variation {i}: {e}")
                results[f"variation_{i}"] = pd.DataFrame()  # Empty DataFrame for failed queries
        
        print(f"[OK] Executed {len(variations)} query variations")
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
    
    def create_histogram_variations(self, column: str, query_results: Dict) -> Dict:
        """Create histogram variations for a specific column."""
        print(f"[BUILD] Creating histogram variations for {column}")
        
        histogram_data = {}
        
        for variation_id, result_df in query_results.items():
            if result_df.empty:
                continue
            
            if column not in result_df.columns:
                continue
            
            # Get data for the column
            data = result_df[column].dropna()
            
            if len(data) == 0:
                continue
            
            # Calculate statistics
            stats = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75)
            }
            
            histogram_data[variation_id] = {
                'data': data,
                'stats': stats,
                'variation_id': variation_id
            }
        
        if histogram_data:
            print(f"[OK] Created {len(histogram_data)} histogram variations for {column}")
        else:
            print(f"[WARNING] No valid histogram data for {column}")
        
        return histogram_data
    
    def create_comparative_histograms(self, column: str, histogram_data: Dict):
        """Create comparative histograms for all variations."""
        print(f"[CHART] Creating comparative histograms for: {column}")
        
        if not histogram_data:
            print(f"[WARNING] No histogram data available for {column}")
            return
        
        # Create output directory
        os.makedirs('../outputs/histogram_outputs', exist_ok=True)
        
        # Create subplot layout
        n_variations = len(histogram_data)
        if n_variations <= 2:
            fig, axes = plt.subplots(1, n_variations, figsize=(15, 6))
            if n_variations == 1:
                axes = [axes]
        else:
            cols = min(3, n_variations)
            rows = (n_variations + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            axes = axes.flatten() if rows > 1 else axes
        
        # Create histograms for each variation
        for i, (var_id, var_data) in enumerate(histogram_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = var_data['data']
            stats = var_data['stats']
            
            # Create histogram
            ax.hist(data, bins=min(20, len(data)//2), alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add mean and median lines
            ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
            ax.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.2f}')
            
            # Add statistics text
            stats_text = f'Count: {stats["count"]}\nMean: {stats["mean"]:.2f}\nStd: {stats["std"]:.2f}\nMin: {stats["min"]:.2f}\nMax: {stats["max"]:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'Variation {i+1}: {column}', fontsize=12, fontweight='bold')
            ax.set_xlabel(column, fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(histogram_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'../outputs/histogram_outputs/comparative_histograms_{column}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Comparative histograms saved for {column}")
    
    def create_statistical_comparison(self, column: str, histogram_data: Dict):
        """Create statistical comparison chart for all variations."""
        print(f"[STATS] Creating statistical comparison for: {column}")
        
        if not histogram_data:
            print(f"[WARNING] No histogram data available for {column}")
            return
        
        # Create output directory
        os.makedirs('../outputs/histogram_outputs', exist_ok=True)
        
        # Prepare data for comparison
        variations = []
        means = []
        medians = []
        stds = []
        counts = []
        
        for var_id, var_data in histogram_data.items():
            stats = var_data['stats']
            variations.append(f'Variation {len(variations)+1}')
            means.append(stats['mean'])
            medians.append(stats['median'])
            stds.append(stats['std'])
            counts.append(stats['count'])
        
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean comparison
        bars1 = ax1.bar(variations, means, color='steelblue', alpha=0.7)
        ax1.set_title('Mean Values Comparison', fontweight='bold')
        ax1.set_ylabel('Mean', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, mean in zip(bars1, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(means),
                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Median comparison
        bars2 = ax2.bar(variations, medians, color='orange', alpha=0.7)
        ax2.set_title('Median Values Comparison', fontweight='bold')
        ax2.set_ylabel('Median', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, median in zip(bars2, medians):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(medians),
                    f'{median:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Standard deviation comparison
        bars3 = ax3.bar(variations, stds, color='green', alpha=0.7)
        ax3.set_title('Standard Deviation Comparison', fontweight='bold')
        ax3.set_ylabel('Standard Deviation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, std in zip(bars3, stds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(stds),
                    f'{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Count comparison
        bars4 = ax4.bar(variations, counts, color='red', alpha=0.7)
        ax4.set_title('Data Count Comparison', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, count in zip(bars4, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'../outputs/histogram_outputs/statistical_comparison_{column}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Statistical comparison saved for {column}")
    
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
        query_results = self.execute_query_variations(variations)
        
        # Analyze numeric columns for histogram generation
        numeric_columns = []
        for result_key, result_df in query_results.items():
            if not result_df.empty:
                cols = self.analyze_numeric_columns(result_df)
                numeric_columns.extend(cols)
        
        # Remove duplicates
        numeric_columns = list(set(numeric_columns))
        
        if not numeric_columns:
            print(f"[WARNING] No suitable numeric columns found for histogram analysis")
            return
        
        print(f"[OK] Found {len(numeric_columns)} numeric columns for analysis")
        
        # Create histograms for each numeric column
        for column in numeric_columns:
            print(f"\n[HISTOGRAM] Creating histograms for column: {column}")
            
            # Create histogram variations
            histogram_data = self.create_histogram_variations(column, query_results)
            
            if histogram_data:
                # Create comparative histograms
                self.create_comparative_histograms(column, histogram_data)
                
                # Create statistical comparison
                self.create_statistical_comparison(column, histogram_data)
                
                # Generate insights
                insights = self.generate_insights(column, histogram_data)
                
                # Store insights
                self.save_insights({column: insights})
        
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
    parser.add_argument('data_source', help='Path to CSV file or SQLite database file')
    parser.add_argument('--query', help='SQL query to analyze')
    
    args = parser.parse_args()
    
    # Create generator
    generator = HistogramPermutationsGenerator(args.data_source)
    
    try:
        # Load data
        df = generator.load_data()
        if df is None:
            return
        
        # Use provided query or default
        if args.query:
            sql_query = args.query
        else:
            # Default query for testing
            if args.data_source.endswith('.csv'):
                # For CSV files, create a simple filter query
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    first_col = numeric_cols[0]
                    sql_query = f"SELECT * FROM data WHERE {first_col} > 0"
                else:
                    sql_query = "SELECT * FROM data"
            else:
                # Get the first table name from the database
                if generator.conn:
                    cursor = generator.conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    if tables:
                        first_table = tables[0][0]
                        sql_query = f"SELECT * FROM {first_table} WHERE id > 0"
                    else:
                        sql_query = "SELECT * FROM data WHERE id > 0"
                else:
                    sql_query = "SELECT * FROM data WHERE id > 0"
        
        print(f"[INFO] Using query: {sql_query}")
        
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