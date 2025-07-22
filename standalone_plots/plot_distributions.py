#!/usr/bin/env python3
"""
Advanced Distribution Plot Generator for SQLite Data
Creates sophisticated distribution plots including box plots, violin plots, and statistical analysis.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "data.sqlite"  # Path to your SQLite database
QUERY = "SELECT category, amount FROM sales LIMIT 100000"  # Your SELECT query here

# Plot settings
CATEGORY_COLUMN = "category"  # Column containing categories
VALUE_COLUMN = "amount"  # Column containing values to plot
FIGURE_SIZE = (15, 10)  # Width, height in inches
OUTPUT_FILE = "distribution_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "comprehensive"  # Options: "box", "violin", "comprehensive", "subplots"
SHOW_BOX_PLOT = True  # Show box plot
SHOW_VIOLIN_PLOT = True  # Show violin plot
SHOW_MEAN_POINTS = True  # Show mean points
SHOW_OUTLIERS = True  # Show outliers
SHOW_STATISTICS = True  # Show statistical information
SHOW_SIGNIFICANCE = True  # Show statistical significance tests
SHOW_DISTRIBUTION = True  # Show distribution curves

# Statistical analysis
PERFORM_ANOVA = True  # Perform one-way ANOVA
PERFORM_TESTS = True  # Perform pairwise tests
CONFIDENCE_LEVEL = 0.95  # Confidence level for tests

# Styling
TITLE = "Distribution Analysis by Category"
X_LABEL = "Category"
Y_LABEL = "Amount ($)"
COLOR_PALETTE = "Set3"  # Color palette
ALPHA = 0.7  # Transparency (0-1)
POINT_SIZE = 50
GRID_ALPHA = 0.3

# Advanced features
SHOW_DENSITY = True  # Show density estimation
SHOW_QUANTILES = True  # Show quantile information
SHOW_NORMALITY = True  # Test for normality
SHOW_EFFECT_SIZE = True  # Calculate effect sizes

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def calculate_group_statistics(df, category_col, value_col):
    """Calculate comprehensive statistics for each group."""
    stats_dict = {}
    
    for category in df[category_col].unique():
        group_data = df[df[category_col] == category][value_col].dropna()
        
        if len(group_data) > 0:
            stats_dict[category] = {
                'count': len(group_data),
                'mean': np.mean(group_data),
                'median': np.median(group_data),
                'std': np.std(group_data),
                'min': np.min(group_data),
                'max': np.max(group_data),
                'q25': np.percentile(group_data, 25),
                'q75': np.percentile(group_data, 75),
                'iqr': np.percentile(group_data, 75) - np.percentile(group_data, 25),
                'skewness': stats.skew(group_data),
                'kurtosis': stats.kurtosis(group_data),
                'cv': np.std(group_data) / np.mean(group_data),  # Coefficient of variation
                'normality_p': stats.normaltest(group_data)[1] if len(group_data) > 8 else None
            }
    
    return stats_dict

def perform_anova_test(df, category_col, value_col):
    """Perform one-way ANOVA test."""
    groups = [group[value_col].values for name, group in df.groupby(category_col) 
              if len(group[value_col].dropna()) > 0]
    
    if len(groups) < 2:
        return None
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate effect size (eta-squared)
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((val - grand_mean)**2 for g in groups for val in g)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < (1 - CONFIDENCE_LEVEL)
    }

def perform_pairwise_tests(df, category_col, value_col):
    """Perform pairwise statistical tests."""
    categories = df[category_col].unique()
    results = {}
    
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            group1 = df[df[category_col] == cat1][value_col].dropna()
            group2 = df[df[category_col] == cat2][value_col].dropna()
            
            if len(group1) > 0 and len(group2) > 0:
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                    (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                results[f"{cat1}_vs_{cat2}"] = {
                    'u_statistic': u_stat,
                    'p_value': u_p,
                    'cohens_d': cohens_d,
                    'significant': u_p < (1 - CONFIDENCE_LEVEL),
                    'effect_size': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'
                }
    
    return results

def create_advanced_distribution():
    """Create advanced distribution plot from SQLite data."""
    try:
        # Connect to database
        print(f"Connecting to database: {DATABASE_PATH}")
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Execute query and load data
        print(f"Executing query: {QUERY}")
        df = pd.read_sql_query(QUERY, conn)
        conn.close()
        
        if df.empty:
            print("Warning: Query returned no data!")
            return
        
        # Check if columns exist
        if CATEGORY_COLUMN not in df.columns:
            print(f"Error: Category column '{CATEGORY_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
            
        if VALUE_COLUMN not in df.columns:
            print(f"Error: Value column '{VALUE_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data
        df = df.dropna(subset=[CATEGORY_COLUMN, VALUE_COLUMN])
        
        if len(df) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Calculate statistics
        group_stats = calculate_group_statistics(df, CATEGORY_COLUMN, VALUE_COLUMN)
        
        # Perform statistical tests
        anova_result = perform_anova_test(df, CATEGORY_COLUMN, VALUE_COLUMN) if PERFORM_ANOVA else None
        pairwise_results = perform_pairwise_tests(df, CATEGORY_COLUMN, VALUE_COLUMN) if PERFORM_TESTS else None
        
        # Create the plot
        if PLOT_TYPE == "subplots":
            fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
            ax_box, ax_violin, ax_stats, ax_tests = axes.flatten()
        else:
            fig, ax_main = plt.subplots(figsize=FIGURE_SIZE)
        
        # Set color palette
        colors = sns.color_palette(COLOR_PALETTE, n_colors=len(df[CATEGORY_COLUMN].unique()))
        
        if PLOT_TYPE == "subplots":
            # Box plot subplot
            if SHOW_BOX_PLOT:
                box_plot = ax_box.boxplot([df[df[CATEGORY_COLUMN] == cat][VALUE_COLUMN].values 
                                         for cat in df[CATEGORY_COLUMN].unique()],
                                        labels=df[CATEGORY_COLUMN].unique(),
                                        patch_artist=True, showmeans=SHOW_MEAN_POINTS,
                                        showfliers=SHOW_OUTLIERS)
                
                # Color the boxes
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(ALPHA)
                
                ax_box.set_title('Box Plot by Category', fontsize=12, fontweight='bold')
                ax_box.set_ylabel(Y_LABEL, fontsize=10)
                ax_box.grid(True, alpha=GRID_ALPHA)
            
            # Violin plot subplot
            if SHOW_VIOLIN_PLOT:
                violin_parts = ax_violin.violinplot([df[df[CATEGORY_COLUMN] == cat][VALUE_COLUMN].values 
                                                   for cat in df[CATEGORY_COLUMN].unique()],
                                                  positions=range(len(df[CATEGORY_COLUMN].unique())),
                                                  showmeans=SHOW_MEAN_POINTS)
                
                # Set labels manually
                ax_violin.set_xticks(range(len(df[CATEGORY_COLUMN].unique())))
                ax_violin.set_xticklabels(df[CATEGORY_COLUMN].unique())
                
                # Color the violins
                for pc, color in zip(violin_parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(ALPHA)
                
                ax_violin.set_title('Violin Plot by Category', fontsize=12, fontweight='bold')
                ax_violin.set_ylabel(Y_LABEL, fontsize=10)
                ax_violin.grid(True, alpha=GRID_ALPHA)
            
            # Statistics subplot
            if SHOW_STATISTICS:
                categories = list(group_stats.keys())
                means = [group_stats[cat]['mean'] for cat in categories]
                stds = [group_stats[cat]['std'] for cat in categories]
                
                x_pos = np.arange(len(categories))
                bars = ax_stats.bar(x_pos, means, yerr=stds, capsize=5, 
                                  color=colors[:len(categories)], alpha=ALPHA)
                
                ax_stats.set_title('Mean ± Standard Deviation', fontsize=12, fontweight='bold')
                ax_stats.set_ylabel(Y_LABEL, fontsize=10)
                ax_stats.set_xticks(x_pos)
                ax_stats.set_xticklabels(categories, rotation=45, ha='right')
                ax_stats.grid(True, alpha=GRID_ALPHA)
            
            # Statistical tests subplot
            if SHOW_SIGNIFICANCE and anova_result:
                test_text = f"""ANOVA Results:
F-statistic: {anova_result['f_statistic']:.3f}
p-value: {anova_result['p_value']:.3e}
η² (effect size): {anova_result['eta_squared']:.3f}
Significant: {'Yes' if anova_result['significant'] else 'No'}"""
                
                ax_tests.text(0.05, 0.95, test_text, transform=ax_tests.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax_tests.set_title('Statistical Tests', fontsize=12, fontweight='bold')
                ax_tests.axis('off')
        
        else:
            # Comprehensive plot
            ax = ax_main
            
            # Create violin plot
            if SHOW_VIOLIN_PLOT:
                violin_parts = ax.violinplot([df[df[CATEGORY_COLUMN] == cat][VALUE_COLUMN].values 
                                           for cat in df[CATEGORY_COLUMN].unique()],
                                          positions=range(len(df[CATEGORY_COLUMN].unique())),
                                          showmeans=SHOW_MEAN_POINTS)
                
                # Set labels manually
                ax.set_xticks(range(len(df[CATEGORY_COLUMN].unique())))
                ax.set_xticklabels(df[CATEGORY_COLUMN].unique())
                
                # Color the violins
                for pc, color in zip(violin_parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(ALPHA)
            
            # Add box plot overlay
            if SHOW_BOX_PLOT:
                box_plot = ax.boxplot([df[df[CATEGORY_COLUMN] == cat][VALUE_COLUMN].values 
                                     for cat in df[CATEGORY_COLUMN].unique()],
                                    labels=df[CATEGORY_COLUMN].unique(),
                                    patch_artist=False, showmeans=False,
                                    showfliers=SHOW_OUTLIERS, widths=0.3)
                
                # Style the box plot elements
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
                    plt.setp(box_plot[element], color='black', linewidth=1.5)
            
            ax.set_title(TITLE, fontsize=14, fontweight='bold')
            ax.set_xlabel(X_LABEL, fontsize=12)
            ax.set_ylabel(Y_LABEL, fontsize=12)
            ax.grid(True, alpha=GRID_ALPHA)
            
            # Add statistical information
            if SHOW_STATISTICS:
                stats_text = f"""Group Statistics:
Total groups: {len(group_stats)}
Total observations: {len(df):,}
"""
                
                for cat, stats in group_stats.items():
                    stats_text += f"\n{cat}: n={stats['count']:,}, μ={stats['mean']:.1f}, σ={stats['std']:.1f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Add ANOVA results
            if SHOW_SIGNIFICANCE and anova_result:
                significance_text = f"ANOVA: F={anova_result['f_statistic']:.2f}, p={anova_result['p_value']:.3e}"
                if anova_result['significant']:
                    significance_text += " (Significant)"
                else:
                    significance_text += " (Not Significant)"
                
                ax.text(0.02, 0.02, significance_text, transform=ax.transAxes, 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen' if anova_result['significant'] else 'lightcoral', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced distribution plot saved as: {OUTPUT_FILE}")
        
        # Print detailed statistics
        if SHOW_STATISTICS:
            print("\n📊 Detailed Statistics:")
            for category, stats in group_stats.items():
                print(f"\n{category}:")
                print(f"  Count: {stats['count']:,}")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  IQR: {stats['iqr']:.2f}")
                print(f"  Skewness: {stats['skewness']:.3f}")
                print(f"  Kurtosis: {stats['kurtosis']:.3f}")
                if stats['normality_p']:
                    print(f"  Normality test p-value: {stats['normality_p']:.3e}")
        
        if anova_result:
            print(f"\n📈 ANOVA Results:")
            print(f"  F-statistic: {anova_result['f_statistic']:.3f}")
            print(f"  p-value: {anova_result['p_value']:.3e}")
            print(f"  Effect size (η²): {anova_result['eta_squared']:.3f}")
            print(f"  Significant: {'Yes' if anova_result['significant'] else 'No'}")
        
        # Show plot (optional - comment out if you don't want to display)
        plt.show()
        
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        print("Check that the database file exists and the query is valid.")
    except FileNotFoundError:
        print(f"Database file not found: {DATABASE_PATH}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    create_advanced_distribution() 