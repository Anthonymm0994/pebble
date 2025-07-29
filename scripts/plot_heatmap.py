#!/usr/bin/env python3
"""
Advanced Heatmap and Correlation Matrix Generator for SQLite Data
Creates sophisticated heatmaps, correlation matrices, and multivariate analysis plots.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "data.sqlite"  # Path to your SQLite database
QUERY = "SELECT amount, profit_margin, quantity, rating FROM sales LIMIT 100000"  # Your SELECT query here

# Plot settings
FIGURE_SIZE = (15, 12)  # Width, height in inches
OUTPUT_FILE = "heatmap_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "comprehensive"  # Options: "correlation", "heatmap", "comprehensive", "subplots"
SHOW_CORRELATION_MATRIX = True  # Show correlation matrix
SHOW_HEATMAP = True  # Show heatmap
SHOW_SCATTER_MATRIX = False  # Show scatter plot matrix
SHOW_PCA = True  # Show PCA analysis
SHOW_STATISTICS = True  # Show statistical information
SHOW_SIGNIFICANCE = True  # Show statistical significance

# Correlation analysis
CORRELATION_METHOD = "pearson"  # Options: "pearson", "spearman", "kendall"
SIGNIFICANCE_LEVEL = 0.05  # Significance level for tests
SHOW_P_VALUES = True  # Show p-values in correlation matrix

# Styling
TITLE = "Correlation Analysis and Heatmap"
COLOR_MAP = "RdBu_r"  # Color map for heatmap
ANNOTATE_VALUES = True  # Annotate correlation values
GRID_ALPHA = 0.3

# Advanced features
PERFORM_PCA = True  # Perform Principal Component Analysis
SHOW_LOADINGS = True  # Show PCA loadings
SHOW_VARIANCE = True  # Show explained variance
SHOW_OUTLIERS = True  # Detect and highlight outliers

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def calculate_correlation_matrix(df, method="pearson"):
    """Calculate correlation matrix with p-values."""
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)
    
    # Calculate p-values
    p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i != j:
                # Calculate correlation and p-value
                if method == "pearson":
                    corr, p_val = stats.pearsonr(df[i].dropna(), df[j].dropna())
                elif method == "spearman":
                    corr, p_val = stats.spearmanr(df[i].dropna(), df[j].dropna())
                elif method == "kendall":
                    corr, p_val = stats.kendalltau(df[i].dropna(), df[j].dropna())
                else:
                    corr, p_val = stats.pearsonr(df[i].dropna(), df[j].dropna())
                
                p_matrix.loc[i, j] = p_val
            else:
                p_matrix.loc[i, j] = 1.0
    
    return corr_matrix, p_matrix

def detect_outliers(df, threshold=3.0):
    """Detect outliers using Mahalanobis distance."""
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.dropna())
    
    # Calculate Mahalanobis distance
    mean = np.mean(df_scaled, axis=0)
    cov = np.cov(df_scaled.T)
    inv_cov = np.linalg.inv(cov)
    
    mahal_dist = []
    for point in df_scaled:
        diff = point - mean
        dist = np.sqrt(diff.dot(inv_cov).dot(diff))
        mahal_dist.append(dist)
    
    # Find outliers
    outlier_threshold = np.mean(mahal_dist) + threshold * np.std(mahal_dist)
    outliers = np.array(mahal_dist) > outlier_threshold
    
    return outliers

def perform_pca_analysis(df):
    """Perform Principal Component Analysis."""
    # Remove rows with missing values
    df_clean = df.dropna()
    
    if len(df_clean) < 2:
        return None
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return {
        'pca_result': pca_result,
        'loadings': pca.components_,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'feature_names': df_clean.columns.tolist(),
        'scaler': scaler
    }

def create_advanced_heatmap():
    """Create advanced heatmap and correlation analysis from SQLite data."""
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
        
        # Clean data
        df = df.dropna()
        
        if len(df) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Calculate correlation matrix
        corr_matrix, p_matrix = calculate_correlation_matrix(df, CORRELATION_METHOD)
        
        # Detect outliers
        outliers = detect_outliers(df) if SHOW_OUTLIERS else np.zeros(len(df), dtype=bool)
        
        # Perform PCA
        pca_result = perform_pca_analysis(df) if PERFORM_PCA else None
        
        # Create the plot
        if PLOT_TYPE == "subplots":
            fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
            ax_corr, ax_heatmap, ax_pca, ax_stats = axes.flatten()
        else:
            fig, ax_main = plt.subplots(figsize=FIGURE_SIZE)
        
        if PLOT_TYPE == "subplots":
            # Correlation matrix subplot
            if SHOW_CORRELATION_MATRIX:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=ANNOTATE_VALUES, 
                           cmap=COLOR_MAP, center=0, square=True, 
                           linewidths=0.5, cbar_kws={"shrink": 0.8},
                           ax=ax_corr)
                ax_corr.set_title(f'{CORRELATION_METHOD.title()} Correlation Matrix', 
                                fontsize=12, fontweight='bold')
            
            # Heatmap subplot
            if SHOW_HEATMAP:
                # Create a different type of heatmap (e.g., normalized data)
                df_normalized = (df - df.mean()) / df.std()
                sns.heatmap(df_normalized.T, cmap='viridis', 
                           cbar_kws={"shrink": 0.8}, ax=ax_heatmap)
                ax_heatmap.set_title('Normalized Data Heatmap', 
                                   fontsize=12, fontweight='bold')
            
            # PCA subplot
            if SHOW_PCA and pca_result:
                # Plot explained variance
                components = range(1, len(pca_result['explained_variance']) + 1)
                ax_pca.plot(components, pca_result['cumulative_variance'], 
                           'bo-', linewidth=2, markersize=8)
                ax_pca.set_xlabel('Number of Components', fontsize=10)
                ax_pca.set_ylabel('Cumulative Explained Variance', fontsize=10)
                ax_pca.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
                ax_pca.grid(True, alpha=GRID_ALPHA)
                
                # Add percentage labels
                for i, (comp, var) in enumerate(zip(components, pca_result['cumulative_variance'])):
                    ax_pca.annotate(f'{var:.1%}', (comp, var), 
                                  textcoords="offset points", xytext=(0,10), 
                                  ha='center', fontsize=9)
            
            # Statistics subplot
            if SHOW_STATISTICS:
                stats_text = f"""Dataset Statistics:
Variables: {len(df.columns)}
Observations: {len(df):,}
Outliers: {np.sum(outliers):,} ({np.sum(outliers)/len(df)*100:.1f}%)

Correlation Summary:
Strong (>0.7): {np.sum(np.abs(corr_matrix.values) > 0.7) - len(corr_matrix):,}
Moderate (0.3-0.7): {np.sum((np.abs(corr_matrix.values) > 0.3) & (np.abs(corr_matrix.values) <= 0.7)):,}
Weak (<0.3): {np.sum(np.abs(corr_matrix.values) < 0.3):,}"""
                
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                ax_stats.set_title('Statistical Summary', fontsize=12, fontweight='bold')
                ax_stats.axis('off')
        
        else:
            # Comprehensive plot
            ax = ax_main
            
            # Create correlation heatmap
            if SHOW_CORRELATION_MATRIX:
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Create heatmap
                sns.heatmap(corr_matrix, mask=mask, annot=ANNOTATE_VALUES, 
                           cmap=COLOR_MAP, center=0, square=True, 
                           linewidths=0.5, cbar_kws={"shrink": 0.8},
                           ax=ax, fmt='.2f')
                
                ax.set_title(TITLE, fontsize=14, fontweight='bold')
            
            # Add statistical information
            if SHOW_STATISTICS:
                # Calculate significant correlations
                significant_corr = (p_matrix < SIGNIFICANCE_LEVEL) & (corr_matrix != 1.0)
                num_significant = np.sum(significant_corr.values)
                
                stats_text = f"""Correlation Analysis:
Method: {CORRELATION_METHOD.title()}
Significance level: {SIGNIFICANCE_LEVEL}
Significant correlations: {num_significant}
Total observations: {len(df):,}
Variables: {len(df.columns)}"""
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Add significance indicators
            if SHOW_SIGNIFICANCE and SHOW_P_VALUES:
                # Add asterisks for significant correlations
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j and p_matrix.iloc[i, j] < SIGNIFICANCE_LEVEL:
                            # Add asterisk to significant correlations
                            if p_matrix.iloc[i, j] < 0.001:
                                marker = "***"
                            elif p_matrix.iloc[i, j] < 0.01:
                                marker = "**"
                            else:
                                marker = "*"
                            
                            # Position the marker
                            ax.text(j + 0.5, i + 0.5, marker, 
                                   ha='center', va='center', fontsize=8, 
                                   color='white', fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced heatmap saved as: {OUTPUT_FILE}")
        
        # Print detailed statistics
        if SHOW_STATISTICS:
            print("\n📊 Correlation Analysis:")
            print(f"Method: {CORRELATION_METHOD.title()}")
            print(f"Dataset: {len(df.columns)} variables, {len(df):,} observations")
            
            # Print correlation summary
            print("\n📈 Correlation Summary:")
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    var1 = corr_matrix.index[i]
                    var2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_matrix.iloc[i, j]
                    
                    significance = ""
                    if p_val < 0.001:
                        significance = "***"
                    elif p_val < 0.01:
                        significance = "**"
                    elif p_val < 0.05:
                        significance = "*"
                    
                    print(f"  {var1} vs {var2}: r = {corr_val:.3f} (p = {p_val:.3e}) {significance}")
        
        if pca_result:
            print(f"\n🔬 PCA Analysis:")
            print(f"Explained variance by component:")
            for i, var in enumerate(pca_result['explained_variance']):
                print(f"  PC{i+1}: {var:.1%}")
            print(f"Cumulative variance: {pca_result['cumulative_variance'][-1]:.1%}")
            
            print(f"\nPCA Loadings:")
            for i, component in enumerate(pca_result['loadings']):
                print(f"  PC{i+1}: {dict(zip(pca_result['feature_names'], component))}")
        
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
    create_advanced_heatmap() 