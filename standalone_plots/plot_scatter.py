#!/usr/bin/env python3
"""
Advanced Scatter Plot Generator for SQLite Data
Creates sophisticated scatter plots with correlation analysis, regression lines, and multiple visualization options.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "data.sqlite"  # Path to your SQLite database
QUERY = "SELECT x, y_positive FROM correlations LIMIT 5000"  # Your SELECT query here

# Plot settings
X_COLUMN = "x"  # Column for x-axis values
Y_COLUMN = "y_positive"  # Column for y-axis values
SIZE_COLUMN = None  # Column for point sizes (optional)
COLOR_COLUMN = None  # Column for point colors (optional)
FIGURE_SIZE = (12, 8)  # Width, height in inches
OUTPUT_FILE = "scatter_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "correlation"  # Options: "basic", "correlation", "regression", "density", "hexbin"
SHOW_REGRESSION_LINE = True  # Add regression line
SHOW_CORRELATION = True  # Show correlation coefficient
SHOW_STATS = True  # Show statistical information
SHOW_DENSITY = False  # Show density contours
SHOW_HEXBIN = False  # Use hexbin instead of scatter
SHOW_CONFIDENCE_INTERVALS = True  # Show confidence intervals for regression

# Styling
TITLE = "Scatter Plot with Correlation Analysis"
X_LABEL = "X Values"
Y_LABEL = "Y Values"
POINT_SIZE = 30
ALPHA = 0.6  # Transparency (0-1)
COLOR_MAP = "viridis"  # Color map for density/color coding
GRID_ALPHA = 0.3

# Advanced features
ANNOTATE_OUTLIERS = True  # Highlight statistical outliers
SHOW_TREND_LINE = True  # Show trend line
SHOW_QUADRANT_LINES = False  # Show quadrant dividing lines
SHOW_DISTRIBUTION_MARGINS = False  # Show distribution on margins

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def calculate_correlation_stats(x, y):
    """Calculate comprehensive correlation statistics."""
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Kendall correlation
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    # R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_err': std_err
    }

def detect_outliers(x, y, threshold=2.0):
    """Detect outliers using Mahalanobis distance."""
    # Combine x and y for multivariate outlier detection
    data = np.column_stack([x, y])
    
    # Calculate Mahalanobis distance
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)
    
    mahal_dist = []
    for point in data:
        diff = point - mean
        dist = np.sqrt(diff.dot(inv_cov).dot(diff))
        mahal_dist.append(dist)
    
    # Find outliers
    outlier_threshold = np.mean(mahal_dist) + threshold * np.std(mahal_dist)
    outliers = np.array(mahal_dist) > outlier_threshold
    
    return outliers

def create_advanced_scatter():
    """Create advanced scatter plot from SQLite data."""
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
        if X_COLUMN not in df.columns:
            print(f"Error: X column '{X_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
            
        if Y_COLUMN not in df.columns:
            print(f"Error: Y column '{Y_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data
        df = df.dropna(subset=[X_COLUMN, Y_COLUMN])
        x = df[X_COLUMN].values
        y = df[Y_COLUMN].values
        
        if len(x) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Calculate statistics
        stats_dict = calculate_correlation_stats(x, y)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Detect outliers
        outliers = detect_outliers(x, y) if ANNOTATE_OUTLIERS else np.zeros(len(x), dtype=bool)
        
        # Create scatter plot based on type
        if PLOT_TYPE == "hexbin":
            # Hexbin plot
            hb = ax.hexbin(x, y, gridsize=50, cmap=COLOR_MAP, alpha=ALPHA)
            plt.colorbar(hb, ax=ax, label='Point Density')
        elif PLOT_TYPE == "density":
            # Density scatter plot
            scatter = ax.scatter(x, y, c=stats.gaussian_kde(np.vstack([x, y]))(np.vstack([x, y])), 
                               s=POINT_SIZE, alpha=ALPHA, cmap=COLOR_MAP, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Density')
        else:
            # Regular scatter plot
            # Plot regular points
            regular_mask = ~outliers
            if np.any(regular_mask):
                ax.scatter(x[regular_mask], y[regular_mask], s=POINT_SIZE, alpha=ALPHA, 
                          color='blue', edgecolors='black', linewidth=0.5, label='Regular Points')
            
            # Plot outliers
            if np.any(outliers):
                ax.scatter(x[outliers], y[outliers], s=POINT_SIZE*1.5, alpha=0.8, 
                          color='red', edgecolors='black', linewidth=1, label='Outliers')
        
        # Add regression line
        if SHOW_REGRESSION_LINE:
            # Fit regression line
            slope = stats_dict['slope']
            intercept = stats_dict['intercept']
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = slope * x_range + intercept
            
            # Plot regression line
            ax.plot(x_range, y_pred, color='red', linewidth=2, alpha=0.8, 
                   label=f'Regression (R² = {stats_dict["r_squared"]:.3f})')
            
            # Add confidence intervals
            if SHOW_CONFIDENCE_INTERVALS:
                # Calculate confidence intervals
                y_pred_full = slope * x + intercept
                residuals = y - y_pred_full
                mse = np.sum(residuals**2) / (len(x) - 2)
                
                # Standard error of regression
                se = np.sqrt(mse * (1/len(x) + (x_range - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
                
                # 95% confidence interval
                t_value = stats.t.ppf(0.975, len(x) - 2)
                ci = t_value * se
                
                ax.fill_between(x_range, y_pred - ci, y_pred + ci, 
                              alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Add trend line (loess-like)
        if SHOW_TREND_LINE:
            # Simple moving average trend
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            
            # Calculate moving average
            window_size = max(1, len(x) // 20)
            trend_y = np.convolve(y_sorted, np.ones(window_size)/window_size, mode='same')
            
            ax.plot(x_sorted, trend_y, color='green', linewidth=2, alpha=0.7, 
                   label='Trend Line (Moving Avg)')
        
        # Add quadrant lines
        if SHOW_QUADRANT_LINES:
            x_mean, y_mean = np.mean(x), np.mean(y)
            ax.axhline(y=y_mean, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=x_mean, color='gray', linestyle='--', alpha=0.5)
        
        # Add density contours
        if SHOW_DENSITY:
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(x, y, bins=20)
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters)
            
            # Plot contours
            contour = ax.contour(X, Y, H.T, colors='black', alpha=0.5, linewidths=1)
            ax.clabel(contour, inline=True, fontsize=8)
        
        # Customize plot
        ax.set_title(TITLE, fontsize=14, fontweight='bold')
        ax.set_xlabel(X_LABEL, fontsize=12)
        ax.set_ylabel(Y_LABEL, fontsize=12)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
        
        # Add statistical information
        if SHOW_STATS:
            stats_text = f"""Correlation Statistics:
Pearson r: {stats_dict['pearson_r']:.3f} (p={stats_dict['pearson_p']:.3e})
Spearman ρ: {stats_dict['spearman_r']:.3f} (p={stats_dict['spearman_p']:.3e})
Kendall τ: {stats_dict['kendall_tau']:.3f} (p={stats_dict['kendall_p']:.3e})
R²: {stats_dict['r_squared']:.3f}
Slope: {stats_dict['slope']:.3f} ± {stats_dict['std_err']:.3f}
n = {len(x):,} points"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add correlation interpretation
        if SHOW_CORRELATION:
            r = stats_dict['pearson_r']
            if abs(r) >= 0.8:
                strength = "Very Strong"
            elif abs(r) >= 0.6:
                strength = "Strong"
            elif abs(r) >= 0.4:
                strength = "Moderate"
            elif abs(r) >= 0.2:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            direction = "Positive" if r > 0 else "Negative"
            
            correlation_text = f"Correlation: {strength} {direction}"
            ax.text(0.02, 0.02, correlation_text, transform=ax.transAxes, 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced scatter plot saved as: {OUTPUT_FILE}")
        
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
    create_advanced_scatter() 