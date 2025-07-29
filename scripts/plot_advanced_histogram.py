#!/usr/bin/env python3
"""
Advanced Histogram Generator for SQLite Data
Creates sophisticated histograms with multiple visualization types for comprehensive data analysis.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "sample_data.sqlite"  # Path to your SQLite database
QUERY = "SELECT amount, profit_margin, quantity, rating FROM sales LIMIT 100000"  # Your SELECT query here

# Plot settings
COLUMN_NAME = "amount"  # Column to plot
FIGURE_SIZE = (15, 12)  # Width, height in inches
OUTPUT_FILE = "advanced_histogram_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "comprehensive"  # Options: "basic", "density", "cumulative", "log", "comprehensive"
SHOW_DENSITY = True  # Show density curve
SHOW_CUMULATIVE = True  # Show cumulative distribution
SHOW_LOG_SCALE = False  # Use log scale
SHOW_NORMAL_FIT = True  # Fit normal distribution
SHOW_KDE = True  # Show kernel density estimation
SHOW_STATISTICS = True  # Show statistical information
SHOW_ANOMALIES = True  # Highlight anomalies
SHOW_QUANTILES = True  # Show quantile lines

# Histogram settings
BIN_COUNT = 50  # Number of bins
BIN_TYPE = "auto"  # Options: "auto", "sturges", "fd", "scott", "sqrt"
SHOW_BIN_EDGES = True  # Show bin edge lines
SHOW_BIN_VALUES = False  # Show bin count values

# Styling
TITLE = "Advanced Histogram Analysis"
X_LABEL = "Values"
Y_LABEL = "Frequency"
COLOR = "steelblue"
ALPHA = 0.7  # Transparency (0-1)
GRID_ALPHA = 0.3

# Advanced features
DETECT_ANOMALIES = True  # Detect statistical anomalies
FIT_DISTRIBUTIONS = True  # Fit multiple distributions
SHOW_CONFIDENCE = True  # Show confidence intervals
SHOW_DISTRIBUTION_COMPARISON = True  # Compare with theoretical distributions

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def detect_histogram_anomalies(values, threshold=2.0):
    """Detect anomalies using z-score method."""
    mean = np.mean(values)
    std = np.std(values)
    z_scores = np.abs((values - mean) / std)
    anomalies = z_scores > threshold
    return anomalies

def fit_normal_distribution(values):
    """Fit normal distribution to data."""
    # Calculate parameters
    mu = np.mean(values)
    sigma = np.std(values)
    
    # Generate x values for plotting
    x = np.linspace(values.min(), values.max(), 100)
    
    # Calculate normal distribution
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Scale to match histogram
    hist, bins = np.histogram(values, bins=BIN_COUNT, density=True)
    y_scaled = y * len(values) * (bins[1] - bins[0])
    
    return x, y_scaled, mu, sigma

def fit_multiple_distributions(values):
    """Fit multiple distributions to data."""
    distributions = {}
    
    # Normal distribution
    mu, sigma = stats.norm.fit(values)
    distributions['normal'] = {'params': (mu, sigma), 'name': 'Normal'}
    
    # Log-normal distribution
    try:
        shape, loc, scale = stats.lognorm.fit(values)
        distributions['lognormal'] = {'params': (shape, loc, scale), 'name': 'Log-Normal'}
    except:
        pass
    
    # Exponential distribution
    try:
        loc, scale = stats.expon.fit(values)
        distributions['exponential'] = {'params': (loc, scale), 'name': 'Exponential'}
    except:
        pass
    
    # Gamma distribution
    try:
        a, loc, scale = stats.gamma.fit(values)
        distributions['gamma'] = {'params': (a, loc, scale), 'name': 'Gamma'}
    except:
        pass
    
    return distributions

def calculate_histogram_statistics(values):
    """Calculate comprehensive histogram statistics."""
    stats_dict = {
        'count': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'cv': np.std(values) / np.mean(values),  # Coefficient of variation
        'normality_p': stats.normaltest(values)[1] if len(values) > 8 else None
    }
    
    return stats_dict

def create_advanced_histogram():
    """Create advanced histogram from SQLite data."""
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
        
        # Check if column exists
        if COLUMN_NAME not in df.columns:
            print(f"Error: Column '{COLUMN_NAME}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data
        df = df.dropna(subset=[COLUMN_NAME])
        values = df[COLUMN_NAME].values
        
        if len(values) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Calculate statistics
        stats_dict = calculate_histogram_statistics(values)
        
        # Detect anomalies
        anomalies = detect_histogram_anomalies(values) if DETECT_ANOMALIES else np.zeros(len(values), dtype=bool)
        
        # Fit distributions
        distributions = fit_multiple_distributions(values) if FIT_DISTRIBUTIONS else {}
        
        # Create the plot
        if PLOT_TYPE == "comprehensive":
            fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
            ax_main, ax_density, ax_cumulative, ax_log, ax_dist, ax_stats = axes.flatten()
        else:
            fig, ax_main = plt.subplots(figsize=FIGURE_SIZE)
        
        # Main histogram
        n, bins, patches = ax_main.hist(values, bins=BIN_COUNT, color=COLOR, alpha=ALPHA, 
                                       edgecolor='black', linewidth=0.5, density=False)
        
        # Add normal fit
        if SHOW_NORMAL_FIT:
            x_fit, y_fit, mu, sigma = fit_normal_distribution(values)
            ax_main.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, 
                        label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
        
        # Add KDE
        if SHOW_KDE:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_kde = np.linspace(values.min(), values.max(), 100)
            y_kde = kde(x_kde) * len(values) * (bins[1] - bins[0])
            ax_main.plot(x_kde, y_kde, 'g-', linewidth=2, alpha=0.8, label='KDE')
        
        # Add quantile lines
        if SHOW_QUANTILES:
            q25, q75 = stats_dict['q25'], stats_dict['q75']
            ax_main.axvline(q25, color='orange', linestyle='--', alpha=0.8, label=f'Q25: {q25:.2f}')
            ax_main.axvline(stats_dict['median'], color='red', linestyle='--', alpha=0.8, label=f'Median: {stats_dict["median"]:.2f}')
            ax_main.axvline(q75, color='orange', linestyle='--', alpha=0.8, label=f'Q75: {q75:.2f}')
        
        ax_main.set_title('Main Histogram', fontsize=12, fontweight='bold')
        ax_main.set_xlabel(X_LABEL, fontsize=10)
        ax_main.set_ylabel(Y_LABEL, fontsize=10)
        ax_main.grid(True, alpha=GRID_ALPHA)
        ax_main.legend()
        
        if PLOT_TYPE == "comprehensive":
            # Density histogram
            if SHOW_DENSITY:
                ax_density.hist(values, bins=BIN_COUNT, color='lightcoral', alpha=ALPHA, 
                              edgecolor='black', linewidth=0.5, density=True)
                ax_density.set_title('Density Histogram', fontsize=12, fontweight='bold')
                ax_density.set_xlabel(X_LABEL, fontsize=10)
                ax_density.set_ylabel('Density', fontsize=10)
                ax_density.grid(True, alpha=GRID_ALPHA)
            
            # Cumulative histogram
            if SHOW_CUMULATIVE:
                ax_cumulative.hist(values, bins=BIN_COUNT, color='lightgreen', alpha=ALPHA, 
                                 edgecolor='black', linewidth=0.5, cumulative=True, density=True)
                ax_cumulative.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
                ax_cumulative.set_xlabel(X_LABEL, fontsize=10)
                ax_cumulative.set_ylabel('Cumulative Probability', fontsize=10)
                ax_cumulative.grid(True, alpha=GRID_ALPHA)
            
            # Log scale histogram
            if SHOW_LOG_SCALE:
                ax_log.hist(values, bins=BIN_COUNT, color='lightblue', alpha=ALPHA, 
                           edgecolor='black', linewidth=0.5)
                ax_log.set_yscale('log')
                ax_log.set_title('Log Scale Histogram', fontsize=12, fontweight='bold')
                ax_log.set_xlabel(X_LABEL, fontsize=10)
                ax_log.set_ylabel('Frequency (log)', fontsize=10)
                ax_log.grid(True, alpha=GRID_ALPHA)
            
            # Distribution comparison
            if SHOW_DISTRIBUTION_COMPARISON and distributions:
                # Plot histogram
                ax_dist.hist(values, bins=BIN_COUNT, color='lightyellow', alpha=ALPHA, 
                           edgecolor='black', linewidth=0.5, density=True, label='Data')
                
                # Plot fitted distributions
                x_plot = np.linspace(values.min(), values.max(), 100)
                colors = ['red', 'blue', 'green', 'purple']
                
                for i, (dist_name, dist_info) in enumerate(distributions.items()):
                    if dist_name == 'normal':
                        mu, sigma = dist_info['params']
                        y = stats.norm.pdf(x_plot, mu, sigma)
                    elif dist_name == 'lognormal':
                        shape, loc, scale = dist_info['params']
                        y = stats.lognorm.pdf(x_plot, shape, loc, scale)
                    elif dist_name == 'exponential':
                        loc, scale = dist_info['params']
                        y = stats.expon.pdf(x_plot, loc, scale)
                    elif dist_name == 'gamma':
                        a, loc, scale = dist_info['params']
                        y = stats.gamma.pdf(x_plot, a, loc, scale)
                    else:
                        continue
                    
                    ax_dist.plot(x_plot, y, color=colors[i % len(colors)], linewidth=2, 
                               alpha=0.8, label=dist_info['name'])
                
                ax_dist.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
                ax_dist.set_xlabel(X_LABEL, fontsize=10)
                ax_dist.set_ylabel('Density', fontsize=10)
                ax_dist.grid(True, alpha=GRID_ALPHA)
                ax_dist.legend()
            
            # Statistics subplot
            if SHOW_STATISTICS:
                stats_text = f"""Histogram Statistics:
Count: {stats_dict['count']:,}
Mean: {stats_dict['mean']:.2f}
Median: {stats_dict['median']:.2f}
Std: {stats_dict['std']:.2f}
Skewness: {stats_dict['skewness']:.3f}
Kurtosis: {stats_dict['kurtosis']:.3f}
Range: {stats_dict['range']:.2f}
IQR: {stats_dict['iqr']:.2f}
CV: {stats_dict['cv']:.3f}"""
                
                if stats_dict['normality_p']:
                    stats_text += f"\nNormality p: {stats_dict['normality_p']:.3e}"
                
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                ax_stats.set_title('Statistical Summary', fontsize=12, fontweight='bold')
                ax_stats.axis('off')
        
        # Add statistical information to main plot
        if SHOW_STATISTICS:
            stats_text = f"Mean: {stats_dict['mean']:.2f}\nStd: {stats_dict['std']:.2f}\nSkewness: {stats_dict['skewness']:.3f}"
            ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add anomaly information
        if SHOW_ANOMALIES and np.any(anomalies):
            anomaly_text = f"Anomalies: {np.sum(anomalies)} ({np.sum(anomalies)/len(values)*100:.1f}%)"
            ax_main.text(0.02, 0.02, anomaly_text, transform=ax_main.transAxes, 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced histogram saved as: {OUTPUT_FILE}")
        
        # Print detailed statistics
        if SHOW_STATISTICS:
            print("\n📊 Histogram Analysis Statistics:")
            print(f"Count: {stats_dict['count']:,}")
            print(f"Mean: {stats_dict['mean']:.2f}")
            print(f"Median: {stats_dict['median']:.2f}")
            print(f"Std: {stats_dict['std']:.2f}")
            print(f"Skewness: {stats_dict['skewness']:.3f}")
            print(f"Kurtosis: {stats_dict['kurtosis']:.3f}")
            print(f"Range: {stats_dict['range']:.2f}")
            print(f"IQR: {stats_dict['iqr']:.2f}")
            print(f"Coefficient of Variation: {stats_dict['cv']:.3f}")
            if stats_dict['normality_p']:
                print(f"Normality test p-value: {stats_dict['normality_p']:.3e}")
            print(f"Anomalies: {np.sum(anomalies):,} ({np.sum(anomalies)/len(values)*100:.1f}%)")
        
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
    create_advanced_histogram() 