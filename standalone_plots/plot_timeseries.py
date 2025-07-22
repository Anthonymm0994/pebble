#!/usr/bin/env python3
"""
Advanced Time Series Plot Generator for SQLite Data
Creates sophisticated time series plots with trend analysis, seasonality detection, and multiple visualization options.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import seaborn as sns
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "data.sqlite"  # Path to your SQLite database
QUERY = "SELECT timestamp, temperature FROM time_series LIMIT 1000"  # Your SELECT query here

# Plot settings
TIME_COLUMN = "timestamp"  # Column containing time/datetime values
VALUE_COLUMN = "temperature"  # Column containing values to plot
FIGURE_SIZE = (15, 10)  # Width, height in inches
OUTPUT_FILE = "timeseries_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "comprehensive"  # Options: "basic", "trend", "seasonal", "comprehensive", "subplots"
SHOW_TREND_LINE = True  # Add trend line
SHOW_MOVING_AVERAGE = True  # Show moving average
SHOW_SEASONALITY = True  # Show seasonality analysis
SHOW_FORECAST = False  # Show simple forecast
SHOW_ANOMALIES = True  # Highlight anomalies
SHOW_STATISTICS = True  # Show statistical information
SHOW_SUBPLOTS = False  # Show multiple subplots

# Moving average settings
MA_WINDOW = 24  # Moving average window size
TREND_WINDOW = 168  # Trend smoothing window (7 days for hourly data)

# Styling
TITLE = "Time Series Analysis"
X_LABEL = "Time"
Y_LABEL = "Temperature (°C)"
LINE_WIDTH = 1.5
ALPHA = 0.7  # Transparency (0-1)
COLOR_PRIMARY = "blue"
COLOR_TREND = "red"
COLOR_MA = "green"
COLOR_ANOMALY = "orange"

# Advanced features
DETECT_SEASONALITY = True  # Detect seasonal patterns
CALCULATE_STATS = True  # Calculate comprehensive statistics
SHOW_CONFIDENCE_BANDS = True  # Show confidence bands
SHOW_DISTRIBUTION = False  # Show value distribution

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def detect_anomalies(values, threshold=2.0):
    """Detect anomalies using z-score method."""
    mean = np.mean(values)
    std = np.std(values)
    z_scores = np.abs((values - mean) / std)
    anomalies = z_scores > threshold
    return anomalies

def calculate_trend(x, y):
    """Calculate linear trend."""
    # Convert time to numeric for regression
    x_numeric = np.arange(len(x))
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y)
    
    # Calculate trend line
    trend_line = slope * x_numeric + intercept
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'trend_line': trend_line,
        'trend_direction': 'increasing' if slope > 0 else 'decreasing'
    }

def detect_seasonality(values, period=24):
    """Detect seasonality using autocorrelation."""
    if len(values) < 2 * period:
        return None
    
    # Calculate autocorrelation
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[len(values)-1:] / autocorr[len(values)-1]
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, len(autocorr)-1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            peaks.append(i)
    
    # Find the most prominent seasonal period
    if peaks:
        seasonal_period = peaks[np.argmax([autocorr[p] for p in peaks])]
        seasonal_strength = autocorr[seasonal_period]
        return {
            'period': seasonal_period,
            'strength': seasonal_strength,
            'autocorr': autocorr
        }
    
    return None

def calculate_statistics(values):
    """Calculate comprehensive time series statistics."""
    stats_dict = {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values),
        'cv': np.std(values) / np.mean(values),  # Coefficient of variation
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25)
    }
    
    return stats_dict

def create_advanced_timeseries():
    """Create advanced time series plot from SQLite data."""
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
        if TIME_COLUMN not in df.columns:
            print(f"Error: Time column '{TIME_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
            
        if VALUE_COLUMN not in df.columns:
            print(f"Error: Value column '{VALUE_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data
        df = df.dropna(subset=[TIME_COLUMN, VALUE_COLUMN])
        df = df.sort_values(TIME_COLUMN)
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[TIME_COLUMN]):
            df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
        
        times = df[TIME_COLUMN].values
        values = df[VALUE_COLUMN].values
        
        if len(values) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Calculate statistics
        stats_dict = calculate_statistics(values)
        
        # Detect anomalies
        anomalies = detect_anomalies(values) if SHOW_ANOMALIES else np.zeros(len(values), dtype=bool)
        
        # Calculate trend
        trend_dict = calculate_trend(times, values) if SHOW_TREND_LINE else None
        
        # Detect seasonality
        seasonality = detect_seasonality(values) if DETECT_SEASONALITY else None
        
        # Create the plot
        if SHOW_SUBPLOTS:
            fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
            ax_main, ax_trend, ax_seasonal = axes
        else:
            fig, ax_main = plt.subplots(figsize=FIGURE_SIZE)
        
        # Main time series plot
        ax_main.plot(times, values, color=COLOR_PRIMARY, linewidth=LINE_WIDTH, 
                    alpha=ALPHA, label='Original Data')
        
        # Add moving average
        if SHOW_MOVING_AVERAGE:
            window_size = min(MA_WINDOW, len(values) // 10)
            if window_size > 1:
                ma_values = pd.Series(values).rolling(window=window_size, center=True).mean()
                ax_main.plot(times, ma_values, color=COLOR_MA, linewidth=2, 
                           alpha=0.8, label=f'Moving Average (window={window_size})')
        
        # Add trend line
        if SHOW_TREND_LINE and trend_dict:
            trend_times = np.arange(len(times))
            ax_main.plot(times, trend_dict['trend_line'], color=COLOR_TREND, 
                        linewidth=2, alpha=0.8, 
                        label=f'Trend (R² = {trend_dict["r_squared"]:.3f})')
            
            # Add confidence bands
            if SHOW_CONFIDENCE_BANDS:
                # Calculate confidence intervals
                residuals = values - trend_dict['trend_line']
                mse = np.sum(residuals**2) / (len(values) - 2)
                
                # Standard error of regression
                x_range = np.arange(len(times))
                se = np.sqrt(mse * (1/len(times) + (x_range - np.mean(x_range))**2 / np.sum((x_range - np.mean(x_range))**2)))
                
                # 95% confidence interval
                t_value = stats.t.ppf(0.975, len(times) - 2)
                ci = t_value * se
                
                ax_main.fill_between(times, trend_dict['trend_line'] - ci, 
                                   trend_dict['trend_line'] + ci, 
                                   alpha=0.3, color=COLOR_TREND, 
                                   label='95% Confidence Interval')
        
        # Highlight anomalies
        if SHOW_ANOMALIES and np.any(anomalies):
            anomaly_times = times[anomalies]
            anomaly_values = values[anomalies]
            ax_main.scatter(anomaly_times, anomaly_values, color=COLOR_ANOMALY, 
                          s=50, alpha=0.8, label=f'Anomalies ({np.sum(anomalies)} points)')
        
        # Customize main plot
        ax_main.set_title(TITLE, fontsize=14, fontweight='bold')
        ax_main.set_ylabel(Y_LABEL, fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Add statistical information
        if SHOW_STATISTICS:
            stats_text = f"""Statistics:
Mean: {stats_dict['mean']:.2f}
Std: {stats_dict['std']:.2f}
Min: {stats_dict['min']:.2f}
Max: {stats_dict['max']:.2f}
Range: {stats_dict['range']:.2f}
CV: {stats_dict['cv']:.3f}
Skewness: {stats_dict['skewness']:.3f}
n = {len(values):,} points"""
            
            ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add trend interpretation
        if SHOW_TREND_LINE and trend_dict:
            trend_text = f"Trend: {trend_dict['trend_direction'].title()} (slope = {trend_dict['slope']:.4f})"
            ax_main.text(0.02, 0.02, trend_text, transform=ax_main.transAxes, 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add seasonality information
        if seasonality:
            seasonal_text = f"Seasonality: Period = {seasonality['period']}, Strength = {seasonality['strength']:.3f}"
            ax_main.text(0.5, 0.02, seasonal_text, transform=ax_main.transAxes, 
                        fontsize=10, fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Create subplots if requested
        if SHOW_SUBPLOTS:
            # Trend subplot
            if trend_dict:
                ax_trend.plot(times, values - trend_dict['trend_line'], 
                            color='purple', linewidth=1, alpha=0.7)
                ax_trend.set_ylabel('Detrended Values', fontsize=10)
                ax_trend.set_title('Detrended Time Series', fontsize=12)
                ax_trend.grid(True, alpha=0.3)
            
            # Seasonal subplot
            if seasonality:
                # Calculate seasonal component
                period = seasonality['period']
                seasonal_component = np.zeros(len(values))
                
                for i in range(len(values)):
                    if i >= period:
                        seasonal_component[i] = values[i] - values[i - period]
                
                ax_seasonal.plot(times, seasonal_component, 
                               color='orange', linewidth=1, alpha=0.7)
                ax_seasonal.set_ylabel('Seasonal Component', fontsize=10)
                ax_seasonal.set_title(f'Seasonal Pattern (Period = {period})', fontsize=12)
                ax_seasonal.grid(True, alpha=0.3)
            
            # Set x-axis label for bottom subplot
            ax_seasonal.set_xlabel(X_LABEL, fontsize=12)
        
        else:
            ax_main.set_xlabel(X_LABEL, fontsize=12)
        
        # Add distribution plot if requested
        if SHOW_DISTRIBUTION:
            # Create inset axes for distribution
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_inset = inset_axes(ax_main, width="30%", height="30%", loc="upper right")
            
            ax_inset.hist(values, bins=30, alpha=0.7, color=COLOR_PRIMARY, edgecolor='black')
            ax_inset.set_title('Value Distribution', fontsize=10)
            ax_inset.set_xlabel('Values', fontsize=8)
            ax_inset.set_ylabel('Frequency', fontsize=8)
            ax_inset.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced time series plot saved as: {OUTPUT_FILE}")
        
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
    create_advanced_timeseries() 