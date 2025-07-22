#!/usr/bin/env python3
"""
Advanced Polar Plot Generator for SQLite Data
Creates sophisticated polar plots with multiple visualization types for comprehensive data analysis.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import griddata
import seaborn as sns

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "sample_data.sqlite"  # Path to your SQLite database
QUERY = "SELECT x as angle, y_positive as radius, size as temperature, color as pressure FROM correlations LIMIT 5000"  # Your SELECT query here

# Plot settings
ANGLE_COLUMN = "angle"  # Column containing angle values (degrees)
RADIUS_COLUMN = "radius"  # Column containing radius values
COLOR_COLUMN = "temperature"  # Column for color coding (optional)
SIZE_COLUMN = "pressure"  # Column for point sizes (optional)
FIGURE_SIZE = (15, 12)  # Width, height in inches
OUTPUT_FILE = "advanced_polar_output.png"  # Output filename

# Plot type and features
PLOT_TYPE = "comprehensive"  # Options: "basic", "density", "contour", "rose", "wind", "comprehensive"
SHOW_DENSITY = True  # Show density contours
SHOW_CONTOURS = True  # Show contour lines
SHOW_ROSE = False  # Show wind rose style
SHOW_STATISTICS = True  # Show statistical information
SHOW_ANOMALIES = True  # Highlight anomalies
SHOW_TRENDS = True  # Show trend analysis
SHOW_SEASONALITY = True  # Show seasonal patterns

# Styling
TITLE = "Advanced Polar Analysis"
COLOR_MAP = "viridis"  # Color map for visualization
MARKER_SIZE = 30
ALPHA = 0.7  # Transparency (0-1)
GRID_ALPHA = 0.3

# Advanced features
DETECT_ANOMALIES = True  # Detect statistical anomalies
CALCULATE_TRENDS = True  # Calculate trend patterns
SHOW_CONFIDENCE = True  # Show confidence regions
SHOW_DISTRIBUTION = True  # Show value distribution

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def detect_polar_anomalies(angles, radii, threshold=2.0):
    """Detect anomalies in polar data using distance from center."""
    # Convert to cartesian coordinates
    x = radii * np.cos(np.radians(angles))
    y = radii * np.sin(np.radians(angles))
    
    # Calculate distance from center
    distances = np.sqrt(x**2 + y**2)
    
    # Detect outliers using z-score
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    z_scores = np.abs((distances - mean_dist) / std_dist)
    anomalies = z_scores > threshold
    
    return anomalies

def calculate_polar_statistics(angles, radii):
    """Calculate comprehensive polar statistics."""
    # Convert to cartesian
    x = radii * np.cos(np.radians(angles))
    y = radii * np.sin(np.radians(angles))
    
    # Calculate statistics
    stats_dict = {
        'mean_radius': np.mean(radii),
        'std_radius': np.std(radii),
        'min_radius': np.min(radii),
        'max_radius': np.max(radii),
        'mean_angle': np.mean(angles),
        'std_angle': np.std(angles),
        'center_x': np.mean(x),
        'center_y': np.mean(y),
        'total_area': np.pi * np.mean(radii)**2,
        'eccentricity': np.std(radii) / np.mean(radii)
    }
    
    return stats_dict

def create_wind_rose(angles, values, ax):
    """Create wind rose style plot."""
    # Create histogram of angles
    angle_bins = np.linspace(0, 360, 13)  # 12 bins (30 degrees each)
    hist, bin_edges = np.histogram(angles, bins=angle_bins)
    
    # Convert to polar coordinates
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_rad = np.radians(bin_centers)
    
    # Create bars
    bars = ax.bar(bin_centers_rad, hist, width=np.radians(30), 
                  alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars by value
    for i, (bar, val) in enumerate(zip(bars, hist)):
        if val > 0:
            bar.set_facecolor(plt.cm.viridis(val / max(hist)))
    
    ax.set_title('Wind Rose Style Distribution', fontsize=12, fontweight='bold')

def create_density_contour(angles, radii, ax):
    """Create density contour plot."""
    # Convert to cartesian coordinates
    x = radii * np.cos(np.radians(angles))
    y = radii * np.sin(np.radians(angles))
    
    # Create grid for contour
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate density
    points = np.column_stack([x, y])
    values = np.ones(len(points))  # Equal weight for density
    
    zi = griddata(points, values, (xi_grid, yi_grid), method='linear', fill_value=0)
    
    # Create contour plot
    contour = ax.contour(xi_grid, yi_grid, zi, levels=10, colors='white', alpha=0.5, linewidths=1)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Fill contours
    ax.contourf(xi_grid, yi_grid, zi, levels=10, cmap='viridis', alpha=0.3)

def create_advanced_polar():
    """Create advanced polar plot from SQLite data."""
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
        if ANGLE_COLUMN not in df.columns:
            print(f"Error: Angle column '{ANGLE_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
            
        if RADIUS_COLUMN not in df.columns:
            print(f"Error: Radius column '{RADIUS_COLUMN}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Clean data
        df = df.dropna(subset=[ANGLE_COLUMN, RADIUS_COLUMN])
        angles = df[ANGLE_COLUMN].values
        radii = df[RADIUS_COLUMN].values
        
        if len(angles) == 0:
            print("Error: No valid data points after cleaning.")
            return
        
        # Convert angles to radians if they're in degrees
        if angles.max() > 2 * np.pi:  # Assume degrees if max > 2π
            angles_rad = np.radians(angles)
        else:
            angles_rad = angles
        
        # Calculate statistics
        stats_dict = calculate_polar_statistics(angles, radii)
        
        # Detect anomalies
        anomalies = detect_polar_anomalies(angles, radii) if DETECT_ANOMALIES else np.zeros(len(angles), dtype=bool)
        
        # Create the plot
        if PLOT_TYPE == "comprehensive":
            fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE, 
                                   subplot_kw={'projection': 'polar'})
            ax_main, ax_density, ax_contour, ax_rose, ax_trend, ax_stats = axes.flatten()
        else:
            fig, ax_main = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
        
        # Main polar plot
        if COLOR_COLUMN in df.columns:
            color_values = df[COLOR_COLUMN].values
            scatter = ax_main.scatter(angles_rad, radii, c=color_values, s=MARKER_SIZE, 
                                    alpha=ALPHA, cmap=COLOR_MAP, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax_main, shrink=0.8, label=COLOR_COLUMN)
        else:
            # Plot regular points
            regular_mask = ~anomalies
            if np.any(regular_mask):
                ax_main.scatter(angles_rad[regular_mask], radii[regular_mask], s=MARKER_SIZE, 
                              alpha=ALPHA, color='blue', edgecolors='black', linewidth=0.5, label='Regular Points')
            
            # Plot anomalies
            if np.any(anomalies):
                ax_main.scatter(angles_rad[anomalies], radii[anomalies], s=MARKER_SIZE*1.5, 
                              alpha=0.8, color='red', edgecolors='black', linewidth=1, label='Anomalies')
        
        ax_main.set_title('Main Polar Plot', fontsize=12, fontweight='bold')
        ax_main.grid(True, alpha=GRID_ALPHA)
        
        if PLOT_TYPE == "comprehensive":
            # Density plot
            if SHOW_DENSITY:
                ax_density.scatter(angles_rad, radii, c=radii, s=MARKER_SIZE, 
                                 alpha=ALPHA, cmap='plasma', edgecolors='black', linewidth=0.5)
                ax_density.set_title('Density by Radius', fontsize=12, fontweight='bold')
                ax_density.grid(True, alpha=GRID_ALPHA)
            
            # Contour plot
            if SHOW_CONTOURS:
                create_density_contour(angles, radii, ax_contour)
                ax_contour.scatter(angles_rad, radii, s=20, alpha=0.6, color='red')
                ax_contour.set_title('Density Contours', fontsize=12, fontweight='bold')
                ax_contour.grid(True, alpha=GRID_ALPHA)
            
            # Wind rose
            if SHOW_ROSE:
                create_wind_rose(angles, radii, ax_rose)
                ax_rose.grid(True, alpha=GRID_ALPHA)
            
            # Trend analysis
            if SHOW_TRENDS:
                # Calculate trend by angle
                angle_bins = np.linspace(0, 360, 13)
                bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
                bin_centers_rad = np.radians(bin_centers)
                
                mean_radii = []
                for i in range(len(angle_bins)-1):
                    mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
                    if np.any(mask):
                        mean_radii.append(np.mean(radii[mask]))
                    else:
                        mean_radii.append(0)
                
                ax_trend.plot(bin_centers_rad, mean_radii, 'o-', linewidth=2, markersize=8, color='green')
                ax_trend.set_title('Trend by Angle', fontsize=12, fontweight='bold')
                ax_trend.grid(True, alpha=GRID_ALPHA)
            
            # Statistics subplot
            if SHOW_STATISTICS:
                stats_text = f"""Polar Statistics:
Mean Radius: {stats_dict['mean_radius']:.2f}
Std Radius: {stats_dict['std_radius']:.2f}
Min Radius: {stats_dict['min_radius']:.2f}
Max Radius: {stats_dict['max_radius']:.2f}
Eccentricity: {stats_dict['eccentricity']:.3f}
Total Points: {len(angles):,}
Anomalies: {np.sum(anomalies):,}"""
                
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                ax_stats.set_title('Statistical Summary', fontsize=12, fontweight='bold')
                ax_stats.axis('off')
        
        # Add statistical information to main plot
        if SHOW_STATISTICS:
            stats_text = f"Mean Radius: {stats_dict['mean_radius']:.2f}\nStd Radius: {stats_dict['std_radius']:.2f}\nEccentricity: {stats_dict['eccentricity']:.3f}"
            ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add anomaly information
        if SHOW_ANOMALIES and np.any(anomalies):
            anomaly_text = f"Anomalies: {np.sum(anomalies)} ({np.sum(anomalies)/len(angles)*100:.1f}%)"
            ax_main.text(0.02, 0.02, anomaly_text, transform=ax_main.transAxes, 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Advanced polar plot saved as: {OUTPUT_FILE}")
        
        # Print detailed statistics
        if SHOW_STATISTICS:
            print("\n📊 Polar Analysis Statistics:")
            print(f"Mean Radius: {stats_dict['mean_radius']:.2f}")
            print(f"Std Radius: {stats_dict['std_radius']:.2f}")
            print(f"Min Radius: {stats_dict['min_radius']:.2f}")
            print(f"Max Radius: {stats_dict['max_radius']:.2f}")
            print(f"Eccentricity: {stats_dict['eccentricity']:.3f}")
            print(f"Total Points: {len(angles):,}")
            print(f"Anomalies: {np.sum(anomalies):,} ({np.sum(anomalies)/len(angles)*100:.1f}%)")
        
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
    create_advanced_polar() 