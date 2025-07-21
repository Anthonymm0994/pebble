#!/usr/bin/env python3
"""
Polar Plot Generator for SQLite Data
Connects to SQLite database, runs a query, and creates a polar plot.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these variables to customize your plot
# =============================================================================

# Database and query settings
DATABASE_PATH = "data.sqlite"  # Path to your SQLite database
QUERY = "SELECT angle_column, radius_column FROM table_name"  # Your SELECT query here

# Plot settings
ANGLE_COLUMN = "angle_column"  # Column containing angle values (degrees)
RADIUS_COLUMN = "radius_column"  # Column containing radius values
FIGURE_SIZE = (10, 8)  # Width, height in inches
OUTPUT_FILE = "polar_output.png"  # Output filename

# Styling
TITLE = "Polar Plot"
COLOR = "darkblue"
MARKER_SIZE = 50
ALPHA = 0.7  # Transparency (0-1)
LINE_WIDTH = 2

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def create_polar_plot():
    """Create polar plot from SQLite data."""
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
        
        # Create the plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=FIGURE_SIZE)
        
        # Convert angles to radians if they're in degrees
        angles = df[ANGLE_COLUMN]
        if angles.max() > 2 * np.pi:  # Assume degrees if max > 2π
            angles = np.radians(angles)
        
        radii = df[RADIUS_COLUMN]
        
        # Create polar plot
        scatter = ax.scatter(angles, radii, c=radii, s=MARKER_SIZE, 
                           alpha=ALPHA, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Radius Values', rotation=270, labelpad=15)
        
        # Customize plot
        ax.set_title(TITLE, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Set angle ticks (every 45 degrees)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
        
        # Add statistics text
        mean_radius = radii.mean()
        max_radius = radii.max()
        ax.text(0.02, 0.98, f'Mean Radius: {mean_radius:.2f}\nMax Radius: {max_radius:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Polar plot saved as: {OUTPUT_FILE}")
        
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
    create_polar_plot() 