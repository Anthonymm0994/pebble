#!/usr/bin/env python3
"""
Histogram Plot Generator for SQLite Data
Connects to SQLite database, runs a query, and creates a histogram plot.
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
QUERY = "SELECT column_name FROM table_name"  # Your SELECT query here

# Plot settings
COLUMN_NAME = "column_name"  # Column to plot (should match your query)
BIN_COUNT = 20  # Number of bins for histogram
FIGURE_SIZE = (10, 6)  # Width, height in inches
OUTPUT_FILE = "histogram_output.png"  # Output filename

# Styling
TITLE = "Histogram of Data"
X_LABEL = "Values"
Y_LABEL = "Frequency"
COLOR = "steelblue"
ALPHA = 0.7  # Transparency (0-1)

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def create_histogram():
    """Create histogram plot from SQLite data."""
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
        
        # Create the plot
        plt.figure(figsize=FIGURE_SIZE)
        
        # Create histogram
        plt.hist(df[COLUMN_NAME], bins=BIN_COUNT, color=COLOR, alpha=ALPHA, 
                edgecolor='black', linewidth=0.5)
        
        # Customize plot
        plt.title(TITLE, fontsize=14, fontweight='bold')
        plt.xlabel(X_LABEL, fontsize=12)
        plt.ylabel(Y_LABEL, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[COLUMN_NAME].mean()
        std_val = df[COLUMN_NAME].std()
        plt.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Histogram saved as: {OUTPUT_FILE}")
        
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
    create_histogram() 