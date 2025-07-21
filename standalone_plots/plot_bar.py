#!/usr/bin/env python3
"""
Bar Chart Generator for SQLite Data
Connects to SQLite database, runs a query, and creates a bar chart.
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
QUERY = "SELECT category_column, value_column FROM table_name"  # Your SELECT query here

# Plot settings
CATEGORY_COLUMN = "category_column"  # Column containing categories/labels
VALUE_COLUMN = "value_column"  # Column containing values to plot
FIGURE_SIZE = (12, 6)  # Width, height in inches
OUTPUT_FILE = "bar_output.png"  # Output filename

# Styling
TITLE = "Bar Chart"
X_LABEL = "Categories"
Y_LABEL = "Values"
COLOR = "skyblue"
ALPHA = 0.8  # Transparency (0-1)
ROTATE_LABELS = 45  # Angle to rotate x-axis labels (0 for horizontal)

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def create_bar_chart():
    """Create bar chart from SQLite data."""
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
        
        # Create the plot
        plt.figure(figsize=FIGURE_SIZE)
        
        # Create bar chart
        bars = plt.bar(df[CATEGORY_COLUMN], df[VALUE_COLUMN], 
                      color=COLOR, alpha=ALPHA, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        plt.title(TITLE, fontsize=14, fontweight='bold')
        plt.xlabel(X_LABEL, fontsize=12)
        plt.ylabel(Y_LABEL, fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if ROTATE_LABELS > 0:
            plt.xticks(rotation=ROTATE_LABELS, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Add statistics text
        total_value = df[VALUE_COLUMN].sum()
        max_value = df[VALUE_COLUMN].max()
        plt.text(0.02, 0.98, f'Total: {total_value:.2f}\nMax: {max_value:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved as: {OUTPUT_FILE}")
        
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
    create_bar_chart() 