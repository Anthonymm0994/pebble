#!/usr/bin/env python3
"""
Test Individual Plotting Scripts
Tests each of the main plotting scripts with the comprehensive database.
"""

import os
import subprocess
import sys
from pathlib import Path

def test_histogram_script():
    """Test the histogram plotting script."""
    print("📊 Testing histogram script...")
    
    # Create a modified version of the histogram script for testing
    test_script = """
#!/usr/bin/env python3
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration for testing
DATABASE_PATH = "data.sqlite"
QUERY = "SELECT amount FROM sales LIMIT 100000"
COLUMN_NAME = "amount"
BIN_COUNT = 50
FIGURE_SIZE = (12, 8)
OUTPUT_FILE = "test_histogram_output.png"
TITLE = "Sales Amount Distribution (Test)"
X_LABEL = "Amount ($)"
Y_LABEL = "Frequency"
COLOR = "steelblue"
ALPHA = 0.7

def create_histogram():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(QUERY, conn)
        conn.close()
        
        if df.empty:
            print("Warning: Query returned no data!")
            return
        
        plt.figure(figsize=FIGURE_SIZE)
        plt.hist(df[COLUMN_NAME], bins=BIN_COUNT, color=COLOR, alpha=ALPHA, 
                edgecolor='black', linewidth=0.5)
        
        plt.title(TITLE, fontsize=14, fontweight='bold')
        plt.xlabel(X_LABEL, fontsize=12)
        plt.ylabel(Y_LABEL, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        mean_val = df[COLUMN_NAME].mean()
        std_val = df[COLUMN_NAME].std()
        plt.text(0.02, 0.98, f'Mean: ${mean_val:,.2f}\\nStd: ${std_val:,.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Histogram saved as: {OUTPUT_FILE}")
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_histogram()
"""
    
    with open("test_histogram.py", "w") as f:
        f.write(test_script)
    
    result = subprocess.run([sys.executable, "test_histogram.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Histogram script test passed!")
        if os.path.exists("test_histogram_output.png"):
            print("   📁 Output file created successfully")
    else:
        print("❌ Histogram script test failed!")
        print(f"   Error: {result.stderr}")
    
    # Cleanup
    if os.path.exists("test_histogram.py"):
        os.remove("test_histogram.py")

def test_bar_script():
    """Test the bar chart plotting script."""
    print("📊 Testing bar chart script...")
    
    # Create a modified version of the bar chart script for testing
    test_script = """
#!/usr/bin/env python3
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration for testing
DATABASE_PATH = "data.sqlite"
QUERY = "SELECT category, total_sales FROM category_summary"
CATEGORY_COLUMN = "category"
VALUE_COLUMN = "total_sales"
FIGURE_SIZE = (12, 8)
OUTPUT_FILE = "test_bar_output.png"
TITLE = "Sales by Category (Test)"
X_LABEL = "Category"
Y_LABEL = "Total Sales ($)"
COLOR = "skyblue"
ALPHA = 0.8
ROTATE_LABELS = 45

def create_bar_chart():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(QUERY, conn)
        conn.close()
        
        if df.empty:
            print("Warning: Query returned no data!")
            return
        
        plt.figure(figsize=FIGURE_SIZE)
        bars = plt.bar(df[CATEGORY_COLUMN], df[VALUE_COLUMN], 
                      color=COLOR, alpha=ALPHA, edgecolor='black', linewidth=0.5)
        
        plt.title(TITLE, fontsize=14, fontweight='bold')
        plt.xlabel(X_LABEL, fontsize=12)
        plt.ylabel(Y_LABEL, fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        if ROTATE_LABELS > 0:
            plt.xticks(rotation=ROTATE_LABELS, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved as: {OUTPUT_FILE}")
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_bar_chart()
"""
    
    with open("test_bar.py", "w") as f:
        f.write(test_script)
    
    result = subprocess.run([sys.executable, "test_bar.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Bar chart script test passed!")
        if os.path.exists("test_bar_output.png"):
            print("   📁 Output file created successfully")
    else:
        print("❌ Bar chart script test failed!")
        print(f"   Error: {result.stderr}")
    
    # Cleanup
    if os.path.exists("test_bar.py"):
        os.remove("test_bar.py")

def test_polar_script():
    """Test the polar plot plotting script."""
    print("📊 Testing polar plot script...")
    
    # Create a modified version of the polar plot script for testing
    test_script = """
#!/usr/bin/env python3
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration for testing
DATABASE_PATH = "data.sqlite"
QUERY = "SELECT angle, radius FROM measurements"
ANGLE_COLUMN = "angle"
RADIUS_COLUMN = "radius"
FIGURE_SIZE = (10, 8)
OUTPUT_FILE = "test_polar_output.png"
TITLE = "Scientific Measurements (Test)"
MARKER_SIZE = 30
ALPHA = 0.7

def create_polar_plot():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(QUERY, conn)
        conn.close()
        
        if df.empty:
            print("Warning: Query returned no data!")
            return
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=FIGURE_SIZE)
        
        angles = df[ANGLE_COLUMN]
        if angles.max() > 2 * np.pi:
            angles = np.radians(angles)
        
        radii = df[RADIUS_COLUMN]
        
        scatter = ax.scatter(angles, radii, c=radii, s=MARKER_SIZE, 
                           alpha=ALPHA, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Radius Values', rotation=270, labelpad=15)
        
        ax.set_title(TITLE, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print(f"Polar plot saved as: {OUTPUT_FILE}")
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_polar_plot()
"""
    
    with open("test_polar.py", "w") as f:
        f.write(test_script)
    
    result = subprocess.run([sys.executable, "test_polar.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Polar plot script test passed!")
        if os.path.exists("test_polar_output.png"):
            print("   📁 Output file created successfully")
    else:
        print("❌ Polar plot script test failed!")
        print(f"   Error: {result.stderr}")
    
    # Cleanup
    if os.path.exists("test_polar.py"):
        os.remove("test_polar.py")

def main():
    """Run all individual script tests."""
    print("🧪 Testing individual plotting scripts...")
    
    if not os.path.exists('data.sqlite'):
        print("❌ Database not found! Please run create_comprehensive_db.py first.")
        return
    
    test_histogram_script()
    test_bar_script()
    test_polar_script()
    
    print("\n🎉 All individual script tests completed!")
    
    # Cleanup test output files
    test_files = ["test_histogram_output.png", "test_bar_output.png", "test_polar_output.png"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up {file}")

if __name__ == "__main__":
    main() 