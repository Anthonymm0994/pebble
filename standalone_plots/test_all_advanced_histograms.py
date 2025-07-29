#!/usr/bin/env python3
"""
Test All Advanced Histogram Generators
=====================================

Comprehensive test script to run all advanced histogram generators and create
various types of professional histograms with different libraries.

This script tests:
- AdvancedBokehHistograms class
- AdvancedSeabornHistograms class  
- AdvancedPlotlyHistograms class
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_script(script_name: str, description: str):
    """Run a Python script and capture output."""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=600)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check return code
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 10 minutes")
    except Exception as e:
        print(f"💥 Error running {script_name}: {e}")
    
    end_time = time.time()
    print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
    print(f"{'='*70}\n")

def check_output_directories():
    """Check if output directories were created and contain files."""
    output_dirs = [
        "bokeh_histogram_outputs",
        "seaborn_histogram_outputs", 
        "plotly_histogram_outputs"
    ]
    
    print("\n📁 Checking Output Directories:")
    print("=" * 50)
    
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"✅ {dir_name}: {len(files)} files found")
            
            # List some files
            for file in files[:5]:  # Show first 5 files
                print(f"   📄 {file.name}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print(f"❌ {dir_name}: Directory not found")

def check_dependencies():
    """Check if required packages are available."""
    print("🔍 Checking Dependencies:")
    print("=" * 30)
    
    dependencies = [
        ("bokeh", "Advanced Bokeh Histograms"),
        ("plotly", "Advanced Plotly Histograms"),
        ("seaborn", "Advanced Seaborn Histograms")
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package} - Available ({description})")
        except ImportError:
            print(f"❌ {package} - Not available ({description})")

def main():
    """Main function to test all advanced histogram generators."""
    print("🧪 Testing All Advanced Histogram Generators")
    print("=" * 70)
    
    # Check dependencies first
    check_dependencies()
    
    # List of scripts to test
    scripts_to_test = [
        {
            "name": "advanced_bokeh_histograms.py",
            "description": "Advanced Bokeh histogram generator with interactive features and professional styling"
        },
        {
            "name": "seaborn_advanced_histograms.py", 
            "description": "Advanced Seaborn histogram generator with sophisticated styling and statistical analysis"
        },
        {
            "name": "plotly_interactive_histograms.py",
            "description": "Advanced Plotly histogram generator with interactive features and animations"
        }
    ]
    
    # Check if scripts exist
    print("\n🔍 Checking script availability:")
    for script in scripts_to_test:
        if Path(script["name"]).exists():
            print(f"✅ {script['name']} - Found")
        else:
            print(f"❌ {script['name']} - Not found")
    
    print("\n🚀 Starting advanced histogram tests...")
    
    # Run each script
    for script in scripts_to_test:
        if Path(script["name"]).exists():
            run_script(script["name"], script["description"])
        else:
            print(f"⚠️  Skipping {script['name']} - file not found")
    
    # Check output directories
    check_output_directories()
    
    print("\n🎉 Advanced histogram testing complete!")
    print("\n📊 Summary of generated outputs:")
    print("- bokeh_histogram_outputs/: Interactive Bokeh histograms with hover tools")
    print("- seaborn_histogram_outputs/: Professional Seaborn histograms with statistical analysis")
    print("- plotly_histogram_outputs/: Interactive Plotly histograms with animations")
    
    print("\n💡 Advanced Features Available:")
    print("- Interactive hover tools and zoom capabilities")
    print("- Distribution fitting with AIC/BIC comparison")
    print("- Statistical analysis with normality tests")
    print("- Professional styling and color schemes")
    print("- Multiple export formats (HTML, PNG, PDF, SVG)")
    print("- Animations and transitions (Plotly)")
    print("- Faceted plots and comparisons")
    print("- Outlier detection and analysis")
    
    print("\n🔧 Usage Tips:")
    print("- Install missing packages: pip install bokeh plotly seaborn")
    print("- Modify DATABASE_PATH in scripts to use your own database")
    print("- Adjust queries and parameters for custom analysis")
    print("- Check output directories for generated interactive plots")
    print("- Use HTML files for interactive exploration")
    print("- Combine multiple libraries for comprehensive analysis")

if __name__ == "__main__":
    main() 