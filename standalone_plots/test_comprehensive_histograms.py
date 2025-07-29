#!/usr/bin/env python3
"""
Test Comprehensive Histogram Scripts
===================================

Test script to run all comprehensive histogram generators and create
various types of histograms with different configurations.

This script tests:
- ComprehensiveHistogram class
- AdvancedHistogramAnalysis class  
- HistogramPermutations class
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_script(script_name: str, description: str):
    """Run a Python script and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
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
        print(f"⏰ {script_name} timed out after 5 minutes")
    except Exception as e:
        print(f"💥 Error running {script_name}: {e}")
    
    end_time = time.time()
    print(f"⏱️  Execution time: {end_time - start_time:.2f} seconds")
    print(f"{'='*60}\n")

def check_output_directories():
    """Check if output directories were created and contain files."""
    output_dirs = [
        "histogram_outputs",
        "advanced_histogram_outputs", 
        "histogram_permutations"
    ]
    
    print("\n📁 Checking Output Directories:")
    print("=" * 40)
    
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

def main():
    """Main function to test all comprehensive histogram scripts."""
    print("🧪 Testing Comprehensive Histogram Scripts")
    print("=" * 60)
    
    # List of scripts to test
    scripts_to_test = [
        {
            "name": "comprehensive_histogram.py",
            "description": "Comprehensive histogram generator with overlapping distributions and statistical analysis"
        },
        {
            "name": "advanced_histogram_analysis.py", 
            "description": "Advanced histogram analysis with bokeh integration and interactive features"
        },
        {
            "name": "histogram_permutations.py",
            "description": "Histogram permutations and filtering generator with query combinations"
        }
    ]
    
    # Check if scripts exist
    print("🔍 Checking script availability:")
    for script in scripts_to_test:
        if Path(script["name"]).exists():
            print(f"✅ {script['name']} - Found")
        else:
            print(f"❌ {script['name']} - Not found")
    
    print("\n🚀 Starting comprehensive histogram tests...")
    
    # Run each script
    for script in scripts_to_test:
        if Path(script["name"]).exists():
            run_script(script["name"], script["description"])
        else:
            print(f"⚠️  Skipping {script['name']} - file not found")
    
    # Check output directories
    check_output_directories()
    
    print("\n🎉 Comprehensive histogram testing complete!")
    print("\n📊 Summary of generated outputs:")
    print("- histogram_outputs/: Basic comprehensive histograms")
    print("- advanced_histogram_outputs/: Advanced analysis with interactive plots")
    print("- histogram_permutations/: Permutation-based histogram analysis")
    
    print("\n💡 Usage Tips:")
    print("- Modify DATABASE_PATH in scripts to use your own database")
    print("- Adjust filters and columns in main() functions for custom analysis")
    print("- Check output directories for generated plots and CSV summaries")
    print("- Use the generated CSV files for further statistical analysis")

if __name__ == "__main__":
    main() 