#!/usr/bin/env python3
"""
Comprehensive Plot Testing Script
Generates various plots from the comprehensive database and saves them to a test folder.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_test_folder():
    """Create test folder for output plots."""
    test_folder = Path("test_plots")
    test_folder.mkdir(exist_ok=True)
    return test_folder

def test_histograms(test_folder):
    """Test various histogram plots."""
    print("📊 Testing histogram plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Sales amounts histogram
    df = pd.read_sql_query("SELECT amount FROM sales LIMIT 100000", conn)
    plt.figure(figsize=(12, 8))
    plt.hist(df['amount'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.title('Sales Amount Distribution (100K samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Amount ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'histogram_sales_amounts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Age distribution
    df = pd.read_sql_query("SELECT age FROM distributions LIMIT 100000", conn)
    plt.figure(figsize=(12, 8))
    plt.hist(df['age'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    plt.title('Age Distribution (100K samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'histogram_age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Income distribution (log scale)
    df = pd.read_sql_query("SELECT income FROM distributions LIMIT 100000", conn)
    plt.figure(figsize=(12, 8))
    plt.hist(df['income'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('Income Distribution (100K samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Income ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'histogram_income_log_scale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 4: Bimodal distribution
    df = pd.read_sql_query("SELECT bimodal_dist FROM distributions LIMIT 100000", conn)
    plt.figure(figsize=(12, 8))
    plt.hist(df['bimodal_dist'], bins=40, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Bimodal Distribution (100K samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'histogram_bimodal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Histogram tests completed!")

def test_bar_charts(test_folder):
    """Test various bar chart plots."""
    print("📊 Testing bar chart plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Sales by category
    df = pd.read_sql_query("SELECT category, total_sales FROM category_summary", conn)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['category'], df['total_sales'], color='skyblue', alpha=0.8, edgecolor='black')
    plt.title('Total Sales by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'bar_chart_sales_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Sales by region
    df = pd.read_sql_query("SELECT region, total_sales FROM region_summary", conn)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['region'], df['total_sales'], color='lightcoral', alpha=0.8, edgecolor='black')
    plt.title('Total Sales by Region', fontsize=14, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'bar_chart_sales_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Average profit margin by category
    df = pd.read_sql_query("SELECT category, avg_profit_margin FROM category_summary", conn)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['category'], df['avg_profit_margin'], color='gold', alpha=0.8, edgecolor='black')
    plt.title('Average Profit Margin by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Average Profit Margin (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'bar_chart_profit_margin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Bar chart tests completed!")

def test_polar_plots(test_folder):
    """Test various polar plot plots."""
    print("📊 Testing polar plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Basic polar plot
    df = pd.read_sql_query("SELECT angle, radius FROM measurements", conn)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    scatter = ax.scatter(df['angle'], df['radius'], c=df['radius'], s=30, 
                        alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    
    ax.set_title('Scientific Measurements - Polar Plot', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Radius Values', rotation=270, labelpad=15)
    
    plt.savefig(test_folder / 'polar_plot_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Temperature vs angle
    df = pd.read_sql_query("SELECT angle, temperature FROM measurements", conn)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    scatter = ax.scatter(df['angle'], df['temperature'], c=df['temperature'], s=40, 
                        alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)
    
    ax.set_title('Temperature vs Angle - Polar Plot', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
    
    plt.savefig(test_folder / 'polar_plot_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Pressure vs angle
    df = pd.read_sql_query("SELECT angle, pressure FROM measurements", conn)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    scatter = ax.scatter(df['angle'], df['pressure'], c=df['pressure'], s=35, 
                        alpha=0.7, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    
    ax.set_title('Pressure vs Angle - Polar Plot', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Pressure (hPa)', rotation=270, labelpad=15)
    
    plt.savefig(test_folder / 'polar_plot_pressure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Polar plot tests completed!")

def test_scatter_plots(test_folder):
    """Test scatter plots (bonus)."""
    print("📊 Testing scatter plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Positive correlation
    df = pd.read_sql_query("SELECT x, y_positive FROM correlations LIMIT 5000", conn)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y_positive'], alpha=0.6, s=20, c='blue')
    plt.title('Positive Correlation Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'scatter_positive_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Negative correlation
    df = pd.read_sql_query("SELECT x, y_negative FROM correlations LIMIT 5000", conn)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y_negative'], alpha=0.6, s=20, c='red')
    plt.title('Negative Correlation Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'scatter_negative_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: No correlation
    df = pd.read_sql_query("SELECT x, y_no_correlation FROM correlations LIMIT 5000", conn)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y_no_correlation'], alpha=0.6, s=20, c='green')
    plt.title('No Correlation Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(test_folder / 'scatter_no_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Scatter plot tests completed!")

def test_time_series(test_folder):
    """Test time series plots (bonus)."""
    print("📊 Testing time series plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Temperature over time
    df = pd.read_sql_query("SELECT timestamp, temperature FROM time_series LIMIT 1000", conn)
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df['temperature'], color='red', linewidth=1, alpha=0.8)
    plt.title('Temperature Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(test_folder / 'time_series_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Humidity over time
    df = pd.read_sql_query("SELECT timestamp, humidity FROM time_series LIMIT 1000", conn)
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df['humidity'], color='blue', linewidth=1, alpha=0.8)
    plt.title('Humidity Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Humidity (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(test_folder / 'time_series_humidity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Time series tests completed!")

def main():
    """Run all tests and generate plots."""
    print("🚀 Starting comprehensive plot testing...")
    
    # Create test folder
    test_folder = create_test_folder()
    print(f"📁 Test plots will be saved to: {test_folder}")
    
    # Check if database exists
    if not os.path.exists('data.sqlite'):
        print("❌ Database not found! Please run create_comprehensive_db.py first.")
        return
    
    # Run all tests
    test_histograms(test_folder)
    test_bar_charts(test_folder)
    test_polar_plots(test_folder)
    test_scatter_plots(test_folder)
    test_time_series(test_folder)
    
    print(f"\n🎉 All tests completed! Check the '{test_folder}' folder for generated plots.")
    print(f"📊 Generated {len(list(test_folder.glob('*.png')))} plot files.")

if __name__ == "__main__":
    main() 