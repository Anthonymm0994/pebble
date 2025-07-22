#!/usr/bin/env python3
"""
Advanced Plot Testing Script
Generates comprehensive advanced plots from the database and saves them to a test folder.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_advanced_test_folder():
    """Create advanced test folder for output plots."""
    test_folder = Path("advanced_test_plots")
    test_folder.mkdir(exist_ok=True)
    return test_folder

def test_advanced_scatter_plots(test_folder):
    """Test various advanced scatter plot configurations."""
    print("📊 Testing advanced scatter plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Positive correlation with all features
    df = pd.read_sql_query("SELECT x, y_positive, size FROM correlations LIMIT 5000", conn)
    plt.figure(figsize=(12, 8))
    
    # Create scatter with size variation
    scatter = plt.scatter(df['x'], df['y_positive'], s=df['size'], alpha=0.6, 
                         c=df['y_positive'], cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df['x'], df['y_positive'], 1)
    p = np.poly1d(z)
    plt.plot(df['x'], p(df['x']), "r--", alpha=0.8, linewidth=2)
    
    plt.title('Advanced Scatter Plot - Positive Correlation', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.colorbar(scatter, label='Y Values')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    corr = df['x'].corr(df['y_positive'])
    plt.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_scatter_positive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Negative correlation
    df = pd.read_sql_query("SELECT x, y_negative FROM correlations LIMIT 5000", conn)
    plt.figure(figsize=(12, 8))
    
    plt.scatter(df['x'], df['y_negative'], alpha=0.6, s=30, c='red', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df['x'], df['y_negative'], 1)
    p = np.poly1d(z)
    plt.plot(df['x'], p(df['x']), "b--", alpha=0.8, linewidth=2)
    
    plt.title('Advanced Scatter Plot - Negative Correlation', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    corr = df['x'].corr(df['y_negative'])
    plt.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_scatter_negative.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Hexbin plot for density
    df = pd.read_sql_query("SELECT x, y_no_correlation FROM correlations LIMIT 10000", conn)
    plt.figure(figsize=(12, 8))
    
    hb = plt.hexbin(df['x'], df['y_no_correlation'], gridsize=50, cmap='plasma', alpha=0.8)
    plt.colorbar(hb, label='Point Density')
    
    plt.title('Hexbin Density Plot - No Correlation', fontsize=14, fontweight='bold')
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_hexbin_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced scatter plot tests completed!")

def test_advanced_timeseries_plots(test_folder):
    """Test various advanced time series plot configurations."""
    print("📊 Testing advanced time series plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Comprehensive time series with multiple variables
    df = pd.read_sql_query("SELECT timestamp, temperature, humidity, pressure FROM time_series LIMIT 2000", conn)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Temperature plot
    axes[0].plot(df['timestamp'], df['temperature'], color='red', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].set_title('Temperature Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add moving average
    ma_temp = df['temperature'].rolling(window=24, center=True).mean()
    axes[0].plot(df['timestamp'], ma_temp, color='darkred', linewidth=2, alpha=0.8, label='24h Moving Average')
    axes[0].legend()
    
    # Humidity plot
    axes[1].plot(df['timestamp'], df['humidity'], color='blue', linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel('Humidity (%)', fontsize=12)
    axes[1].set_title('Humidity Over Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add moving average
    ma_humidity = df['humidity'].rolling(window=24, center=True).mean()
    axes[1].plot(df['timestamp'], ma_humidity, color='darkblue', linewidth=2, alpha=0.8, label='24h Moving Average')
    axes[1].legend()
    
    # Pressure plot
    axes[2].plot(df['timestamp'], df['pressure'], color='green', linewidth=1.5, alpha=0.8)
    axes[2].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_title('Pressure Over Time', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add moving average
    ma_pressure = df['pressure'].rolling(window=24, center=True).mean()
    axes[2].plot(df['timestamp'], ma_pressure, color='darkgreen', linewidth=2, alpha=0.8, label='24h Moving Average')
    axes[2].legend()
    
    # Rotate x-axis labels
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_timeseries_multi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Seasonal decomposition
    df = pd.read_sql_query("SELECT timestamp, temperature FROM time_series LIMIT 1000", conn)
    
    # Calculate seasonal components
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.dayofyear
    
    # Group by hour for daily pattern
    hourly_avg = df.groupby('hour')['temperature'].mean()
    
    plt.figure(figsize=(12, 8))
    plt.plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, markersize=8, color='purple')
    plt.title('Daily Temperature Pattern (Seasonal Component)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 3))
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_timeseries_seasonal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced time series tests completed!")

def test_advanced_distribution_plots(test_folder):
    """Test various advanced distribution plot configurations."""
    print("📊 Testing advanced distribution plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Comprehensive violin plot with statistics
    df = pd.read_sql_query("SELECT category, amount FROM sales LIMIT 100000", conn)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Violin plot
    import seaborn as sns
    sns.violinplot(data=df, x='category', y='amount', ax=ax1, palette='Set3')
    ax1.set_title('Sales Amount Distribution by Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Amount ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Box plot with statistics
    box_plot = ax2.boxplot([df[df['category'] == cat]['amount'].values for cat in df['category'].unique()],
                           labels=df['category'].unique(), patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Sales Amount Box Plot by Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Amount ($)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_distribution_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Statistical distribution comparison
    df = pd.read_sql_query("SELECT normal_dist, exponential_dist, uniform_dist, bimodal_dist FROM distributions LIMIT 100000", conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Normal distribution
    axes[0, 0].hist(df['normal_dist'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Normal Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Values', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Exponential distribution
    axes[0, 1].hist(df['exponential_dist'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('Exponential Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Values', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uniform distribution
    axes[1, 0].hist(df['uniform_dist'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title('Uniform Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Values', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bimodal distribution
    axes[1, 1].hist(df['bimodal_dist'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Bimodal Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Values', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced distribution tests completed!")

def test_advanced_heatmap_plots(test_folder):
    """Test various advanced heatmap plot configurations."""
    print("📊 Testing advanced heatmap plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Correlation matrix heatmap
    df = pd.read_sql_query("SELECT amount, profit_margin, quantity, rating FROM sales LIMIT 100000", conn)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Sales Data Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_heatmap_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multi-dimensional data heatmap
    df = pd.read_sql_query("SELECT category, region, AVG(amount) as avg_amount, COUNT(*) as count FROM sales GROUP BY category, region", conn)
    
    # Pivot data for heatmap
    pivot_data = df.pivot(index='category', columns='region', values='avg_amount')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Average Sales Amount by Category and Region', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_heatmap_pivot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Time series heatmap
    df = pd.read_sql_query("SELECT timestamp, temperature, humidity, pressure FROM time_series LIMIT 1000", conn)
    
    # Create time-based aggregation
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    
    # Aggregate by hour and day
    hourly_temp = df.groupby('hour')['temperature'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_temp.index, hourly_temp.values, 'o-', linewidth=2, markersize=8, color='red')
    plt.title('Average Temperature by Hour of Day', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 3))
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_heatmap_temporal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced heatmap tests completed!")

def test_advanced_polar_plots(test_folder):
    """Test various advanced polar plot configurations."""
    print("📊 Testing advanced polar plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Complex polar pattern with multiple variables
    df = pd.read_sql_query("SELECT angle, radius, temperature, pressure FROM measurements", conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': 'polar'})
    
    # Basic radius plot
    scatter1 = axes[0, 0].scatter(df['angle'], df['radius'], c=df['radius'], s=30, 
                                  alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 0].set_title('Radius vs Angle', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0, 0], shrink=0.8)
    
    # Temperature plot
    scatter2 = axes[0, 1].scatter(df['angle'], df['temperature'], c=df['temperature'], s=30, 
                                  alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)
    axes[0, 1].set_title('Temperature vs Angle', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[0, 1], shrink=0.8)
    
    # Pressure plot
    scatter3 = axes[1, 0].scatter(df['angle'], df['pressure'], c=df['pressure'], s=30, 
                                  alpha=0.7, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_title('Pressure vs Angle', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=axes[1, 0], shrink=0.8)
    
    # Combined plot
    scatter4 = axes[1, 1].scatter(df['angle'], df['radius'], c=df['temperature'], s=30, 
                                  alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Radius vs Angle (colored by Temperature)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter4, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_polar_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced polar plot tests completed!")

def main():
    """Run all advanced tests and generate plots."""
    print("🚀 Starting advanced plot testing...")
    
    # Create test folder
    test_folder = create_advanced_test_folder()
    print(f"📁 Advanced test plots will be saved to: {test_folder}")
    
    # Check if database exists
    if not os.path.exists('data.sqlite'):
        print("❌ Database not found! Please run create_comprehensive_db.py first.")
        return
    
    # Run all advanced tests
    test_advanced_scatter_plots(test_folder)
    test_advanced_timeseries_plots(test_folder)
    test_advanced_distribution_plots(test_folder)
    test_advanced_heatmap_plots(test_folder)
    test_advanced_polar_plots(test_folder)
    
    print(f"\n🎉 All advanced tests completed! Check the '{test_folder}' folder for generated plots.")
    print(f"📊 Generated {len(list(test_folder.glob('*.png')))} advanced plot files.")

if __name__ == "__main__":
    main() 