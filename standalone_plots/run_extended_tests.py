#!/usr/bin/env python3
"""
Extended Advanced Plot Testing Script
Generates comprehensive advanced polar plots and histograms with multiple configurations.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_extended_test_folder():
    """Create extended test folder for output plots."""
    test_folder = Path("extended_test_plots")
    test_folder.mkdir(exist_ok=True)
    return test_folder

def test_advanced_polar_variations(test_folder):
    """Test various advanced polar plot configurations."""
    print("📊 Testing advanced polar plot variations...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Temperature-based polar plot
    df = pd.read_sql_query("SELECT angle, radius, temperature FROM measurements", conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': 'polar'})
    
    # Main temperature plot
    angles_rad = np.radians(df['angle'])
    scatter1 = axes[0, 0].scatter(angles_rad, df['radius'], c=df['temperature'], s=30, 
                                  alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 0].set_title('Temperature vs Radius', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0, 0], shrink=0.8, label='Temperature')
    
    # Pressure-based plot
    df_pressure = pd.read_sql_query("SELECT angle, radius, pressure FROM measurements", conn)
    angles_rad_p = np.radians(df_pressure['angle'])
    scatter2 = axes[0, 1].scatter(angles_rad_p, df_pressure['radius'], c=df_pressure['pressure'], s=30, 
                                  alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)
    axes[0, 1].set_title('Pressure vs Radius', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[0, 1], shrink=0.8, label='Pressure')
    
    # Humidity-based plot
    df_humidity = pd.read_sql_query("SELECT angle, radius, humidity FROM measurements", conn)
    angles_rad_h = np.radians(df_humidity['angle'])
    scatter3 = axes[1, 0].scatter(angles_rad_h, df_humidity['radius'], c=df_humidity['humidity'], s=30, 
                                  alpha=0.7, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_title('Humidity vs Radius', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=axes[1, 0], shrink=0.8, label='Humidity')
    
    # Combined plot
    scatter4 = axes[1, 1].scatter(angles_rad, df['radius'], c=df['temperature'], s=30, 
                                  alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Combined Analysis', fontsize=12, fontweight='bold')
    plt.colorbar(scatter4, ax=axes[1, 1], shrink=0.8, label='Temperature')
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_polar_variations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Wind rose style plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': 'polar'})
    
    # Create wind rose for different variables
    variables = ['temperature', 'pressure', 'humidity', 'radius']
    titles = ['Temperature Rose', 'Pressure Rose', 'Humidity Rose', 'Radius Rose']
    cmaps = ['viridis', 'plasma', 'coolwarm', 'viridis']
    
    for i, (var, title, cmap) in enumerate(zip(variables, titles, cmaps)):
        df_var = pd.read_sql_query(f"SELECT angle, {var} FROM measurements", conn)
        
        # Create histogram of angles
        angle_bins = np.linspace(0, 360, 13)  # 12 bins (30 degrees each)
        hist, bin_edges = np.histogram(df_var['angle'], bins=angle_bins, weights=df_var[var])
        
        # Convert to polar coordinates
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rad = np.radians(bin_centers)
        
        # Create bars
        bars = axes[i//2, i%2].bar(bin_centers_rad, hist, width=np.radians(30), 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        
        # Color bars by value
        for j, (bar, val) in enumerate(zip(bars, hist)):
            if val > 0:
                bar.set_facecolor(plt.cm.get_cmap(cmap)(val / max(hist)))
        
        axes[i//2, i%2].set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_polar_wind_roses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced polar plot variations completed!")

def test_advanced_histogram_variations(test_folder):
    """Test various advanced histogram configurations."""
    print("📊 Testing advanced histogram variations...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Multiple variable histograms
    variables = ['amount', 'profit_margin', 'quantity', 'rating']
    titles = ['Sales Amount', 'Profit Margin', 'Quantity', 'Rating']
    colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
        df = pd.read_sql_query(f"SELECT {var} FROM sales LIMIT 100000", conn)
        values = df[var].dropna().values
        
        if len(values) > 0:
            # Create histogram
            n, bins, patches = axes[i//2, i%2].hist(values, bins=50, color=color, alpha=0.7, 
                                                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[i//2, i%2].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                                   label=f'Mean: {mean_val:.2f}')
            axes[i//2, i%2].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.8, 
                                   label=f'Mean+Std: {mean_val + std_val:.2f}')
            axes[i//2, i%2].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.8, 
                                   label=f'Mean-Std: {mean_val - std_val:.2f}')
            
            axes[i//2, i%2].set_title(title, fontsize=12, fontweight='bold')
            axes[i//2, i%2].set_xlabel(var.replace('_', ' ').title(), fontsize=10)
            axes[i//2, i%2].set_ylabel('Frequency', fontsize=10)
            axes[i//2, i%2].grid(True, alpha=0.3)
            axes[i//2, i%2].legend()
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_histogram_multiple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Distribution comparison histograms
    df_dist = pd.read_sql_query("SELECT normal_dist, exponential_dist, uniform_dist, bimodal_dist FROM distributions LIMIT 100000", conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    distributions = ['normal_dist', 'exponential_dist', 'uniform_dist', 'bimodal_dist']
    dist_titles = ['Normal Distribution', 'Exponential Distribution', 'Uniform Distribution', 'Bimodal Distribution']
    dist_colors = ['blue', 'red', 'green', 'purple']
    
    for i, (dist, title, color) in enumerate(zip(distributions, dist_titles, dist_colors)):
        values = df_dist[dist].dropna().values
        
        if len(values) > 0:
            # Create histogram
            axes[i//2, i%2].hist(values, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            
            # Add fitted normal curve
            mu, sigma = np.mean(values), np.std(values)
            x = np.linspace(values.min(), values.max(), 100)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            y_scaled = y * len(values) * (x[1] - x[0])
            axes[i//2, i%2].plot(x, y_scaled, 'r-', linewidth=2, alpha=0.8, label='Normal Fit')
            
            axes[i//2, i%2].set_title(title, fontsize=12, fontweight='bold')
            axes[i//2, i%2].set_xlabel('Values', fontsize=10)
            axes[i//2, i%2].set_ylabel('Frequency', fontsize=10)
            axes[i//2, i%2].grid(True, alpha=0.3)
            axes[i//2, i%2].legend()
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_histogram_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 3: Log-scale and cumulative histograms
    df_sales = pd.read_sql_query("SELECT amount FROM sales LIMIT 100000", conn)
    values = df_sales['amount'].dropna().values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Regular histogram
    axes[0, 0].hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Regular Histogram', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Amount', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-scale histogram
    axes[0, 1].hist(values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Log-Scale Histogram', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Amount', fontsize=10)
    axes[0, 1].set_ylabel('Frequency (log)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative histogram
    axes[1, 0].hist(values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5, 
                    cumulative=True, density=True)
    axes[1, 0].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Amount', fontsize=10)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Density histogram
    axes[1, 1].hist(values, bins=50, alpha=0.7, color='gold', edgecolor='black', linewidth=0.5, density=True)
    axes[1, 1].set_title('Density Histogram', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Amount', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'advanced_histogram_variations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Advanced histogram variations completed!")

def test_specialized_polar_plots(test_folder):
    """Test specialized polar plot types."""
    print("📊 Testing specialized polar plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Radar chart style
    df = pd.read_sql_query("SELECT category, AVG(amount) as avg_amount, AVG(profit_margin) as avg_profit FROM sales GROUP BY category", conn)
    
    # Create radar chart
    categories = df['category'].values
    avg_amounts = df['avg_amount'].values
    avg_profits = df['avg_profit'].values
    
    # Normalize values for radar chart
    amount_norm = (avg_amounts - avg_amounts.min()) / (avg_amounts.max() - avg_amounts.min())
    profit_norm = (avg_profits - avg_profits.min()) / (avg_profits.max() - avg_profits.min())
    
    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot radar chart - ensure arrays have same length
    amount_norm = np.append(amount_norm, amount_norm[0])  # Complete the circle
    profit_norm = np.append(profit_norm, profit_norm[0])  # Complete the circle
    
    ax.plot(angles, amount_norm, 'o-', linewidth=2, label='Average Amount', color='blue')
    ax.fill(angles, amount_norm, alpha=0.25, color='blue')
    
    ax.plot(angles, profit_norm, 'o-', linewidth=2, label='Average Profit', color='red')
    ax.fill(angles, profit_norm, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Sales Performance Radar Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(test_folder / 'specialized_polar_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Polar heatmap
    df_heat = pd.read_sql_query("SELECT angle, radius, temperature FROM measurements", conn)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Create polar heatmap
    angles_rad = np.radians(df_heat['angle'])
    scatter = ax.scatter(angles_rad, df_heat['radius'], c=df_heat['temperature'], s=50, 
                        alpha=0.8, cmap='viridis', edgecolors='black', linewidth=0.5)
    
    ax.set_title('Polar Temperature Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Temperature')
    
    plt.tight_layout()
    plt.savefig(test_folder / 'specialized_polar_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Specialized polar plots completed!")

def test_specialized_histogram_plots(test_folder):
    """Test specialized histogram plot types."""
    print("📊 Testing specialized histogram plots...")
    
    conn = sqlite3.connect('data.sqlite')
    
    # Test 1: Stacked histogram by category
    df = pd.read_sql_query("SELECT category, amount FROM sales LIMIT 100000", conn)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = df['category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Create stacked histogram
    for i, (cat, color) in enumerate(zip(categories, colors)):
        cat_data = df[df['category'] == cat]['amount'].values
        if len(cat_data) > 0:
            ax.hist(cat_data, bins=30, alpha=0.7, color=color, label=cat, 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_title('Stacked Histogram by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Amount', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'specialized_histogram_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test 2: Multiple histogram comparison
    df_multi = pd.read_sql_query("SELECT amount, profit_margin, quantity, rating FROM sales LIMIT 100000", conn)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    variables = ['amount', 'profit_margin', 'quantity', 'rating']
    titles = ['Sales Amount', 'Profit Margin', 'Quantity', 'Rating']
    colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
        values = df_multi[var].dropna().values
        
        if len(values) > 0:
            # Create histogram with KDE
            axes[i//2, i%2].hist(values, bins=30, alpha=0.7, color=color, 
                                edgecolor='black', linewidth=0.5, density=True, label='Histogram')
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_kde = np.linspace(values.min(), values.max(), 100)
            y_kde = kde(x_kde)
            axes[i//2, i%2].plot(x_kde, y_kde, 'r-', linewidth=2, alpha=0.8, label='KDE')
            
            # Add normal fit
            mu, sigma = np.mean(values), np.std(values)
            x_norm = np.linspace(values.min(), values.max(), 100)
            y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
            axes[i//2, i%2].plot(x_norm, y_norm, 'g--', linewidth=2, alpha=0.8, label='Normal Fit')
            
            axes[i//2, i%2].set_title(title, fontsize=12, fontweight='bold')
            axes[i//2, i%2].set_xlabel(var.replace('_', ' ').title(), fontsize=10)
            axes[i//2, i%2].set_ylabel('Density', fontsize=10)
            axes[i//2, i%2].legend()
            axes[i//2, i%2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(test_folder / 'specialized_histogram_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    conn.close()
    print("✅ Specialized histogram plots completed!")

def main():
    """Run all extended tests and generate plots."""
    print("🚀 Starting extended advanced plot testing...")
    
    # Create test folder
    test_folder = create_extended_test_folder()
    print(f"📁 Extended test plots will be saved to: {test_folder}")
    
    # Check if database exists
    if not os.path.exists('data.sqlite'):
        print("❌ Database not found! Please run create_comprehensive_db.py first.")
        return
    
    # Run all extended tests
    test_advanced_polar_variations(test_folder)
    test_advanced_histogram_variations(test_folder)
    test_specialized_polar_plots(test_folder)
    test_specialized_histogram_plots(test_folder)
    
    print(f"\n🎉 All extended tests completed! Check the '{test_folder}' folder for generated plots.")
    print(f"📊 Generated {len(list(test_folder.glob('*.png')))} extended plot files.")

if __name__ == "__main__":
    main() 