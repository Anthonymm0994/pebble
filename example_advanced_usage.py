#!/usr/bin/env python3
"""
Example Advanced Dataset Analyzer Usage
======================================

This script demonstrates how to use the advanced dataset analyzer
with custom configurations and detailed field exploration.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from advanced_dataset_analyzer import AdvancedDatasetAnalyzer
from analysis_config import ANALYSIS_CONFIG, get_config_for_financial_data, get_config_for_log_data

def create_advanced_sample_database():
    """Create a more complex sample database with various transformation patterns."""
    print("🔧 Creating advanced sample database...")
    
    # Create database
    conn = sqlite3.connect('advanced_sample.db')
    
    # Create source table with various data types
    conn.execute('''
        CREATE TABLE IF NOT EXISTS source_transactions (
            id INTEGER PRIMARY KEY,
            transaction_id TEXT,
            transaction_time TEXT,
            user_id TEXT,
            amount REAL,
            currency TEXT,
            category TEXT,
            status TEXT,
            priority INTEGER,
            metadata TEXT
        )
    ''')
    
    # Create derived table with transformations
    conn.execute('''
        CREATE TABLE IF NOT EXISTS processed_transactions (
            id INTEGER PRIMARY KEY,
            processed_id TEXT,
            processed_time TEXT,
            user_id TEXT,
            total_amount_usd REAL,
            risk_score REAL,
            category_group TEXT,
            is_high_value BOOLEAN,
            processing_delay_seconds INTEGER
        )
    ''')
    
    # Generate source data with various patterns
    source_data = []
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    categories = ['electronics', 'clothing', 'food', 'services', 'transport']
    statuses = ['pending', 'completed', 'failed', 'cancelled']
    
    for i in range(500):
        # Add processing delay (0-120 seconds)
        delay = random.randint(0, 120)
        transaction_time = base_time + timedelta(minutes=i, seconds=random.randint(0, 30))
        processed_time = transaction_time + timedelta(seconds=delay)
        
        amount = random.uniform(10, 1000)
        currency = random.choice(currencies)
        
        # Convert to USD (simplified conversion rates)
        usd_rate = {'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3, 'JPY': 0.007}
        amount_usd = amount * usd_rate[currency]
        
        # Calculate risk score based on amount and category
        risk_score = min(100, (amount_usd / 100) + random.uniform(-10, 10))
        
        # Determine if high value
        is_high_value = amount_usd > 500
        
        # Group categories
        category_group = 'high_value' if amount_usd > 500 else 'standard'
        
        source_data.append({
            'id': i + 1,
            'transaction_id': f'TXN_{i:06d}',
            'transaction_time': transaction_time.strftime('%H:%M:%S.%f')[:-3],
            'user_id': f'USER_{random.randint(1, 50):03d}',
            'amount': amount,
            'currency': currency,
            'category': random.choice(categories),
            'status': random.choice(statuses),
            'priority': random.randint(1, 5),
            'metadata': f'{{"source": "web", "session_id": "sess_{i}"}}'
        })
        
        # Only include completed transactions in derived dataset (filtering)
        if source_data[-1]['status'] == 'completed':
            processed_data.append({
                'id': len(processed_data) + 1,
                'processed_id': f'PROC_{source_data[-1]["transaction_id"]}',
                'processed_time': processed_time.strftime('%H:%M:%S.%f')[:-3],
                'user_id': source_data[-1]['user_id'],
                'total_amount_usd': amount_usd,
                'risk_score': risk_score,
                'category_group': category_group,
                'is_high_value': is_high_value,
                'processing_delay_seconds': delay
            })
    
    # Insert source data
    df_source = pd.DataFrame(source_data)
    df_source.to_sql('source_transactions', conn, if_exists='replace', index=False)
    
    # Insert processed data
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_sql('processed_transactions', conn, if_exists='replace', index=False)
    
    print(f"✅ Created source_transactions: {len(df_source)} rows")
    print(f"✅ Created processed_transactions: {len(df_processed)} rows")
    
    conn.close()
    return 'advanced_sample.db'

def run_basic_analysis():
    """Run basic analysis with default configuration."""
    print("\n" + "="*80)
    print("🔍 RUNNING BASIC ANALYSIS")
    print("="*80)
    
    # Create sample database
    db_path = create_advanced_sample_database()
    
    # Create analyzer with default config
    analyzer = AdvancedDatasetAnalyzer(db_path)
    
    try:
        # Connect and load datasets
        analyzer.connect()
        analyzer.load_datasets('source_transactions', 'processed_transactions')
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis()
        
        # Generate report
        report = analyzer.generate_report()
        
        print("\n📊 BASIC ANALYSIS RESULTS:")
        print(f"  - Filtering detected: {results.get('basic_comparison', {}).get('filtering_detected', False)}")
        print(f"  - Column relationships: {len(results.get('column_relationships', {}).get('exact_matches', []))}")
        print(f"  - Temporal relationships: {len(results.get('temporal_analysis', {}).get('time_delays', []))}")
        
    except Exception as e:
        print(f"❌ Error during basic analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

def run_financial_analysis():
    """Run analysis optimized for financial data."""
    print("\n" + "="*80)
    print("💰 RUNNING FINANCIAL DATA ANALYSIS")
    print("="*80)
    
    # Create sample database
    db_path = create_advanced_sample_database()
    
    # Get financial-specific configuration
    financial_config = get_config_for_financial_data()
    
    # Create analyzer with financial config
    analyzer = AdvancedDatasetAnalyzer(db_path, financial_config)
    
    try:
        # Connect and load datasets
        analyzer.connect()
        analyzer.load_datasets('source_transactions', 'processed_transactions')
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis()
        
        # Generate report
        report = analyzer.generate_report()
        
        print("\n💰 FINANCIAL ANALYSIS RESULTS:")
        print(f"  - Value transformations: {len(results.get('value_transformations', {}).get('mathematical_transforms', []))}")
        print(f"  - Currency conversions detected: {len([t for t in results.get('value_transformations', {}).get('mathematical_transforms', []) if 'scaling' in str(t)])}")
        print(f"  - Risk scoring patterns: {len([t for t in results.get('value_transformations', {}).get('categorical_transforms', []) if 'risk' in str(t)])}")
        
    except Exception as e:
        print(f"❌ Error during financial analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

def explore_specific_fields():
    """Demonstrate field-specific exploration."""
    print("\n" + "="*80)
    print("🔍 FIELD-SPECIFIC EXPLORATION")
    print("="*80)
    
    # Create sample database
    db_path = create_advanced_sample_database()
    
    # Create analyzer
    analyzer = AdvancedDatasetAnalyzer(db_path)
    
    try:
        # Connect and load datasets
        analyzer.connect()
        analyzer.load_datasets('source_transactions', 'processed_transactions')
        
        # Explore specific fields
        fields_to_explore = [
            'transaction_time',      # Temporal field
            'amount',               # Numeric field
            'currency',             # Categorical field
            'user_id',              # Identifier field
            'total_amount_usd'      # Transformed field
        ]
        
        for field in fields_to_explore:
            print(f"\n{'='*60}")
            analyzer.explore_specific_field(field, max_samples=20)
        
    except Exception as e:
        print(f"❌ Error during field exploration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

def run_custom_analysis():
    """Run analysis with custom configuration."""
    print("\n" + "="*80)
    print("⚙️ RUNNING CUSTOM ANALYSIS")
    print("="*80)
    
    # Create sample database
    db_path = create_advanced_sample_database()
    
    # Custom configuration
    custom_config = ANALYSIS_CONFIG.copy()
    custom_config.update({
        'correlation_threshold': 0.6,      # Lower threshold for more relationships
        'similarity_threshold': 0.4,       # Lower threshold for name matching
        'time_window_seconds': 180,        # Shorter time window
        'focus_areas': {
            'temporal_analysis': True,
            'value_transformations': True,
            'filtering_detection': True,
            'aggregation_detection': False,  # Disable aggregation detection
            'correlation_analysis': True,
            'pattern_detection': True,
            'heuristic_analysis': True,
        }
    })
    
    # Create analyzer with custom config
    analyzer = AdvancedDatasetAnalyzer(db_path, custom_config)
    
    try:
        # Connect and load datasets
        analyzer.connect()
        analyzer.load_datasets('source_transactions', 'processed_transactions')
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis()
        
        # Generate report
        report = analyzer.generate_report()
        
        print("\n⚙️ CUSTOM ANALYSIS RESULTS:")
        print(f"  - Custom correlation threshold: {custom_config['correlation_threshold']}")
        print(f"  - Custom similarity threshold: {custom_config['similarity_threshold']}")
        print(f"  - Time window: {custom_config['time_window_seconds']} seconds")
        print(f"  - Aggregation detection disabled: {not custom_config['focus_areas']['aggregation_detection']}")
        
    except Exception as e:
        print(f"❌ Error during custom analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

def demonstrate_interactive_features():
    """Demonstrate interactive features and customization."""
    print("\n" + "="*80)
    print("🎯 INTERACTIVE FEATURES DEMONSTRATION")
    print("="*80)
    
    # Create sample database
    db_path = create_advanced_sample_database()
    
    # Create analyzer
    analyzer = AdvancedDatasetAnalyzer(db_path)
    
    try:
        # Connect and load datasets
        analyzer.connect()
        analyzer.load_datasets('source_transactions', 'processed_transactions')
        
        # Run analysis
        results = analyzer.comprehensive_analysis()
        
        # Demonstrate interactive features
        print("\n🎯 Interactive Features:")
        print("1. Field exploration - explore specific fields in detail")
        print("2. Custom configuration - modify analysis parameters")
        print("3. Focus areas - enable/disable specific analysis types")
        print("4. Visualization customization - modify plot styles and colors")
        print("5. Report customization - control what's included in reports")
        
        # Show how to explore a specific field
        print("\n📊 Example: Exploring 'amount' field transformation:")
        analyzer.explore_specific_field('amount')
        
        print("\n📊 Example: Exploring 'transaction_time' temporal patterns:")
        analyzer.explore_specific_field('transaction_time')
        
    except Exception as e:
        print(f"❌ Error during interactive demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

def main():
    """Run all demonstration examples."""
    print("🚀 ADVANCED DATASET ANALYZER DEMONSTRATION")
    print("=" * 80)
    
    # Run different types of analysis
    run_basic_analysis()
    run_financial_analysis()
    explore_specific_fields()
    run_custom_analysis()
    demonstrate_interactive_features()
    
    print("\n" + "="*80)
    print("✅ All demonstrations completed!")
    print("📁 Check the generated files:")
    print("  - advanced_analysis_report.txt")
    print("  - basic_comparison.png")
    print("  - correlation_analysis.png")
    print("  - temporal_analysis.png")
    print("  - interactive_comparison.html")
    print("="*80)

if __name__ == "__main__":
    main() 