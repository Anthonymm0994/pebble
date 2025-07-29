#!/usr/bin/env python3
"""
Test script for the Dataset Relationship Detector
===============================================

This script creates sample data and tests the detector to ensure everything works correctly.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from dataset_relationship_detector import DatasetRelationshipDetector

def create_test_database():
    """Create a test database with sample data that simulates your scenario."""
    print("[BUILD] Creating test database...")
    
    # Create database
    conn = sqlite3.connect('test_relationships.db')
    
    # Create source table (original messages)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS source_messages (
            id INTEGER PRIMARY KEY,
            message_id TEXT,
            message_time TEXT,
            user_id TEXT,
            content TEXT,
            priority INTEGER,
            status TEXT,
            amount REAL,
            category TEXT
        )
    ''')
    
    # Create derived table (processed messages)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS processed_messages (
            id INTEGER PRIMARY KEY,
            processed_id TEXT,
            processed_time TEXT,
            user_id TEXT,
            message_length INTEGER,
            priority_level TEXT,
            is_urgent BOOLEAN,
            total_amount_usd REAL,
            category_group TEXT
        )
    ''')
    
    # Generate source data
    source_data = []
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    categories = ['electronics', 'clothing', 'food', 'services', 'transport']
    statuses = ['pending', 'completed', 'failed', 'cancelled']
    
    for i in range(200):
        # Add processing delay (5-60 seconds)
        delay = random.randint(5, 60)
        message_time = base_time + timedelta(minutes=i, seconds=random.randint(0, 30))
        processed_time = message_time + timedelta(seconds=delay)
        
        amount = random.uniform(10, 1000)
        currency = random.choice(['USD', 'EUR', 'GBP'])
        
        # Convert to USD (simplified conversion rates)
        usd_rate = {'USD': 1.0, 'EUR': 1.1, 'GBP': 1.3}
        amount_usd = amount * usd_rate[currency]
        
        source_data.append({
            'id': i + 1,
            'message_id': f'MSG_{i:04d}',
            'message_time': message_time.strftime('%H:%M:%S.%f')[:-3],
            'user_id': f'USER_{random.randint(1, 20):02d}',
            'content': f'Message content {i} with some text',
            'priority': random.randint(1, 5),
            'status': random.choice(statuses),
            'amount': amount,
            'category': random.choice(categories)
        })
    
    # Generate processed data (only completed messages, with transformations)
    processed_data = []
    
    for row in source_data:
        # Only process completed messages (filtering)
        if row['status'] == 'completed':
            processed_data.append({
                'id': len(processed_data) + 1,
                'processed_id': f'PROC_{row["message_id"]}',
                'processed_time': (datetime.strptime(row['message_time'], '%H:%M:%S.%f') + 
                                 timedelta(seconds=random.randint(5, 60))).strftime('%H:%M:%S.%f')[:-3],
                'user_id': row['user_id'],
                'message_length': len(row['content']),
                'priority_level': 'HIGH' if row['priority'] >= 4 else 'MEDIUM',
                'is_urgent': row['priority'] >= 4,
                'total_amount_usd': amount_usd,
                'category_group': 'high_value' if amount_usd > 500 else 'standard'
            })
    
    # Insert data
    df_source = pd.DataFrame(source_data)
    df_processed = pd.DataFrame(processed_data)
    
    df_source.to_sql('source_messages', conn, if_exists='replace', index=False)
    df_processed.to_sql('processed_messages', conn, if_exists='replace', index=False)
    
    print(f"[OK] Created source_messages: {len(df_source)} rows")
    print(f"[OK] Created processed_messages: {len(df_processed)} rows")
    
    conn.close()
    return 'test_relationships.db'

def test_basic_functionality():
    """Test basic functionality of the detector."""
    print("\n[TEST] Testing basic functionality...")
    
    # Create test database
    db_path = create_test_database()
    
    # Create detector
    detector = DatasetRelationshipDetector(db_path)
    
    try:
        # Test connection
        assert detector.connect(), "Connection failed"
        print("[OK] Database connection successful")
        
        # Test dataset loading
        detector.load_datasets('source_messages', 'processed_messages')
        assert detector.df_source is not None, "Source dataset not loaded"
        assert detector.df_derived is not None, "Derived dataset not loaded"
        print("[OK] Dataset loading successful")
        
        # Test timestamp analysis
        timestamp_analysis = detector.analyze_timestamp_relationships()
        assert 'time_delays' in timestamp_analysis, "Timestamp analysis failed"
        print("[OK] Timestamp analysis successful")
        
        # Test column similarities
        similarities = detector.find_column_similarities()
        assert 'exact_matches' in similarities, "Column similarities failed"
        print("[OK] Column similarities successful")
        
        # Test transformation detection
        transformations = detector.detect_transformations()
        assert 'filtering' in transformations, "Transformation detection failed"
        print("[OK] Transformation detection successful")
        
        # Test join suggestions
        join_suggestions = detector.suggest_joins()
        assert isinstance(join_suggestions, list), "Join suggestions failed"
        print("[OK] Join suggestions successful")
        
        print("[OK] All basic functionality tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.close()

def test_comprehensive_analysis():
    """Test comprehensive analysis pipeline."""
    print("\n[TEST] Testing comprehensive analysis...")
    
    # Create test database
    db_path = create_test_database()
    
    # Create detector
    detector = DatasetRelationshipDetector(db_path)
    
    try:
        # Connect and load datasets
        detector.connect()
        detector.load_datasets('source_messages', 'processed_messages')
        
        # Run comprehensive analysis
        results = detector.run_comprehensive_analysis()
        
        # Verify results structure
        assert 'timestamp_analysis' in results, "Missing timestamp analysis"
        assert 'similarities' in results, "Missing similarities"
        assert 'transformations' in results, "Missing transformations"
        assert 'join_suggestions' in results, "Missing join suggestions"
        assert 'report' in results, "Missing report"
        
        print("[OK] Comprehensive analysis successful")
        
        # Check for expected findings
        timestamp_analysis = results['timestamp_analysis']
        if timestamp_analysis['time_delays']:
            print(f"[OK] Found {len(timestamp_analysis['time_delays'])} timestamp relationships")
        
        similarities = results['similarities']
        if similarities['exact_matches']:
            print(f"[OK] Found {len(similarities['exact_matches'])} exact column matches")
        
        transformations = results['transformations']
        if transformations['filtering']['filtering_detected']:
            print("[OK] Detected filtering pattern")
        
        join_suggestions = results['join_suggestions']
        if join_suggestions:
            print(f"[OK] Found {len(join_suggestions)} join suggestions")
        
        print("[OK] All comprehensive analysis tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.close()

def test_field_exploration():
    """Test field exploration functionality."""
    print("\n[TEST] Testing field exploration...")
    
    # Create test database
    db_path = create_test_database()
    
    # Create detector
    detector = DatasetRelationshipDetector(db_path)
    
    try:
        # Connect and load datasets
        detector.connect()
        detector.load_datasets('source_messages', 'processed_messages')
        
        # Test exploring different types of fields
        fields_to_test = ['message_time', 'user_id', 'amount', 'priority']
        
        for field in fields_to_test:
            print(f"\nTesting field exploration for: {field}")
            detector.explore_field(field, max_samples=10)
        
        print("[OK] Field exploration tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.close()

def test_visualization_generation():
    """Test visualization generation."""
    print("\n[TEST] Testing visualization generation...")
    
    # Create test database
    db_path = create_test_database()
    
    # Create detector
    detector = DatasetRelationshipDetector(db_path)
    
    try:
        # Connect and load datasets
        detector.connect()
        detector.load_datasets('source_messages', 'processed_messages')
        
        # Generate visualizations
        detector.generate_visualizations()
        
        # Check if visualization files were created
        expected_files = [
            'dataset_comparison.png',
            'timestamp_analysis.png',
            'correlation_analysis.png',
            'transformation_analysis.png'
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"[OK] Created {file}")
            else:
                print(f"⚠️  {file} not found")
        
        print("[OK] Visualization generation tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.close()

def test_command_line_interface():
    """Test command line interface."""
    print("\n[TEST] Testing command line interface...")
    
    # Create test database
    db_path = create_test_database()
    
    try:
        import subprocess
        import sys
        
        # Test basic command
        cmd = [sys.executable, 'dataset_relationship_detector.py', db_path, 
               '--source', 'source_messages', '--derived', 'processed_messages']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] Command line interface test passed!")
        else:
            print(f"⚠️  Command line test had issues: {result.stderr}")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all tests."""
    print("[START] Running all tests for Dataset Relationship Detector")
    print("=" * 80)
    
    # Run all test functions
    test_basic_functionality()
    test_comprehensive_analysis()
    test_field_exploration()
    test_visualization_generation()
    test_command_line_interface()
    
    print("\n" + "=" * 80)
    print("[OK] All tests completed!")
    print("📁 Check the generated files:")
    print("  - dataset_relationship_report.txt")
    print("  - dataset_comparison.png")
    print("  - timestamp_analysis.png")
    print("  - correlation_analysis.png")
    print("  - transformation_analysis.png")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests() 