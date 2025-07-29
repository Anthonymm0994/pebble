#!/usr/bin/env python3
"""
Example usage of the Table Relationship Analyzer
===============================================

This script demonstrates how to use the analyzer with a sample database.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

def create_sample_database():
    """Create a sample database with two related tables."""
    print("🔧 Creating sample database...")
    
    # Create database
    conn = sqlite3.connect('sample_analysis.db')
    
    # Create original table (source data)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS source_messages (
            id INTEGER PRIMARY KEY,
            message_id TEXT,
            message_time TEXT,
            user_id TEXT,
            content TEXT,
            priority INTEGER,
            status TEXT
        )
    ''')
    
    # Create derived table (processed data)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS processed_messages (
            id INTEGER PRIMARY KEY,
            processed_id TEXT,
            processed_time TEXT,
            user_id TEXT,
            message_length INTEGER,
            priority_level TEXT,
            is_urgent BOOLEAN
        )
    ''')
    
    # Generate sample data for source table
    source_data = []
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    for i in range(100):
        # Add some processing delay (0-30 seconds)
        delay = random.randint(0, 30)
        message_time = base_time + timedelta(minutes=i, seconds=delay)
        processed_time = message_time + timedelta(seconds=random.randint(5, 60))
        
        source_data.append({
            'id': i + 1,
            'message_id': f'MSG_{i:04d}',
            'message_time': message_time.strftime('%H:%M:%S.%f')[:-3],
            'user_id': f'USER_{random.randint(1, 10):02d}',
            'content': f'Message content {i}',
            'priority': random.randint(1, 5),
            'status': random.choice(['pending', 'sent', 'delivered'])
        })
    
    # Insert source data
    df_source = pd.DataFrame(source_data)
    df_source.to_sql('source_messages', conn, if_exists='replace', index=False)
    
    # Generate processed data (derived from source with transformations)
    processed_data = []
    
    for i, row in df_source.iterrows():
        # Only process messages with priority > 2 (filtering)
        if row['priority'] > 2:
            processed_data.append({
                'id': len(processed_data) + 1,
                'processed_id': f'PROC_{row["message_id"]}',
                'processed_time': (datetime.strptime(row['message_time'], '%H:%M:%S.%f') + 
                                 timedelta(seconds=random.randint(5, 60))).strftime('%H:%M:%S.%f')[:-3],
                'user_id': row['user_id'],
                'message_length': len(row['content']),
                'priority_level': 'HIGH' if row['priority'] >= 4 else 'MEDIUM',
                'is_urgent': row['priority'] >= 4
            })
    
    # Insert processed data
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_sql('processed_messages', conn, if_exists='replace', index=False)
    
    print(f"✅ Created source_messages: {len(df_source)} rows")
    print(f"✅ Created processed_messages: {len(df_processed)} rows")
    
    conn.close()
    return 'sample_analysis.db'

def run_analysis():
    """Run the table relationship analysis."""
    from table_relationship_analyzer import TableRelationshipAnalyzer
    
    # Create sample database
    db_path = create_sample_database()
    
    print("\n" + "="*80)
    print("🔍 RUNNING TABLE RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # Create analyzer and run analysis
    analyzer = TableRelationshipAnalyzer(db_path)
    
    try:
        results = analyzer.run_comprehensive_analysis('source_messages', 'processed_messages')
        
        if results:
            print("\n📊 ANALYSIS RESULTS SUMMARY:")
            print(f"  - Column similarities: {len(results['similarities'])}")
            print(f"  - Timestamp relationships: {len(results['timestamp_analysis'])}")
            print(f"  - Join suggestions: {len(results['join_suggestions'])}")
            
            # Show some key findings
            if results['timestamp_analysis']:
                print("\n🕒 KEY TIMESTAMP FINDINGS:")
                for pair, analysis in results['timestamp_analysis'].items():
                    print(f"  {pair}: {analysis['match_count']} matches, "
                          f"avg delay: {analysis['mean_delay']:.2f}s")
            
            if results['join_suggestions']:
                print("\n🔗 TOP JOIN SUGGESTIONS:")
                for i, suggestion in enumerate(results['join_suggestions'][:3]):
                    print(f"  {i+1}. {suggestion['type']}: "
                          f"{suggestion['columns'][0]} ↔ {suggestion['columns'][1]} "
                          f"(confidence: {suggestion['confidence']:.3f})")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    run_analysis() 