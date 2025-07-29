#!/usr/bin/env python3
"""
Create sample database from CSV file for testing histogram permutations.
"""

import pandas as pd
import sqlite3

def create_sample_database():
    """Create a sample SQLite database from the CSV file."""
    print("Creating sample database from sample_data.csv...")
    
    # Read the CSV file
    df = pd.read_csv('sample_data.csv')
    print(f"Loaded {len(df)} rows from sample_data.csv")
    
    # Create SQLite database
    conn = sqlite3.connect('sample_data.db')
    
    # Write to database
    df.to_sql('sample_data', conn, if_exists='replace', index=False)
    
    # Verify the data
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sample_data")
    count = cursor.fetchone()[0]
    print(f"Database created with {count} rows")
    
    # Show table structure
    cursor.execute("PRAGMA table_info(sample_data)")
    columns = cursor.fetchall()
    print("Table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    print("Sample database 'sample_data.db' created successfully!")

if __name__ == "__main__":
    create_sample_database() 