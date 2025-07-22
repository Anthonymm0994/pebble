#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data.sqlite')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Available tables:")
for table in tables:
    print(f"  - {table[0]}")
    
    # Get column info for each table
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print(f"    Columns: {[col[1] for col in columns]}")
    print()

conn.close() 