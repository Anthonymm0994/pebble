#!/usr/bin/env python3
"""
Fix Unicode issues by replacing emoji characters with text equivalents.
"""

import re

def fix_unicode_in_file(filename):
    """Replace emoji characters with text equivalents."""
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace emoji characters with text equivalents
    replacements = {
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🕒': '[TIME]',
        '🔍': '[SEARCH]',
        '🔄': '[TRANSFORM]',
        '🔗': '[JOIN]',
        '📊': '[DATA]',
        '📝': '[REPORT]',
        '🚀': '[START]',
        '🎯': '[TARGET]',
        '🔧': '[BUILD]',
        '🧪': '[TEST]',
        '💡': '[IDEA]',
        '📋': '[SUMMARY]',
        '📈': '[CHART]',
        '⚙️': '[CONFIG]',
        '💰': '[MONEY]',
        '🔌': '[CONNECT]'
    }
    
    for emoji, text in replacements.items():
        content = content.replace(emoji, text)
    
    # Write the fixed content back
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode issues in {filename}")

if __name__ == "__main__":
    # Fix the main detector file
    fix_unicode_in_file('dataset_relationship_detector.py')
    fix_unicode_in_file('test_dataset_analyzer.py') 