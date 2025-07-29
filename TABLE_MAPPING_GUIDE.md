# Table Mapping Guide: Finding Relationships Between Derived Tables

## Overview
This guide helps you map relationships between two SQLite tables where:
- **tableA** is the original table
- **tableB** is derived from tableA (plus other sources)
- **Many-to-one relationship**: Multiple rows in tableA → One row in tableB
- **Some rows get dropped**: Not all tableA rows appear in tableB
- **Some columns are the same**: Direct mappings exist
- **Most columns are different**: Transformations and aggregations occur

## Prerequisites
- SQLite database with your two tables
- Python with Anaconda libraries (pandas, numpy, matplotlib, seaborn, scipy)
- All scripts from the `scripts/` directory

---

## Step 1: Initial Table Analysis

### 1.1 Basic Table Information
First, understand your tables' structure:

```bash
# Navigate to scripts directory
cd scripts/

# Check what tables exist in your database
python data_explorer.py your_database.sqlite --list-tables

# Get detailed info about both tables
python data_explorer.py your_database.sqlite --table tableA
python data_explorer.py your_database.sqlite --table tableB
```

**What this tells you:**
- Row counts (tableA should have more rows than tableB)
- Column names and data types
- Basic statistics for each column

### 1.2 Data Profiling
Get deeper insights into your data:

```bash
# Profile tableA
python data_profiler.py your_database.sqlite --table tableA

# Profile tableB  
python data_profiler.py your_database.sqlite --table tableB
```

**What this tells you:**
- Data quality issues
- Missing values patterns
- Value distributions
- Potential transformation clues

---

## Step 2: Comprehensive Relationship Analysis

### 2.1 Run the Table Relationship Analyzer
This is your main tool for finding the mapping:

```bash
python table_relationship_analyzer.py your_database.sqlite tableA tableB
```

**What this reveals:**
- **Exact column matches** between tableA and tableB
- **Similar column names** (fuzzy matching)
- **Timestamp correlations** (for processing delays)
- **Transformation patterns** (filters, aggregations)
- **Join suggestions** for connecting tables

### 2.2 Review the Analysis Results
Check the generated files in `../outputs/relationship_outputs/`:
- `relationship_summary.txt` - Text summary of findings
- `relationship_analysis.json` - Detailed JSON data
- `table_comparison.png` - Visual comparison
- `column_similarity_heatmap.png` - Column relationship heatmap

**Key things to look for:**
- Which columns have exact matches?
- What's the correlation between timestamp columns?
- Are there clear filtering patterns?
- What aggregation patterns are detected?

---

## Step 3: Deep Dive Analysis

### 3.1 Analyze Specific Columns
If you found potential matches, analyze them in detail:

```bash
# For numeric columns that might be related
python advanced_histogram_analysis.py your_database.sqlite --table tableA --column column_name
python advanced_histogram_analysis.py your_database.sqlite --table tableB --column column_name

# Compare distributions between tables
python advanced_histogram_analysis.py your_database.sqlite --table tableA --column column_name --compare tableB.column_name
```

### 3.2 Test Join Hypotheses
Based on the relationship analysis, test potential joins:

```bash
# If you found timestamp correlations, test temporal joins
python interactive_plots.py your_database.sqlite --table tableA

# Create custom queries to test relationships
python data_explorer.py your_database.sqlite --query "SELECT * FROM tableA WHERE column_name IN (SELECT column_name FROM tableB)"
```

---

## Step 4: Transformation Detection

### 4.1 Identify Filtering Patterns
Look for rows that exist in tableA but not tableB:

```bash
# Use the relationship analyzer results to identify filters
# Check the 'filters' section in the JSON output

# Test specific filtering hypotheses
python data_explorer.py your_database.sqlite --query "
SELECT COUNT(*) as tableA_count FROM tableA 
WHERE column_name NOT IN (SELECT column_name FROM tableB)
"
```

### 4.2 Detect Aggregations
Identify many-to-one relationships:

```bash
# Compare row counts
python data_explorer.py your_database.sqlite --query "
SELECT 
  (SELECT COUNT(*) FROM tableA) as tableA_rows,
  (SELECT COUNT(*) FROM tableB) as tableB_rows,
  (SELECT COUNT(*) FROM tableA) - (SELECT COUNT(*) FROM tableB) as difference
"

# Look for aggregation patterns in numeric columns
python histogram_permutations.py your_database.sqlite --query "
SELECT 
  source_column,
  COUNT(*) as count,
  AVG(value_column) as avg_value,
  SUM(value_column) as sum_value
FROM tableA 
GROUP BY source_column
"
```

---

## Step 5: Validation and Testing

### 5.1 Create Test Queries
Based on your findings, create queries to validate the mapping:

```bash
# Test exact column matches
python data_explorer.py your_database.sqlite --query "
SELECT 
  tableA.column_name,
  tableB.column_name,
  COUNT(*) as match_count
FROM tableA 
JOIN tableB ON tableA.column_name = tableB.column_name
GROUP BY tableA.column_name, tableB.column_name
"

# Test timestamp-based joins with delays
python data_explorer.py your_database.sqlite --query "
SELECT 
  tableA.timestamp_column,
  tableB.timestamp_column,
  COUNT(*) as matches
FROM tableA 
JOIN tableB ON ABS(JULIANDAY(tableA.timestamp_column) - JULIANDAY(tableB.timestamp_column)) < 0.1
GROUP BY tableA.timestamp_column, tableB.timestamp_column
"
```

### 5.2 Visual Validation
Create visualizations to confirm relationships:

```bash
# Create comparative histograms for matched columns
python plot_permutation_builder.py your_database.sqlite --histogram matched_column --filter category

# Create timeline plots for temporal relationships
python interactive_plots.py your_database.sqlite --table tableA
```

---

## Step 6: Building the Complete Mapping

### 6.1 Document Your Findings
Based on the analysis, document:

1. **Exact column matches** (same name, same data)
2. **Similar columns** (different names, similar data)
3. **Transformation patterns** (aggregations, filters)
4. **Join strategies** (how to connect tables)
5. **Dropped rows** (what gets filtered out)

### 6.2 Create Mapping Queries
Build SQL queries that map tableA to tableB:

```sql
-- Example mapping query based on findings
SELECT 
  tableA.id as tableA_id,
  tableB.id as tableB_id,
  tableA.exact_match_column,
  tableA.transformed_column,
  tableB.aggregated_column,
  tableA.timestamp_column,
  tableB.processed_timestamp
FROM tableA 
LEFT JOIN tableB ON 
  tableA.exact_match_column = tableB.exact_match_column
  AND ABS(JULIANDAY(tableA.timestamp_column) - JULIANDAY(tableB.processed_timestamp)) < 0.1
WHERE tableA.some_filter_condition = 'value'
```

---

## Step 7: Advanced Analysis (If Needed)

### 7.1 Complex Transformation Detection
If simple analysis doesn't reveal the full picture:

```bash
# Use business intelligence reporting for complex patterns
python business_intelligence_reporter.py your_database.sqlite --table tableA
python business_intelligence_reporter.py your_database.sqlite --table tableB

# Assess data quality differences
python data_quality_assessor.py your_database.sqlite --table tableA
python data_quality_assessor.py your_database.sqlite --table tableB
```

### 7.2 Custom Analysis Scripts
If you need more specific analysis, modify the existing scripts:

```bash
# Example: Create custom analysis for your specific columns
python -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('your_database.sqlite')
tableA = pd.read_sql_query('SELECT * FROM tableA', conn)
tableB = pd.read_sql_query('SELECT * FROM tableB', conn)

# Your custom analysis here
print('TableA shape:', tableA.shape)
print('TableB shape:', tableB.shape)
print('Common columns:', set(tableA.columns) & set(tableB.columns))
"
```

---

## Troubleshooting Common Issues

### Issue: No clear relationships found
**Solution:**
- Check data types match between similar columns
- Look for encoding issues in text columns
- Verify timestamp formats are consistent
- Try different similarity thresholds

### Issue: Too many false positives
**Solution:**
- Use more specific filtering criteria
- Check for data quality issues
- Verify the many-to-one assumption
- Look for intermediate transformation tables

### Issue: Timestamp correlations are weak
**Solution:**
- Check for different time zones
- Look for processing delays that vary
- Consider date-only vs datetime comparisons
- Check for different timestamp formats

---

## Expected Output Examples

### Successful Analysis Results:
```
[MATCHES] Exact column matches: {'id', 'product_name', 'category'}
[CORRELATION] timestamp_A vs timestamp_B: 0.892, avg delay: 2.34 hours
[TRANSFORM] Found aggregation: tableA.value_column → tableB.sum_value
[FILTER] Found filter: tableA.status = 'active' → tableB (dropped inactive)
```

### Mapping Summary:
- **Direct mappings**: 3 columns (id, product_name, category)
- **Transformed mappings**: 2 columns (value → sum_value, timestamp → processed_timestamp)
- **Join strategy**: Exact match on id + timestamp correlation
- **Filtering**: Only active records from tableA appear in tableB
- **Aggregation**: Multiple tableA rows → Single tableB row based on grouping

---

## Tips for Success

1. **Start simple**: Begin with exact column matches
2. **Use timestamps**: They're often the most reliable join keys
3. **Check data quality**: Clean data reveals clearer relationships
4. **Document everything**: Keep notes of what you find
5. **Test incrementally**: Validate each relationship before moving on
6. **Use visualizations**: Plots often reveal patterns that numbers don't
7. **Be patient**: Complex transformations take time to uncover

---

## Next Steps

Once you've mapped the relationships:
1. **Create a mapping document** with your findings
2. **Build validation queries** to test the mapping
3. **Set up monitoring** to detect when the relationship changes
4. **Consider automation** if this is a recurring analysis need

Remember: The goal is to understand **how** tableB is derived from tableA, not just **that** it is derived. This understanding will help you with future data analysis and quality assurance. 