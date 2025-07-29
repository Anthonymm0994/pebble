# Table Relationship Analyzer

A comprehensive Python tool to analyze relationships between two SQLite tables, with special focus on timestamp-based connections and pattern detection. This tool helps uncover how one table might be derived from another through various transformations like filtering, aggregation, or data processing.

## Features

### 🔍 **Timestamp Analysis**
- Automatically detects timestamp columns using pattern matching
- Analyzes time delays between related timestamps (e.g., processing delays)
- Supports multiple timestamp formats including 24-hour format (16:07:34.053)
- Finds potential matches within configurable time windows

### 🔗 **Relationship Detection**
- **Column Similarities**: Compares column names, data types, and value distributions
- **Transformation Detection**: Identifies filtering, aggregation, and value transformations
- **Join Suggestions**: Recommends potential joins between tables with confidence scores
- **Pattern Recognition**: Detects mathematical transformations (scaling, offsets)

### 📊 **Visualization & Reporting**
- Generates comprehensive visualizations comparing table structures
- Creates detailed analysis reports
- Provides confidence scores for all relationships
- Saves results to files for further analysis

## Requirements

All libraries are included in Anaconda:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `sqlite3` - Database connectivity (built-in)

## Installation

No additional installation required! Just ensure you have Anaconda installed with the standard data science packages.

## Usage

### Basic Usage

```python
from table_relationship_analyzer import TableRelationshipAnalyzer

# Create analyzer
analyzer = TableRelationshipAnalyzer('your_database.db')

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis('table1', 'table2')

# Close connection
analyzer.close()
```

### Command Line Usage

```bash
# Analyze all tables in database
python table_relationship_analyzer.py your_database.db

# Analyze specific tables
python table_relationship_analyzer.py your_database.db --table1 source_table --table2 derived_table
```

### Example with Sample Data

```bash
# Run the example to see the analyzer in action
python example_usage.py
```

## Output Files

The analyzer generates several output files:

1. **`table_relationship_report.txt`** - Comprehensive text report with all findings
2. **`table_relationship_analysis.png`** - Visualizations comparing table structures

## Key Analysis Features

### Timestamp Relationship Analysis

The analyzer is particularly powerful for timestamp-based relationships:

```python
# Example: Finding processing delays between "Message Time" columns
timestamp_analysis = analyzer.analyze_timestamp_relationships()

for pair, analysis in timestamp_analysis.items():
    print(f"{pair}: {analysis['match_count']} matches")
    print(f"Average delay: {analysis['mean_delay']:.2f}s")
    print(f"Delay range: {analysis['min_delay']:.2f}s - {analysis['max_delay']:.2f}s")
```

### Transformation Detection

Detects various types of transformations:

- **Filtering**: Identifies if one table is a subset of another
- **Aggregation**: Detects if values are summed, averaged, or grouped
- **Column Mapping**: Maps columns between tables with confidence scores
- **Value Transformations**: Identifies scaling, offsets, or other mathematical changes

### Join Suggestions

Provides intelligent join recommendations:

```python
join_suggestions = analyzer.suggest_joins()

for suggestion in join_suggestions:
    print(f"Type: {suggestion['type']}")
    print(f"Columns: {suggestion['columns'][0]} ↔ {suggestion['columns'][1]}")
    print(f"Confidence: {suggestion['confidence']:.3f}")
```

## Sample Report Output

```
================================================================================
TABLE RELATIONSHIP ANALYSIS REPORT
================================================================================

📋 TABLE OVERVIEW
----------------------------------------
Table 1: 100 rows, 7 columns
Table 2: 60 rows, 7 columns

🔍 COLUMN SIMILARITIES
----------------------------------------
user_id ↔ user_id: 1.000
message_time ↔ processed_time: 0.850
message_id ↔ processed_id: 0.750

🕒 TIMESTAMP RELATIONSHIPS
----------------------------------------
message_time ↔ processed_time:
  - Average delay: 32.45s
  - Match count: 60
  - Delay range: 5.12s - 60.00s

🔄 DETECTED TRANSFORMATIONS
----------------------------------------
Filtering:
  - row_count_ratio: 0.60
  - filtering_likelihood: high

Column Mapping:
  - user_id → user_id (confidence: 1.000)
  - message_time → processed_time (confidence: 0.850)

🔗 JOIN SUGGESTIONS
----------------------------------------
Type: timestamp_join
Columns: message_time ↔ processed_time
Confidence: 0.600

Type: exact_match
Columns: user_id ↔ user_id
Confidence: 1.000
```

## Advanced Usage

### Custom Analysis

```python
# Load specific tables
analyzer.load_tables('source_table', 'derived_table')

# Find timestamp columns
timestamp_cols = analyzer.find_timestamp_columns()

# Analyze specific relationships
similarities = analyzer.find_column_similarities()
transformations = analyzer.detect_transformations()
join_suggestions = analyzer.suggest_joins()

# Generate custom visualizations
analyzer.generate_visualizations()
```

### Working with Your Data

For your specific case with "Message Time" columns:

1. **Timestamp Format**: The analyzer automatically detects 24-hour format timestamps like `16:07:34.053`
2. **Processing Delays**: It will find the variable delays you mentioned between source and processed timestamps
3. **Column Mapping**: It will help identify which columns correspond between tables
4. **Transformation Detection**: It will show how values change between the source and derived tables

## Tips for Best Results

1. **Column Naming**: Use descriptive column names that hint at relationships (e.g., "message_time" vs "processed_time")
2. **Data Quality**: Ensure timestamp columns are in consistent formats
3. **Sample Size**: The analyzer works best with reasonable amounts of data (100+ rows per table)
4. **Unique Identifiers**: Include unique IDs or keys that can help establish relationships

## Troubleshooting

### Common Issues

1. **No timestamp relationships found**: Check that your timestamp columns contain actual datetime data
2. **Low confidence scores**: Ensure column names are descriptive and data types are consistent
3. **Memory issues**: For very large tables, consider sampling the data first

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
results = analyzer.run_comprehensive_analysis('table1', 'table2')
```

## Example Scenarios

### Scenario 1: Message Processing Pipeline
- **Source Table**: Raw messages with timestamps
- **Derived Table**: Processed messages with processing delays
- **Analysis**: Finds processing delays, filters, and column mappings

### Scenario 2: Data Aggregation
- **Source Table**: Detailed transaction records
- **Derived Table**: Daily summaries
- **Analysis**: Detects aggregation patterns and grouping

### Scenario 3: Data Filtering
- **Source Table**: Complete dataset
- **Derived Table**: Subset based on conditions
- **Analysis**: Identifies filtering criteria and row count ratios

## Contributing

This tool is designed to be extensible. You can add custom analysis methods by extending the `TableRelationshipAnalyzer` class.

## License

This tool is provided as-is for data analysis purposes. Feel free to modify and adapt for your specific needs. 