# Advanced Dataset Relationship Analyzer

A comprehensive Python tool for uncovering how one dataset was derived from another, using advanced visualizations, comparisons, correlation analysis, and heuristics. This tool provides deep insights into data transformations, filtering patterns, aggregation, and temporal relationships.

## 🚀 Key Features

### **Advanced Analysis Capabilities**
- **Comprehensive Transformation Detection**: Identifies mathematical, categorical, and temporal transformations
- **Statistical Correlation Analysis**: Analyzes relationships between corresponding columns
- **Temporal Pattern Recognition**: Detects processing delays and time-based transformations
- **Heuristic Analysis**: Applies domain-specific heuristics to understand derivation patterns
- **Interactive Visualizations**: Creates both static and interactive plots for deep exploration

### **Customization & Flexibility**
- **Configurable Analysis**: Easy-to-modify configuration files for different use cases
- **Field-Specific Exploration**: Deep dive into specific fields or columns
- **Domain-Specific Configurations**: Pre-built configs for financial, log, and ecommerce data
- **Performance Optimization**: Configurable sampling and processing for large datasets

### **Comprehensive Reporting**
- **Detailed Analysis Reports**: Text reports with confidence scores and recommendations
- **Multiple Visualization Types**: Basic comparisons, correlations, transformations, and temporal analysis
- **Interactive Dashboards**: HTML-based interactive visualizations
- **Export Options**: Multiple output formats for further analysis

## 📋 Requirements

All libraries are included in Anaconda:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scipy` - Statistical functions
- `scikit-learn` - Machine learning utilities
- `plotly` - Interactive visualizations
- `sqlite3` - Database connectivity (built-in)

## 🛠️ Installation

No additional installation required! Just ensure you have Anaconda installed with the standard data science packages.

## 📖 Usage

### Basic Usage

```python
from advanced_dataset_analyzer import AdvancedDatasetAnalyzer

# Create analyzer
analyzer = AdvancedDatasetAnalyzer('your_database.db')

# Load datasets
analyzer.connect()
analyzer.load_datasets('source_table', 'derived_table')

# Run comprehensive analysis
results = analyzer.comprehensive_analysis()

# Generate report
report = analyzer.generate_report()

# Close connection
analyzer.close()
```

### Command Line Usage

```bash
# Basic analysis
python advanced_dataset_analyzer.py your_database.db --source source_table --derived derived_table

# With sampling for large datasets
python advanced_dataset_analyzer.py your_database.db --source source_table --derived derived_table --sample-size 5000

# Explore specific field
python advanced_dataset_analyzer.py your_database.db --source source_table --derived derived_table --explore-field amount
```

### Custom Configuration

```python
from advanced_dataset_analyzer import AdvancedDatasetAnalyzer
from analysis_config import ANALYSIS_CONFIG, get_config_for_financial_data

# Use financial-specific configuration
financial_config = get_config_for_financial_data()
analyzer = AdvancedDatasetAnalyzer('your_database.db', financial_config)

# Or customize your own configuration
custom_config = ANALYSIS_CONFIG.copy()
custom_config.update({
    'correlation_threshold': 0.6,
    'similarity_threshold': 0.4,
    'focus_areas': {
        'temporal_analysis': True,
        'value_transformations': True,
        'filtering_detection': True,
        'aggregation_detection': False,
    }
})
analyzer = AdvancedDatasetAnalyzer('your_database.db', custom_config)
```

## 🔍 Analysis Types

### 1. Basic Dataset Comparison
- Row and column count comparisons
- Data type distribution analysis
- Missing data assessment
- Memory usage comparison

### 2. Column Relationship Analysis
- **Exact Matches**: Identical column names
- **Similar Columns**: Name-based similarity matching
- **Transformed Columns**: Columns with mathematical or categorical transformations
- **New/Dropped Columns**: Columns added or removed in derived dataset

### 3. Temporal Analysis
- **Timestamp Detection**: Automatic identification of time-related columns
- **Processing Delays**: Analysis of time differences between corresponding timestamps
- **Temporal Patterns**: Detection of time-based transformations and aggregations

### 4. Value Transformation Analysis
- **Mathematical Transformations**: Scaling, offsets, rounding, logarithmic changes
- **Categorical Transformations**: Value mappings, encoding changes, binning
- **Statistical Comparisons**: Distribution analysis and correlation testing

### 5. Statistical Correlation Analysis
- **Column Correlations**: Statistical relationships between corresponding columns
- **Value Distribution Comparison**: Analysis of how value distributions change
- **Statistical Tests**: Hypothesis testing for transformation patterns

### 6. Pattern Detection
- **Filtering Patterns**: Detection of row filtering and subset creation
- **Aggregation Patterns**: Identification of summarization and grouping
- **Transformation Patterns**: Recognition of systematic value changes
- **Temporal Patterns**: Time-based processing patterns

### 7. Heuristic Analysis
- **Derivation Hypothesis**: Intelligent guesses about how datasets relate
- **Confidence Scores**: Quantitative measures of analysis reliability
- **Recommendations**: Suggestions for further investigation

## 🎯 Field-Specific Exploration

### Explore Specific Fields

```python
# Explore a temporal field
analyzer.explore_specific_field('message_time', max_samples=50)

# Explore a numeric field
analyzer.explore_specific_field('amount', max_samples=100)

# Explore a categorical field
analyzer.explore_specific_field('category', max_samples=20)
```

### Field Exploration Features

- **Value Distribution Analysis**: Compare value distributions between source and derived
- **Transformation Detection**: Identify specific transformations applied to the field
- **Statistical Comparison**: Calculate correlation and statistical measures
- **Sample Value Display**: Show actual values for manual inspection

## 📊 Visualization Types

### 1. Basic Comparison Plots
- Row and column count comparisons
- Data type distribution analysis
- Missing data comparison
- Memory usage analysis

### 2. Correlation Analysis Plots
- Scatter plots for numeric columns
- Correlation coefficient displays
- Distribution comparison plots
- Statistical relationship visualizations

### 3. Transformation Analysis Plots
- Mathematical transformation detection
- Categorical transformation mapping
- Value distribution changes
- Transformation pattern identification

### 4. Temporal Analysis Plots
- Processing delay histograms
- Time series comparisons
- Temporal pattern identification
- Time-based transformation analysis

### 5. Interactive Plots
- Interactive dashboards using Plotly
- Zoomable and filterable visualizations
- Multi-dimensional analysis views
- Exportable interactive charts

## ⚙️ Configuration Options

### Analysis Configuration

```python
ANALYSIS_CONFIG = {
    'timestamp_patterns': ['time', 'date', 'timestamp', 'message'],
    'correlation_threshold': 0.7,
    'similarity_threshold': 0.5,
    'time_window_seconds': 3600,
    'max_sample_size': 10000,
    'focus_areas': {
        'temporal_analysis': True,
        'value_transformations': True,
        'filtering_detection': True,
        'aggregation_detection': True,
        'correlation_analysis': True,
        'pattern_detection': True,
        'heuristic_analysis': True,
    }
}
```

### Domain-Specific Configurations

```python
# Financial data configuration
financial_config = get_config_for_financial_data()

# Log data configuration
log_config = get_config_for_log_data()

# Ecommerce data configuration
ecommerce_config = get_config_for_ecommerce_data()
```

### Custom Field Mappings

```python
FIELD_MAPPINGS = {
    'source_field_name': 'derived_field_name',
    'message_time': 'processed_time',
    'user_id': 'user_identifier',
    'amount': 'total_amount',
}
```

## 📈 Example Output

### Sample Report

```
================================================================================
ADVANCED DATASET RELATIONSHIP ANALYSIS REPORT
================================================================================

📋 EXECUTIVE SUMMARY
--------------------------------------------------
Source Dataset: 1000 rows, 8 columns
Derived Dataset: 600 rows, 7 columns
Row Count Ratio: 0.600

🔍 KEY FINDINGS
--------------------------------------------------
✅ FILTERING DETECTED: 60.0% of source rows retained
✅ MATHEMATICAL TRANSFORMATIONS: 3 detected
✅ CATEGORICAL TRANSFORMATIONS: 2 detected
✅ TEMPORAL PROCESSING: 2 timestamp relationships found

📊 DETAILED ANALYSIS
--------------------------------------------------
Column Relationships:
  - Exact matches: 3
  - Similar columns: 2
  - New columns: 2
  - Dropped columns: 1

Derivation Hypothesis:
FILTERING: The derived dataset appears to be a filtered subset of the source dataset.
MATHEMATICAL TRANSFORMATIONS: Some columns show mathematical transformations (scaling, offsets).
TEMPORAL PROCESSING: There are consistent time delays, suggesting processing or transformation time.

Confidence Scores:
  - Filtering: 0.400
  - Transformations: 0.600
  - Temporal: 0.800
  - Overall: 0.600

💡 RECOMMENDATIONS
--------------------------------------------------
1. Investigate potential filtering criteria by examining value distributions in categorical columns.
2. Look for mathematical relationships between numeric columns that might indicate transformations.
3. Examine timestamp columns more closely to understand temporal processing patterns.
4. Compare value distributions for key columns to understand transformation patterns.
```

### Generated Files

- **`advanced_analysis_report.txt`** - Comprehensive text report
- **`basic_comparison.png`** - Basic dataset comparison visualizations
- **`correlation_analysis.png`** - Correlation analysis plots
- **`temporal_analysis.png`** - Temporal relationship analysis
- **`interactive_comparison.html`** - Interactive dashboard

## 🎯 Use Cases

### 1. Message Processing Pipeline
- **Source**: Raw messages with timestamps
- **Derived**: Processed messages with delays
- **Analysis**: Processing delays, filtering, transformations

### 2. Financial Data Processing
- **Source**: Raw transaction data
- **Derived**: Processed financial summaries
- **Analysis**: Currency conversions, risk scoring, aggregations

### 3. Ecommerce Data Analysis
- **Source**: Raw order data
- **Derived**: Customer analytics
- **Analysis**: Customer segmentation, order aggregation, filtering

### 4. Log Data Processing
- **Source**: Raw log entries
- **Derived**: Aggregated log summaries
- **Analysis**: Time-based aggregation, filtering, pattern detection

## 🔧 Customization Examples

### Financial Data Analysis

```python
# Use financial-specific configuration
financial_config = get_config_for_financial_data()
analyzer = AdvancedDatasetAnalyzer('financial.db', financial_config)

# Focus on currency conversions and risk scoring
analyzer.load_datasets('raw_transactions', 'processed_transactions')
results = analyzer.comprehensive_analysis()
```

### Log Data Analysis

```python
# Use log-specific configuration
log_config = get_config_for_log_data()
analyzer = AdvancedDatasetAnalyzer('logs.db', log_config)

# Focus on temporal patterns and filtering
analyzer.load_datasets('raw_logs', 'aggregated_logs')
results = analyzer.comprehensive_analysis()
```

### Custom Field Exploration

```python
# Explore specific fields in detail
analyzer.explore_specific_field('transaction_amount', max_samples=100)
analyzer.explore_specific_field('log_timestamp', max_samples=50)
analyzer.explore_specific_field('user_category', max_samples=20)
```

## 🚀 Performance Tips

### For Large Datasets
```python
# Use sampling for large datasets
analyzer.load_datasets('large_source', 'large_derived', sample_size=5000)

# Customize performance settings
custom_config = ANALYSIS_CONFIG.copy()
custom_config.update({
    'max_sample_size': 5000,
    'use_multiprocessing': True,
    'chunk_size': 1000,
})
```

### For Real-time Analysis
```python
# Focus on specific analysis types
custom_config = ANALYSIS_CONFIG.copy()
custom_config['focus_areas'] = {
    'temporal_analysis': True,
    'filtering_detection': True,
    'value_transformations': False,  # Disable for speed
    'correlation_analysis': False,   # Disable for speed
}
```

## 🔍 Troubleshooting

### Common Issues

1. **No relationships found**: Lower similarity and correlation thresholds
2. **Memory issues**: Reduce sample size or enable chunking
3. **Slow performance**: Disable unused analysis types
4. **Missing visualizations**: Check if Plotly is installed for interactive plots

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
results = analyzer.comprehensive_analysis()
```

## 📚 Advanced Features

### Custom Analysis Functions

```python
# Add custom analysis functions
def custom_business_logic_detection(analyzer):
    # Your custom analysis logic
    pass

# Extend the analyzer
analyzer.custom_analysis = custom_business_logic_detection
```

### Export Customization

```python
# Customize export options
EXPORT_CONFIG = {
    'export_formats': ['txt', 'html', 'json'],
    'include_raw_data': True,
    'compress_output': True,
}
```

## 🤝 Contributing

This tool is designed to be extensible. You can:

1. Add custom analysis functions
2. Create domain-specific configurations
3. Extend visualization capabilities
4. Add new transformation detection patterns

## 📄 License

This tool is provided as-is for data analysis purposes. Feel free to modify and adapt for your specific needs.

## 🎯 Quick Start

1. **Install**: Ensure Anaconda is installed
2. **Run Example**: `python example_advanced_usage.py`
3. **Customize**: Modify `analysis_config.py` for your needs
4. **Analyze**: Use with your own database

```bash
# Quick start with your database
python advanced_dataset_analyzer.py your_database.db --source source_table --derived derived_table
```

This advanced analyzer provides the most comprehensive toolkit for understanding how one dataset was derived from another, with extensive customization options and detailed insights into all types of data transformations. 