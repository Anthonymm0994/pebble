# Dataset Relationship Analysis Toolkit

A comprehensive Python toolkit for analyzing relationships between SQLite datasets and creating detailed visualizations. This toolkit helps you reverse-engineer how one dataset was derived from another, with a focus on timestamp relationships, column mapping, and pattern detection.

## 🚀 Key Features

### **Dataset Relationship Analysis**
- **Timestamp Analysis**: Detects processing delays and temporal relationships
- **Column Mapping**: Identifies similar columns across datasets
- **Transformation Detection**: Finds filtering, aggregation, and value transformations
- **Join Suggestions**: Recommends high-confidence joins between tables
- **Comprehensive Reporting**: Generates detailed analysis reports

### **Histogram Analysis**
- **Multiple Histogram Types**: Basic, overlay, cumulative, log-scale, and binned histograms
- **Statistical Summaries**: Comprehensive statistical analysis for all numeric columns
- **Timestamp Distributions**: Special analysis for time-based data
- **High-Quality Visualizations**: 300 DPI PNG outputs

### **Easy to Use**
- **Anaconda Compatible**: Uses only standard data science libraries
- **Command Line Interface**: Simple CLI for quick analysis
- **Customizable**: Easy to modify for specific use cases
- **Comprehensive Testing**: Full test suite included

## 📋 Requirements

All libraries are included in Anaconda:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scipy` - Statistical functions
- `sqlite3` - Database connectivity (built-in)

## 🛠️ Installation

No additional installation required! Just ensure you have Anaconda installed with the standard data science packages.

## 📖 Usage

### 1. Dataset Relationship Analysis

#### Basic Usage
```bash
# Analyze relationships between two tables
python dataset_relationship_detector.py your_database.db --source source_table --derived derived_table

# With sampling for large datasets
python dataset_relationship_detector.py your_database.db --source source_table --derived derived_table --sample-size 5000

# Explore specific field
python dataset_relationship_detector.py your_database.db --source source_table --derived derived_table --explore-field amount
```

#### Python API
```python
from dataset_relationship_detector import DatasetRelationshipDetector

# Create analyzer
analyzer = DatasetRelationshipDetector('your_database.db')

# Load datasets
analyzer.connect()
analyzer.load_datasets('source_table', 'derived_table')

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis()

# Explore specific field
analyzer.explore_field('message_time', max_samples=50)

# Close connection
analyzer.close()
```

### 2. Histogram Analysis

#### Basic Usage
```bash
# Create comprehensive histograms for all numeric columns
python histogram_analysis.py your_database.db --source source_table --derived derived_table

# Analyze specific columns
python histogram_analysis.py your_database.db --source source_table --derived derived_table --columns amount priority message_length
```

#### Python API
```python
from histogram_analysis import HistogramAnalyzer

# Create analyzer
analyzer = HistogramAnalyzer('your_database.db')

# Load datasets
analyzer.connect()
analyzer.load_datasets('source_table', 'derived_table')

# Create specific histogram types
analyzer.create_basic_histograms('amount')
analyzer.create_overlay_histograms('amount')
analyzer.create_cumulative_histograms('amount')
analyzer.create_log_scale_histograms('amount')
analyzer.create_binned_histograms('amount')

# Create comprehensive analysis
analyzer.create_comprehensive_histogram_analysis()

# Close connection
analyzer.close()
```

## 🔍 Analysis Types

### Dataset Relationship Analysis

1. **Timestamp Analysis**
   - Detects timestamp columns automatically
   - Calculates processing delays between corresponding timestamps
   - Identifies temporal patterns and relationships

2. **Column Similarity Analysis**
   - Finds exact column matches
   - Identifies similar column names
   - Detects value overlaps between columns

3. **Transformation Detection**
   - **Filtering**: Detects if one dataset is a filtered subset
   - **Aggregation**: Identifies summarization patterns
   - **Value Transformations**: Finds mathematical and categorical transformations

4. **Join Suggestions**
   - Timestamp-based joins with confidence scores
   - Exact value match joins
   - Name similarity-based joins

### Histogram Analysis

1. **Basic Histograms**
   - Individual histograms for source and derived datasets
   - Statistical summaries included
   - High-quality visualizations

2. **Overlay Histograms**
   - Side-by-side comparison of distributions
   - Normalized density plots
   - Easy pattern identification

3. **Cumulative Histograms**
   - Shows cumulative distribution functions
   - Useful for understanding data spread
   - Helps identify transformation patterns

4. **Log-Scale Histograms**
   - For wide-ranging data
   - Reveals patterns in skewed distributions
   - Useful for financial or scientific data

5. **Binned Histograms**
   - Custom binning for specific analysis
   - Controlled granularity
   - Consistent bin sizes

6. **Statistical Summary Histograms**
   - Overview of all numeric columns
   - Quick comparison across datasets
   - Comprehensive statistical analysis

## 📊 Output Files

### Dataset Relationship Analysis
- **`dataset_relationship_report.txt`** - Comprehensive text report
- **`dataset_comparison.png`** - Basic dataset comparison visualizations
- **`timestamp_analysis.png`** - Temporal relationship analysis
- **`correlation_analysis.png`** - Correlation analysis plots
- **`transformation_analysis.png`** - Transformation detection plots

### Histogram Analysis
- **`basic_histogram_[column].png`** - Basic histograms for each column
- **`overlay_histogram_[column].png`** - Overlay histograms for each column
- **`cumulative_histogram_[column].png`** - Cumulative histograms for each column
- **`log_histogram_[column].png`** - Log-scale histograms for each column
- **`binned_histogram_[column].png`** - Binned histograms for each column
- **`statistical_summary_histograms.png`** - Summary of all numeric columns
- **`timestamp_histograms.png`** - Timestamp distribution analysis

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

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_dataset_analyzer.py
```

This will:
- Create test databases with sample data
- Test all analysis functions
- Verify visualization generation
- Check command line interface
- Validate Unicode handling

## 📈 Example Output

### Sample Analysis Report
```
====================================================================================================
DATASET RELATIONSHIP ANALYSIS REPORT
====================================================================================================

[SUMMARY] EXECUTIVE SUMMARY
--------------------------------------------------
Source Dataset: 200 rows, 9 columns
Derived Dataset: 45 rows, 9 columns
Row Count Ratio: 0.225

[TIME] TIMESTAMP ANALYSIS
--------------------------------------------------
message_time -> processed_time:
  - Mean delay: 1687.20s
  - Match count: 4576
  - Confidence: 1.000

[SEARCH] COLUMN SIMILARITIES
--------------------------------------------------
Exact matches: 2
Similar columns: 6
Value overlaps: 2
New columns: 7
Dropped columns: 7

[TRANSFORM] DETECTED TRANSFORMATIONS
--------------------------------------------------
[OK] FILTERING DETECTED: 22.5% of source rows retained
[OK] AGGREGATION DETECTED: 4 columns aggregated
[OK] MATHEMATICAL TRANSFORMATIONS: 4 detected
[OK] CATEGORICAL TRANSFORMATIONS: 1 detected

[JOIN] JOIN SUGGESTIONS
--------------------------------------------------
1. TIMESTAMP_JOIN:
   Source: message_time
   Derived: processed_time
   Confidence: 1.000
   Details: {'mean_delay': 1687.204326923077, 'join_condition': 'ABS(message_time - processed_time) <= 3600'}
```

## 🔧 Customization

### Configuration Options
```python
# Customize analysis parameters
analyzer = DatasetRelationshipDetector('your_database.db')

# Modify thresholds
analyzer.correlation_threshold = 0.6
analyzer.similarity_threshold = 0.4
analyzer.time_window_seconds = 180
```

### Histogram Customization
```python
# Customize histogram parameters
analyzer.create_basic_histograms('amount', bins=50)
analyzer.create_binned_histograms('amount', num_bins=20)
```

## 🚀 Performance Tips

### For Large Datasets
```bash
# Use sampling for large datasets
python dataset_relationship_detector.py your_database.db --source source_table --derived derived_table --sample-size 5000
```

### For Real-time Analysis
```python
# Focus on specific analysis types
analyzer = DatasetRelationshipDetector('your_database.db')
analyzer.focus_areas = {
    'temporal_analysis': True,
    'filtering_detection': True,
    'value_transformations': False,  # Disable for speed
}
```

## 🔍 Troubleshooting

### Common Issues

1. **No relationships found**: Lower similarity and correlation thresholds
2. **Memory issues**: Reduce sample size or enable chunking
3. **Slow performance**: Disable unused analysis types
4. **Unicode errors**: Fixed in latest version

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
results = analyzer.run_comprehensive_analysis()
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

### Field-Specific Exploration
```python
# Explore specific fields in detail
analyzer.explore_field('transaction_amount', max_samples=100)
analyzer.explore_field('log_timestamp', max_samples=50)
analyzer.explore_field('user_category', max_samples=20)
```

## 🤝 Contributing

This toolkit is designed to be extensible. You can:

1. Add custom analysis functions
2. Create domain-specific configurations
3. Extend visualization capabilities
4. Add new transformation detection patterns

## 📄 License

This toolkit is provided as-is for data analysis purposes. Feel free to modify and adapt for your specific needs.

## 🎯 Quick Start

1. **Install**: Ensure Anaconda is installed
2. **Test**: `python test_dataset_analyzer.py`
3. **Analyze**: `python dataset_relationship_detector.py your_database.db --source source_table --derived derived_table`
4. **Visualize**: `python histogram_analysis.py your_database.db --source source_table --derived derived_table`

## 📁 File Structure

```
├── dataset_relationship_detector.py    # Main relationship analysis tool
├── histogram_analysis.py              # Comprehensive histogram analysis
├── test_dataset_analyzer.py           # Test suite
├── fix_unicode.py                     # Unicode fix utility
├── check_db.py                        # Database inspection tool
├── README.md                          # This documentation
└── *.png                              # Generated visualizations
```

This comprehensive toolkit provides everything you need to analyze relationships between datasets and create detailed visualizations for data exploration and analysis. 