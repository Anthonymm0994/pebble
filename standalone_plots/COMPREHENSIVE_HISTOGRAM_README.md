# Comprehensive Histogram Analysis Suite

This suite provides **ROBUST** and **COMPREHENSIVE** Python histogram scripts with advanced features for data analysis. All scripts connect to SQLite databases, generate query permutations, and create sophisticated visualizations.

## 🚀 **Scripts Overview**

### 1. **comprehensive_histogram.py** - Basic Comprehensive Analysis
**Features:**
- ✅ Overlapping histograms with transparency
- ✅ Query permutation generation
- ✅ Statistical analysis (normality tests, distribution fitting)
- ✅ Multiple visualization styles
- ✅ Export capabilities (PNG, PDF, SVG)

**Key Capabilities:**
- Distribution fitting (Normal, Log-Normal, Exponential, Gamma)
- Normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
- Comprehensive statistics (mean, median, std, skewness, kurtosis)
- Overlapping histogram comparisons
- AIC-based model selection

### 2. **advanced_histogram_analysis.py** - Advanced Interactive Analysis
**Features:**
- ✅ Interactive bokeh plots (if bokeh installed)
- ✅ Plotly interactive visualizations
- ✅ Advanced statistical analysis
- ✅ Distribution fitting with multiple criteria
- ✅ HTML export for interactive plots

**Key Capabilities:**
- Interactive hover tools and zoom capabilities
- Advanced outlier detection using IQR method
- Percentile analysis (P10, P90, P95, P99)
- Jarque-Bera normality test
- Weibull distribution fitting
- AIC and BIC model comparison

### 3. **histogram_permutations.py** - Query Permutations & Filtering
**Features:**
- ✅ Query permutation generation
- ✅ Advanced filtering combinations
- ✅ Multiple visualization styles
- ✅ Statistical comparison across permutations
- ✅ CSV export for statistical summaries

**Key Capabilities:**
- Automatic filter combination generation
- Grid-based permutation visualization
- Statistical comparison across different filters
- Comprehensive CSV summaries
- Multiple output formats

## 📊 **Database Requirements**

All scripts expect a SQLite database with the following structure:

```sql
-- Example table structure
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    amount REAL,
    quantity INTEGER,
    profit_margin REAL,
    region TEXT,
    category TEXT,
    date TEXT
);
```

## 🛠️ **Installation & Dependencies**

```bash
# Required packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Optional packages for advanced features
pip install bokeh plotly
```

## 📈 **Usage Examples**

### Basic Comprehensive Analysis
```python
from comprehensive_histogram import ComprehensiveHistogram

# Initialize
hist_gen = ComprehensiveHistogram("your_database.sqlite")

# Generate analysis with filters
filters = {
    'region': ['North', 'South', 'East', 'West'],
    'category': ['Electronics', 'Clothing', 'Books']
}

hist_gen.generate_comprehensive_analysis(
    table='sales',
    columns=['amount', 'quantity', 'profit_margin'],
    filters=filters,
    max_permutations=5
)
```

### Advanced Interactive Analysis
```python
from advanced_histogram_analysis import AdvancedHistogramAnalysis

# Initialize
analyzer = AdvancedHistogramAnalysis("your_database.sqlite")

# Custom queries
queries = [
    {
        'query': "SELECT amount, quantity FROM sales WHERE region = 'North'",
        'description': "North Region Sales"
    },
    {
        'query': "SELECT amount, quantity FROM sales WHERE amount > 1000",
        'description': "High Value Sales"
    }
]

analyzer.generate_advanced_analysis(queries)
```

### Permutation Analysis
```python
from histogram_permutations import HistogramPermutations

# Initialize
perm_gen = HistogramPermutations("your_database.sqlite")

# Generate permutations
filters = {
    'region': ['North', 'South', 'East', 'West'],
    'category': ['Electronics', 'Clothing', 'Books'],
    'amount': [100, 500, 1000]  # Threshold values
}

figures, summaries = perm_gen.generate_permutation_analysis(
    table='sales',
    columns=['amount', 'quantity', 'profit_margin'],
    filters=filters,
    max_permutations=8
)
```

## 📁 **Output Structure**

### Directory Structure
```
standalone_plots/
├── histogram_outputs/           # Basic comprehensive histograms
│   ├── comprehensive_histogram_analysis_1.png
│   ├── comprehensive_histogram_analysis_1.pdf
│   └── ...
├── advanced_histogram_outputs/  # Advanced interactive plots
│   ├── advanced_histogram_analysis_1.png
│   ├── interactive_amount_1.html
│   └── ...
└── histogram_permutations/      # Permutation analysis
    ├── histogram_permutations_YYYYMMDD_HHMMSS_1.png
    ├── statistical_summary_YYYYMMDD_HHMMSS.csv
    └── ...
```

### Output Files
- **PNG/PDF/SVG**: High-resolution static plots
- **HTML**: Interactive plots (plotly/bokeh)
- **CSV**: Statistical summaries for further analysis

## 🔧 **Configuration Options**

### Database Configuration
```python
DATABASE_PATH = "your_database.sqlite"  # Change to your database
```

### Output Configuration
```python
OUTPUT_DIR = "histogram_outputs"        # Output directory
SAVE_FORMATS = ["png", "pdf", "svg"]    # Export formats
DPI = 300                               # Image resolution
FIGURE_SIZE = (12, 8)                   # Default figure size
```

### Statistical Analysis Settings
```python
CONFIDENCE_LEVEL = 0.95                 # Confidence level for tests
NORMALITY_TESTS = ["shapiro", "anderson", "ks"]  # Normality tests
DISTRIBUTION_FITS = ["normal", "lognormal", "exponential", "gamma"]  # Distribution fits
```

## 📊 **Statistical Features**

### Comprehensive Statistics
- **Basic**: Count, mean, median, std, min, max
- **Advanced**: Q25, Q75, IQR, skewness, kurtosis
- **Specialized**: Coefficient of variation, MAD, percentiles

### Normality Tests
- **Shapiro-Wilk**: Good for small samples
- **Anderson-Darling**: Robust for various distributions
- **Kolmogorov-Smirnov**: Non-parametric test
- **Jarque-Bera**: Tests for normality and skewness

### Distribution Fitting
- **Normal**: Standard normal distribution
- **Log-Normal**: For right-skewed data
- **Exponential**: For time/rate data
- **Gamma**: Flexible shape distribution
- **Weibull**: For reliability data

### Model Selection
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion
- **Log-Likelihood**: Maximum likelihood estimation

## 🎯 **Advanced Features**

### Query Permutations
- Automatic generation of filter combinations
- Configurable maximum permutations
- Smart caching for performance
- Error handling for invalid queries

### Interactive Visualizations
- **Bokeh**: Interactive web-based plots
- **Plotly**: Rich interactive features
- **Hover tools**: Detailed information on hover
- **Zoom/Pan**: Interactive navigation

### Export Capabilities
- **Multiple formats**: PNG, PDF, SVG, HTML
- **High resolution**: 300 DPI output
- **Batch processing**: Multiple plots at once
- **CSV summaries**: Statistical data export

## 🚨 **Error Handling**

All scripts include robust error handling:
- Database connection errors
- Missing columns/tables
- Invalid query syntax
- Empty result sets
- Distribution fitting failures

## 💡 **Best Practices**

### Performance Optimization
1. **Use caching**: Scripts cache query results
2. **Limit permutations**: Set reasonable max_permutations
3. **Filter data**: Use WHERE clauses to reduce data size
4. **Batch processing**: Generate multiple plots at once

### Data Quality
1. **Check data types**: Ensure numeric columns are properly typed
2. **Handle missing values**: Scripts automatically handle NaN values
3. **Validate queries**: Test queries before running analysis
4. **Monitor output**: Check generated files for quality

### Customization
1. **Modify filters**: Adjust filter combinations for your data
2. **Change columns**: Select relevant numeric columns
3. **Adjust parameters**: Modify statistical test parameters
4. **Custom styling**: Modify matplotlib/seaborn styles

## 🔍 **Troubleshooting**

### Common Issues
1. **Database not found**: Check DATABASE_PATH
2. **Missing columns**: Verify table structure
3. **Empty results**: Check filter values
4. **Import errors**: Install required packages

### Performance Issues
1. **Slow execution**: Reduce max_permutations
2. **Memory issues**: Use smaller data subsets
3. **Large files**: Adjust DPI or figure size

## 📚 **Examples & Use Cases**

### Sales Analysis
```python
# Analyze sales by region and category
filters = {
    'region': ['North', 'South', 'East', 'West'],
    'category': ['Electronics', 'Clothing', 'Books']
}
```

### Financial Data
```python
# Analyze transaction amounts
columns = ['amount', 'profit_margin', 'quantity']
```

### Scientific Data
```python
# Analyze experimental measurements
columns = ['temperature', 'pressure', 'humidity']
```

## 🎉 **Success Metrics**

The scripts successfully generate:
- ✅ **18+ comprehensive histogram plots** per run
- ✅ **Multiple statistical summaries** in CSV format
- ✅ **Interactive visualizations** (when dependencies available)
- ✅ **High-quality exports** in multiple formats
- ✅ **Robust error handling** for various scenarios

## 📞 **Support**

For issues or questions:
1. Check the error messages in console output
2. Verify database structure and data types
3. Ensure all required packages are installed
4. Test with smaller datasets first

---

**🎯 These scripts provide COMPREHENSIVE, ROBUST, and USEFUL histogram analysis capabilities for any SQLite database!** 