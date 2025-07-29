# Pebble Data Analysis Toolkit

A comprehensive Python toolkit for data analysis, visualization, and reporting using only Anaconda-compatible libraries.

## 📁 Repository Structure

```
pebble/
├── scripts/                    # All data analysis and visualization scripts
│   ├── interactive_plots.py   # Fancy timeline & polar plots with 24-hour format
│   ├── data_profiler.py       # Comprehensive data profiling
│   ├── business_intelligence_reporter.py  # BI reports & KPIs
│   ├── data_explorer.py       # Data exploration & insights
│   ├── data_quality_assessor.py  # Data quality assessment
│   ├── histogram_permutations.py  # Advanced histogram analysis
│   ├── advanced_histogram_analysis.py  # Advanced histogram analysis
│   ├── comprehensive_histogram.py  # Comprehensive histogram analysis
│   ├── seaborn_advanced_histograms.py  # Seaborn advanced histograms
│   ├── plotly_interactive_histograms.py  # Plotly interactive histograms
│   ├── plot_*.py              # Individual plot types (bar, scatter, heatmap, etc.)
│   └── plot_advanced_*.py     # Advanced plot types (polar, distributions, etc.)
├── test_data/                 # Test data and generated outputs
│   ├── sample_data.csv        # Sample data for testing
│   ├── sample_time_data.csv   # Sample data with 24-hour time format
│   └── outputs/              # All generated visualizations and reports
└── README.md
```

## 🚀 Quick Start

### Interactive Plots (Timeline & Polar)
```bash
cd scripts
python interactive_plots.py ../your_data.csv
```

### Data Profiling
```bash
python data_profiler.py ../your_data.csv
```

### Business Intelligence Reports
```bash
python business_intelligence_reporter.py ../your_data.csv
```

### Data Exploration
```bash
python data_explorer.py ../your_data.csv
```

### Data Quality Assessment
```bash
python data_quality_assessor.py ../your_data.csv
```

### Advanced Histogram Analysis
```bash
python histogram_permutations.py ../your_data.csv
python advanced_histogram_analysis.py ../your_data.csv
python comprehensive_histogram.py ../your_data.csv
```

### Individual Plot Types
```bash
python plot_histogram.py ../your_data.csv
python plot_scatter.py ../your_data.csv
python plot_heatmap.py ../your_data.csv
python plot_timeseries.py ../your_data.csv
python plot_polar.py ../your_data.csv
python plot_bar.py ../your_data.csv
python plot_distributions.py ../your_data.csv
```

## 🎨 Features

### Interactive Plots
- **Timeline Plots**: Support for 24-hour time format (HH:MM:SS.mmm)
- **Polar Plots**: Beautiful circular visualizations
- **Radar Plots**: Multi-dimensional analysis
- **24-Hour Heatmaps**: Activity pattern analysis

### Data Analysis
- **Comprehensive Profiling**: Data types, statistics, patterns
- **Business Intelligence**: KPIs, trends, forecasts
- **Data Exploration**: Automated insights and correlations
- **Quality Assessment**: Completeness, accuracy, consistency

### Advanced Visualizations
- **Histogram Permutations**: SQL-based filtering and analysis
- **Advanced Histograms**: Multiple histogram types and styles
- **Interactive Charts**: Plotly-based interactive visualizations
- **Statistical Analysis**: Comprehensive statistical comparisons

## 📊 Output Organization

All generated outputs are organized in `test_data/outputs/`:
- `interactive_outputs/` - Timeline, polar, radar plots
- `profiler_outputs/` - Data profiling results
- `bi_outputs/` - Business intelligence reports
- `explorer_outputs/` - Data exploration visualizations
- `quality_outputs/` - Quality assessment reports
- `histogram_outputs/` - Histogram analysis results

## 🔧 Requirements

- Python 3.7+
- Anaconda distribution (includes all required libraries)
- Libraries: pandas, numpy, matplotlib, seaborn, scipy

## 📝 Usage Examples

### Timeline Analysis with 24-Hour Format
```python
# Your data should have time columns in format: "16:07:34.053"
python interactive_plots.py your_data.csv
```

### Comprehensive Data Analysis
```python
# Run all analysis types
python data_profiler.py your_data.csv
python business_intelligence_reporter.py your_data.csv
python data_explorer.py your_data.csv
python data_quality_assessor.py your_data.csv
```

### Advanced Histogram Analysis
```python
# Generate histogram permutations based on SQL queries
python histogram_permutations.py your_data.csv

# Advanced histogram analysis
python advanced_histogram_analysis.py your_data.csv
```

## 🎯 Key Features

- **24-Hour Time Format Support**: Perfect for time-series analysis
- **Professional Visualizations**: High-quality plots with consistent styling
- **Comprehensive Analysis**: Multiple analysis types in one toolkit
- **Anaconda Compatible**: Uses only standard Anaconda libraries
- **Clean Organization**: Well-structured repository with clear separation

## 📈 Generated Visualizations

- **Timeline Plots**: Time-series analysis with trend lines
- **Polar Plots**: Circular data visualizations
- **Heatmaps**: Activity patterns and correlations
- **Histograms**: Multiple histogram types and styles
- **Statistical Charts**: Comparative analysis and insights
- **Quality Dashboards**: Data quality assessment reports
- **BI Reports**: Business intelligence with KPIs and trends

This toolkit provides everything you need for comprehensive data analysis and visualization using only Anaconda-compatible libraries! 