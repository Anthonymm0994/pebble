#!/usr/bin/env python3
"""
Configuration file for the Advanced Dataset Analyzer
==================================================

This file contains configuration options that can be easily modified
to customize the analysis behavior and focus on specific aspects.
"""

# Analysis Configuration
ANALYSIS_CONFIG = {
    # Timestamp detection patterns
    'timestamp_patterns': [
        'time', 'date', 'timestamp', 'created', 'updated', 'message', 
        'log', 'event', 'when', 'at', 'ts', 'received', 'sent'
    ],
    
    # Thresholds for relationship detection
    'correlation_threshold': 0.7,      # Minimum correlation to consider columns related
    'similarity_threshold': 0.5,       # Minimum name similarity for column matching
    'time_window_seconds': 3600,      # Maximum time window for timestamp matching
    
    # Sampling and performance
    'max_sample_size': 10000,          # Maximum rows to analyze (for large datasets)
    'pca_components': 3,               # Number of PCA components for dimensionality reduction
    
    # Visualization settings
    'plot_style': 'seaborn-v0_8',     # Matplotlib style for plots
    'output_dir': 'analysis_output',   # Directory for output files
    'dpi': 300,                       # DPI for saved plots
    
    # Analysis focus areas (set to True to enable, False to disable)
    'focus_areas': {
        'temporal_analysis': True,     # Focus on timestamp relationships
        'value_transformations': True, # Focus on value changes
        'filtering_detection': True,   # Focus on filtering patterns
        'aggregation_detection': True, # Focus on aggregation patterns
        'correlation_analysis': True,  # Focus on statistical correlations
        'pattern_detection': True,     # Focus on general patterns
        'heuristic_analysis': True,    # Focus on heuristic insights
    }
}

# Custom field mappings (for known relationships)
FIELD_MAPPINGS = {
    # Example: 'source_field_name': 'derived_field_name'
    # 'message_time': 'processed_time',
    # 'user_id': 'user_identifier',
    # 'amount': 'total_amount',
}

# Custom transformation patterns to look for
TRANSFORMATION_PATTERNS = {
    'mathematical': {
        'scaling': True,      # Look for multiplication/division patterns
        'offset': True,       # Look for addition/subtraction patterns
        'rounding': True,     # Look for rounding patterns
        'logarithm': True,    # Look for logarithmic transformations
    },
    'categorical': {
        'mapping': True,      # Look for value mappings
        'binning': True,      # Look for binning/grouping
        'encoding': True,     # Look for encoding changes
    },
    'temporal': {
        'timezone_conversion': True,  # Look for timezone changes
        'format_conversion': True,    # Look for format changes
        'aggregation': True,          # Look for time-based aggregation
    }
}

# Custom heuristics for specific domains
DOMAIN_HEURISTICS = {
    'financial': {
        'currency_conversion': True,
        'interest_calculation': True,
        'tax_calculation': True,
    },
    'ecommerce': {
        'order_processing': True,
        'inventory_tracking': True,
        'customer_segmentation': True,
    },
    'logistics': {
        'route_optimization': True,
        'delivery_tracking': True,
        'warehouse_management': True,
    }
}

# Visualization customization
VISUALIZATION_CONFIG = {
    'color_scheme': {
        'source': '#1f77b4',      # Blue for source data
        'derived': '#ff7f0e',     # Orange for derived data
        'correlation': '#2ca02c',  # Green for correlations
        'transformation': '#d62728' # Red for transformations
    },
    'figure_size': (15, 12),
    'font_size': 12,
    'save_format': 'png',  # 'png', 'pdf', 'svg'
    'interactive': True,   # Generate interactive plots
}

# Report customization
REPORT_CONFIG = {
    'include_confidence_scores': True,
    'include_recommendations': True,
    'include_visualizations': True,
    'include_detailed_analysis': True,
    'max_examples_per_category': 10,
    'confidence_threshold': 0.3,  # Minimum confidence to include in report
}

# Performance tuning
PERFORMANCE_CONFIG = {
    'use_multiprocessing': False,  # Enable for very large datasets
    'chunk_size': 1000,           # Process data in chunks
    'memory_limit_gb': 4,         # Memory limit for analysis
    'timeout_seconds': 300,       # Timeout for analysis
}

# Custom analysis functions (can be extended)
CUSTOM_ANALYSIS = {
    'business_logic_detection': True,  # Look for business rule patterns
    'data_quality_assessment': True,   # Assess data quality changes
    'anomaly_detection': True,         # Detect anomalies in transformations
    'trend_analysis': True,           # Analyze trends in transformations
}

# Export configuration
EXPORT_CONFIG = {
    'export_formats': ['txt', 'html', 'json'],  # Export formats
    'include_raw_data': False,                   # Include raw data in exports
    'compress_output': False,                    # Compress output files
    'backup_original': True,                     # Backup original files
}

# Example usage functions
def get_config_for_financial_data():
    """Get configuration optimized for financial data analysis."""
    config = ANALYSIS_CONFIG.copy()
    config['focus_areas']['value_transformations'] = True
    config['focus_areas']['temporal_analysis'] = True
    config['correlation_threshold'] = 0.8
    return config

def get_config_for_log_data():
    """Get configuration optimized for log data analysis."""
    config = ANALYSIS_CONFIG.copy()
    config['focus_areas']['temporal_analysis'] = True
    config['focus_areas']['filtering_detection'] = True
    config['time_window_seconds'] = 7200  # 2 hours for log processing
    return config

def get_config_for_ecommerce_data():
    """Get configuration optimized for ecommerce data analysis."""
    config = ANALYSIS_CONFIG.copy()
    config['focus_areas']['aggregation_detection'] = True
    config['focus_areas']['value_transformations'] = True
    config['similarity_threshold'] = 0.6
    return config

# Custom field exploration templates
FIELD_EXPLORATION_TEMPLATES = {
    'timestamp_field': {
        'analysis_types': ['temporal', 'format', 'timezone'],
        'visualizations': ['timeline', 'distribution', 'correlation'],
        'metrics': ['mean_delay', 'std_delay', 'format_consistency']
    },
    'numeric_field': {
        'analysis_types': ['statistical', 'transformation', 'distribution'],
        'visualizations': ['histogram', 'scatter', 'boxplot'],
        'metrics': ['mean_diff', 'std_diff', 'correlation']
    },
    'categorical_field': {
        'analysis_types': ['mapping', 'encoding', 'filtering'],
        'visualizations': ['bar_chart', 'heatmap', 'venn_diagram'],
        'metrics': ['unique_ratio', 'mapping_accuracy', 'coverage']
    }
}

# Example usage
if __name__ == "__main__":
    print("Configuration file loaded successfully!")
    print(f"Default analysis config: {len(ANALYSIS_CONFIG)} parameters")
    print(f"Field mappings: {len(FIELD_MAPPINGS)} custom mappings")
    print(f"Transformation patterns: {len(TRANSFORMATION_PATTERNS)} categories") 