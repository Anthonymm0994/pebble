STANDALONE SQLITE PLOTTING SCRIPTS
====================================

This folder contains three standalone Python scripts for creating plots from SQLite database data.
All scripts use standard libraries (pandas, matplotlib, numpy) that come with Anaconda/Spyder.

FILES:
- plot_histogram.py  - Creates histogram plots
- plot_polar.py      - Creates polar plots  
- plot_bar.py        - Creates bar charts
- README.txt         - This file

REQUIREMENTS:
- Python 3.6+
- pandas
- matplotlib
- numpy
- sqlite3 (built-in)

USAGE:
======

1. Edit the configuration variables at the top of each script:
   - DATABASE_PATH: Path to your SQLite database file
   - QUERY: Your SELECT query to pull data
   - Column names, styling options, output filename

2. Run the script:
   python plot_histogram.py
   python plot_polar.py
   python plot_bar.py

EXAMPLE SETUP:
==============

For a database with a table called 'sales' with columns 'amount' and 'category':

HISTOGRAM:
- QUERY = "SELECT amount FROM sales"
- COLUMN_NAME = "amount"
- BIN_COUNT = 20

BAR CHART:
- QUERY = "SELECT category, SUM(amount) as total FROM sales GROUP BY category"
- CATEGORY_COLUMN = "category"
- VALUE_COLUMN = "total"

POLAR PLOT:
- QUERY = "SELECT angle, radius FROM measurements"
- ANGLE_COLUMN = "angle"
- RADIUS_COLUMN = "radius"

CONFIGURATION VARIABLES:
========================

Each script has these main configuration sections:

DATABASE SETTINGS:
- DATABASE_PATH: Path to your .sqlite file
- QUERY: SQL SELECT statement

PLOT SETTINGS:
- Column names to plot
- Figure size
- Output filename

STYLING:
- Colors, transparency, labels
- Titles and axis labels

ERROR HANDLING:
===============

All scripts include error handling for:
- Missing database file
- Invalid SQL queries
- Missing columns in data
- Empty query results

The scripts will show helpful error messages and available columns if something goes wrong.

OUTPUT:
========

Each script will:
1. Connect to your database
2. Execute the query
3. Create the plot
4. Save as PNG file (300 DPI)
5. Display the plot (optional)

TIPS:
======

- Start with a simple query to test your database connection
- Use LIMIT in your query to test with a small dataset first
- Check the available columns by running a simple SELECT * query
- Adjust figure sizes if labels are cut off
- Comment out plt.show() if you don't want the plot to display

TROUBLESHOOTING:
=================

1. "Database file not found": Check DATABASE_PATH is correct
2. "Column not found": Check your query returns the expected columns
3. "No data": Your query returned empty results
4. Plot looks wrong: Check data types and ranges in your columns 