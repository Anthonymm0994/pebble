# Pebble - SQLite Viewer & Builder

A clean, modular, professional Rust desktop application for viewing and building SQLite databases.

## Features

### Viewer Mode (Read-Only)
- Open and explore SQLite databases safely
- Browse tables and views in the sidebar
- Execute read-only SQL queries
- Paginated results with configurable page size
- Export query results to CSV or JSON
- Multiple simultaneous query windows

### Builder Mode (Read/Write)
- Create new SQLite databases
- Import CSV files with automatic type inference
- Override detected column types before import
- Create indexes on columns
- Full database modification capabilities

## Usage

1. Run the application:
   ```bash
   cargo run
   ```

2. Use the File menu to:
   - Open an existing SQLite database
   - Create a new database
   - Import CSV files (in Builder mode)

3. Switch between Viewer and Builder modes using the Mode menu

4. Click on tables or views in the sidebar to open query windows

5. Edit queries and execute them with Ctrl+Enter or the Execute button

6. Export results using the Export button in query windows

## Building

```bash
cargo build --release
```

## Requirements

- Rust 1.70 or later
- Windows, macOS, or Linux

## Architecture

The project is organized into modular components:

- `core/` - Database and CSV abstractions
- `infer/` - Type inference for CSV columns
- `ui/` - User interface components
- `app/` - Application state management

## License

This project is licensed under the MIT License. 