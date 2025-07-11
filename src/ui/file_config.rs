use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use egui::{Context, Id};
use crate::core::{Database, CsvReader};
use crate::infer::{TypeInferrer, ColumnType};

#[derive(Clone)]
pub struct FileConfig {
    pub path: PathBuf,
    pub table_name: String,
    pub header_row: usize,
    pub delimiter: char,
    pub sample_size: usize,
    pub columns: Vec<ColumnConfig>,
    pub null_values: Vec<String>,
    pub preview_data: Option<PreviewData>,
}

#[derive(Clone)]
pub struct ColumnConfig {
    pub name: String,
    pub data_type: ColumnType,
    pub included: bool,
    pub create_index: bool,
    pub is_primary_key: bool,
    pub not_null: bool,
    pub unique: bool,
}

#[derive(Clone)]
pub struct PreviewData {
    pub rows: Vec<Vec<String>>,
}

impl FileConfig {
    pub fn new(path: PathBuf) -> Self {
        let table_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("table")
            .to_string();
        
        Self {
            path,
            table_name,
            header_row: 0,
            delimiter: ',',
            sample_size: 1000,
            columns: Vec::new(),
            null_values: vec!["", "NULL", "null", "N/A"].into_iter().map(String::from).collect(),
            preview_data: None,
        }
    }
    
    pub fn file_name(&self) -> String {
        self.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    }
}

pub struct FileConfigDialog {
    id: Id,
    pub show: bool,
    pub database_path: Option<PathBuf>,
    pub files: Vec<FileConfig>,
    pub current_file_index: usize,
    pub create_database: bool,
    
    // UI state
    null_value_input: String,
    error: Option<String>,
    processing_state: Arc<Mutex<ProcessingState>>,
    needs_resampling: bool,
    pk_changed_index: Option<usize>,
}

#[derive(Clone)]
pub enum ProcessingState {
    Idle,
    Loading(f32, String),
    Processing(String, f32),
    Complete,
    Error(String),
}

impl FileConfigDialog {
    pub fn new(id: Id) -> Self {
        Self {
            id,
            show: false,
            database_path: None,
            files: Vec::new(),
            current_file_index: 0,
            create_database: false,
            null_value_input: String::new(),
            error: None,
            processing_state: Arc::new(Mutex::new(ProcessingState::Idle)),
            needs_resampling: false,
            pk_changed_index: None,
        }
    }
    
    pub fn open(&mut self, path: PathBuf) {
        self.database_path = Some(path);
        self.show = true;
        self.create_database = false;
        self.files.clear();
        self.current_file_index = 0;
    }
    
    pub fn open_with_csv_selection(&mut self) {
        // First, let user select CSV files
        if let Some(csv_files) = rfd::FileDialog::new()
            .add_filter("CSV files", &["csv"])
            .set_title("Select CSV files to import")
            .pick_files()
        {
            if !csv_files.is_empty() {
                // Reset dialog state
                self.reset();
                
                // Set default database path in Documents folder
                let default_db_path = if let Some(docs_dir) = dirs::document_dir() {
                    docs_dir.join("pebble.db")
                } else {
                    PathBuf::from("pebble.db")
                };
                
                self.database_path = Some(default_db_path);
                self.show = true;
                self.create_database = false;
                self.files.clear();
                self.current_file_index = 0;
                
                // Add all selected CSV files
                for csv_path in csv_files {
                    self.add_file(csv_path);
                }
            }
        }
    }
    
    fn reset(&mut self) {
        self.files.clear();
        self.current_file_index = 0;
        self.error = None;
        self.null_value_input.clear();
        self.needs_resampling = false;
        self.pk_changed_index = None;
        if let Ok(mut state) = self.processing_state.lock() {
            *state = ProcessingState::Idle;
        }
    }
    
    pub fn add_file(&mut self, path: PathBuf) {
        let config = FileConfig::new(path);
        self.files.push(config);
        self.current_file_index = self.files.len() - 1;
        self.load_preview_for_current_file();
    }
    
    pub fn show(&mut self, ctx: &Context) -> Option<PathBuf> {
        if !self.show {
            return None;
        }
        
        let mut created_db_path = None;
        
        // Check processing state first
        let current_state = if let Ok(state) = self.processing_state.lock() {
            match &*state {
                ProcessingState::Idle => {
                    // eprintln!("Current state: Idle");
                    ProcessingState::Idle
                }
                ProcessingState::Processing(msg, progress) => {
                    // eprintln!("Current state: Processing - {}", msg);
                    ctx.request_repaint();
                    ProcessingState::Processing(msg.clone(), *progress)
                }
                ProcessingState::Complete => {
                    // eprintln!("UI: Processing complete, closing dialog");
                    // Database creation completed successfully
                    created_db_path = self.database_path.clone();
                    self.show = false;
                    
                    // Reset state for next time
                    ProcessingState::Complete
                }
                ProcessingState::Error(error_msg) => {
                    // eprintln!("Current state: Error - {}", error_msg);
                    // Show error and reset state
                    self.error = Some(error_msg.clone());
                    ProcessingState::Error(error_msg.clone())
                }
                ProcessingState::Loading(progress, msg) => {
                    ProcessingState::Loading(*progress, msg.clone())
                }
            }
        } else {
            ProcessingState::Idle
        };
        
        // Reset state after Complete or Error
        match current_state {
            ProcessingState::Complete | ProcessingState::Error(_) => {
                if let Ok(mut state) = self.processing_state.lock() {
                    *state = ProcessingState::Idle;
                }
                
                // If complete, return the path immediately
                if matches!(current_state, ProcessingState::Complete) {
                    return created_db_path;
                }
            }
            _ => {}
        }
        
        // Show progress overlay if processing
        if let ProcessingState::Processing(message, progress) = &current_state {
            // Show progress dialog with less dark overlay
            egui::Area::new(egui::Id::new("progress_overlay"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .show(ctx, |ui| {
                    let screen_rect = ctx.screen_rect();
                    ui.painter().rect_filled(
                        screen_rect,
                        0.0,
                        egui::Color32::from_black_alpha(120) // Much lighter overlay
                    );
                });
            
            egui::Window::new("Processing")
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .fixed_size([400.0, 200.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.heading("Creating Database");
                        ui.add_space(20.0);
                        
                        // Progress bar
                        let progress_bar = egui::ProgressBar::new(*progress)
                            .text(format!("{:.0}%", progress * 100.0))
                            .desired_width(350.0);
                        ui.add(progress_bar);
                        
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new(message).size(14.0));
                        
                        ui.add_space(20.0);
                        ui.label(egui::RichText::new("Please wait...").size(12.0).color(egui::Color32::from_gray(150)));
                    });
                });
            
            // Don't show the main dialog content while processing
            return None;
        }
        
        // Show main dialog only if not processing
        egui::CentralPanel::default().show(ctx, |ui| {
            // Title bar
            ui.horizontal(|ui| {
                ui.heading("Create Database from CSV");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("✖").clicked() {
                        self.show = false;
                    }
                });
            });
            ui.separator();
            
            self.render_content(ui);
            
            if self.create_database {
                self.start_database_creation();
                self.create_database = false;
            }
        });
        
        created_db_path
    }
    
    fn render_content(&mut self, ui: &mut egui::Ui) {
        // Use vertical layout with bottom panel for buttons
        egui::TopBottomPanel::bottom("bottom_buttons")
            .show_inside(ui, |ui| {
                ui.add_space(10.0);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.show = false;
                    }
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_create = self.database_path.is_some() && 
                            !self.files.is_empty() && 
                            self.files.iter().all(|f| !f.table_name.is_empty() && 
                                f.columns.iter().any(|c| c.included));
                        
                        let file_count = self.files.len();
                        let create_button = egui::Button::new(
                            egui::RichText::new(format!("✅ Create Database with {} Table{}", 
                                file_count, 
                                if file_count == 1 { "" } else { "s" }))
                                .size(16.0)
                                .color(egui::Color32::WHITE)
                        )
                        .fill(egui::Color32::from_rgb(76, 175, 80))
                        .rounding(egui::Rounding::same(6.0));
                        
                        if ui.add_enabled(can_create, create_button).clicked() {
                            // Validate constraints before creating
                            if let Some(error) = self.validate_constraints() {
                                self.error = Some(error);
                            } else {
                                self.create_database = true;
                            }
                        }
                        
                        ui.add_space(10.0);
                        ui.label(
                            egui::RichText::new("💡 Configure each file's import settings before creating the database")
                                .size(12.0)
                                .color(egui::Color32::from_gray(150))
                        );
                    });
                });
                ui.add_space(5.0);
            });
        
        // Main content in central panel
        egui::CentralPanel::default()
            .show_inside(ui, |ui| {
                // Database path section
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Database Path:");
                        if let Some(path) = &self.database_path {
                            ui.label(path.display().to_string());
                        } else {
                            ui.label("No database selected");
                        }
                        
                        if ui.button("Browse...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("SQLite Database", &["db", "sqlite", "sqlite3"])
                                .set_title("Save database as...")
                                .save_file()
                            {
                                self.database_path = Some(path);
                            }
                        }
                    });
                });
                
                // Error display
                if let Some(error) = &self.error.clone() {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::from_rgb(255, 100, 100), format!("❌ {}", error));
                        if ui.small_button("✖").clicked() {
                            self.error = None;
                        }
                    });
                }
                
                ui.separator();
                
                // Two-column layout with more vertical space
                let available_height = ui.available_height();
                ui.horizontal_top(|ui| {
                    // Left side - file configuration
                    ui.vertical(|ui| {
                        ui.set_width(500.0);
                        ui.set_height(available_height);
                        
                        // File selector dropdown
                        ui.horizontal(|ui| {
                            ui.label("CSV File:");
                            
                            let file_names: Vec<String> = self.files.iter()
                                .enumerate()
                                .map(|(_idx, config)| {
                                    let configured = !config.columns.is_empty();
                                    format!("{}{}", 
                                        config.file_name(),
                                        if configured { " ✓" } else { "" }
                                    )
                                })
                                .collect();
                            
                            if !file_names.is_empty() {
                                let current_text = file_names.get(self.current_file_index)
                                    .cloned()
                                    .unwrap_or_else(|| "No file selected".to_string());
                                
                                egui::ComboBox::new("file_selector_combo", "CSV File")
                                    .selected_text(&current_text)
                                    .show_ui(ui, |ui| {
                                        for (idx, name) in file_names.iter().enumerate() {
                                            if ui.selectable_value(&mut self.current_file_index, idx, name).clicked() {
                                                // Load preview for newly selected file
                                                self.load_preview_for_current_file();
                                            }
                                        }
                                    });
                            } else {
                                ui.label("No files added");
                            }
                            
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Add Files...").clicked() {
                                    if let Some(paths) = rfd::FileDialog::new()
                                        .add_filter("CSV files", &["csv"])
                                        .set_title("Select CSV files")
                                        .pick_files()
                                    {
                                        for path in paths {
                                            self.add_file(path);
                                        }
                                    }
                                }
                            });
                        });
                        
                        ui.label(format!("Files to import: {} total, {} configured", 
                            self.files.len(),
                            self.files.iter().filter(|f| !f.columns.is_empty()).count()
                        ));
                        
                        ui.separator();
                        ui.add_space(5.0);
                        
                        if let Some(config) = self.files.get_mut(self.current_file_index) {
                            // Table name
                            ui.horizontal(|ui| {
                                ui.label("Table Name:");
                                ui.text_edit_singleline(&mut config.table_name);
                            });
                            
                            ui.add_space(10.0);
                            
                            // Header configuration
                            ui.group(|ui| {
                                ui.set_width(ui.available_width());
                                ui.label(egui::RichText::new("Header Configuration").size(16.0).strong());
                                ui.add_space(5.0);
                                
                                ui.horizontal(|ui| {
                                    ui.label("Header Row:");
                                    
                                    // Convert to 1-indexed for display
                                    let mut header_row_display = config.header_row + 1;
                                    let max_rows = config.preview_data.as_ref()
                                        .map(|p| p.rows.len())
                                        .unwrap_or(10) as i32;
                                    
                                    let response = ui.add(
                                        egui::DragValue::new(&mut header_row_display)
                                            .range(1..=max_rows)
                                            .speed(1)
                                    );
                                    
                                    if response.changed() {
                                        config.header_row = (header_row_display - 1).max(0) as usize;
                                        // Trigger resampling
                                        self.needs_resampling = true;
                                    }
                                    
                                    ui.label(format!("(1-{})", max_rows));
                                });
                                
                                ui.add_space(5.0);
                                ui.label(
                                    egui::RichText::new("The green highlighted row in the preview is your header")
                                        .size(12.0)
                                        .color(egui::Color32::from_gray(150))
                                );
                            });
                            
                            ui.add_space(10.0);
                            
                            // Sample size
                            ui.horizontal(|ui| {
                                ui.label("Sample Size:");
                                let response = ui.add(
                                    egui::DragValue::new(&mut config.sample_size)
                                        .range(100..=10000)
                                        .speed(10)
                                );
                                if response.changed() {
                                    self.needs_resampling = true;
                                }
                                ui.label("rows");
                                
                                if ui.button("🔄 Resample").clicked() {
                                    self.needs_resampling = true;
                                }
                            });
                            
                            ui.add_space(10.0);
                            
                            // Delimiter
                            ui.horizontal(|ui| {
                                ui.label("Delimiter:");
                                ui.radio_value(&mut config.delimiter, ',', "Comma");
                                ui.radio_value(&mut config.delimiter, '\t', "Tab");
                                ui.radio_value(&mut config.delimiter, ';', "Semicolon");
                                ui.radio_value(&mut config.delimiter, '|', "Pipe");
                            });
                            
                            ui.add_space(10.0);
                            
                            // Null values
                            ui.group(|ui| {
                                ui.set_width(ui.available_width());
                                ui.label(egui::RichText::new("Null Values").size(14.0));
                                ui.label(egui::RichText::new("Values to treat as NULL:").size(12.0));
                                
                                egui::ScrollArea::vertical()
                                    .id_salt(format!("null_scroll_{}", self.current_file_index))
                                    .max_height(100.0)
                                    .show(ui, |ui| {
                                        let mut to_remove = None;
                                        for (idx, pattern) in config.null_values.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                if ui.small_button("×").clicked() {
                                                    to_remove = Some(idx);
                                                }
                                                ui.label(if pattern.is_empty() { "[empty string]" } else { pattern });
                                            });
                                        }
                                        if let Some(idx) = to_remove {
                                            config.null_values.remove(idx);
                                        }
                                    });
                                
                                ui.horizontal(|ui| {
                                    ui.text_edit_singleline(&mut self.null_value_input);
                                    if ui.button("Add").clicked() && !self.null_value_input.trim().is_empty() {
                                        config.null_values.push(self.null_value_input.clone());
                                        self.null_value_input.clear();
                                    }
                                });
                            });
                            
                            ui.add_space(10.0);
                            
                            // Column selection
                            ui.group(|ui| {
                                ui.set_width(ui.available_width());
                                ui.label(egui::RichText::new("Column Selection").size(14.0));
                                
                                if !config.columns.is_empty() {
                                    let selected_count = config.columns.iter()
                                        .filter(|c| c.included)
                                        .count();
                                    
                                    ui.horizontal(|ui| {
                                        if ui.button("Select All").clicked() {
                                            for col in &mut config.columns {
                                                col.included = true;
                                            }
                                        }
                                        if ui.button("Deselect All").clicked() {
                                            for col in &mut config.columns {
                                                col.included = false;
                                            }
                                        }
                                        ui.label(format!("{} / {} selected", selected_count, config.columns.len()));
                                    });
                                    
                                    ui.separator();
                                    
                                    let available_height = ui.available_height();
                                    egui::ScrollArea::vertical()
                                        .id_salt(format!("column_scroll_{}", self.current_file_index))
                                        .max_height(available_height)
                                        .show(ui, |ui| {
                                            use egui_extras::{TableBuilder, Column};
                                            
                                            TableBuilder::new(ui)
                                                .striped(true)
                                                .resizable(true)
                                                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                                                .column(Column::auto().at_least(60.0)) // Include
                                                .column(Column::auto().at_least(100.0).resizable(true)) // Column
                                                .column(Column::auto().at_least(100.0)) // Type
                                                .column(Column::auto().at_least(40.0)) // PK
                                                .column(Column::auto().at_least(70.0)) // Not Null
                                                .column(Column::auto().at_least(60.0)) // Unique
                                                .column(Column::auto().at_least(50.0)) // Index
                                                .header(20.0, |mut header| {
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Include").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Column").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Type").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("PK").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Not Null").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Unique").strong());
                                                    });
                                                    header.col(|ui| {
                                                        ui.label(egui::RichText::new("Index").strong());
                                                    });
                                                })
                                                .body(|mut body| {
                                                    for (col_idx, col) in config.columns.iter_mut().enumerate() {
                                                        body.row(25.0, |mut row| {
                                                            row.col(|ui| {
                                                                ui.checkbox(&mut col.included, "");
                                                            });
                                                            row.col(|ui| {
                                                                ui.label(&col.name);
                                                            });
                                                            row.col(|ui| {
                                                                // Type dropdown
                                                                egui::ComboBox::new(format!("type_{}_{}", self.current_file_index, col_idx), "")
                                                                    .selected_text(format!("{:?}", col.data_type))
                                                                    .width(90.0)
                                                                    .show_ui(ui, |ui| {
                                                                        ui.set_max_height(200.0); // Force dropdown to open downward
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::Text, "Text");
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::Integer, "Integer");
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::Real, "Real");
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::Boolean, "Boolean");
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::Date, "Date");
                                                                        ui.selectable_value(&mut col.data_type, ColumnType::DateTime, "DateTime");
                                                                    });
                                                            });
                                                            row.col(|ui| {
                                                                // Primary key checkbox
                                                                let mut pk_changed = false;
                                                                let was_pk = col.is_primary_key;
                                                                if ui.checkbox(&mut col.is_primary_key, "").changed() {
                                                                    if col.is_primary_key && !was_pk {
                                                                        pk_changed = true;
                                                                    }
                                                                }
                                                                
                                                                // Store the index if PK was checked
                                                                if pk_changed {
                                                                    self.pk_changed_index = Some(col_idx);
                                                                }
                                                            });
                                                            row.col(|ui| {
                                                                ui.checkbox(&mut col.not_null, "");
                                                            });
                                                            row.col(|ui| {
                                                                ui.checkbox(&mut col.unique, "");
                                                            });
                                                            row.col(|ui| {
                                                                ui.checkbox(&mut col.create_index, "");
                                                            });
                                                        });
                                                    }
                                                });
                                        });
                                }
                            });
                            
                            // Handle primary key changes after the loop
                            if let Some(pk_idx) = self.pk_changed_index.take() {
                                for (idx, col) in config.columns.iter_mut().enumerate() {
                                    if idx != pk_idx {
                                        col.is_primary_key = false;
                                    }
                                }
                            }
                        }
                    });
                    
                    ui.separator();
                    
            // Right side - data preview
            ui.vertical(|ui| {
                ui.set_height(available_height);
                ui.label(egui::RichText::new("Data Preview").size(16.0).strong());
                ui.add_space(8.0);
                
                let preview_height = ui.available_height();
                
                if let Some(config) = self.files.get(self.current_file_index) {
                    if let Some(preview) = &config.preview_data {
                        egui::ScrollArea::both()
                            .id_salt(format!("preview_scroll_{}", self.current_file_index))
                            .max_height(preview_height)
                            .show(ui, |ui| {
                                // Use TableBuilder for proper vertical separators
                                use egui_extras::{TableBuilder, Column};
                                
                                // Calculate number of columns (row number + data columns)
                                let num_columns = if let Some(first_row) = preview.rows.first() {
                                    first_row.len() + 1 // +1 for row number column
                                } else {
                                    1
                                };
                                
                                TableBuilder::new(ui)
                                    .striped(true)
                                    .resizable(true)
                                    .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                                    .column(Column::auto().at_least(40.0)) // Row number column
                                    .columns(Column::auto().at_least(100.0).resizable(true), num_columns - 1) // Data columns
                                    .vscroll(false) // We're already in a scroll area
                                    .body(|mut body| {
                                        for (row_idx, row) in preview.rows.iter().enumerate() {
                                            let is_header = row_idx == config.header_row;
                                            let color = if is_header {
                                                egui::Color32::from_rgb(100, 200, 100)
                                            } else {
                                                egui::Color32::from_gray(200)
                                            };
                                            
                                            body.row(20.0, |mut table_row| {
                                                // Row number
                                                table_row.col(|ui| {
                                                    let row_text = egui::RichText::new((row_idx + 1).to_string())
                                                        .color(if is_header { color } else { egui::Color32::from_gray(150) });
                                                    ui.label(if is_header { row_text.strong() } else { row_text });
                                                });
                                                
                                                // Row data
                                                for cell in row.iter() {
                                                    table_row.col(|ui| {
                                                        let cell_text = egui::RichText::new(cell)
                                                            .color(if is_header { color } else { egui::Color32::from_gray(200) });
                                                        ui.label(if is_header { cell_text.strong() } else { cell_text });
                                                    });
                                                }
                                            });
                                        }
                                    });
                            });
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Loading preview...");
                        });
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("No file selected");
                    });
                }
            });
        });
            });  // This closes the CentralPanel show_inside
        
        // Process resampling if needed
        if self.needs_resampling {
            self.needs_resampling = false;
            self.load_preview_for_current_file();
        }
    }
    
    fn load_preview_for_current_file(&mut self) {
        if let Some(config) = self.files.get_mut(self.current_file_index) {
            let path = config.path.clone();
            let delimiter = config.delimiter;
            let sample_size = config.sample_size;
            let header_row = config.header_row;
            
            // Load preview data
            match CsvReader::from_path(&path) {
                Ok(mut reader) => {
                    reader.set_delimiter(delimiter);
                    
                    // Read preview rows
                    let mut preview_rows: Vec<Vec<String>> = Vec::new();
                    if let Ok(records) = reader.sample_records(sample_size.min(50)) {
                        for record in records {
                            preview_rows.push(record.iter().map(|s| s.to_string()).collect());
                        }
                    }
                    
                    // Get headers from the specified row
                    if let Some(header_row_data) = preview_rows.get(header_row) {
                        let headers = header_row_data.clone();
                        
                        // Sample data for type inference (skip header row)
                        let mut sample_data = Vec::new();
                        for (idx, row) in preview_rows.iter().enumerate() {
                            if idx > header_row && sample_data.len() < sample_size {
                                sample_data.push(row.clone());
                            }
                        }
                        
                        // Infer types
                        let inferred_types = TypeInferrer::infer_column_types(&headers, &sample_data);
                        
                        // Update columns
                        config.columns.clear();
                        for (header, (_name, data_type)) in headers.iter().zip(inferred_types.iter()) {
                            config.columns.push(ColumnConfig {
                                name: header.clone(),
                                data_type: data_type.clone(),
                                included: true,
                                create_index: false,
                                is_primary_key: false,
                                not_null: false,
                                unique: false,
                            });
                        }
                    }
                    
                    config.preview_data = Some(PreviewData { rows: preview_rows });
                }
                Err(e) => {
                    self.error = Some(format!("Failed to load preview: {}", e));
                }
            }
        }
    }
    
    fn start_database_creation(&mut self) -> Option<PathBuf> {
        let db_path = self.database_path.clone()?;
        let files = self.files.clone();
        let processing_state = self.processing_state.clone();
        
        // Start processing in background thread
        std::thread::spawn(move || {
            Self::create_database_in_thread(db_path, files, processing_state);
        });
        
        None // Return None for now, actual path will be handled differently
    }
    
    fn create_database_in_thread(
        db_path: PathBuf,
        files: Vec<FileConfig>,
        processing_state: Arc<Mutex<ProcessingState>>
    ) {
        // Update state to processing
        if let Ok(mut state) = processing_state.lock() {
            *state = ProcessingState::Processing("Initializing database...".to_string(), 0.0);
        }
        
        match Database::open_writable(&db_path) {
            Ok(mut db) => {
                let total_files = files.len();
                
                // Set SQLite pragmas for maximum performance during bulk insert
                let _ = db.execute_sql("PRAGMA synchronous = OFF");
                let _ = db.execute_sql("PRAGMA journal_mode = MEMORY");
                let _ = db.execute_sql("PRAGMA cache_size = -64000"); // 64MB cache
                let _ = db.execute_sql("PRAGMA temp_store = MEMORY");
                
                // Begin a single transaction for all operations (much faster)
                if let Err(e) = db.begin_transaction() {
                    if let Ok(mut state) = processing_state.lock() {
                        *state = ProcessingState::Error(format!("Failed to begin transaction: {}", e));
                    }
                    return;
                }
                
                // First create all tables
                if let Ok(mut state) = processing_state.lock() {
                    *state = ProcessingState::Processing("Creating tables...".to_string(), 0.05);
                }
                
                for (file_idx, config) in files.iter().enumerate() {
                    // Update progress for table creation
                    if let Ok(mut state) = processing_state.lock() {
                        *state = ProcessingState::Processing(
                            format!("Creating table '{}' ({}/{})", config.table_name, file_idx + 1, total_files),
                            (file_idx + 1) as f32 / total_files as f32 * 0.2 // First 20% for table creation
                        );
                    }
                    
                    // Create columns definition with constraints
                    let mut column_defs = Vec::new();
                    let mut primary_key_col = None;
                    
                    for col in config.columns.iter().filter(|c| c.included) {
                        let mut col_def = format!("{} {}", col.name, col.data_type.to_sql_type());
                        
                        if col.is_primary_key {
                            primary_key_col = Some(col.name.clone());
                        }
                        
                        if col.not_null {
                            col_def.push_str(" NOT NULL");
                        }
                        
                        if col.unique && !col.is_primary_key {
                            col_def.push_str(" UNIQUE");
                        }
                        
                        column_defs.push(col_def);
                    }
                    
                    // Add primary key constraint
                    if let Some(pk_col) = primary_key_col {
                        column_defs.push(format!("PRIMARY KEY ({})", pk_col));
                    }
                    
                    if column_defs.is_empty() {
                        continue;
                    }
                    
                    // Create table with constraints
                    let create_sql = format!(
                        "CREATE TABLE IF NOT EXISTS '{}' ({})",
                        config.table_name,
                        column_defs.join(", ")
                    );
                    
                    if let Err(e) = db.execute_sql(&create_sql) {
                        if let Ok(mut state) = processing_state.lock() {
                            *state = ProcessingState::Error(format!("Failed to create table {}: {}", config.table_name, e));
                        }
                        let _ = db.rollback_transaction();
                        return;
                    }
                }
                
                // Now import data for each file
                for (file_idx, config) in files.iter().enumerate() {
                    // eprintln!("Starting import for file {}: {}", file_idx + 1, config.file_name());
                    
                    // Update initial progress for this file
                    if let Ok(mut state) = processing_state.lock() {
                        *state = ProcessingState::Processing(
                            format!("Importing {} ({}/{})", config.file_name(), file_idx + 1, total_files),
                            0.2 + (0.7 * file_idx as f32 / total_files as f32)
                        );
                    }
                    
                    if let Ok(mut reader) = CsvReader::from_path(&config.path) {
                        reader.set_delimiter(config.delimiter);
                        
                        // Skip to header row and then skip one more to get to data
                        for _ in 0..=config.header_row {
                            let _ = reader.read_record();
                        }
                        
                        // Get column indices for included columns
                        let column_indices: Vec<usize> = config.columns.iter()
                            .enumerate()
                            .filter_map(|(idx, c)| if c.included { Some(idx) } else { None })
                            .collect();
                        
                        // Import records in batches for better performance
                        let mut row_count = 0;
                        let mut batch_values = Vec::new();
                        const BATCH_SIZE: usize = 10000; // Increased from 1000 for better performance
                        
                        while let Ok(Some(record)) = reader.read_record() {
                            let values: Vec<String> = column_indices.iter()
                                .map(|&idx| {
                                    let value = record.get(idx).unwrap_or("");
                                    // Check if it's a null value
                                    if config.null_values.contains(&value.to_string()) {
                                        String::new()
                                    } else {
                                        value.to_string()
                                    }
                                })
                                .collect();
                            
                            batch_values.push(values);
                            row_count += 1;
                            
                            // Insert batch when it reaches BATCH_SIZE
                            if batch_values.len() >= BATCH_SIZE {
                                if let Err(e) = db.batch_insert(&config.table_name, &batch_values) {
                                    if let Ok(mut state) = processing_state.lock() {
                                        *state = ProcessingState::Error(format!("Failed to insert batch: {}", e));
                                    }
                                    let _ = db.rollback_transaction();
                                    return;
                                }
                                
                                batch_values.clear();
                                
                                // Update progress - show actual row count for this file
                                if let Ok(mut state) = processing_state.lock() {
                                    let overall_progress = 0.2 + (0.7 * file_idx as f32 / total_files as f32) + 
                                                         (0.7 / total_files as f32 * 0.5); // Estimate halfway through file
                                    *state = ProcessingState::Processing(
                                        format!("Importing {} - {} rows", config.file_name(), row_count),
                                        overall_progress
                                    );
                                }
                            }
                        }
                        
                        // Insert remaining records
                        if !batch_values.is_empty() {
                            if let Err(e) = db.batch_insert(&config.table_name, &batch_values) {
                                if let Ok(mut state) = processing_state.lock() {
                                    *state = ProcessingState::Error(format!("Failed to insert final batch: {}", e));
                                }
                                let _ = db.rollback_transaction();
                                return;
                            }
                        }
                        
                        // Final progress update for this file
                        if let Ok(mut state) = processing_state.lock() {
                            let overall_progress = 0.2 + (0.7 * (file_idx + 1) as f32 / total_files as f32);
                            *state = ProcessingState::Processing(
                                format!("Completed {} - {} rows", config.file_name(), row_count),
                                overall_progress
                            );
                        }
                        
                        // eprintln!("Completed importing {} with {} rows", config.file_name(), row_count);
                    } else {
                        // eprintln!("Failed to open CSV file: {}", config.path.display());
                    }
                }
                
                // eprintln!("All files imported, creating indexes...");
                
                // Create indexes after all data is imported (faster)
                if let Ok(mut state) = processing_state.lock() {
                    *state = ProcessingState::Processing("Creating indexes...".to_string(), 0.9);
                }
                
                for config in &files {
                    for col in &config.columns {
                        if col.included && col.create_index {
                            if let Err(_e) = db.create_index(&config.table_name, &col.name) {
                                // eprintln!("Failed to create index on {}.{}: {}", config.table_name, col.name, e);
                            }
                        }
                    }
                }
                
                // Commit the transaction
                if let Ok(mut state) = processing_state.lock() {
                    *state = ProcessingState::Processing("Finalizing database...".to_string(), 0.95);
                }
                
                // eprintln!("Committing transaction...");
                
                if let Err(e) = db.commit_transaction() {
                    // eprintln!("Failed to commit transaction: {}", e);
                    if let Ok(mut state) = processing_state.lock() {
                        *state = ProcessingState::Error(format!("Failed to commit transaction: {}", e));
                    }
                    let _ = db.rollback_transaction();
                    return;
                }
                
                // eprintln!("Transaction committed successfully");
                
                // Restore safe SQLite settings
                let _ = db.execute_sql("PRAGMA synchronous = NORMAL");
                let _ = db.execute_sql("PRAGMA journal_mode = DELETE");
                
                // eprintln!("Database creation completed successfully");
                
                if let Ok(mut state) = processing_state.lock() {
                    *state = ProcessingState::Complete;
                    // eprintln!("State set to Complete");
                }
            }
            Err(e) => {
                // eprintln!("Failed to create database: {}", e);
                if let Ok(mut state) = processing_state.lock() {
                    *state = ProcessingState::Error(format!("Failed to create database: {}", e));
                }
            }
        }
        
        // eprintln!("create_database_in_thread finished");
    }

    fn validate_constraints(&self) -> Option<String> {
        for config in &self.files {
            // Check for multiple primary keys
            let primary_key_count = config.columns.iter()
                .filter(|c| c.included && c.is_primary_key)
                .count();
            
            if primary_key_count > 1 {
                return Some(format!(
                    "Table '{}' has multiple primary key columns. Only one primary key is allowed per table.",
                    config.table_name
                ));
            }
            
            // Check for NOT NULL columns that might have null values
            if let Some(preview) = &config.preview_data {
                for col in &config.columns {
                    if !col.included || !col.not_null {
                        continue;
                    }
                    
                    // Find column index
                    if let Some(col_idx) = config.columns.iter().position(|c| c.name == col.name) {
                        // Check data rows (skip header)
                        let mut has_null = false;
                        for (row_idx, row) in preview.rows.iter().enumerate() {
                            if row_idx <= config.header_row {
                                continue;
                            }
                            
                            if let Some(value) = row.get(col_idx) {
                                if config.null_values.contains(value) || value.is_empty() {
                                    has_null = true;
                                    break;
                                }
                            }
                        }
                        
                        if has_null {
                            return Some(format!(
                                "Column '{}' in table '{}' is marked as NOT NULL but contains null values. Please either remove the NOT NULL constraint or clean your data.",
                                col.name, config.table_name
                            ));
                        }
                    }
                }
            }
            
            // Validate table name
            if config.table_name.is_empty() {
                return Some(format!("File '{}' has an empty table name", config.file_name()));
            }
            
            // Check for invalid characters in table name
            if !config.table_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return Some(format!(
                    "Table name '{}' contains invalid characters. Use only letters, numbers, and underscores.",
                    config.table_name
                ));
            }
            
            // Check that at least one column is included
            if !config.columns.iter().any(|c| c.included) {
                return Some(format!(
                    "Table '{}' has no columns selected. Select at least one column to include.",
                    config.table_name
                ));
            }
        }
        
        // Check for duplicate table names
        let mut table_names = std::collections::HashSet::new();
        for config in &self.files {
            if !table_names.insert(&config.table_name) {
                return Some(format!(
                    "Duplicate table name '{}'. Each table must have a unique name.",
                    config.table_name
                ));
            }
        }
        
        None
    }
} 