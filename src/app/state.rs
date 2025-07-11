use egui::{Context, Id};
use crate::core::{Database, TableInfo};
use crate::ui::{Sidebar, QueryWindow, CsvImportDialog, FileConfigDialog};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Viewer,
    Builder,
}

pub struct PebbleApp {
    mode: AppMode,
    database: Option<Arc<Database>>,
    writable_database: Option<Arc<Database>>,
    database_path: Option<std::path::PathBuf>,
    tables: Vec<TableInfo>,
    views: Vec<String>,
    sidebar: Sidebar,
    query_windows: Vec<QueryWindow>,
    csv_import_dialog: Option<CsvImportDialog>,
    file_config_dialog: FileConfigDialog,
    next_window_id: usize,
    error: Option<String>,
    pebble_texture: Option<egui::TextureHandle>,
}

impl PebbleApp {
    pub fn new() -> Self {
        Self {
            mode: AppMode::Viewer,  // Changed from Builder to Viewer
            database: None,
            writable_database: None,
            database_path: None,
            tables: Vec::new(),
            views: Vec::new(),
            sidebar: Sidebar::new(),
            query_windows: Vec::new(),
            csv_import_dialog: None,
            file_config_dialog: FileConfigDialog::new(Id::new("file_config_dialog")),
            next_window_id: 0,
            error: None,
            pebble_texture: None,
        }
    }
    
    pub fn update(&mut self, ctx: &Context) {
        // Apply dark theme
        ctx.set_visuals(egui::Visuals::dark());
        
        // Load pebble texture on first frame
        if self.pebble_texture.is_none() {
            if let Ok(image_data) = std::fs::read("media/peb.png") {
                if let Ok(image) = image::load_from_memory(&image_data) {
                    let size = [image.width() as _, image.height() as _];
                    let image_buffer = image.to_rgba8();
                    let pixels = image_buffer.as_flat_samples();
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        size,
                        pixels.as_slice(),
                    );
                    self.pebble_texture = Some(ctx.load_texture(
                        "pebble_logo",
                        color_image,
                        egui::TextureOptions::default(),
                    ));
                }
            }
        }
        
        // Top panel with menu
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.label(egui::RichText::new("Pebble").size(16.0).strong());
                ui.separator();
                
                ui.menu_button("File", |ui| {
                    if ui.button("Open Database...").clicked() {
                        self.open_database();
                        ui.close_menu();
                    }
                    
                    if ui.button("New Database...").clicked() {
                        self.new_database_from_csv();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Import CSV to Current Database...").clicked() {
                        self.show_csv_import();
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    
                    if ui.button("Exit").clicked() {
                        std::process::exit(0);
                    }
                });
                
                ui.separator();  // Added separator between File and Mode
                
                ui.menu_button("Mode", |ui| {
                    ui.radio_value(&mut self.mode, AppMode::Viewer, "Viewer (Read-only)");
                    ui.radio_value(&mut self.mode, AppMode::Builder, "Builder (Read/Write)");
                });
                
                ui.separator();
                
                if let Some(path) = &self.database_path {
                    ui.label(format!("Database: {}", path.file_name().unwrap_or_default().to_string_lossy()));
                    ui.separator();
                    ui.label(format!("Mode: {:?}", self.mode));
                }
            });
        });
        
        // Error display
        if let Some(error) = self.error.clone() {
            egui::TopBottomPanel::top("error_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(255, 100, 100), format!("✗ {}", error));
                    if ui.button("×").clicked() {
                        self.error = None;
                    }
                });
            });
        }
        
        // Sidebar
        if self.database.is_some() {
            egui::SidePanel::left("sidebar")
                .default_width(200.0)
                .min_width(150.0)
                .max_width(300.0)
                .resizable(true)
                .show(ctx, |ui| {
                    // Set darker background for the sidebar panel
                    ui.visuals_mut().widgets.noninteractive.bg_fill = egui::Color32::from_gray(30);
                    
                    if let Some(table_to_open) = self.sidebar.show(ctx, ui, &self.tables, &self.views) {
                        self.open_query_window(&table_to_open);
                    }
                });
        }
        
        // Main content area
        if !self.file_config_dialog.show {
            egui::CentralPanel::default().show(ctx, |ui| {
                if self.database.is_none() {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            // Display pebble image if loaded
                            if let Some(texture) = &self.pebble_texture {
                                let desired_size = egui::vec2(120.0, 120.0); // Adjust size as needed
                                let image_size = texture.size_vec2();
                                let scale = (desired_size.x / image_size.x).min(desired_size.y / image_size.y);
                                let scaled_size = image_size * scale;
                                
                                ui.add(egui::Image::new(texture).fit_to_exact_size(scaled_size));
                                ui.add_space(20.0);
                            }
                            
                            ui.heading(egui::RichText::new("Pebble").size(32.0));
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("SQLite Viewer & Builder").size(16.0).color(egui::Color32::from_gray(180)));
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("\"From raw to refined. Smooth by design.\"").size(14.0).italics().color(egui::Color32::from_gray(150)));
                            ui.add_space(30.0);
                            
                            ui.group(|ui| {
                                ui.set_width(300.0);
                                if ui.button(egui::RichText::new("Open Database").size(16.0))
                                    .on_hover_text("Open an existing SQLite database")
                                    .clicked() {
                                    self.open_database();
                                }
                                
                                ui.add_space(10.0);
                                
                                if ui.button(egui::RichText::new("Create Database from CSV").size(16.0))
                                    .on_hover_text("Import CSV files to create a new database")
                                    .clicked() {
                                    self.file_config_dialog.open_with_csv_selection();
                                }
                            });
                            
                            ui.add_space(30.0);
                            
                            ui.label(egui::RichText::new("Tips:").size(14.0).strong().color(egui::Color32::from_gray(200)));
                            ui.label(egui::RichText::new("• Import multiple CSV files at once").color(egui::Color32::from_gray(160)));
                            ui.label(egui::RichText::new("• Configure data types and constraints").color(egui::Color32::from_gray(160)));
                            ui.label(egui::RichText::new("• Export query results to CSV or JSON").color(egui::Color32::from_gray(160)));
                            ui.label(egui::RichText::new("• Use Ctrl+Enter to execute queries").color(egui::Color32::from_gray(160)));
                        });
                    });
                }
            });
        }
        
        // Query windows
        if let Some(db) = &self.database {
            self.query_windows.retain_mut(|window| {
                window.show(ctx, db.clone())
            });
        }
        
        // Show CSV import dialog if active
        if let Some(dialog) = &mut self.csv_import_dialog {
            if !dialog.show(ctx) {
                self.csv_import_dialog = None;
                self.load_tables(); // Refresh after import
            }
        }
        
        // File config dialog
        if let Some(path) = self.file_config_dialog.show(ctx) {
            self.mode = AppMode::Builder;
            self.load_database(path);
        }
    }
    
    fn open_database(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("SQLite Database", &["db", "sqlite", "sqlite3", "db3"])
            .pick_file()
        {
            match if self.mode == AppMode::Builder {
                Database::open_writable(&path)
            } else {
                Database::open_readonly(&path)
            } {
                Ok(db) => {
                    self.database = Some(Arc::new(db));
                    self.database_path = Some(path);
                    self.load_tables();
                    self.error = None;
                }
                Err(e) => {
                    self.error = Some(format!("Failed to open database: {}", e));
                }
            }
        }
    }
    
    fn new_database_from_csv(&mut self) {
        // This method is no longer needed as we handle it directly in the menu
    }
    
    fn load_database(&mut self, path: std::path::PathBuf) {
        match self.mode {
            AppMode::Viewer => {
                match Database::open_readonly(&path) {
                    Ok(db) => {
                        self.database = Some(Arc::new(db));
                        self.database_path = Some(path);
                        self.load_tables();
                        self.error = None;
                    }
                    Err(e) => {
                        self.error = Some(format!("Failed to open database: {}", e));
                    }
                }
            }
            AppMode::Builder => {
                match Database::open_writable(&path) {
                    Ok(db) => {
                        self.writable_database = Some(Arc::new(db));
                        
                        // Also open a read-only version for queries
                        match Database::open_readonly(&path) {
                            Ok(ro_db) => {
                                self.database = Some(Arc::new(ro_db));
                                self.database_path = Some(path);
                                self.load_tables();
                                self.error = None;
                            }
                            Err(e) => {
                                self.error = Some(format!("Failed to open read-only database: {}", e));
                            }
                        }
                    }
                    Err(e) => {
                        self.error = Some(format!("Failed to open database: {}", e));
                    }
                }
            }
        }
    }
    
    fn open_query_window(&mut self, table_name: &str) {
        if let Some(_db) = &self.database {
            let window = QueryWindow::new(
                self.next_window_id,
                table_name.to_string(),
                format!("SELECT * FROM '{}'", table_name),
            );
            self.query_windows.push(window);
            self.next_window_id += 1;
        }
    }
    
    fn show_csv_import(&mut self) {
        if self.database_path.is_some() && self.mode == AppMode::Builder {
            self.csv_import_dialog = Some(CsvImportDialog::new(Id::new("csv_import_dialog")));
        } else {
            self.error = Some("Open a database in Builder mode to import CSV files".to_string());
        }
    }

    fn load_tables(&mut self) {
        if let Some(db) = &self.database {
            match db.get_tables() {
                Ok(tables) => self.tables = tables,
                Err(e) => self.error = Some(format!("Failed to load tables: {}", e)),
            }
            
            match db.get_views() {
                Ok(views) => {
                    self.views = views.into_iter().map(|v| v.name).collect();
                },
                Err(e) => self.error = Some(format!("Failed to load views: {}", e)),
            }
        }
    }
} 