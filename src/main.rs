mod app;
mod core;
mod infer;
mod ui;

use eframe::egui;
use app::PebbleApp;
use ui::apply_theme;

fn main() -> Result<(), eframe::Error> {
    // Load icon from peb.png
    let icon_data = load_icon_from_png();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Pebble - SQLite Viewer & Builder")
            .with_icon(icon_data),
        ..Default::default()
    };
    
    eframe::run_native(
        "Pebble",
        options,
        Box::new(|cc| {
            // Apply the dark theme
            apply_theme(&cc.egui_ctx);
            Ok(Box::new(PebbleApp::new()))
        }),
    )
}

fn load_icon_from_png() -> egui::IconData {
    // Try to load the peb.png file
    if let Ok(image_data) = std::fs::read("media/peb.png") {
        if let Ok(image) = image::load_from_memory(&image_data) {
            let image = image.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);
            let image_buffer = image.to_rgba8();
            return egui::IconData {
                rgba: image_buffer.into_raw(),
                width: 32,
                height: 32,
            };
        }
    }
    
    // Fallback to generated icon if loading fails
    create_pebble_icon()
}

fn create_pebble_icon() -> egui::IconData {
    // Keep the existing generated icon as fallback
    let size = 32u32;
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    
    // Draw a cute rounded pebble shape
    for y in 0..size {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            let cx = size as f32 / 2.0;
            let cy = size as f32 / 2.0;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            
            // Create a rounded, slightly irregular pebble shape
            let dist_x = dx.abs() / 14.0;
            let dist_y = dy.abs() / 12.0;
            let combined = (dist_x.powf(2.5) + dist_y.powf(2.5)).powf(1.0/2.5);
            
            if combined < 1.0 {
                // Main pebble color (light gray like in the image)
                let base_color = 180u8;
                let edge_factor = combined;
                let color = (base_color as f32 * (1.0 - edge_factor * 0.3)) as u8;
                
                pixels[idx] = color;       // R
                pixels[idx + 1] = color;   // G
                pixels[idx + 2] = color;   // B
                pixels[idx + 3] = 255;     // A
                
                // Add cute face features
                let fx = x as i32;
                let fy = y as i32;
                
                // Eyes (oval shaped)
                let left_eye_x = 11;
                let right_eye_x = 21;
                let eye_y = 13;
                
                // Left eye
                if ((fx - left_eye_x).abs() <= 1 && (fy - eye_y).abs() <= 2) ||
                   ((fx - left_eye_x).abs() <= 2 && (fy - eye_y).abs() <= 1) {
                    pixels[idx] = 60;
                    pixels[idx + 1] = 60;
                    pixels[idx + 2] = 60;
                }
                
                // Right eye
                if ((fx - right_eye_x).abs() <= 1 && (fy - eye_y).abs() <= 2) ||
                   ((fx - right_eye_x).abs() <= 2 && (fy - eye_y).abs() <= 1) {
                    pixels[idx] = 60;
                    pixels[idx + 1] = 60;
                    pixels[idx + 2] = 60;
                }
                
                // Cute smile (curved line)
                if fy == 19 && fx >= 12 && fx <= 20 {
                    let smile_curve = ((fx - 16).abs() as f32 / 4.0).powf(2.0);
                    if (fy as f32 - 19.0 - smile_curve).abs() < 1.0 {
                        pixels[idx] = 60;
                        pixels[idx + 1] = 60;
                        pixels[idx + 2] = 60;
                    }
                }
                if fy == 20 && (fx == 11 || fx == 21) {
                    pixels[idx] = 60;
                    pixels[idx + 1] = 60;
                    pixels[idx + 2] = 60;
                }
                
                // Blush/cheeks (optional, lighter pink/red tint)
                if ((fx - 8).pow(2) + (fy - 16).pow(2) <= 9) ||
                   ((fx - 24).pow(2) + (fy - 16).pow(2) <= 9) {
                    pixels[idx] = (pixels[idx] as u16 * 11 / 10).min(255) as u8;
                    pixels[idx + 1] = (pixels[idx + 1] as u16 * 9 / 10) as u8;
                    pixels[idx + 2] = (pixels[idx + 2] as u16 * 9 / 10) as u8;
                }
            }
        }
    }
    
    egui::IconData {
        rgba: pixels,
        width: size,
        height: size,
    }
}

impl eframe::App for PebbleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update(ctx);
    }
}
