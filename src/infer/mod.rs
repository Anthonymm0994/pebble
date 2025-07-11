use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnType {
    Integer,
    Real,
    Text,
    Boolean,
    Date,
    DateTime,
    Blob,
}

impl ColumnType {
    pub fn to_sql_type(&self) -> &'static str {
        match self {
            ColumnType::Integer => "INTEGER",
            ColumnType::Real => "REAL",
            ColumnType::Text => "TEXT",
            ColumnType::Boolean => "BOOLEAN",
            ColumnType::Date => "DATE",
            ColumnType::DateTime => "DATETIME",
            ColumnType::Blob => "BLOB",
        }
    }
}

pub struct TypeInferrer;

impl TypeInferrer {
    pub fn infer_column_types(
        headers: &[String],
        samples: &[Vec<String>],
    ) -> Vec<(String, ColumnType)> {
        headers
            .iter()
            .enumerate()
            .map(|(idx, header)| {
                let column_type = Self::infer_column_type(samples, idx);
                (header.clone(), column_type)
            })
            .collect()
    }
    
    fn infer_column_type(samples: &[Vec<String>], col_idx: usize) -> ColumnType {
        let mut is_int = true;
        let mut is_float = true;
        let mut is_bool = true;
        let mut is_date = true;
        let mut is_datetime = true;
        let mut non_empty_count = 0;
        let mut has_decimal = false;
        
        for row in samples {
            if let Some(value) = row.get(col_idx) {
                if value.is_empty() || value.to_lowercase() == "null" {
                    continue;
                }
                
                non_empty_count += 1;
                
                // Check boolean
                if is_bool {
                    let lower = value.to_lowercase();
                    if !matches!(lower.as_str(), "true" | "false" | "1" | "0" | "yes" | "no" | "y" | "n") {
                        is_bool = false;
                    }
                }
                
                // Check integer
                if is_int {
                    if value.parse::<i64>().is_err() {
                        is_int = false;
                    }
                }
                
                // Check float (but track if we see decimals)
                if is_float {
                    match value.parse::<f64>() {
                        Ok(_) => {
                            if value.contains('.') {
                                has_decimal = true;
                            }
                        }
                        Err(_) => is_float = false,
                    }
                }
                
                // Check date
                if is_date && !Self::is_date(value) {
                    is_date = false;
                }
                
                // Check datetime
                if is_datetime && !Self::is_datetime(value) {
                    is_datetime = false;
                }
            }
        }
        
        // Return the most specific type that matches
        if non_empty_count == 0 {
            ColumnType::Text
        } else if is_bool {
            ColumnType::Boolean
        } else if is_int {
            ColumnType::Integer
        } else if is_float {
            // Only use Real if we actually saw decimal values
            if has_decimal {
                ColumnType::Real
            } else {
                ColumnType::Integer
            }
        } else if is_datetime {
            ColumnType::DateTime
        } else if is_date {
            ColumnType::Date
        } else {
            ColumnType::Text
        }
    }
    
    fn is_date(value: &str) -> bool {
        // Common date formats
        let formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
        ];
        
        for format in &formats {
            if NaiveDate::parse_from_str(value, format).is_ok() {
                return true;
            }
        }
        
        false
    }
    
    fn is_datetime(value: &str) -> bool {
        // Common datetime formats
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
        ];
        
        for format in &formats {
            if NaiveDateTime::parse_from_str(value, format).is_ok() {
                return true;
            }
        }
        
        // Check for Unix timestamp (seconds or milliseconds)
        if let Ok(ts) = value.parse::<i64>() {
            if ts > 946684800 && ts < 4102444800 {
                // Between 2000 and 2100 in seconds
                return true;
            }
            if ts > 946684800000 && ts < 4102444800000 {
                // Between 2000 and 2100 in milliseconds
                return true;
            }
        }
        
        false
    }
} 