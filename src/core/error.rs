use std::fmt;

#[derive(Debug)]
pub enum PebbleError {
    Database(String),
    Csv(String),
    Io(String),
    InvalidQuery(String),
    ReadOnly,
    Custom(String),
}

impl fmt::Display for PebbleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PebbleError::Database(msg) => write!(f, "Database error: {}", msg),
            PebbleError::Csv(msg) => write!(f, "CSV error: {}", msg),
            PebbleError::Io(msg) => write!(f, "IO error: {}", msg),
            PebbleError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
            PebbleError::ReadOnly => write!(f, "Database is read-only"),
            PebbleError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for PebbleError {}

impl From<rusqlite::Error> for PebbleError {
    fn from(err: rusqlite::Error) -> Self {
        PebbleError::Database(err.to_string())
    }
}

impl From<csv::Error> for PebbleError {
    fn from(err: csv::Error) -> Self {
        PebbleError::Csv(err.to_string())
    }
}

impl From<std::io::Error> for PebbleError {
    fn from(err: std::io::Error) -> Self {
        PebbleError::Io(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, PebbleError>; 