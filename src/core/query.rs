use crate::core::{Database, error::{Result, PebbleError}};
use std::sync::Arc;

pub struct QueryExecutor;

impl QueryExecutor {
    pub fn execute(db: &Arc<Database>, query: &str) -> Result<QueryResult> {
        let conn = db.conn();
        let mut stmt = conn.prepare(query)?;
        
        let column_names: Vec<String> = stmt.column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        
        let mut rows = Vec::new();
        let mut rows_iter = stmt.query([])?;
        
        while let Some(row) = rows_iter.next()? {
            let mut row_data = Vec::new();
            for i in 0..column_names.len() {
                let value: rusqlite::types::Value = row.get(i)?;
                row_data.push(match value {
                    rusqlite::types::Value::Null => "NULL".to_string(),
                    rusqlite::types::Value::Integer(i) => i.to_string(),
                    rusqlite::types::Value::Real(f) => f.to_string(),
                    rusqlite::types::Value::Text(s) => s,
                    rusqlite::types::Value::Blob(b) => format!("<BLOB {} bytes>", b.len()),
                });
            }
            rows.push(row_data);
        }
        
        Ok(QueryResult {
            columns: column_names,
            rows,
            total_rows: None,
        })
    }
    
    pub fn execute_with_pagination(
        db: &Arc<Database>,
        query: &str,
        page: usize,
        page_size: usize,
    ) -> Result<QueryResult> {
        // First, get the total count
        let count_query = format!("SELECT COUNT(*) FROM ({})", query);
        let total_rows = match db.conn().query_row(&count_query, [], |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(count) => Some(count as usize),
            Err(_) => None,
        };
        
        // Then get the paginated results
        let paginated_query = format!("{} LIMIT {} OFFSET {}", query, page_size, page * page_size);
        let mut result = Self::execute(db, &paginated_query)?;
        result.total_rows = total_rows;
        
        Ok(result)
    }
    
    pub fn validate_read_only(query: &str) -> Result<()> {
        let query_upper = query.to_uppercase();
        let forbidden_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
            "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA"
        ];
        
        for keyword in &forbidden_keywords {
            if query_upper.contains(keyword) {
                return Err(PebbleError::InvalidQuery(
                    format!("Query contains forbidden keyword '{}' in read-only mode", keyword)
                ));
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub total_rows: Option<usize>,
} 