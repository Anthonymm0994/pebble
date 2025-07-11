use rusqlite::{Connection, OpenFlags};
use std::path::Path;
use crate::core::error::Result;

#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub is_primary_key: bool,
}

#[derive(Debug, Clone)]
pub struct TableInfo {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
    pub row_count: i64,
}

#[derive(Debug, Clone)]
pub struct ViewInfo {
    pub name: String,
    pub sql: String,
}

pub struct Database {
    conn: Connection,
    is_readonly: bool,
}

impl Database {
    pub fn open_readonly<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        
        Ok(Self {
            conn,
            is_readonly: true,
        })
    }
    
    pub fn open_writable<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        Ok(Self {
            conn,
            is_readonly: false,
        })
    }
    
    pub fn create_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        
        Ok(Self {
            conn,
            is_readonly: false,
        })
    }
    
    pub fn is_readonly(&self) -> bool {
        self.is_readonly
    }
    
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
    
    pub fn get_tables(&self) -> Result<Vec<TableInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )?;
        
        let table_names: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        
        let mut tables = Vec::new();
        for name in table_names {
            let columns = self.get_table_columns(&name)?;
            let row_count = self.get_table_row_count(&name)?;
            
            tables.push(TableInfo {
                name,
                columns,
                row_count,
            });
        }
        
        Ok(tables)
    }
    
    pub fn get_views(&self) -> Result<Vec<ViewInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name"
        )?;
        
        let views = stmt
            .query_map([], |row| {
                Ok(ViewInfo {
                    name: row.get(0)?,
                    sql: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        
        Ok(views)
    }
    
    fn get_table_columns(&self, table_name: &str) -> Result<Vec<ColumnInfo>> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info('{}')", table_name))?;
        
        let columns = stmt
            .query_map([], |row| {
                Ok(ColumnInfo {
                    name: row.get(1)?,
                    data_type: row.get(2)?,
                    is_nullable: row.get::<_, i32>(3)? == 0,
                    is_primary_key: row.get::<_, i32>(5)? == 1,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        
        Ok(columns)
    }
    
    fn get_table_row_count(&self, table_name: &str) -> Result<i64> {
        let count: i64 = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM '{}'", table_name),
            [],
            |row| row.get(0),
        )?;
        
        Ok(count)
    }
    
    pub fn execute_query(&self, query: &str) -> Result<Vec<Vec<rusqlite::types::Value>>> {
        let mut stmt = self.conn.prepare(query)?;
        let column_count = stmt.column_count();
        
        let rows = stmt
            .query_map([], |row| {
                let mut values = Vec::new();
                for i in 0..column_count {
                    values.push(row.get::<_, rusqlite::types::Value>(i)?);
                }
                Ok(values)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        
        Ok(rows)
    }
    
    pub fn get_column_names(&self, query: &str) -> Result<Vec<String>> {
        let stmt = self.conn.prepare(query)?;
        let names = stmt
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        
        Ok(names)
    }
    
    pub fn execute_sql(&mut self, sql: &str) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot execute SQL in read-only mode".to_string()
            ));
        }
        
        self.conn.execute(sql, [])?;
        Ok(())
    }
    
    pub fn create_table(&mut self, table_name: &str, columns: &[(&str, &str)]) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot create table in read-only mode".to_string()
            ));
        }
        
        let column_defs: Vec<String> = columns
            .iter()
            .map(|(name, dtype)| format!("{} {}", name, dtype))
            .collect();
        
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS '{}' ({})",
            table_name,
            column_defs.join(", ")
        );
        
        self.conn.execute(&sql, [])?;
        Ok(())
    }
    
    pub fn create_index(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot create index in read-only mode".to_string()
            ));
        }
        
        let index_name = format!("idx_{}_{}", table_name, column_name);
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS '{}' ON '{}' ('{}')",
            index_name, table_name, column_name
        );
        
        self.conn.execute(&sql, [])?;
        Ok(())
    }
    
    pub fn begin_transaction(&mut self) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot begin transaction in read-only mode".to_string()
            ));
        }
        
        self.conn.execute("BEGIN", [])?;
        Ok(())
    }
    
    pub fn commit_transaction(&mut self) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot commit transaction in read-only mode".to_string()
            ));
        }
        
        self.conn.execute("COMMIT", [])?;
        Ok(())
    }
    
    pub fn rollback_transaction(&mut self) -> Result<()> {
        self.conn.execute("ROLLBACK", [])?;
        Ok(())
    }
    
    pub fn prepare_insert(&mut self, table_name: &str, column_count: usize) -> Result<rusqlite::Statement> {
        let placeholders: Vec<String> = (0..column_count).map(|_| "?".to_string()).collect();
        let sql = format!(
            "INSERT INTO '{}' VALUES ({})",
            table_name,
            placeholders.join(", ")
        );
        
        Ok(self.conn.prepare(&sql)?)
    }
    
    pub fn insert_record(&mut self, table_name: &str, values: &[String]) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot insert records in read-only mode".to_string()
            ));
        }
        
        let placeholders: Vec<String> = (0..values.len()).map(|_| "?".to_string()).collect();
        let sql = format!(
            "INSERT INTO '{}' VALUES ({})",
            table_name,
            placeholders.join(", ")
        );
        
        let params: Vec<&dyn rusqlite::ToSql> = values
            .iter()
            .map(|v| v as &dyn rusqlite::ToSql)
            .collect();
        
        self.conn.execute(&sql, params.as_slice())?;
        Ok(())
    }
    
    pub fn batch_insert(&mut self, table_name: &str, all_values: &[Vec<String>]) -> Result<()> {
        if self.is_readonly {
            return Err(crate::core::error::PebbleError::Custom(
                "Cannot insert records in read-only mode".to_string()
            ));
        }
        
        if all_values.is_empty() {
            return Ok(());
        }
        
        let column_count = all_values[0].len();
        let placeholders: Vec<String> = (0..column_count).map(|_| "?".to_string()).collect();
        let sql = format!(
            "INSERT INTO '{}' VALUES ({})",
            table_name,
            placeholders.join(", ")
        );
        
        let mut stmt = self.conn.prepare(&sql)?;
        
        for values in all_values {
            let params: Vec<&dyn rusqlite::ToSql> = values
                .iter()
                .map(|v| v as &dyn rusqlite::ToSql)
                .collect();
            
            stmt.execute(params.as_slice())?;
        }
        
        Ok(())
    }
} 