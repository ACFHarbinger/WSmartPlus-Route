/// Data loading commands for simulation logs, CSVs, and training metrics.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::commands::sim_watcher::{parse_day_log_line, DayLogEntry};

/// Matches the `CsvFile` TypeScript interface in DataExplorer.tsx
#[derive(Debug, Serialize, Deserialize)]
pub struct CsvFile {
    pub path: String,
    pub headers: Vec<String>,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
}

/// Matches the `OutputDir` TypeScript interface in ExperimentTracker.tsx
#[derive(Debug, Serialize, Deserialize)]
pub struct OutputDir {
    pub name: String,
    pub path: String,
    pub created_at: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRun {
    pub name: String,
    pub path: String,
    pub has_metrics: bool,
    pub has_hparams: bool,
}

#[tauri::command]
pub fn load_simulation_log(path: String) -> Result<Vec<DayLogEntry>, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<DayLogEntry> = content
        .lines()
        .filter_map(parse_day_log_line)
        .collect();
    Ok(entries)
}

#[tauri::command]
pub fn load_csv_file(path: String) -> Result<CsvFile, String> {
    let mut reader = csv::Reader::from_path(&path).map_err(|e| e.to_string())?;
    let header_record = reader.headers().map_err(|e| e.to_string())?.clone();
    let headers: Vec<String> = header_record.iter().map(|h| h.to_string()).collect();

    let mut rows: Vec<HashMap<String, serde_json::Value>> = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let mut row = HashMap::new();
        for (header, field) in headers.iter().zip(record.iter()) {
            let value = if let Ok(n) = field.parse::<f64>() {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(n)
                        .unwrap_or(serde_json::Number::from(0)),
                )
            } else {
                serde_json::Value::String(field.to_string())
            };
            row.insert(header.clone(), value);
        }
        rows.push(row);
    }
    Ok(CsvFile { path, headers, rows })
}

#[tauri::command]
pub fn list_output_dirs(output_path: String) -> Result<Vec<OutputDir>, String> {
    let base = Path::new(&output_path);
    if !base.exists() {
        return Ok(vec![]);
    }

    let mut dirs: Vec<OutputDir> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let meta = std::fs::metadata(&path).ok();
            let created_at = meta
                .as_ref()
                .and_then(|m| m.created().ok())
                .and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_secs())
                })
                .map(|s| {
                    // ISO-8601 approximation from epoch seconds
                    let dt = s;
                    format!("{}", dt)
                })
                .unwrap_or_default();

            // Sum size of all files in the dir
            let size_bytes: u64 = std::fs::read_dir(&path)
                .map(|rd| {
                    rd.filter_map(|f| f.ok())
                        .filter_map(|f| std::fs::metadata(f.path()).ok())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);

            OutputDir {
                name,
                path: path.to_string_lossy().to_string(),
                created_at,
                size_bytes,
            }
        })
        .collect();

    dirs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(dirs)
}

#[tauri::command]
pub fn list_training_runs(logs_path: String) -> Result<Vec<TrainingRun>, String> {
    let base = Path::new(&logs_path);
    if !base.exists() {
        return Ok(vec![]);
    }

    let mut runs: Vec<TrainingRun> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let has_metrics = path.join("metrics.csv").exists();
            let has_hparams = path.join("hparams.yaml").exists();
            TrainingRun {
                name,
                path: path.to_string_lossy().to_string(),
                has_metrics,
                has_hparams,
            }
        })
        .collect();

    runs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(runs)
}

#[tauri::command]
pub fn load_training_metrics(
    run_path: String,
) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
    let metrics_path = Path::new(&run_path).join("metrics.csv");
    if !metrics_path.exists() {
        return Ok(vec![]);
    }
    load_csv_file(metrics_path.to_string_lossy().to_string()).map(|f| f.rows)
}
