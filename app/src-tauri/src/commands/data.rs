/// Data loading commands for simulation logs, CSVs, and training metrics.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use zip::ZipArchive;

use crate::commands::process::resolve_python;
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

/// Read any text file (YAML, TOML, plain-text log) and return its content as a string.
/// Used by ConfigEditor and OutputBrowser to display file contents without a CSV parser.
#[tauri::command]
pub fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{e}: {path}"))
}

/// Write (overwrite) any text file — used by ConfigEditor to save changes to YAML configs.
#[tauri::command]
pub fn write_text_file(path: String, content: String) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(&path, &content).map_err(|e| format!("{e}: {path}"))
}

/// List all files (non-recursive) inside a directory, returning name + extension + size.
#[derive(Debug, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub extension: String,
}

#[tauri::command]
pub fn list_dir(path: String) -> Result<Vec<DirEntry>, String> {
    let base = Path::new(&path);
    if !base.exists() {
        return Ok(vec![]);
    }
    let mut entries: Vec<DirEntry> = std::fs::read_dir(base)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| {
            let p = e.path();
            let meta = std::fs::metadata(&p).ok();
            DirEntry {
                name: e.file_name().to_string_lossy().to_string(),
                path: p.to_string_lossy().to_string(),
                is_dir: p.is_dir(),
                size_bytes: meta.map(|m| m.len()).unwrap_or(0),
                extension: p
                    .extension()
                    .map(|x| x.to_string_lossy().to_string())
                    .unwrap_or_default(),
            }
        })
        .collect();
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    Ok(entries)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetPreviewStats {
    pub num_instances: usize,
    pub num_nodes: Option<usize>,
    pub demand_mean: Option<f64>,
    pub demand_std: Option<f64>,
    pub demand_histogram: Vec<f64>,
    pub distance_mean: Option<f64>,
    pub file_size_bytes: u64,
}

const PREVIEW_DATASET_SCRIPT: &str = r#"
import json, sys, pickle, os
path = sys.argv[1]
size = os.path.getsize(path)
stats = {
    "num_instances": 0, "num_nodes": None, "demand_mean": None,
    "demand_std": None, "demand_histogram": [], "distance_mean": None,
    "file_size_bytes": size,
}
try:
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
except Exception:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(0)

instances = data if isinstance(data, list) else [data]
stats["num_instances"] = len(instances)

def _tensor_stats(t):
    import torch
    if hasattr(t, "detach"):
        flat = t.detach().float().flatten()
        if flat.numel() == 0:
            return None, None, []
        vals = flat.tolist()
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        hist_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        hist = [0] * (len(hist_edges) - 1)
        for v in vals:
            for i in range(len(hist_edges) - 1):
                if hist_edges[i] <= v < hist_edges[i + 1]:
                    hist[i] += 1
                    break
        return mean, var ** 0.5, [h / max(len(vals), 1) for h in hist]
    return None, None, []

for inst in instances[:5]:
    d = dict(inst) if hasattr(inst, "keys") else (inst if isinstance(inst, dict) else {})
    demand = d.get("demand") or d.get("prize") or d.get("waste")
    loc = d.get("loc")
    if demand is not None:
        m, s, h = _tensor_stats(demand)
        if m is not None:
            stats["demand_mean"] = m
            stats["demand_std"] = s
            stats["demand_histogram"] = h
    if loc is not None and hasattr(loc, "shape"):
        stats["num_nodes"] = int(loc.shape[-2]) if len(loc.shape) >= 2 else int(loc.shape[0])
    dist = d.get("dist") or d.get("distance_matrix")
    if dist is not None:
        m, _, _ = _tensor_stats(dist)
        if m is not None:
            stats["distance_mean"] = m
    if stats["num_nodes"] is not None:
        break

print(json.dumps(stats))
"#;

/// Inspect a generated `.pkl`/`.pt` dataset and return summary statistics for the preview panel.
#[tauri::command]
pub fn preview_dataset_stats(
    path: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<DatasetPreviewStats, String> {
    if !Path::new(&path).exists() {
        return Err(format!("File not found: {path}"));
    }
    let python = resolve_python(&project_root, python_executable);
    let output = Command::new(&python)
        .arg("-c")
        .arg(PREVIEW_DATASET_SCRIPT)
        .arg(&path)
        .current_dir(&project_root)
        .output()
        .map_err(|e| format!("Failed to run preview script: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        return Err("Preview script produced no output".to_string());
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteBundleFile {
    pub path: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteBundleInfo {
    pub path: String,
    pub version: Option<String>,
    pub created_at: Option<String>,
    pub files: Vec<WsrouteBundleFile>,
}

/// Inspect a `.wsroute` zip bundle produced by `logic/gen/export_for_studio.py`.
#[tauri::command]
pub fn inspect_wsroute_bundle(path: String) -> Result<WsrouteBundleInfo, String> {
    let file = File::open(&path).map_err(|e| format!("Failed to open bundle: {e}"))?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid zip bundle: {e}"))?;

    let mut files = Vec::new();
    let mut version = None;
    let mut created_at = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        let size = entry.size();

        if name == "manifest.json" {
            let mut buf = String::new();
            if entry.read_to_string(&mut buf).is_ok() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&buf) {
                    version = v
                        .get("version")
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    created_at = v
                        .get("created_at")
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                }
            }
        }

        files.push(WsrouteBundleFile {
            path: name,
            size_bytes: size,
        });
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(WsrouteBundleInfo {
        path,
        version,
        created_at,
        files,
    })
}
