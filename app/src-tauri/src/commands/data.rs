/// Data loading commands for simulation logs, CSVs, and training metrics.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

use crate::commands::arrow;
use crate::commands::process::resolve_python;
use crate::commands::sim_watcher::{parse_day_log_line, parse_policy_viz_line, DayLogEntry, PolicyVizEntry};

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
pub fn load_policy_viz_log(path: String) -> Result<Vec<PolicyVizEntry>, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let entries: Vec<PolicyVizEntry> = content
        .lines()
        .filter_map(parse_policy_viz_line)
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
    pub arrow_sidecars: Option<u32>,
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
    let mut arrow_sidecars = None;

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
                    arrow_sidecars = v
                        .get("arrow_sidecars")
                        .and_then(|x| x.as_u64())
                        .map(|n| n as u32);
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
        arrow_sidecars,
        files,
    })
}

const BUNDLE_EXTENSIONS: &[&str] = &[
    "arrow", "csv", "json", "jsonl", "yaml", "yml", "npz", "pkl", "pt", "parquet",
];

fn is_bundle_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| BUNDLE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn collect_bundle_files(
    dir: &Path,
    base: &Path,
    out: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_bundle_files(&path, base, out)?;
        } else if is_bundle_extension(&path) {
            let rel = path.strip_prefix(base).map_err(|e| e.to_string())?;
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            out.push((rel_str, path));
        }
    }
    Ok(())
}

fn iso_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Package a run output directory into a `.wsroute` zip bundle (§G.8).
#[tauri::command]
pub fn create_wsroute_bundle(
    source_dir: String,
    output_path: String,
    include_arrow: Option<bool>,
) -> Result<WsrouteBundleInfo, String> {
    let source = Path::new(&source_dir);
    if !source.is_dir() {
        return Err(format!("Source directory not found: {source_dir}"));
    }

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_bundle_files(source, source, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut arrow_sidecars: Vec<(String, PathBuf)> = Vec::new();
    if include_arrow.unwrap_or(false) {
        let staging = std::env::temp_dir().join(format!(
            "wsroute_arrow_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

        for (rel, abs) in &files {
            let ext = abs.extension().and_then(|e| e.to_str()).unwrap_or("");
            let arrow_rel = Path::new(rel)
                .with_extension("arrow")
                .to_string_lossy()
                .replace('\\', "/");
            if files.iter().any(|(r, _)| r == &arrow_rel) {
                continue;
            }
            let arrow_abs = staging.join(&arrow_rel);
            let path_str = abs.to_string_lossy();
            match ext {
                "csv" => arrow::write_csv_arrow_sidecar(&path_str, &arrow_abs)?,
                "jsonl" => arrow::write_simulation_log_arrow_sidecar(&path_str, &arrow_abs)?,
                _ => continue,
            };
            arrow_sidecars.push((arrow_rel, arrow_abs));
        }
    }

    let mut packaged: Vec<(String, PathBuf)> = files.clone();
    packaged.extend(arrow_sidecars.clone());
    packaged.sort_by(|a, b| a.0.cmp(&b.0));

    let rel_paths: Vec<String> = packaged.iter().map(|(r, _)| r.clone()).collect();
    let manifest = serde_json::json!({
        "version": "1",
        "created_at": iso_timestamp(),
        "source": source_dir,
        "file_count": rel_paths.len(),
        "arrow_sidecars": arrow_sidecars.len(),
        "files": rel_paths,
    });

    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let out_file = File::create(&output_path).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(out_file);
    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("manifest.json", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(manifest.to_string().as_bytes())
        .map_err(|e| e.to_string())?;

    for (rel, abs) in &packaged {
        zip.start_file(rel, options).map_err(|e| e.to_string())?;
        let mut f = File::open(abs).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        zip.write_all(&buf).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    if let Some((_, path)) = arrow_sidecars.first() {
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
    }
    inspect_wsroute_bundle(output_path)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WsrouteExtractResult {
    pub dest_dir: String,
    pub extracted_files: Vec<String>,
    pub log_path: Option<String>,
}

/// Extract a `.wsroute` bundle to a destination directory; returns first `.jsonl` path if found.
#[tauri::command]
pub fn extract_wsroute_bundle(path: String, dest_dir: String) -> Result<WsrouteExtractResult, String> {
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let dest = Path::new(&dest_dir);

    let file = File::open(&path).map_err(|e| format!("Failed to open bundle: {e}"))?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid zip bundle: {e}"))?;

    let mut extracted = Vec::new();
    let mut log_path: Option<String> = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        if name.ends_with('/') {
            continue;
        }

        let out_path = dest.join(&name);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let mut out_file = File::create(&out_path).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out_file).map_err(|e| e.to_string())?;

        let out_str = out_path.to_string_lossy().to_string();
        extracted.push(out_str.clone());

        if log_path.is_none() && name.ends_with(".jsonl") {
            log_path = Some(out_str);
        }
    }

    extracted.sort();

    Ok(WsrouteExtractResult {
        dest_dir,
        extracted_files: extracted,
        log_path,
    })
}

const PARQUET_SCRIPT: &str = r#"
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
df.to_parquet(sys.argv[2], engine="pyarrow", index=False)
"#;

fn run_parquet_export(
    project_root: &str,
    csv_path: &str,
    output_path: &str,
) -> Result<String, String> {
    let python = resolve_python(project_root, None);
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let output = Command::new(&python)
        .args(["-c", PARQUET_SCRIPT, csv_path, output_path])
        .output()
        .map_err(|e| format!("Failed to run Python for Parquet export: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Parquet export failed (is pyarrow installed?): {stderr}"
        ));
    }

    Ok(output_path.to_string())
}

/// Convert an on-disk CSV file to Parquet via pandas/pyarrow (§G.7).
#[tauri::command]
pub fn export_csv_to_parquet(
    project_root: String,
    csv_path: String,
    output_path: String,
) -> Result<String, String> {
    if !Path::new(&csv_path).is_file() {
        return Err(format!("CSV file not found: {csv_path}"));
    }
    run_parquet_export(&project_root, &csv_path, &output_path)
}

fn write_table_csv(
    headers: &[String],
    rows: &[HashMap<String, serde_json::Value>],
    path: &Path,
) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path).map_err(|e| e.to_string())?;
    writer.write_record(headers).map_err(|e| e.to_string())?;
    for row in rows {
        let record: Vec<String> = headers
            .iter()
            .map(|h| match row.get(h) {
                Some(serde_json::Value::Number(n)) => n.to_string(),
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Bool(b)) => b.to_string(),
                Some(serde_json::Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            })
            .collect();
        writer.write_record(&record).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Export in-memory tabular data to Parquet (writes a temp CSV first) (§G.7).
#[tauri::command]
pub fn export_table_parquet(
    project_root: String,
    headers: Vec<String>,
    rows: Vec<HashMap<String, serde_json::Value>>,
    output_path: String,
) -> Result<String, String> {
    let temp_dir = std::env::temp_dir().join("wsmart-studio-parquet");
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let temp_csv = temp_dir.join(format!(
        "table_{}.csv",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    write_table_csv(&headers, &rows, &temp_csv)?;
    let result = run_parquet_export(&project_root, temp_csv.to_str().unwrap(), &output_path);
    let _ = fs::remove_file(&temp_csv);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_sample_run(dir: &Path) -> std::io::Result<()> {
        let log = dir.join("sim_log.jsonl");
        let mut f = File::create(log)?;
        writeln!(
            f,
            r#"GUI_DAY_LOG_START:test_policy,0,1,{{"profit":10.0,"km":5.0,"overflows":0,"kg":42.0,"kg/km":8.4,"cost":-32.0,"ncol":10,"kg_lost":0.0}}"#,
        )?;
        writeln!(
            f,
            r#"GUI_DAY_LOG_START:test_policy,0,2,{{"profit":11.5,"km":5.2,"overflows":1,"kg":40.0,"kg/km":7.7,"cost":-28.5,"ncol":9,"kg_lost":0.5}}"#,
        )?;
        Ok(())
    }

    fn write_sample_csv(dir: &Path) -> std::io::Result<()> {
        let csv = dir.join("metrics.csv");
        let mut f = File::create(csv)?;
        writeln!(f, "policy,profit,overflows")?;
        writeln!(f, "gurobi,12.5,0")?;
        writeln!(f, "alns,10.1,2")?;
        Ok(())
    }

    #[test]
    fn wsroute_bundle_round_trip_preserves_jsonl() {
        let base = std::env::temp_dir().join(format!(
            "wsroute_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let source = base.join("source");
        let bundle_path = base.join("run.wsroute");
        let extract_dir = base.join("extracted");

        fs::create_dir_all(&source).expect("create source");
        write_sample_run(&source).expect("write sample run");

        let info = create_wsroute_bundle(
            source.to_string_lossy().to_string(),
            bundle_path.to_string_lossy().to_string(),
            None,
        )
        .expect("create bundle");
        assert!(info.files.iter().any(|f| f.path.ends_with(".jsonl")));

        let extracted = extract_wsroute_bundle(
            bundle_path.to_string_lossy().to_string(),
            extract_dir.to_string_lossy().to_string(),
        )
        .expect("extract bundle");

        let log_path = extracted.log_path.expect("jsonl path");
        let content = fs::read_to_string(&log_path).expect("read extracted log");
        assert!(content.contains("GUI_DAY_LOG_START:test_policy"));

        let entries = load_simulation_log(log_path).expect("parse extracted log");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].policy, "test_policy");
        assert_eq!(entries[0].day, 1);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn simulation_arrow_sidecar_row_parity() {
        let base = std::env::temp_dir().join(format!(
            "sim_arrow_parity_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).expect("create base");
        write_sample_run(&base).expect("write sample run");

        let log_path = base.join("sim_log.jsonl");
        let arrow_path = base.join("sim_log.arrow");
        let entries = load_simulation_log(log_path.to_string_lossy().to_string())
            .expect("load simulation log");
        let row_count =
            arrow::write_simulation_log_arrow_sidecar(&log_path.to_string_lossy(), &arrow_path)
                .expect("write simulation arrow sidecar");
        assert_eq!(row_count, entries.len());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn wsroute_bundle_with_arrow_sidecars() {
        let base = std::env::temp_dir().join(format!(
            "wsroute_arrow_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let source = base.join("source");
        let bundle_path = base.join("run.wsroute");

        fs::create_dir_all(&source).expect("create source");
        write_sample_run(&source).expect("write sample run");
        write_sample_csv(&source).expect("write sample csv");

        let info = create_wsroute_bundle(
            source.to_string_lossy().to_string(),
            bundle_path.to_string_lossy().to_string(),
            Some(true),
        )
        .expect("create bundle with arrow");

        assert_eq!(info.arrow_sidecars, Some(2));
        assert!(info.files.iter().any(|f| f.path.ends_with("metrics.arrow")));
        assert!(info.files.iter().any(|f| f.path.ends_with("sim_log.arrow")));

        let _ = fs::remove_dir_all(&base);
    }
}
