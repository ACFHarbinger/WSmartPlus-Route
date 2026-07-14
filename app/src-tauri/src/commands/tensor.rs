/// TensorDict / NumPy archive inspection and slice streaming (§G.5.1).
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use arrow::array::{Float64Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zip::ZipArchive;

use crate::commands::arrow::{export_batch, ArrowIpcFile};
use crate::commands::process::resolve_python;

const INSPECT_TD_SCRIPT: &str = r#"
import json, sys
path = sys.argv[1]
try:
    import torch
except ImportError as e:
    print(json.dumps({"error": f"PyTorch required for .td files: {e}"}))
    sys.exit(0)
try:
    td = torch.load(path, map_location="cpu", weights_only=False)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
if not hasattr(td, "keys"):
    print(json.dumps({"error": "File does not contain a TensorDict-like object"}))
    sys.exit(0)
arrays = []
total_bytes = 0
for key in sorted(td.keys()):
    t = td[key]
    if not hasattr(t, "shape"):
        continue
    shape = list(t.shape)
    dtype = str(getattr(t, "dtype", "float32")).replace("torch.", "")
    nbytes = int(t.numel() * t.element_size())
    arrays.append({"key": key, "shape": shape, "dtype": dtype, "size_bytes": nbytes})
    total_bytes += nbytes
print(json.dumps({
    "path": path,
    "arrays": arrays,
    "total_bytes": total_bytes,
    "used_memmap": False,
}))
"#;

const LOAD_TD_SLICE_SCRIPT: &str = r#"
import json, sys
path, key, indices_json, max_dim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
indices = json.loads(indices_json)
try:
    import torch
    import numpy as np
except ImportError as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
try:
    td = torch.load(path, map_location="cpu", weights_only=False)
    arr = td[key].detach().float().cpu().numpy()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)
shape = list(arr.shape)
if len(shape) == 0:
    print(json.dumps({"error": "Empty tensor"}))
    sys.exit(0)
if len(shape) == 1:
    plane = arr.reshape(-1, 1)
elif len(shape) == 2:
    plane = arr
else:
    leading = len(shape) - 2
    if len(indices) < leading:
        print(json.dumps({"error": f"Need {leading} leading indices, got {len(indices)}"}))
        sys.exit(0)
    rows, cols = shape[-2], shape[-1]
    prefix = indices[:leading]
    out = []
    for r in range(rows):
        for c in range(cols):
            idx = tuple(prefix + [r, c])
            out.append(float(arr[idx]))
    plane = np.array(out, dtype=np.float64).reshape(rows, cols)
rows, cols = plane.shape
stride_r = max(1, rows // max_dim)
stride_c = max(1, cols // max_dim)
grid = plane[::stride_r, ::stride_c]
values = grid.tolist()
flat = [v for row in values for v in row]
vmin = float(np.min(flat)) if flat else 0.0
vmax = float(np.max(flat)) if flat else 0.0
print(json.dumps({
    "key": key,
    "full_shape": shape,
    "slice_shape": [len(values), len(values[0]) if values else 0],
    "indices": indices,
    "values": values,
    "min": vmin,
    "max": vmax,
}))
"#;

fn is_td_path(path: &Path) -> bool {
    path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("td"))
        .unwrap_or(false)
}

fn inspect_td_archive(
    path: &str,
    project_root: &str,
    python_executable: Option<String>,
) -> Result<NpzArchiveInfo, String> {
    let python = resolve_python(project_root, python_executable);
    let output = Command::new(&python)
        .arg("-c")
        .arg(INSPECT_TD_SCRIPT)
        .arg(path)
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Failed to inspect .td archive: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("TensorDict inspect produced no output: {stderr}"));
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid inspect JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

fn load_td_slice(
    path: &str,
    key: &str,
    indices: &[usize],
    max_dim: usize,
    project_root: &str,
    python_executable: Option<String>,
) -> Result<TensorSlicePreview, String> {
    let start = Instant::now();
    let python = resolve_python(project_root, python_executable);
    let indices_json =
        serde_json::to_string(indices).map_err(|e| format!("Index serialisation failed: {e}"))?;
    let output = Command::new(&python)
        .arg("-c")
        .arg(LOAD_TD_SLICE_SCRIPT)
        .arg(path)
        .arg(key)
        .arg(&indices_json)
        .arg(max_dim.to_string())
        .current_dir(project_root)
        .output()
        .map_err(|e| format!("Failed to load .td slice: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("TensorDict slice produced no output: {stderr}"));
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid slice JSON: {e}"))?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }

    let mut preview: TensorSlicePreview =
        serde_json::from_value(value).map_err(|e| format!("Slice parse error: {e}"))?;
    preview.rust_ms = start.elapsed().as_millis() as u64;
    Ok(preview)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NpzArrayInfo {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NpzArchiveInfo {
    pub path: String,
    pub arrays: Vec<NpzArrayInfo>,
    pub total_bytes: u64,
    pub used_memmap: bool,
}

#[derive(Debug, Serialize)]
pub struct NpzVectorData {
    pub key: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorSlicePreview {
    pub key: String,
    pub full_shape: Vec<usize>,
    pub slice_shape: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    pub rust_ms: u64,
}

fn parse_npy_header(raw: &[u8]) -> Result<(Vec<usize>, String, usize), String> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic header".to_string());
    }
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;
    if raw.len() < header_end {
        return Err("Truncated NPY header".to_string());
    }
    let header = String::from_utf8_lossy(&raw[header_start..header_end]);
    let shape = parse_shape_from_header(&header)?;
    let dtype = parse_descr_from_header(&header).unwrap_or_else(|| "unknown".to_string());
    Ok((shape, dtype, header_end))
}

fn parse_shape_from_header(header: &str) -> Result<Vec<usize>, String> {
    let start = header.find("'shape'").ok_or("shape missing in NPY header")?;
    let rest = &header[start..];
    let open = rest.find('(').ok_or("shape tuple missing")?;
    let close = rest[open..].find(')').ok_or("shape tuple unclosed")?;
    let inner = &rest[open + 1..open + close];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    Ok(inner
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect())
}

fn parse_descr_from_header(header: &str) -> Option<String> {
    let start = header.find("'descr'")?;
    let rest = &header[start..];
    let q1 = rest.find('\'')?;
    let rest2 = &rest[q1 + 1..];
    let q2 = rest2.find('\'')?;
    Some(rest2[..q2].to_string())
}

fn read_zip_npy_entry(
    entry: &mut zip::read::ZipFile<'_>,
) -> Result<(Vec<usize>, String, Vec<u8>), String> {
    let mut raw = Vec::new();
    entry.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let (shape, dtype, offset) = parse_npy_header(&raw)?;
    Ok((shape, dtype, raw[offset..].to_vec()))
}

fn load_array_from_bytes(bytes: &[u8]) -> Result<ArrayD<f64>, String> {
    let mut cursor = Cursor::new(bytes);
    ArrayD::read_npy(&mut cursor).map_err(|e| format!("NPY parse error: {e}"))
}

fn load_array_from_path(path: &Path) -> Result<ArrayD<f64>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    ArrayD::read_npy(&mut file).map_err(|e| format!("NPY read error: {e}"))
}

fn select_2d_slice(arr: &ArrayD<f64>, indices: &[usize]) -> Result<(Vec<usize>, Vec<f64>), String> {
    let shape = arr.shape().to_vec();
    if shape.is_empty() {
        return Err("Empty tensor".to_string());
    }
    if shape.len() == 1 {
        let n = shape[0];
        return Ok((vec![n, 1], arr.iter().copied().collect()));
    }
    if shape.len() == 2 {
        return Ok((shape.clone(), arr.iter().copied().collect()));
    }

    let leading = shape.len() - 2;
    if indices.len() < leading {
        return Err(format!(
            "Need {} leading indices for rank-{} tensor, got {}",
            leading,
            shape.len(),
            indices.len()
        ));
    }

    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut idx: Vec<usize> = indices[..leading].to_vec();
            idx.push(r);
            idx.push(c);
            let v = arr
                .get(IxDyn(&idx))
                .copied()
                .ok_or_else(|| format!("Index out of bounds: {idx:?}"))?;
            out.push(v);
        }
    }
    Ok((vec![rows, cols], out))
}

fn downsample_2d(
    flat: &[f64],
    rows: usize,
    cols: usize,
    max_dim: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    if rows == 0 || cols == 0 {
        return (vec![], vec![0, 0]);
    }
    let stride_r = (rows / max_dim).max(1);
    let stride_c = (cols / max_dim).max(1);
    let out_rows = (rows + stride_r - 1) / stride_r;
    let out_cols = (cols + stride_c - 1) / stride_c;
    let mut grid = vec![vec![0.0; out_cols]; out_rows];
    for (oi, r) in (0..rows).step_by(stride_r).enumerate() {
        for (oj, c) in (0..cols).step_by(stride_c).enumerate() {
            grid[oi][oj] = flat[r * cols + c];
        }
    }
    (grid, vec![out_rows, out_cols])
}

fn min_max(grid: &[Vec<f64>]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in grid {
        for &v in row {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
    }
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 0.0;
    }
    (min, max)
}

fn open_npz_array(
    path: &str,
    key: &str,
) -> Result<(Vec<usize>, ArrayD<f64>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let npy_name = format!("{key}.npy");
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();
    let resolved = if names.iter().any(|n| n == &npy_name) {
        npy_name
    } else if names.iter().any(|n| n == key) {
        key.to_string()
    } else {
        return Err(format!("Key '{key}' not found in NPZ"));
    };
    let mut entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    let (shape, _dtype, payload) = read_zip_npy_entry(&mut entry)?;
    let arr = load_array_from_bytes(&payload)?;
    if arr.shape().to_vec() != shape {
        // Header shape is authoritative for reporting; array may still be usable.
    }
    Ok((shape, arr))
}

#[tauri::command]
pub fn inspect_npz_archive(
    path: String,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<NpzArchiveInfo, String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("File not found: {path}"));
    }

    if is_td_path(p) {
        let root = project_root.filter(|s| !s.is_empty()).ok_or_else(|| {
            "Project root required to inspect TensorDict (.td) archives".to_string()
        })?;
        return inspect_td_archive(&path, &root, python_executable);
    }

    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "npy" {
        let file_meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
        let mut file = File::open(p).map_err(|e| e.to_string())?;
        let mut header_buf = vec![0u8; 256];
        let n = file.read(&mut header_buf).map_err(|e| e.to_string())?;
        header_buf.truncate(n);
        let (shape, dtype, _) = parse_npy_header(&header_buf)?;
        let key = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("array")
            .to_string();
        return Ok(NpzArchiveInfo {
            path,
            arrays: vec![NpzArrayInfo {
                key,
                shape,
                dtype,
                size_bytes: file_meta.len(),
            }],
            total_bytes: file_meta.len(),
            used_memmap: file_meta.len() > 8 * 1024 * 1024,
        });
    }

    let file = File::open(p).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let mut arrays = Vec::new();
    let mut total_bytes = 0u64;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let key = name.trim_end_matches(".npy").to_string();
        let (shape, dtype, payload) = read_zip_npy_entry(&mut entry)?;
        let size_bytes = payload.len() as u64;
        total_bytes += size_bytes;
        arrays.push(NpzArrayInfo {
            key,
            shape,
            dtype,
            size_bytes,
        });
    }

    arrays.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(NpzArchiveInfo {
        path,
        arrays,
        total_bytes,
        used_memmap: false,
    })
}

#[tauri::command]
pub fn load_tensor_slice(
    path: String,
    key: Option<String>,
    indices: Option<Vec<usize>>,
    max_dim: Option<usize>,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<TensorSlicePreview, String> {
    let start = Instant::now();
    let max_dim = max_dim.unwrap_or(64).clamp(8, 256);
    let indices = indices.unwrap_or_default();

    let p = Path::new(&path);
    if is_td_path(p) {
        let root = project_root.filter(|s| !s.is_empty()).ok_or_else(|| {
            "Project root required to load TensorDict (.td) slices".to_string()
        })?;
        let k = key.ok_or("TensorDict (.td) requires a key")?;
        return load_td_slice(&path, &k, &indices, max_dim, &root, python_executable);
    }

    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (resolved_key, full_shape, arr) = if ext == "npy" {
        let arr = load_array_from_path(p)?;
        let shape = arr.shape().to_vec();
        let k = key.unwrap_or_else(|| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("array")
                .to_string()
        });
        (k, shape, arr)
    } else {
        let k = key.ok_or("NPZ requires a key")?;
        let (shape, arr) = open_npz_array(&path, &k)?;
        (k, shape, arr)
    };

    let (plane_shape, flat) = select_2d_slice(&arr, &indices)?;
    let rows = plane_shape[0];
    let cols = plane_shape.get(1).copied().unwrap_or(1);
    let (values, slice_shape) = downsample_2d(&flat, rows, cols, max_dim);
    let (min, max) = min_max(&values);

    Ok(TensorSlicePreview {
        key: resolved_key,
        full_shape,
        slice_shape,
        indices,
        values,
        min,
        max,
        rust_ms: start.elapsed().as_millis() as u64,
    })
}

#[tauri::command]
pub fn tensor_slice_to_arrow_ipc(
    path: String,
    key: Option<String>,
    indices: Option<Vec<usize>>,
    max_dim: Option<usize>,
    project_root: Option<String>,
    python_executable: Option<String>,
) -> Result<ArrowIpcFile, String> {
    let start = Instant::now();
    let preview = load_tensor_slice(
        path,
        key,
        indices,
        max_dim,
        project_root,
        python_executable,
    )?;
    let rows = preview.slice_shape[0];
    let cols = preview.slice_shape.get(1).copied().unwrap_or(1);

    let n = rows * cols;
    let mut row_b = Int32Builder::with_capacity(n);
    let mut col_b = Int32Builder::with_capacity(n);
    let mut val_b = Float64Builder::with_capacity(n);

    for (r, row) in preview.values.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            row_b.append_value(r as i32);
            col_b.append_value(c as i32);
            val_b.append_value(v);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("row", DataType::Int32, false),
        Field::new("col", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(row_b.finish()),
            Arc::new(col_b.finish()),
            Arc::new(val_b.finish()),
        ],
    )
    .map_err(|e| e.to_string())?;

    export_batch(&batch, "tensor", start)
}

/// Load 1-D (or scalar) arrays from an NPZ archive — used for θ axes and BPC markers (§G.5.2).
#[tauri::command]
pub fn load_npz_vectors(path: String, keys: Vec<String>) -> Result<Vec<NpzVectorData>, String> {
    let mut out = Vec::new();
    for key in keys {
        let (shape, arr) = open_npz_array(&path, &key)?;
        let flat: Vec<f64> = arr.iter().copied().collect();
        if flat.is_empty() {
            return Err(format!("Key '{key}' is empty"));
        }
        if shape.len() > 1 {
            return Err(format!(
                "Key '{key}' has rank {} — expected 0-D or 1-D vector",
                shape.len()
            ));
        }
        out.push(NpzVectorData {
            key,
            values: flat,
        });
    }
    Ok(out)
}

/// Memory-map probe: reports whether a standalone `.npy` is large enough for mmap handoff.
#[tauri::command]
pub fn probe_npy_mmap(path: String) -> Result<bool, String> {
    let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
    if meta.len() < 128 {
        return Ok(false);
    }
    let mut file = File::open(&path).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    Ok(magic == *b"\x93NUMPY" && meta.len() > 8 * 1024 * 1024)
}
