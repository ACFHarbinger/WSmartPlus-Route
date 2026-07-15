/// TensorDict / NumPy archive inspection and slice streaming (§G.5.1).
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use memmap2::Mmap;
use arrow::array::{Float64Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zip::read::ZipFile;
use zip::{CompressionMethod, ZipArchive};

use crate::commands::arrow::{export_batch, ArrowIpcFile};
use crate::commands::process::resolve_python;

#[cfg(test)]
const MMAP_THRESHOLD_BYTES: u64 = 64;
#[cfg(not(test))]
const MMAP_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;

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
    preview.used_memmap = false;
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
    pub used_memmap: bool,
    #[serde(default)]
    pub used_decompress_slice: bool,
}

fn itemsize_from_descr(descr: &str) -> Result<usize, String> {
    let d = descr.trim_matches(|c| c == '\'' || c == '"');
    if d.ends_with("f8") || d.ends_with("i8") || d.ends_with("u8") || d.ends_with("l") {
        Ok(8)
    } else if d.ends_with("f4") || d.ends_with("i4") || d.ends_with("u4") || d.ends_with("f") {
        Ok(4)
    } else if d.ends_with("f2") || d.ends_with("i2") || d.ends_with("u2") {
        Ok(2)
    } else if d.ends_with("i1") || d.ends_with("u1") || d.ends_with("b1") {
        Ok(1)
    } else {
        Err(format!("Unsupported NPY dtype for mmap slice: {descr}"))
    }
}

fn c_order_offset(shape: &[usize], indices: &[usize]) -> Result<usize, String> {
    if shape.len() < 2 {
        return Ok(0);
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
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut offset = 0usize;
    for i in 0..leading {
        offset += indices[i] * strides[i];
    }
    Ok(offset)
}

fn read_f64_plane(bytes: &[u8], itemsize: usize) -> Result<Vec<f64>, String> {
    if bytes.len() % itemsize != 0 {
        return Err("Mmap plane byte length is not aligned to dtype width".to_string());
    }
    let count = bytes.len() / itemsize;
    let mut out = Vec::with_capacity(count);
    match itemsize {
        8 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        4 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
            }
        }
        2 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f16_to_f64(u16::from_le_bytes(chunk.try_into().unwrap())));
            }
        }
        1 => {
            for &b in bytes {
                out.push(b as f64);
            }
        }
        _ => return Err(format!("Unsupported itemsize {itemsize}")),
    }
    Ok(out)
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (frac as f64) / 1024.0 * 2.0_f64.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if frac == 0 {
            if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }
    let val = (1.0 + (frac as f64) / 1024.0) * 2.0_f64.powi(exp as i32 - 15);
    if sign == 1 { -val } else { val }
}

struct NpzNpyEntryLoc {
    data_start: u64,
    compression: CompressionMethod,
    uncompressed_size: u64,
}

fn resolve_npz_npy_key(archive: &mut ZipArchive<File>, key: &str) -> Result<String, String> {
    let npy_name = format!("{key}.npy");
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();
    if names.iter().any(|n| n == &npy_name) {
        Ok(npy_name)
    } else if names.iter().any(|n| n == key) {
        Ok(key.to_string())
    } else {
        Err(format!("Key '{key}' not found in NPZ"))
    }
}

fn resolve_npz_npy_entry(path: &Path, key: &str) -> Result<NpzNpyEntryLoc, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let resolved = resolve_npz_npy_key(&mut archive, key)?;
    let entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    Ok(npz_entry_loc(&entry))
}

fn npz_entry_loc(entry: &ZipFile<'_>) -> NpzNpyEntryLoc {
    NpzNpyEntryLoc {
        data_start: entry.data_start(),
        compression: entry.compression(),
        uncompressed_size: entry.size(),
    }
}

fn load_plane_from_npy_bytes(
    raw: &[u8],
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>), String> {
    let (shape, dtype, data_offset) = parse_npy_header(raw)?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > raw.len() {
        return Err("NPY plane slice exceeds buffer bounds".to_string());
    }
    let flat = read_f64_plane(&raw[byte_start..byte_end], itemsize)?;
    Ok((shape, vec![rows, cols], flat))
}

fn load_npz_plane_decompress(
    path: &Path,
    key: &str,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>, bool), String> {
    let loc = resolve_npz_npy_entry(path, key)?;
    if loc.uncompressed_size <= MMAP_THRESHOLD_BYTES {
        return Err("NPZ entry below decompress-slice threshold".to_string());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid NPZ: {e}"))?;
    let resolved = resolve_npz_npy_key(&mut archive, key)?;
    let mut entry = archive
        .by_name(&resolved)
        .map_err(|e| format!("Key '{key}' not found in NPZ: {e}"))?;
    let mut raw = Vec::new();
    entry.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let (shape, plane_shape, flat) = load_plane_from_npy_bytes(&raw, indices)?;
    Ok((shape, plane_shape, flat, true))
}

fn load_npz_plane_mmap(
    path: &Path,
    key: &str,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>, bool), String> {
    let loc = resolve_npz_npy_entry(path, key)?;
    if loc.compression != CompressionMethod::Stored {
        return Err("NPZ entry is compressed — mmap slice unavailable".to_string());
    }
    if loc.uncompressed_size <= MMAP_THRESHOLD_BYTES {
        return Err("NPZ entry below mmap threshold".to_string());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
    let npy_start = loc.data_start as usize;
    if npy_start >= mmap.len() {
        return Err("NPZ data offset out of bounds".to_string());
    }
    let (shape, dtype, data_offset) = parse_npy_header(&mmap[npy_start..])?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot mmap-slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = npy_start + data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > mmap.len() {
        return Err("Mmap NPZ slice exceeds file bounds".to_string());
    }
    let flat = read_f64_plane(&mmap[byte_start..byte_end], itemsize)?;
    Ok((shape, vec![rows, cols], flat, true))
}

fn load_npy_plane_mmap(
    path: &Path,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<f64>, bool), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
    let (shape, dtype, data_offset) = parse_npy_header(&mmap)?;
    let itemsize = itemsize_from_descr(&dtype)?;
    let rows = *shape.get(shape.len().saturating_sub(2)).unwrap_or(&0);
    let cols = shape.last().copied().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return Err("Cannot mmap-slice empty tensor plane".to_string());
    }
    let elem_offset = c_order_offset(&shape, indices)?;
    let plane_elems = rows * cols;
    let byte_start = data_offset + elem_offset * itemsize;
    let byte_end = byte_start + plane_elems * itemsize;
    if byte_end > mmap.len() {
        return Err("Mmap slice exceeds file bounds".to_string());
    }
    let flat = read_f64_plane(&mmap[byte_start..byte_end], itemsize)?;
    Ok((vec![rows, cols], flat, true))
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
    let rest = &header[start + "'descr'".len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('\'') {
        return None;
    }
    let value = &after[1..];
    let end = value.find('\'')?;
    Some(value[..end].to_string())
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
            used_memmap: file_meta.len() > MMAP_THRESHOLD_BYTES,
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
    let used_memmap = arrays.iter().any(|a| a.size_bytes > MMAP_THRESHOLD_BYTES);
    Ok(NpzArchiveInfo {
        path,
        arrays,
        total_bytes,
        used_memmap,
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

    let resolved_key = if ext == "npy" {
        key.unwrap_or_else(|| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("array")
                .to_string()
        })
    } else {
        key.ok_or("NPZ requires a key")?
    };

    let meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
    let use_npy_mmap = ext == "npy" && meta.len() > MMAP_THRESHOLD_BYTES;

    let (full_shape, plane_shape, flat, used_memmap, used_decompress_slice) = if use_npy_mmap {
        let (plane_shape, flat, mmap_used) = load_npy_plane_mmap(p, &indices)?;
        let mut header_buf = vec![0u8; 256];
        let mut file = File::open(p).map_err(|e| e.to_string())?;
        let n = file.read(&mut header_buf).map_err(|e| e.to_string())?;
        header_buf.truncate(n);
        let (shape, _, _) = parse_npy_header(&header_buf)?;
        (shape, plane_shape, flat, mmap_used, false)
    } else if ext == "npz" {
        match load_npz_plane_mmap(p, &resolved_key, &indices) {
            Ok((shape, plane_shape, flat, mmap_used)) => {
                (shape, plane_shape, flat, mmap_used, false)
            }
            Err(_) => match load_npz_plane_decompress(p, &resolved_key, &indices) {
                Ok((shape, plane_shape, flat, decomp_used)) => {
                    (shape, plane_shape, flat, false, decomp_used)
                }
                Err(_) => {
                    let (shape, arr) = open_npz_array(&path, &resolved_key)?;
                    let (plane_shape, flat) = select_2d_slice(&arr, &indices)?;
                    (shape, plane_shape, flat, false, false)
                }
            },
        }
    } else {
        let arr = load_array_from_path(p)?;
        let shape = arr.shape().to_vec();
        let (plane_shape, flat) = select_2d_slice(&arr, &indices)?;
        (shape, plane_shape, flat, false, false)
    };

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
        used_memmap,
        used_decompress_slice,
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

/// Load an NPZ array of any rank as a flattened vector — used by the native
/// Report Studio dataset statistics engine (§H.1 raw waste matrices).
#[tauri::command]
pub fn load_npz_flat(path: String, key: String) -> Result<Vec<f64>, String> {
    let (_, arr) = open_npz_array(&path, &key)?;
    Ok(arr.iter().copied().collect())
}

/// Memory-map probe: reports whether a `.npy` or stored `.npz` entry is large enough for mmap.
#[tauri::command]
pub fn probe_npy_mmap(path: String) -> Result<bool, String> {
    let p = Path::new(&path);
    let meta = std::fs::metadata(p).map_err(|e| e.to_string())?;
    if meta.len() < 128 {
        return Ok(false);
    }
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "npz" {
        let file = File::open(p).map_err(|e| e.to_string())?;
        let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
        for i in 0..archive.len() {
            let entry = archive.by_index(i).map_err(|e| e.to_string())?;
            if !entry.name().ends_with(".npy") {
                continue;
            }
            let loc = npz_entry_loc(&entry);
            if loc.uncompressed_size > MMAP_THRESHOLD_BYTES {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    let mut file = File::open(p).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    Ok(magic == *b"\x93NUMPY" && meta.len() > MMAP_THRESHOLD_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_npy(path: &Path, shape: &[usize], values: &[f64]) {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let header = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({shape_str},), }}"
        );
        let pad = (16 - (10 + header.len()) % 16) % 16;
        let header_padded = format!("{header}{}", " ".repeat(pad));
        let header_len = header_padded.len() as u16;
        let mut file = File::create(path).unwrap();
        file.write_all(b"\x93NUMPY\x01\x00").unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header_padded.as_bytes()).unwrap();
        for v in values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn mmap_plane_reads_trailing_2d_slice() {
        let dir = std::env::temp_dir().join(format!("wsroute_mmap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tensor.npy");
        // shape (2, 3, 4) — plane at index 1 is 3×4
        let mut vals = Vec::new();
        for b in 0..2 {
            for r in 0..3 {
                for c in 0..4 {
                    vals.push((b * 100 + r * 10 + c) as f64);
                }
            }
        }
        write_test_npy(&path, &[2, 3, 4], &vals);

        let (plane_shape, flat, used) = load_npy_plane_mmap(&path, &[1]).unwrap();
        assert!(used);
        assert_eq!(plane_shape, vec![3, 4]);
        assert_eq!(flat.len(), 12);
        assert!((flat[0] - 100.0).abs() < 1e-9);
        assert!((flat[11] - 123.0).abs() < 1e-9);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn npz_mmap_plane_reads_trailing_2d_slice() {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        use zip::ZipWriter;

        let dir = std::env::temp_dir().join(format!("wsroute_npz_mmap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let npy_path = dir.join("tensor.npy");
        let mut vals = Vec::new();
        for b in 0..2 {
            for r in 0..3 {
                for c in 0..4 {
                    vals.push((b * 100 + r * 10 + c) as f64);
                }
            }
        }
        write_test_npy(&npy_path, &[2, 3, 4], &vals);

        let npz_path = dir.join("tensor.npz");
        let npy_bytes = std::fs::read(&npy_path).unwrap();
        let file = File::create(&npz_path).unwrap();
        let mut zip = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        zip.start_file("tensor.npy", options).unwrap();
        zip.write_all(&npy_bytes).unwrap();
        zip.finish().unwrap();

        let (shape, plane_shape, flat, used) =
            load_npz_plane_mmap(&npz_path, "tensor", &[1]).unwrap();
        assert!(used);
        assert_eq!(shape, vec![2, 3, 4]);
        assert_eq!(plane_shape, vec![3, 4]);
        assert_eq!(flat.len(), 12);
        assert!((flat[0] - 100.0).abs() < 1e-9);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn npz_decompress_plane_reads_trailing_2d_slice() {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        use zip::ZipWriter;

        let dir =
            std::env::temp_dir().join(format!("wsroute_npz_deflate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let npy_path = dir.join("tensor.npy");
        let mut vals = Vec::new();
        for b in 0..2 {
            for r in 0..3 {
                for c in 0..4 {
                    vals.push((b * 100 + r * 10 + c) as f64);
                }
            }
        }
        write_test_npy(&npy_path, &[2, 3, 4], &vals);

        let npz_path = dir.join("tensor_deflate.npz");
        let npy_bytes = std::fs::read(&npy_path).unwrap();
        let file = File::create(&npz_path).unwrap();
        let mut zip = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        zip.start_file("tensor.npy", options).unwrap();
        zip.write_all(&npy_bytes).unwrap();
        zip.finish().unwrap();

        let (shape, plane_shape, flat, used) =
            load_npz_plane_decompress(&npz_path, "tensor", &[1]).unwrap();
        assert!(used);
        assert_eq!(shape, vec![2, 3, 4]);
        assert_eq!(plane_shape, vec![3, 4]);
        assert_eq!(flat.len(), 12);
        assert!((flat[11] - 123.0).abs() < 1e-9);

        let _ = std::fs::remove_dir_all(dir);
    }
}
