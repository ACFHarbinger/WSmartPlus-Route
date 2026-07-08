/// System inspection commands: validate paths, probe Python version, Hydra config dump.
use std::process::Command;

use super::process::resolve_python;

/// Check that `path` is a directory containing `main.py`.
/// Returns an Ok confirmation or a descriptive error.
#[tauri::command]
pub fn validate_project_root(path: String) -> Result<String, String> {
    let dir = std::path::Path::new(&path);
    if !dir.exists() {
        return Err(format!("Directory does not exist: {path}"));
    }
    if !dir.is_dir() {
        return Err(format!("Path is not a directory: {path}"));
    }
    if !dir.join("main.py").exists() {
        return Err(format!("main.py not found in {path}"));
    }
    Ok(format!("main.py found in {path}"))
}

/// Run `<python_path> --version` synchronously and return the version string.
/// Works on both Python 2 (stderr) and Python 3 (stdout).
#[tauri::command]
pub fn probe_python(python_path: String) -> Result<String, String> {
    let output = Command::new(&python_path)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to run '{python_path}': {e}"))?;

    // Python 3 prints to stdout; Python 2 prints to stderr.
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let version = if !stdout.is_empty() { stdout } else { stderr };

    if version.is_empty() {
        return Err(format!("'{python_path} --version' produced no output"));
    }
    Ok(version)
}

/// Run `python main.py <task> --cfg job` and return the resolved Hydra config as YAML.
#[tauri::command]
pub fn dump_hydra_config(
    task: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<String, String> {
    let python = resolve_python(&project_root, python_executable);
    let output = Command::new(&python)
        .args(["main.py", &task, "--cfg", "job"])
        .current_dir(&project_root)
        .output()
        .map_err(|e| format!("Failed to run Hydra config dump: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let msg = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("Hydra --cfg job failed: {msg}"));
    }

    if stdout.trim().is_empty() {
        return Err("Hydra --cfg job produced no output".to_string());
    }

    Ok(stdout)
}
