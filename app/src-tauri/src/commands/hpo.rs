/// Optuna HPO commands — list studies and load trial data via Python subprocess.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;

use super::process::resolve_python;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OptunaStudySummary {
    pub name: String,
    pub n_trials: i32,
    pub n_complete: i32,
    pub best_value: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OptunaTrial {
    pub number: i32,
    pub value: Option<f64>,
    pub state: String,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OptunaStudyData {
    pub name: String,
    pub trials: Vec<OptunaTrial>,
    pub importances: HashMap<String, f64>,
    pub best_value: Option<f64>,
    pub best_params: HashMap<String, serde_json::Value>,
}

const LIST_STUDIES_SCRIPT: &str = r#"
import json, sys
try:
    import optuna
except ImportError:
    print(json.dumps({"error": "optuna not installed"}))
    sys.exit(0)
storage = sys.argv[1]
optuna.logging.set_verbosity(optuna.logging.WARNING)
summaries = optuna.study.get_all_study_summaries(storage=storage)
out = []
for s in summaries:
    # Re-load each study briefly for complete count and best value
    try:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        best = study.best_value if done else None
        out.append({
            "name": s.study_name,
            "n_trials": len(study.trials),
            "n_complete": len(done),
            "best_value": best,
        })
    except Exception:
        out.append({
            "name": s.study_name,
            "n_trials": s.n_trials,
            "n_complete": 0,
            "best_value": None,
        })
print(json.dumps(out))
"#;

const LOAD_STUDY_SCRIPT: &str = r#"
import json, sys
try:
    import optuna
    from optuna.importance import FanovaImportanceEvaluator
except ImportError:
    print(json.dumps({"error": "optuna not installed"}))
    sys.exit(0)
storage, name = sys.argv[1], sys.argv[2]
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name=name, storage=storage)
trials = []
for t in study.trials:
    trials.append({
        "number": t.number,
        "value": t.value,
        "state": t.state.name,
        "params": {k: v for k, v in t.params.items()},
    })
completed = [t for t in study.trials if t.state.name == "COMPLETE"]
importances = {}
if len(completed) >= 2:
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator()
        )
    except Exception:
        pass
best_params = {}
if completed:
    best_params = {k: v for k, v in study.best_trial.params.items()}
print(json.dumps({
    "name": name,
    "trials": trials,
    "importances": importances,
    "best_value": study.best_value if completed else None,
    "best_params": best_params,
}))
"#;

fn run_python_json(
    python: &str,
    working_dir: &str,
    script: &str,
    args: &[&str],
) -> Result<serde_json::Value, String> {
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .args(args)
        .current_dir(working_dir)
        .output()
        .map_err(|e| format!("Failed to run Python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !output.status.success() && stdout.is_empty() {
        return Err(if stderr.is_empty() {
            format!("Python exited with code {:?}", output.status.code())
        } else {
            stderr
        });
    }

    let value: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("Invalid JSON from Python: {e}\n{stdout}"))?;

    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(err.to_string());
    }

    Ok(value)
}

/// List all Optuna studies in a storage backend.
#[tauri::command]
pub fn list_optuna_studies(
    storage_url: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<Vec<OptunaStudySummary>, String> {
    let python = resolve_python(&project_root, python_executable);
    let value = run_python_json(&python, &project_root, LIST_STUDIES_SCRIPT, &[&storage_url])?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}

/// Load all trials and parameter importances for a single Optuna study.
#[tauri::command]
pub fn load_optuna_study(
    storage_url: String,
    study_name: String,
    project_root: String,
    python_executable: Option<String>,
) -> Result<OptunaStudyData, String> {
    let python = resolve_python(&project_root, python_executable);
    let value = run_python_json(
        &python,
        &project_root,
        LOAD_STUDY_SCRIPT,
        &[&storage_url, &study_name],
    )?;
    serde_json::from_value(value).map_err(|e| e.to_string())
}
