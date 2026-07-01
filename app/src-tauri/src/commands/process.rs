/// Process management commands: spawn Python subprocesses, stream stdout, cancel.
///
/// Replaces the Streamlit file-tailer pattern with event-driven stdout streaming.
/// Each spawned process receives a unique `id`; stdout lines are emitted as
/// `process:stdout` events and status changes as `process:status` events.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProcessStatus {
    Running,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessEntry {
    pub id: String,
    pub command: String,
    pub pid: u32,
    pub status: ProcessStatus,
    pub start_time: u64,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StdoutLine {
    pub id: String,
    pub line: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StatusUpdate {
    pub id: String,
    pub status: ProcessStatus,
    pub exit_code: Option<i32>,
}

type Registry = Arc<Mutex<HashMap<String, (u32, tokio::sync::watch::Sender<bool>)>>>;

static PROCESS_REGISTRY: std::sync::OnceLock<Registry> = std::sync::OnceLock::new();

fn registry() -> &'static Registry {
    PROCESS_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

#[tauri::command]
pub async fn spawn_python_process(
    id: String,
    python_args: Vec<String>,
    working_dir: String,
    app: AppHandle,
) -> Result<u32, String> {
    let python = which_python();
    let mut cmd = Command::new(&python);
    cmd.args(&python_args)
        .current_dir(&working_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(false);

    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let pid = child.id().unwrap_or(0);

    let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
    {
        let mut reg = registry().lock().unwrap();
        reg.insert(id.clone(), (pid, cancel_tx));
    }

    let task_id = id.clone();
    let task_app = app.clone();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    tokio::spawn(async move {
        // Stream stdout
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = task_app.emit(
                    "process:stdout",
                    StdoutLine { id: task_id.clone(), line },
                );
            }
        }

        // Stream stderr to the same event for simplicity
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = task_app.emit(
                    "process:stdout",
                    StdoutLine {
                        id: task_id.clone(),
                        line: format!("[stderr] {line}"),
                    },
                );
            }
        }

        let exit_status = tokio::select! {
            status = child.wait() => status.ok(),
            _ = cancel_rx.changed() => {
                let _ = child.kill().await;
                None
            }
        };

        let (status, code) = match exit_status {
            Some(s) if s.success() => (ProcessStatus::Completed, Some(0)),
            Some(s) => (
                ProcessStatus::Failed,
                s.code(),
            ),
            None => (ProcessStatus::Cancelled, None),
        };

        {
            let mut reg = registry().lock().unwrap();
            reg.remove(&task_id);
        }

        let _ = task_app.emit(
            "process:status",
            StatusUpdate { id: task_id, status, exit_code: code },
        );
    });

    Ok(pid)
}

#[tauri::command]
pub fn cancel_process(id: String) -> Result<(), String> {
    let mut reg = registry().lock().unwrap();
    if let Some((_, tx)) = reg.remove(&id) {
        let _ = tx.send(true);
        Ok(())
    } else {
        Err(format!("No running process with id '{id}'"))
    }
}

#[tauri::command]
pub fn list_processes() -> Vec<String> {
    registry()
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn which_python() -> String {
    for candidate in &["python3", "python", ".venv/bin/python", "venv/bin/python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python3".to_string()
}
