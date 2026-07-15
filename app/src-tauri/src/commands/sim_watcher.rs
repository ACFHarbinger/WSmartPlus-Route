/// Real-time simulation log watcher.
///
/// Polls the JSONL simulation output file for new `GUI_DAY_LOG_START:` lines
/// and emits a `sim:day_update` Tauri event on each new entry. This replaces
/// the Streamlit auto-refresh polling loop with an event-driven design that
/// delivers new simulation day data in < 200 ms without re-running any Python.
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

/// Mirrors the Python `DayLogEntry` dataclass from `logic/src/ui/services/log_parser.py`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DayLogEntry {
    pub policy: String,
    pub sample_id: i32,
    pub day: i32,
    pub data: serde_json::Value,
}

/// Policy iteration telemetry from ``PolicyVizMixin`` (§A.3).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyVizEntry {
    pub policy: String,
    pub sample_id: i32,
    pub day: i32,
    pub policy_type: String,
    pub data: serde_json::Value,
}

pub fn parse_day_log_line(line: &str) -> Option<DayLogEntry> {
    let line = line.trim();
    if !line.starts_with("GUI_DAY_LOG_START:") {
        return None;
    }
    let content = &line["GUI_DAY_LOG_START:".len()..];
    // Split into exactly 4 parts: policy, sample_id, day, json_payload
    let parts: Vec<&str> = content.splitn(4, ',').collect();
    if parts.len() < 4 {
        return None;
    }
    let policy = parts[0].trim().to_string();
    let sample_id: i32 = parts[1].trim().parse().ok()?;
    let day: i32 = parts[2].trim().parse().ok()?;
    let data: serde_json::Value = serde_json::from_str(parts[3].trim()).ok()?;
    Some(DayLogEntry { policy, sample_id, day, data })
}

pub fn parse_policy_viz_line(line: &str) -> Option<PolicyVizEntry> {
    let line = line.trim();
    if !line.starts_with("POLICY_VIZ_START:") {
        return None;
    }
    let content = &line["POLICY_VIZ_START:".len()..];
    // policy, sample_id, day, policy_type, json_payload
    let parts: Vec<&str> = content.splitn(5, ',').collect();
    if parts.len() < 5 {
        return None;
    }
    let policy = parts[0].trim().to_string();
    let sample_id: i32 = parts[1].trim().parse().ok()?;
    let day: i32 = parts[2].trim().parse().ok()?;
    let policy_type = parts[3].trim().to_string();
    let data: serde_json::Value = serde_json::from_str(parts[4].trim()).ok()?;
    Some(PolicyVizEntry {
        policy,
        sample_id,
        day,
        policy_type,
        data,
    })
}

/// Global watcher cancellation flag (path → stop signal).
static WATCHER_STOP: std::sync::OnceLock<Arc<Mutex<Option<String>>>> = std::sync::OnceLock::new();

#[tauri::command]
pub async fn start_sim_watcher(path: String, app: AppHandle) -> Result<(), String> {
    let stop_cell = WATCHER_STOP.get_or_init(|| Arc::new(Mutex::new(None)));
    {
        let mut guard = stop_cell.lock().unwrap();
        *guard = None; // clear any previous stop request
    }
    let stop = Arc::clone(stop_cell);
    let watch_path = path.clone();

    tokio::spawn(async move {
        let mut lines_seen: usize = 0;

        loop {
            // Check for stop signal
            if stop.lock().map(|g| g.is_some()).unwrap_or(false) {
                break;
            }

            if let Ok(content) = tokio::fs::read_to_string(&watch_path).await {
                let lines: Vec<&str> = content.lines().collect();
                if lines.len() > lines_seen {
                    for line in &lines[lines_seen..] {
                        if let Some(entry) = parse_day_log_line(line) {
                            let _ = app.emit("sim:day_update", &entry);
                        }
                        if let Some(viz) = parse_policy_viz_line(line) {
                            let _ = app.emit("sim:policy_viz_update", &viz);
                        }
                    }
                    lines_seen = lines.len();
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }
    });

    Ok(())
}

#[tauri::command]
pub fn stop_sim_watcher() -> Result<(), String> {
    if let Some(stop) = WATCHER_STOP.get() {
        let mut guard = stop.lock().unwrap();
        *guard = Some("stop".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_day_log_line, parse_policy_viz_line};

    #[test]
    fn parse_policy_viz_line_valid() {
        let line = r#"POLICY_VIZ_START:ALNS + Ftsp,0,2,alns,{"iteration":[0,1],"best_cost":[10.0,9.0]}"#;
        let entry = parse_policy_viz_line(line).expect("parse");
        assert_eq!(entry.policy, "ALNS + Ftsp");
        assert_eq!(entry.sample_id, 0);
        assert_eq!(entry.day, 2);
        assert_eq!(entry.policy_type, "alns");
        assert_eq!(entry.data["best_cost"][0], 10.0);
    }

    #[test]
    fn parse_policy_viz_line_ignores_day_log() {
        let line = r#"GUI_DAY_LOG_START:greedy,0,1,{"profit":10}"#;
        assert!(parse_policy_viz_line(line).is_none());
        assert!(parse_day_log_line(line).is_some());
    }
}
