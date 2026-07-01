mod commands;

use commands::{data, process, sim_watcher, system};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            // Data loading
            data::load_simulation_log,
            data::load_csv_file,
            data::list_output_dirs,
            data::list_training_runs,
            data::load_training_metrics,
            data::read_text_file,
            data::list_dir,
            // Simulation file watcher
            sim_watcher::start_sim_watcher,
            sim_watcher::stop_sim_watcher,
            // Process management
            process::spawn_python_process,
            process::cancel_process,
            process::list_processes,
            // System inspection
            system::validate_project_root,
            system::probe_python,
        ])
        .run(tauri::generate_context!())
        .expect("error while running WSmart-Route Studio");
}
