mod commands;

use commands::{data, hpo, mlflow, policies, process, sim_watcher, system, zenml};

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
            data::write_text_file,
            data::list_dir,
            data::preview_dataset_stats,
            // Simulation file watcher
            sim_watcher::start_sim_watcher,
            sim_watcher::stop_sim_watcher,
            // Process management
            process::spawn_python_process,
            process::cancel_process,
            process::list_processes,
            // Policy registry
            policies::list_sim_policies,
            // Optuna HPO
            hpo::list_optuna_studies,
            hpo::load_optuna_study,
            // MLflow experiment tracking
            mlflow::list_mlflow_runs,
            mlflow::list_mlflow_metric_keys,
            mlflow::load_mlflow_metric_history,
            // ZenML pipeline tracking
            zenml::list_zenml_pipeline_runs,
            zenml::load_zenml_run_steps,
            // System inspection
            system::validate_project_root,
            system::probe_python,
            system::dump_hydra_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running WSmart-Route Studio");
}
