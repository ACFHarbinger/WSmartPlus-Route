# WSmart+ Route Framework - Justfile

set unstable := true

red := '\033[0;31m'
green := '\033[0;32m'
yellow := '\033[0;33m'
blue := '\033[0;34m'
purple := '\033[0;35m'
cyan := '\033[0;36m'
bold := '\033[1m'
reset := '\033[0m'

# Default variables (can be overridden: just train problem=wcvrp)

problem := "vrpp"
model := "am"
encoder := "gat"
decoder := "glimpse"
area := "figueiradafoz"
num_loc := "350"
sim_distribution := "gamma3"
batch_size := "64"
temporal_horizon := "0"
samples := "1"
seed := "42"
marker := "fast"
strategy := "greedy"
n_cores := "20"
hpo_policy := "bpc"
hpo_method := "nsgaii"
hpo_workers := "1"
hpo_selection := ""
hpo_acceptance := ""
hpo_improver := ""
hpo_trials := "10"
hpo_samples := "5"
hpo_policy_kw := "max_cg_iterations"
hpo_selection_kw := ""
hpo_acceptance_kw := ""
hpo_improver_kw := ""
policies := "aco_hh,alns,bpc,hgs,pg_clns,psoma,sans,swc_tcf"
results_dir := "assets/output/30_days/riomaior_100"
distribution := ""
constructor := ""
ms_strategy := ""
improver := ""
dry_run := "false"
quiet := "false"

# --- Submodules ---

mod benchmark 'tools/benchmark'
mod ci 'tools/ci'
mod controller 'tools/controller'
mod database 'tools/database'
mod export 'tools/export'
mod helper 'tools/helper'
mod infrastructure 'tools/infrastructure'
mod reducer 'tools/reducer'
mod script 'tools/script'
mod test 'tools/test'
mod ui 'tools/ui'
mod validation 'tools/validation'

# --- Help ---

# Print available commands with descriptions
help: helper::_print_header
    just helper::help

# --- Shorthands ---

# Initialize environment and install all dependencies
setup: helper::_print_header
    just infrastructure::setup

# Sync dependencies using uv
sync: helper::_print_header
    just infrastructure::sync

# Train a model with Hydra configs (problem=vrpp model=am)
train problem=problem model=model encoder=encoder decoder=decoder batch_size=batch_size temporal_horizon=temporal_horizon: helper::_print_header
    just problem='{{ problem }}' model='{{ model }}' encoder='{{ encoder }}' decoder='{{ decoder }}' batch_size='{{ batch_size }}' temporal_horizon='{{ temporal_horizon }}' controller::train

# Run model evaluation with Hydra configs
eval model_path="" dataset="" problem=problem strategy=strategy: helper::_print_header
    just model_path='{{ model_path }}' dataset='{{ dataset }}' problem='{{ problem }}' strategy='{{ strategy }}' controller::eval

# Run multi-day simulator test
test-sim policies=policies area=area samples=samples n_cores=n_cores num_loc=num_loc sim_distribution=sim_distribution: helper::_print_header
    just policies='{{ policies }}' area='{{ area }}' samples='{{ samples }}' n_cores='{{ n_cores }}' num_loc='{{ num_loc }}' sim_distribution='{{ sim_distribution }}' controller::test-sim

# Remove targeted simulation runs from output artefacts
clean-results results_dir=results_dir distribution=distribution constructor=constructor ms_strategy=ms_strategy improver=improver dry_run=dry_run quiet=quiet: helper::_print_header
    just results_dir='{{ results_dir }}' distribution='{{ distribution }}' constructor='{{ constructor }}' ms_strategy='{{ ms_strategy }}' improver='{{ improver }}' dry_run='{{ dry_run }}' quiet='{{ quiet }}' controller::clean-results

# Run simulation policy HPO
hpo-sim policy=hpo_policy trials=hpo_trials method=hpo_method workers=hpo_workers selection=hpo_selection acceptance=hpo_acceptance improver=hpo_improver policy_kw=hpo_policy_kw selection_kw=hpo_selection_kw acceptance_kw=hpo_acceptance_kw improver_kw=hpo_improver_kw area=area samples=hpo_samples: helper::_print_header
    just policy='{{ policy }}' trials='{{ trials }}' method='{{ method }}' workers='{{ workers }}' selection='{{ selection }}' acceptance='{{ acceptance }}' improver='{{ improver }}' policy_kw='{{ policy_kw }}' selection_kw='{{ selection_kw }}' acceptance_kw='{{ acceptance_kw }}' improver_kw='{{ improver_kw }}' area='{{ area }}' samples='{{ samples }}' controller::hpo-sim

# Generate a dataset
gen-data problem=problem: helper::_print_header
    just problem='{{ problem }}' controller::gen-data

# Generate Figueira da Foz plastic datasets
gen-data-figfoz-plastic: helper::_print_header
    just controller::gen-data-figfoz-plastic

# Launch the PySide6 GUI
gui: helper::_print_header
    just ui::gui

# Launch the Streamlit dashboard
dashboard: helper::_print_header
    just ui::dashboard

# Run fast unit tests (use `just test::test` for the full suite)
test-fast: helper::_print_header
    just test::test-fast

# Check code quality with ruff
lint: helper::_print_header
    just ci::lint

# Format code with ruff
format: helper::_print_header
    just ci::format

# Clean caches and build artifacts
clean: helper::_print_header
    just reducer::clean

# Remove simulation output files
clean-outputs: helper::_print_header
    just reducer::clean-outputs

# Remove large outputs (checkpoints and realtime logs)
clean-large-outputs: helper::_print_header
    just reducer::clean-large-outputs

# Build stripped-simulator executable (run after pruning)
package: helper::_print_header
    just infrastructure::package

# Run automated algorithm export (creates algo-export branch)
algo-export constructors="" selectors="" improvement="" acceptance="" joint="" models="" rl_algorithms="" imitation_policies="" drop_features="" envs="" sim_datasets="" distributions="" network="" skip_build="false" dry_run="false": helper::_print_header
    just constructors='{{ constructors }}' selectors='{{ selectors }}' improvement='{{ improvement }}' acceptance='{{ acceptance }}' joint='{{ joint }}' models='{{ models }}' rl_algorithms='{{ rl_algorithms }}' imitation_policies='{{ imitation_policies }}' drop_features='{{ drop_features }}' envs='{{ envs }}' sim_datasets='{{ sim_datasets }}' distributions='{{ distributions }}' network='{{ network }}' skip_build='{{ skip_build }}' dry_run='{{ dry_run }}' export::algo-export

# Generic run command — pass any main.py arguments directly
run *args: helper::_print_header
    uv run python main.py {{ args }}

# Commit using the .gitmessage template
commit message: helper::_print_header
    just message='{{ message }}' helper::commit
