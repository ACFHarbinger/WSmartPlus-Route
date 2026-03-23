# WSmart-Route Justfile

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
size := "100"
area := "riomaior"
epochs := "31"
batch_size := "64"
temporal_horizon := "0"
days := "30"
samples := "1"
seed := "42"
marker := "fast"
strategy := "greedy"
distribution := "gamma3"
n_cores := "20"
policies := "abc,aco_hh,aco_ks,ahvpl,aks,alns,bb,bc,bp,bpc,cf_rs,cvrp,de,ema,es_mcl,es_mkl,es_mpl,fa,filo,ga,gd,genius,gihh,gls,gphh,hgs_alns,hgs_rr,hgs,hmm_gd_hh,hms,hs,hulk,hvpl,ie,ils_rvnd_sp,ils,kgls,ks,lahc,lb_vns,lb,lca,ma_dp,ma_im,ma_ts,ma,oba,oi,popmusic,pso,psoda,psoma,qde,rens,rl_ahvpl,rl_alns,rl_gd_hh,rl_hvpl,rrt,rts,sa,sans,sca,schc,sisr,slc,ss_hh,swc_tcf,ta,ts,tsp,vns,vpl"

# --- Setup & Environment ---

# Sync dependencies using uv
sync:
    uv sync --all-groups --all-extras

# Install dependencies via pip
install:
    uv pip install -r requirements.txt || uv pip install -e .

# --- Primary Execution Commands (Hydra-based) ---
# Train a model with Hydra configs

# Usage: just train problem=wcvrp model=am size=50 epochs=100
train problem=problem model=model size=size epochs=epochs encoder=encoder decoder=decoder batch_size=batch_size temporal_horizon=temporal_horizon area=area distribution=distribution train_data_size="12800" val_data_size="128":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🚀 STARTING HYDRA TRAINING SESSION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model:" "{{ model }} ({{ encoder }})"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Graph Size:" "{{ size }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Area:" "{{ area }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Epochs:" "{{ epochs }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Batch Size:" "{{ batch_size }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" && \
    uv run python main.py train \
        envs={{ problem }} \
        models={{ model }} \
        model.encoder.type={{ encoder }} \
        env.num_loc={{ size }} \
        env.graph.num_loc={{ size }} \
        env.graph.area={{ area }} \
        train.data_distribution={{ distribution }} \
        train.train_data_size={{ train_data_size }} \
        train.val_data_size={{ val_data_size }} \
        hpo.n_trials=0 \
        train.n_epochs={{ epochs }} \
        train.batch_size={{ batch_size }}

# Run model evaluation with Hydra configs

# Usage: just eval model_path=./weights/best.pt dataset=data/test.pkl problem=wcvrp size=50 strategy=greedy
eval model_path="" dataset="" problem=problem size=size strategy=strategy:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "📊 STARTING MODEL EVALUATION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model Path:" "{{ model_path }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Dataset:" "{{ dataset }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Graph Size:" "{{ size }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Strategy:" "{{ strategy }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py eval \
        eval.policy.model.load_path={{ model_path }} \
        eval.datasets=[{{ dataset }}] \
        eval.problem={{ problem }} \
        eval.graph.num_loc={{ size }} \
        eval.val_size={{ samples }} \
        eval.decoding.strategy={{ strategy }}

# Run simulator testing with Hydra configs

# Usage: just test-sim policies="vrpp,alns" days=31 area=riomaior
test-sim policies=policies days=days area=area size=size samples=samples problem=problem n_cores=n_cores data_distribution=distribution:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🧪 STARTING SIMULATION TESTING"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Policies:" "{{ policies }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Days:" "{{ days }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Size:" "{{ size }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Area:" "{{ area }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Samples:" "{{ samples }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Distribution:" "{{ distribution }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py test_sim \
        sim.policies=[{{ policies }}] \
        sim.days={{ days }} \
        sim.n_samples={{ samples }} \
        sim.data_distribution={{ distribution }} \
        sim.graph.area={{ area }} \
        sim.graph.num_loc={{ size }} \
        sim.problem={{ problem }} \
        sim.cpu_cores={{ n_cores }}

# Generate data with Hydra configs

# Usage: just gen-data wcvrp 100 emp test_simulator 10 31
gen-data problem=problem distribution=distribution data_type="test_simulator":
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "📁 GENERATING DATASET"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Problem:" "{{ problem }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Distribution:" "{{ distribution }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Dataset Type:" "{{ data_type }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py gen_data \
        data.dataset_type={{ data_type }} \
        data.problem={{ problem }} \
        data.data_distributions=[{{ distribution }}]

# Launch the GUI
gui:
    uv run python main.py gui

# Launch the dashboard
dashboard:
    uv run streamlit run dashboard.py

# Count lines of code and comments
count-loc:
    uv run python logic/src/utils/validation/count_loc.py logic/src

# Tree view of lines of code and comments
tree-loc:
    uv run python logic/src/utils/validation/tree_loc.py logic/src

# Check docstring coverage
check-docs:
    uv run python logic/src/utils/docs/check_docstrings.py logic/src
    uv run python logic/src/utils/docs/check_docstrings.py gui/src

# Check Google style docstrings
check-google-docs:
    uv run python logic/src/utils/docs/check_google_style.py logic/src

# Check for multiple classes in one file
check-multi-classes:
    uv run python logic/src/utils/validation/check_multi_classes.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test

# Check for nested imports
check-nested-imports:
    uv run python logic/src/utils/validation/check_nested_imports.py logic/src --exclude pipeline/simulations/wsmart_bin_analysis/test --ignore_factories

# --- Advanced Testing & Benchmarks ---

# Run mutation tests using mutmut
mutation-test:
    export PYTHONPATH=. && uv run mutmut run --paths-to-mutate logic/src/

# Show mutation test results
mutation-report:
    uv run mutmut show all

# Run performance benchmarks
benchmark:
    export PYTHONPATH=. && uv run python -m logic.benchmark.run_all

# --- Script Runners ---

# Run training script (scripts/train.sh)
run-train:
    bash scripts/train.sh

# Run evaluation script (scripts/evaluation.sh)
run-eval:
    bash scripts/evaluation.sh

# Run data generation script (scripts/gen_data.sh)
run-gen-data:
    bash scripts/gen_data.sh

# Run meta-training script (scripts/meta_train.sh)
run-meta:
    bash scripts/meta_train.sh

# Run HPO script (scripts/hyperparam_optim.sh)
run-hpo:
    bash scripts/hyperparam_optim.sh

# Run simulation script (scripts/test_sim.sh)
run-sim:
    bash scripts/test_sim.sh

# --- Test & Quality ---

# Run all tests
test:
    uv run pytest --cov=logic/src --cov=gui/src --cov-report=xml --cov-report=term-missing

# Run fast unit tests
test-fast:
    uv run pytest -m "fast or unit"

# Run logic tests
test-logic:
    uv run pytest logic/test/

# Run GUI tests
test-gui:
    uv run pytest gui/test/

# Run tests with a specific marker
test-marker marker=marker:
    uv run pytest -m "{{ marker }}"

# Check code quality with ruff
lint:
    uv run ruff check . --fix --exclude ".venv"

build-docs:
    cd logic/docs && uv run make html

# Format code with black and ruff
format:
    uv run ruff format . --exclude ".venv"

# Build the dashboard docker image
docker-build:
    docker build -t wsmart-dashboard .

# Run the dashboard in a container with live log mounting
docker-dashboard:
    docker run -p 8501:8501 \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/assets/output:/app/assets/output \
      wsmart-dashboard

# --- Maintenance ---

# Clean caches and artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".hypothesis" -exec rm -rf {} +
    find . -type f -name "coverage.xml" -exec rm {} +
    find . -type f -name ".coverage" -exec rm {} +
    rm -rf build/
    rm -rf dist/
    rm -rf temp/
    rm -rf wandb/
    rm -rf mlruns/
    rm -rf outputs/
    rm -rf checkpoints/
    rm -rf *.egg-info
    rm -rf logs/
    rm -rf model_weights/

# --- Tracking Database ---

# Print an overview of experiments, runs, and record counts
db-inspect:
    uv run python -m logic.src.tracking.database inspect

# Delete all tracking data while preserving the schema
db-clean:
    uv run python -m logic.src.tracking.database clean

# Integrity check, WAL checkpoint, and VACUUM
db-compact:
    uv run python -m logic.src.tracking.database compact

# Remove stale runs  (override: just db-prune older_than=7 status=failed experiment=AM-VRPP-50)
db-prune older_than="30" status="failed" experiment="" dry_run="":
    uv run python -m logic.src.tracking.database prune \
        --older-than {{ older_than }} \
        --status {{ status }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }} \
        {{ if dry_run != "" { "--dry-run" } else { "" } }}

# Export a run to JSON  (override: just db-export run_id=a1b2c3d4 output=run.json)
db-export run_id="" experiment="" output="" latest="":
    uv run python -m logic.src.tracking.database export \
        {{ if run_id != "" { "--run-id " + run_id } else { "" } }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }} \
        {{ if output != "" { "--output " + output } else { "" } }} \
        {{ if latest != "" { "--latest" } else { "" } }}

# Show comprehensive database statistics  (override: just db-stats experiment=AM-VRPP-50)
db-stats experiment="":
    uv run python -m logic.src.tracking.database stats \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }}

# Show per-metric statistics  (override: just db-metrics key=train/cost experiment=AM-VRPP-50)
db-metrics key="" experiment="":
    uv run python -m logic.src.tracking.database metrics \
        {{ if key != "" { "--key " + key } else { "" } }} \
        {{ if experiment != "" { "--experiment " + experiment } else { "" } }}

# Generic run command
run *args:
    uv run python main.py {{ args }}
