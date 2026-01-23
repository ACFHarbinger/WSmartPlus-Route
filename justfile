# WSmart-Route Justfile

# Default variables (can be overridden: just train problem=wcvrp)
problem := "vrpp"
model := "am"
size := "50"
area := "riomaior"
epochs := "100"
days := "31"
seed := "42"
marker := "fast"

# --- Setup & Environment ---

# Sync dependencies using uv
sync:
    uv sync

# Install dependencies via pip
install:
    uv pip install -r requirements.txt || uv pip install -e .

# --- Primary Execution ---

# Run model training (using new Lightning pipeline)
train problem=problem model=model size=size epochs=epochs:
    uv run python main.py train_lightning model.name={{model}} env.name={{problem}} env.num_loc={{size}} train.n_epochs={{epochs}}

# Run model evaluation
eval:
    uv run python main.py eval

# Run simulator testing
test-sim policies="regular gurobi alns" days=days:
    uv run python main.py test_sim --policies {{policies}} --days {{days}}

# Generate virtual graph data
gen-data problem=problem size=size:
    uv run python main.py generate_data virtual --problem {{problem}} --graph_sizes {{size}}

# Launch the GUI
gui:
    uv run python main.py gui

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
    uv run pytest

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
    uv run pytest -m "{{marker}}"

# Check code quality with ruff
lint:
    uv run ruff check . --fix --exclude ".venv"

# Check docstring coverage
check-docs:
    uv run python logic/src/utils/check_docstrings.py logic/

# Format code with black and ruff
format:
    uv run ruff format . --exclude ".venv"

# --- Maintenance ---

# Clean caches and artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    rm -rf build/
    rm -rf dist/
    rm -rf temp/
    rm -rf wandb/
    rm -rf outputs/
    rm -rf checkpoints/
    rm -rf *.egg-info

# Generic run command
run *args:
    uv run python main.py {{args}}
