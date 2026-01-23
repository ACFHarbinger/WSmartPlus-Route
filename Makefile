# WSmart+ Route Framework - Makefile
# System for solving complex Combinatorial Optimization (CO) problems

SHELL := /bin/bash

# Colors for terminal output
BLUE         := \033[0;34m
CYAN         := \033[0;36m
GREEN        := \033[0;32m
YELLOW       := \033[1;33m
RED          := \033[0;31m
MAGENTA      := \033[0;35m
NC           := \033[0m # No Color

# Configuration
PYTHON       := uv run python
PIP          := uv pip
UV           := uv
MAIN         := main.py

# Default Parameters (can be overridden: make train PROBLEM=wcvrp)
PROBLEM      ?= vrpp
MODEL        ?= am
SIZE         ?= 50
AREA         ?= riomaior
WTYPE        ?= plastic
EPOCHS       ?= 100
DAYS         ?= 31
SEED         ?= 42

# --- Macros ---
define print_header
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                  WSmart+ Route Framework Control                     ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
endef

.PHONY: help setup sync train eval test-sim gen-data gui lint format clean test-logic test-gui test-fast test-marker

help:
	$(call print_header)
	@echo -e ""
	@echo -e "$(CYAN)Usage:$(NC) make <target> [VARIABLE=value]"
	@echo -e ""
	@echo -e "$(YELLOW)Setup & Environment:$(NC)"
	@echo -e "  $(GREEN)setup$(NC)         - Initialize environment and install dependencies using uv"
	@echo -e "  $(GREEN)sync$(NC)          - Sync dependencies with uv.lock"
	@echo -e "  $(GREEN)install$(NC)       - Install dependencies using uv pip"
	@echo -e ""
	@echo -e "$(YELLOW)Task Execution (main.py wrappers):$(NC)"
	@echo -e "  $(GREEN)train$(NC)         - Run model training (Default: PROBLEM=$(PROBLEM), MODEL=$(MODEL), SIZE=$(SIZE))"
	@echo -e "  $(GREEN)eval$(NC)          - Run model evaluation"
	@echo -e "  $(GREEN)test-sim$(NC)      - Run simulator testing"
	@echo -e "  $(GREEN)gen-data$(NC)      - Generate virtual graph data"
	@echo -e "  $(GREEN)gui$(NC)           - Launch the PySide6 Graphical User Interface"
	@echo -e ""
	@echo -e "$(YELLOW)Script Wrappers (scripts/*.sh):$(NC)"
	@echo -e "  $(GREEN)run-train$(NC)     - Execute scripts/train.sh"
	@echo -e "  $(GREEN)run-eval$(NC)      - Execute scripts/evaluation.sh"
	@echo -e "  $(GREEN)run-gen$(NC)       - Execute scripts/gen_data.sh (Alias: gen_data)"
	@echo -e "  $(GREEN)run-meta$(NC)      - Execute scripts/meta_train.sh"
	@echo -e "  $(GREEN)run-hpo$(NC)       - Execute scripts/hyperparam_optim.sh"
	@echo -e ""
	@echo -e "$(YELLOW)Testing & Quality:$(NC)"
	@echo -e "  $(GREEN)test$(NC)          - Run all tests using pytest"
	@echo -e "  $(GREEN)test-fast$(NC)     - Run only fast unit tests"
	@echo -e "  $(GREEN)test-logic$(NC)    - Run backend logic tests"
	@echo -e "  $(GREEN)test-gui$(NC)      - Run GUI-specific tests"
	@echo -e "  $(GREEN)test-marker$(NC)   - Run tests with specific marker (e.g., make test-marker MARKER=slow)"
	@echo -e "  $(GREEN)lint$(NC)          - Check code quality with ruff"
	@echo -e "  $(GREEN)format$(NC)        - Format code with black and ruff"
	@echo -e ""
	@echo -e "$(YELLOW)Maintenance:$(NC)"
	@echo -e "  $(GREEN)clean$(NC)         - Remove temporary files and caches"
	@echo -e ""
	@echo -e "$(CYAN)Variables:$(NC)"
	@echo -e "  PROBLEM=$(PROBLEM), MODEL=$(MODEL), SIZE=$(SIZE), AREA=$(AREA), EPOCHS=$(EPOCHS)"

# --- Setup ---

setup:
	$(call print_header)
	@echo -e "$(BLUE)🔧 Setting up environment with uv...$(NC)"
	$(UV) sync
	@echo -e "$(GREEN)✓ Environment ready.$(NC)"

sync:
	$(call print_header)
	@echo -e "$(BLUE)🔄 Syncing dependencies...$(NC)"
	$(UV) sync

install:
	$(call print_header)
	@echo -e "$(BLUE)📦 Installing dependencies with pip...$(NC)"
	$(PIP) install -r requirements.txt || $(UV) pip install -e .

# --- Primary Execution ---

train:
	$(call print_header)
	@echo -e "$(BLUE)🚀 Starting Training [$(PROBLEM)/$(MODEL)/$(SIZE)]...$(NC)"
	$(PYTHON) $(MAIN) train --model $(MODEL) --problem $(PROBLEM) --graph_size $(SIZE) --n_epochs $(EPOCHS)

eval:
	$(call print_header)
	@echo -e "$(BLUE)📊 Starting Evaluation...$(NC)"
	$(PYTHON) $(MAIN) eval

test-sim:
	$(call print_header)
	@echo -e "$(BLUE)🎮 Starting Simulation Test...$(NC)"
	$(PYTHON) $(MAIN) test_sim --policies regular gurobi alns --days $(DAYS)

gen-data:
	$(call print_header)
	@echo -e "$(BLUE)📂 Generating Virtual Data...$(NC)"
	$(PYTHON) $(MAIN) generate_data virtual --problem $(PROBLEM) --graph_sizes $(SIZE)

gui:
	$(call print_header)
	@echo -e "$(BLUE)🖥️  Launching WSmart+ GUI...$(NC)"
	$(PYTHON) $(MAIN) gui

# --- Script Runners ---

run-train:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running training script...$(NC)"
	bash scripts/train.sh

run-eval:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running evaluation script...$(NC)"
	bash scripts/evaluation.sh

run-gen:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running data generation script...$(NC)"
	bash scripts/gen_data.sh

gen_data: run-gen

run-meta:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running meta-training script...$(NC)"
	bash scripts/meta_train.sh

run-hpo:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running HPO script...$(NC)"
	bash scripts/hyperparam_optim.sh


run-sim:
	$(call print_header)
	@echo -e "$(BLUE)📜 Running Simulation script...$(NC)"
	bash scripts/test_sim.sh

# --- Test & Quality ---

test:
	$(call print_header)
	@echo -e "$(BLUE)🧪 Running all tests...$(NC)"
	$(UV) run pytest

test-fast:
	$(call print_header)
	@echo -e "$(BLUE)🧪 Running fast unit tests...$(NC)"
	$(UV) run pytest -m "fast or unit"

test-logic:
	$(call print_header)
	@echo -e "$(BLUE)🧪 Running logic tests...$(NC)"
	$(UV) run pytest logic/test/

test-gui:
	$(call print_header)
	@echo -e "$(BLUE)🧪 Running GUI tests...$(NC)"
	$(UV) run pytest gui/test/

test-marker:
	$(call print_header)
	@echo -e "$(BLUE)🧪 Running tests with marker [$(MARKER)]...$(NC)"
	$(UV) run pytest -m "$(MARKER)"

lint:
	$(call print_header)
	@echo -e "$(BLUE)🔍 Linting with ruff...$(NC)"
	$(UV) run ruff check . --fix --exclude ".venv"

format:
	$(call print_header)
	@echo -e "$(BLUE)✨ Formatting with black & ruff...$(NC)"
	$(UV) run ruff format . --exclude ".venv"

# --- Maintenance ---

clean:
	$(call print_header)
	@echo -e "$(BLUE)🧹 Cleaning caches and artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	@echo -e "$(GREEN)✓ Cleanup complete.$(NC)"
