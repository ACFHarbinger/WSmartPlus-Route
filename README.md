<div align="center">

<img src="https://raw.githubusercontent.com/acfharbinger/WSmartPlus-Route/main/assets/images/logo-wsmartroute-white.png" alt="WSMart+ Route Logo" style="width: 35%; height: auto;">

# WSmart+ Route

**A High-Performance Framework for Combinatorial Optimization in Waste Collection Vehicle Routing.**

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.gurobi.com/"><img alt="Gurobi" src="https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white"></a>
<a href="https://www.hexaly.com/"><img alt="Hexaly" src="https://img.shields.io/badge/Hexaly-Optimizer-0078D7"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</br>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white"></a>
<a href="https://doc.qt.io/qtforpython-6/"><img alt="PySide6" src="https://img.shields.io/badge/PySide6-Qt-41CD52?logo=qt&logoColor=white"></a>
<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/managed%20by-uv-261230.svg"></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-RTX_4080-76B900?logo=nvidia&logoColor=white"></a>
<a href="https://mypy-lang.org/"><img alt="MyPy" src="https://img.shields.io/badge/MyPy-checked-2f4f4f.svg"></a>
<a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white"></a>

<p>
  <a href="#-documentation-hub"><strong>📚 Documentation</strong></a> |
  <a href="#-overview"><strong>Overview</strong></a> |
  <a href="#-key-features"><strong>Features</strong></a> |
  <a href="#-quickstart"><strong>Quickstart</strong></a> |
  <a href="#-model-ecosystem"><strong>Models</strong></a> |
  <a href="#-setup-dependencies"><strong>Setup</strong></a> |
  <a href="#-program-usage"><strong>Usage</strong></a>
</p>

</div>

---

## 📚 Documentation Hub

Start here! Our documentation covers every aspect of the system.

| Document | Description | Target Audience |
| :--- | :--- | :--- |
| **[AGENTS.md](AGENTS.md)** | **The AI Intelligence Guide.** Complete registry of neural models, classical policies, and environment physics. | Researchers, ML Engineers |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | **The System Blueprint.** High-level design, data flow diagrams, and module boundaries. | Architects, Contributors |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | **The Developer Handbook.** Code style, PR process, and development guidelines. | Contributors |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | **The Field Repair Manual.** Common issues, diagnostic steps, and quick fixes. | Everyone |
| **[TUTORIAL.md](TUTORIAL.md)** | **The Developer Encyclopedia.** Deep dives into modules, code snippets, and implementation details. | Developers |

---

## 🎯 Overview

**WSmart+ Route** is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically the **Vehicle Routing Problem with Profits (VRPP)** and **Capacitated Waste Collection VRP (CWC VRP)**.

The project bridges **Deep Reinforcement Learning (DRL)** with **Operations Research (OR)**, providing a benchmarking and deployment environment where neural models (PyTorch) interact with traditional solvers (Gurobi, Hexaly).

### Mission

- 🧠 **Research Platform**: Benchmark neural routing agents against classical OR solvers
- 🚛 **Real-World Application**: Optimize waste collection routes for municipalities
- 📊 **Simulation Environment**: Test policies on realistic multi-day scenarios
- 🖥️ **User-Friendly Interface**: PySide6 GUI for training, evaluation, and visualization

---

## 🚀 Key Features

### 🧠 Intelligence

| Capability | Description |
|------------|-------------|
| **Attention-Based Models** | Transformer architectures (AM, TransGCN, DeepDecoder) for constructive routing |
| **Graph Neural Networks** | GAT, GCN, GGAC encoders for spatial relationship modeling |
| **Hierarchical RL** | Manager-Worker architecture with GAT-LSTM for temporal decision-making |
| **Meta-Learning** | MetaRNN for generalization across different problem distributions |
| **PPO & REINFORCE** | Policy gradient algorithms with multiple baseline strategies |

### 🏛️ Optimization Solvers

| Solver | Type | Description |
|--------|------|-------------|
| **Gurobi** | Exact | Branch-Cut-and-Price for optimal solutions |
| **Hexaly** | Hybrid | High-performance local search optimization |
| **ALNS** | Metaheuristic | Adaptive Large Neighborhood Search |
| **HGS** | Genetic | Hybrid Genetic Search with local search |

### 🎮 Simulation

- **Multi-Day Scenarios**: Test policies over extended time horizons (31-365 days)
- **Stochastic Fill Rates**: Gamma and empirical distributions for bin level modeling
- **Real Road Networks**: OpenStreetMap integration for realistic distance matrices
- **Parallel Execution**: Multi-core simulation with checkpointing support

### 🖥️ User Interface

- **PySide6 GUI**: Modern Qt-based desktop application
- **Training Dashboard**: Real-time loss curves and validation metrics
- **Simulation Viewer**: Interactive route visualization with Folium maps
- **Analysis Tools**: Comparative policy evaluation and statistics

---

## ⚡ Quickstart

Get up and running in 3 steps:

```bash
# 1. Clone and sync dependencies
git clone https://github.com/ACFHarbinger/WSmart-Route.git
cd WSmart-Route
uv sync

# 2. Activate the environment
source .venv/bin/activate

# 3. Launch the GUI
python main.py gui
```

### Verify Installation

```bash
# Run the test suite
python main.py test_suite

# Check code quality
uv run ruff check .
```

---

## 🧠 Model Ecosystem

We provide a comprehensive library of neural architectures and classical policies:

### Neural Models

| Model | Architecture | Function |
|-------|--------------|----------|
| **AttentionModel** | Transformer (Encoder-Decoder) | Constructive routing with Multi-Head Attention |
| **GATLSTManager** | GAT + LSTM | High-level temporal gating for HRL |
| **TemporalAM** | Transformer | Time-dependent attention mechanism |
| **MetaRNN** | RNN/LSTM | Meta-learning for distribution generalization |
| **DeepDecoderAM** | Deep Transformer | Enhanced decoder depth |
| **PointerNetwork** | RNN + Attention | Traditional pointer mechanism |

### Graph Encoders

| Encoder | Type | Description |
|---------|------|-------------|
| **GATEncoder** | Graph Attention | Multi-head attention for node embeddings |
| **GCNEncoder** | Graph Convolution | Standard GCN with aggregation |
| **GGACEncoder** | Gated Graph | Edge-node interaction with gating |
| **TGCEncoder** | Transformer-GCN | Hybrid spatial encoding |

### Classical Policies

| Policy | Type | Description |
|--------|------|-------------|
| **LookAhead** | Rolling Horizon | N-day planning with sub-optimization |
| **ALNS** | Metaheuristic | Destroy-repair operators with adaptive weights |
| **BCP** | Exact | Branch-Cut-and-Price via Gurobi/OR-Tools |
| **HGS** | Genetic | Evolutionary operators with local search |
| **Regular** | Baseline | Fixed-schedule periodic collection |
| **LastMinute** | Reactive | Threshold-triggered collection |

---

## 🔧 Setup Dependencies

Choose your preferred installation method:

### ⚡ UV (Recommended)

Fastest setup using the [uv package manager](https://github.com/astral-sh/uv).

```bash
# Sync the project and create a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat # Windows CMD
.venv\Scripts\Activate.ps1 # Windows PowerShell

# List installed packages
uv pip list
```

To deactivate and/or delete the virtual environment:
```bash
deactivate
rm -rf .venv
```

#### UV Installation

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 🐍 Anaconda

```bash
conda env create --file env/environment.yml -y --name wsr
conda activate wsr
conda list
```

To remove the environment:
```bash
conda deactivate
conda remove -n wsr --all -y
```

### 📦 Virtual Environment (Standard)

> [!NOTE]
> This method requires Python 3.9+ pre-installed on your system.

```bash
python3 -m venv env/.wsr
source env/.wsr/bin/activate
pip install -r env/requirements.txt
pip install -r env/pip_requirements.txt
```

### 🛠️ Setup Scripts

For automated environment setup:

```bash
# Linux
bash scripts/setup_env.sh <uv|conda|venv>

# Windows
scripts\setup_env.bat <uv|conda|venv>
```

---

## 🔑 Setup Optimizers

### Gurobi

1. [Create an account](https://portal.gurobi.com/iam/login/) on the Gurobi website
2. [Request a license](https://portal.gurobi.com/iam/licenses/list) (free academic licenses available)
3. [Download the software](https://www.gurobi.com/downloads/) and install

### Hexaly

1. [Create an account](https://www.hexaly.com/login) on the Hexaly website
2. [Request a license](https://www.hexaly.com/account/on-premise-licenses)

### CUDA Drivers

> [!IMPORTANT]
> CUDA drivers are required for GPU acceleration with NVIDIA hardware.

Download from the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and follow the [installation guide](https://docs.nvidia.com/cuda/index.html).

---

## 📖 Program Usage

### 🗂️ Generating Data

```bash
# Pre-generate training data for 50-vertex graphs
python main.py generate_data virtual --problem vrpp --graph_sizes 50 --n_epochs 10 --seed 42 --data_distribution gamma1

# Generate validation and test data
python main.py generate_data val --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
python main.py generate_data test --problem all --graph_sizes 20 50 --seed 42 --data_distribution gamma1
```

### 🎓 Training

```bash
# Train Attention Model on VRPP with 50 vertices
python main.py train --graph_size 50 --baseline rollout --train_dataset virtual --val_dataset data/vrpp/vrpp20_val_seed1234 --data_distribution gamma1 --n_epochs 10

# Train TransGCN with graph edges
python main.py train --model transgcn --graph_size 20 --edge_threshold 0.2 --edge_method "knn" --n_epochs 20 --data_distribution gamma1
```

#### Resume Training

```bash
python main.py train --model transgcn --graph_size 20 --edge_threshold 0.2 --edge_method "knn" --n_epochs 5 --epoch_start 20 --load_path "results/vrpp_20/run_{datetime}/epoch-19.pt" --data_distribution gamma1
```

#### Multiple GPUs

By default, training uses all available GPUs. Control GPU usage with:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py train

# Disable CUDA
python main.py train --no_cuda
```

### 📊 Evaluation

```bash
# Greedy evaluation
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy greedy --data_distribution gamma1

# Sampling evaluation (best of 1280)
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy sample --width 1280 --eval_batch_size 1 --data_distribution gamma1

# Beam search
python main.py eval data/vrpp/vrpp20_test_seed1234.pkl --model assets/model_weights/vrpp_20/am --decode_strategy bs --width 16 --data_distribution gamma1
```

### 🧪 Simulation Testing

```bash
# Test classical policies on 20-bin network for 31 days
python main.py test_sim --policies policy_last_minute policy_regular policy_look_ahead_a gurobi --problem vrpp --size 20 --days 31 --data_distribution gamma1 --cf 50 70 90 --lvl 2 3 6 --n_vehicles 1

# Test neural model on 365-day scenario
python main.py test_sim --policies transgcn --problem vrpp --size 20 --edge_threshold 0.2 --edge_method "knn" --days 365 --data_distribution emp --cpu_cores 1

# Multi-sample testing with resume
python main.py test_sim --policies gurobi alns --problem vrpp --size 100 --days 365 --data_distribution gamma2 --n_samples 10 --resume
```

### 🖥️ Graphical User Interface

```bash
python main.py gui [--test_only]
```

### ✅ Test Suite

```bash
# Run all tests
python main.py test_suite

# Run specific module/class/test
python main.py test_suite --module <module_name>
python main.py test_suite --class <class_name>
python main.py test_suite --test <test_name>
python main.py test_suite --markers <marker_name>
```

---

## 📜 Scripts

| Script | Description |
|--------|-------------|
| [gen_data.sh](scripts/gen_data.sh) | Generate datasets for training, validation, or testing |
| [train.sh](scripts/train.sh) | Train Deep Learning models |
| [hyperparam_optim.sh](scripts/hyperparam_optim.sh) | Hyperparameter optimization |
| [test_sim.sh](scripts/test_sim.sh) | Test policies on the simulator |
| [slurm.sh](scripts/slurm.sh) | Run on Slurm cluster |

Windows equivalents available as `.bat` files.

---

## 📦 Build Distribution

```bash
# Build source and binary distribution
uv build

# Create executable with PyInstaller
pyinstaller build.spec [--clean] [--noconsole]
```

---

## 🙏 Acknowledgments

This project adapts code and ideas from:

- [Attention, Learn to Solve Routing Problems](https://github.com/wouterkool/attention-learn-to-route)
- [Heterogeneous Attentions for Solving PDP via DRL](https://github.com/jingwenli0312/Heterogeneous-Attentions-PDP-DRL)
- [POMO: Policy Optimization with Multiple Optima](https://github.com/yd-kwon/POMO/tree/master)
- [WSmart+ Bin Analysis](https://github.com/ACFPeacekeeper/wsmart_bin_analysis)
- [Do We Need Anisotropic Graph Neural Networks?](https://github.com/shyam196/egc)
- [Learning TSP Requires Rethinking Generalization](https://github.com/chaitjo/learning-tsp)
- [HGS-CVRP](https://github.com/vidalt/HGS-CVRP)

---

<div align="center">
<strong>WSmart+ Route</strong> - Bridging AI and Operations Research for Smarter Waste Collection
</div>
