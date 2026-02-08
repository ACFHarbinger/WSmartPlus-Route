# WSmart-Route Notebooks Index

> **Purpose**: Guide to all Jupyter notebooks in the WSmart-Route project
> **Last Updated**: 2026-02-08

This directory contains Jupyter notebooks for learning, experimentation, and analysis. Notebooks are organized by difficulty and purpose.

---

## 🎓 Getting Started (New Users Start Here!)

If you're new to WSmart-Route, follow the numbered tutorials in order:

| # | Notebook | Description | Time | Difficulty |
|---|----------|-------------|------|------------|
| 1 | [01_data_generation.ipynb](tutorials/01_data_generation.ipynb) | Generate problem instances for VRPP, WCVRP | 15 min | ⭐ Beginner |
| 2 | [02_environments.ipynb](tutorials/02_environments.ipynb) | Understand environment APIs and state transitions | 20 min | ⭐ Beginner |
| 3 | [03_models_and_policies.ipynb](tutorials/03_models_and_policies.ipynb) | Neural models (AM, TAM) vs classical policies (HGS, ALNS) | 25 min | ⭐⭐ Intermediate |
| 4 | [04_training_with_lightning.ipynb](tutorials/04_training_with_lightning.ipynb) | Train an Attention Model with PyTorch Lightning | 30 min | ⭐⭐ Intermediate |
| 5 | [05_evaluation_and_decoding.ipynb](tutorials/05_evaluation_and_decoding.ipynb) | Evaluate models with greedy, sampling, beam search | 20 min | ⭐⭐ Intermediate |
| 6 | [06_simulation_testing.ipynb](tutorials/06_simulation_testing.ipynb) | Multi-day waste collection simulations | 25 min | ⭐⭐ Intermediate |
| 7 | [07_extending_the_codebase.ipynb](tutorials/07_extending_the_codebase.ipynb) | Add custom models, encoders, environments | 40 min | ⭐⭐⭐ Advanced |

**Recommended Path**:
- **First-time users**: Start with Tutorial 1-2
- **Want to train models**: Tutorials 1-4
- **Researchers extending the code**: All tutorials 1-7

---

## 📊 Data & Dataset Analysis

Explore and visualize problem instances and datasets:

| Notebook | Description | Use Case |
|----------|-------------|----------|
| [datasets.ipynb](datasets.ipynb) | Load, inspect, and visualize VRPP/WCVRP datasets | Understanding data formats |
| [data_files.ipynb](data_files.ipynb) | Dataset file structure and metadata | Data management |
| [generate_maps.ipynb](generate_maps.ipynb) | Visualize routes on interactive maps (Folium) | Result visualization |

---

## 🚀 Policy Demonstrations

See different routing strategies in action:

| Notebook | Policy Type | Problem | Difficulty |
|----------|-------------|---------|------------|
| [VRP_Policy_Regular3day.ipynb](VRP_Policy_Regular3day.ipynb) | Regular (fixed schedule) | Waste Collection | ⭐⭐ Intermediate |
| [look_ahead_vrpp.ipynb](look_ahead_vrpp.ipynb) | Look-ahead (predictive) | VRPP | ⭐⭐ Intermediate |
| [VPP_OneFlow_Lookahead_Dynamic.ipynb](VPP_OneFlow_Lookahead_Dynamic.ipynb) | Dynamic look-ahead | VRPP Multi-day | ⭐⭐⭐ Advanced |
| [VRPP_Dynamic_Ajustado.ipynb](VRPP_Dynamic_Ajustado.ipynb) | Adjusted dynamic policy | VRPP | ⭐⭐⭐ Advanced |

**Key Differences**:
- **Regular**: Collect every N days (e.g., Monday-Wednesday-Friday)
- **Look-ahead**: Predict bin overflow and collect proactively
- **Dynamic**: Adapt routing decisions based on real-time bin levels

---

## 🔬 Experimental & Research

Advanced notebooks for experiments and comparisons:

| Notebook | Purpose | Audience |
|----------|---------|----------|
| [optimization.ipynb](optimization.ipynb) | Compare optimization solvers (Gurobi, HGS, ALNS) | Researchers |
| [collectioncompare.ipynb](collectioncompare.ipynb) | Compare collection policies (performance metrics) | Policy evaluation |
| [collectioncompare_V.ipynb](collectioncompare_V.ipynb) | Variant of collection comparison | Alternative metrics |
| [lightning_rl_training_tutorial.ipynb](lightning_rl_training_tutorial.ipynb) | Detailed RL training walkthrough (PyTorch Lightning) | Deep learners |

---

## 📈 Output Analysis

Analyze simulation and training results:

| Notebook | Purpose |
|----------|---------|
| [output.ipynb](output.ipynb) | Parse and analyze simulation output logs |
| [output_show.ipynb](output_show.ipynb) | Visualize training metrics and convergence |

---

## 🧪 Prototyping

| Notebook | Purpose | Status |
|----------|---------|--------|
| [prototype.ipynb](prototype.ipynb) | Experimental features and quick tests | ⚠️ Unstable |

**Note**: Prototype notebook may contain incomplete or deprecated code. Use for inspiration only.

---

## 🛠️ Prerequisites

Before running notebooks, ensure you have:

1. **Environment Setup**:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

2. **Jupyter Installation**:
   ```bash
   pip install jupyter ipykernel ipywidgets
   python -m ipykernel install --user --name wsmart-route
   ```

3. **Data Generation** (if needed):
   ```bash
   python main.py gen_data --problem vrpp --graph_sizes 20 50 --samples 1000
   ```

---

## 📚 Additional Resources

- **[docs/HYDRA_GUIDE.md](../docs/HYDRA_GUIDE.md)**: Hydra configuration system
- **[CLAUDE.md](../CLAUDE.md)**: Architecture and coding standards
- **[README.md](../README.md)**: Main project documentation
- **[COMPATIBILITY.md](../COMPATIBILITY.md)**: Model-environment compatibility matrix

---

## 🤝 Contributing

Found a notebook that's out of date or want to add a new tutorial?

1. Create your notebook in the appropriate directory
2. Add an entry to this README
3. Include clear markdown explanations and comments
4. Test all cells execute successfully
5. Submit a pull request

---

## 📝 Notebook Naming Conventions

- **Tutorials**: `0X_topic_name.ipynb` (numbered sequence)
- **Demonstrations**: `descriptive_name.ipynb` (lowercase with underscores)
- **Experiments**: Prefix with experiment type (e.g., `ablation_`, `benchmark_`)

---

## ⚡ Quick Start Recipes

### Train Your First Model

```bash
# Follow Tutorial 04
jupyter notebook tutorials/04_training_with_lightning.ipynb
```

### Compare Neural vs Classical Policies

```bash
# Use optimization.ipynb
jupyter notebook optimization.ipynb
```

### Visualize Routes on a Map

```bash
# Run simulation first
python main.py test_sim --days 7 --policies neural gurobi

# Then visualize
jupyter notebook generate_maps.ipynb
```

---

## 💡 Tips

- **GPU Required**: Tutorials 4-7 benefit from CUDA for faster training
- **Data Size**: Start with small instances (20 nodes) for faster iteration
- **Kernel Crashes**: Reduce batch size if you encounter OOM errors
- **Stale Outputs**: Clear outputs before committing: `jupyter nbconvert --clear-output --inplace *.ipynb`

---

**Happy exploring! 🚀**
