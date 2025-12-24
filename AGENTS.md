# AGENTS.md - Instructions for Coding Assistant LLMs

## 1. Project Overview & Mission
WSmart+ Route is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically the Vehicle Routing Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP).
The project mission is to bridge Deep Reinforcement Learning (DRL) with Operations Research (OR). It provides a benchmarking and deployment environment where neural models (PyTorch) interact with traditional solvers (Gurobi, Hexaly).

## 2. Technical Stack & Environmental Governance
Runtime: Python 3.9+ managed strictly via uv. Use uv sync for dependency resolution.
Primary Frameworks:
- DRL/DL: PyTorch (2.2.2) optimized for NVIDIA RTX 4080 (CUDA acceleration).
- OR Solvers: Gurobi Optimizer (11.0.3) and Hexaly.
- GUI: PySide6 (Qt for Python).

Quality Control:
- Linter: ruff (Mandatory compliance).
- Formatter: black.

Testing: pytest (Rooted in logic/test and gui/test).

## 3. Core Architectural Boundaries
Maintain strict separation of concerns across these primary modules:
- **Logic Layer (logic/src/)**
    - models/: Neural architecture implementations.
    - subnets/: Discrete components like Encoders (GAT, GCN, MLP) and Decoders.
    - modules/: Atomic utility layers (Normalization, Multi-Head Attention).
    - problems/: The environment "Physics."
    - state_*.py: Critical logic for state transitions, node masking, and reward calculation.
    - policies/: Traditional heuristic and exact algorithms (ALNS, Branch-Cut-and-Price).
    - pipeline/: Orchestration logic for train.py, eval.py, and test_sim.
- **GUI Layer (gui/src/)**
    - tabs/: Module-specific UI views (Analysis, Training, Simulator).
    - helpers/: Threaded workers (e.g., chart_worker.py) for non-blocking UI.
    - windows/: Application window management.

## 4. Key CLI Entry Points (Operational Playbook)
Always reference these commands when proposing code changes or workflows:
| Action | Command |
| --- | --- |
| Sync Environment | uv sync |
| Data Generation | python main.py generate_data virtual --problem vrpp --graph_sizes 50 |
| Model Training | python main.py train --model am --problem vrpp --graph_size 50 |
| Simulation Test | python main.py test_sim --policies regular gurobi alns --days 31 |
| Launch GUI | python main.py gui |

## 5. External Access and Browser Usage Rules
The agent is authorized to use the following external tools to assist in development:

### Web Search and Documentation
Authorization: Use the @search tool (Google Search) to retrieve the latest documentation for Gurobi 11+, PySide6 API references, and PyTorch 2.2+ best practices.

Verification: When suggesting a solution involving a third-party library, perform a search to verify that the suggested API methods are not deprecated.

Problem Solving: If a CUDA error or a specific Linux driver conflict (NVIDIA 550/560) is detected in logs, use web search to find relevant GitHub issues or system-level workarounds for Ubuntu 24.04.

### Knowledge Cutoff Management
Directive: Always cross-reference your internal training data with a web search if the technology was updated after January 2024 (e.g., specific Gurobi performance tunings).

## 6. Domain-Specific Coding Standards
### Mathematical & DRL Integrity
- Invalid Move Prevention: Decoders must implement masking via logic/src/utils/boolmask.py before sampling nodes to ensure feasibility.
- Activation Scaling: Prefer custom modules in logic/src/models/modules/normalization.py over generic nn.LayerNorm for consistency across 1M token context logic.
- State Transitions: Never modify state_*.py files without ensuring logic/test/test_problems.py still passes.

### Performance & Hardware
- GPU Offloading: Optimization targets the RTX 3090ti and the RTX 4080 (laptop version). Ensure tensors are explicitly moved to device using setup_utils.py.
- GUI Threading: Heavy computations (training/loading) must inherit from QThread. Never run blocking CO logic on the main Qt thread.

## 7. AI Review & Severity Protocol
Categorize your feedback and edits using these severity levels:
- CRITICAL: Breaking state_*.py transition logic; exposing credentials; cryptographic flaws in fs_cryptography.py.
- HIGH: CUDA memory leaks; incorrect skip_connection.py usage; pyproject.toml version mismatches.
- MEDIUM: Suboptimal Pandas operations in pandas_model.py; deviations from ruff formatting.
- LOW: Documentation typos; redundant imports; UI padding/margin adjustments in globals.py.

## 8. Known Constraints & "No-Go" Areas
- Legacy Preservation: Never edit files with copy.py suffixes or those inside legacy/ folders.
- Slurm Sensitivity: Cluster scripts (scripts/slurm.sh) use specific path mappings; verify before modifying.
- Linux Stability: In the MSI Vector environment, always include --use-angle=vulkan and --disable-gpu-sandbox when      suggesting debug flags for the GUI.

## 9. Usage Note
Reference this file during project-wide analysis. When refactoring components, ensure they align with the subnets/ hierarchy and Normalization standards defined here.
