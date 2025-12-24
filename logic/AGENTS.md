# Logic Submodule Agents

This document maintains a registry of the intelligent agents, orchestration components, and environment physics within the `logic/src/` submodule.

## 1. Pipeline Orchestrators (logic/src/pipeline/)
These components manage the training, evaluation, and optimization lifecycles.

| Agent Name | File | Responsibilities |
| :--- | :--- | :--- |
| **Trainer** | `pipeline/train.py` | The central entry point for all training operations (`train`, `mrl_train`, `hp_optim`). Manages device selection, data loading, and the outer epoch loop. |
| **Evaluator** | `pipeline/eval.py` | Handles model evaluation on test datasets. Supports `greedy`, `sampling`, and `beam_search` decoding strategies. |
| **Epoch Manager** | `pipeline/reinforcement_learning/epoch.py` | Orchestrates the inner training loop. Updates "Time" in temporal problems and manages the interaction between the Optimizer and the Model during a single epoch. |
| **HyperParam Optimizer** | `pipeline/reinforcement_learning/hpo.py` | Implements auto-tuning algorithms: `Random Search`, `Grid Search`, and `DEHB` (Differential Evolution Hyperband) to find optimal hyperparameters. |
| **Reinforce** | `pipeline/reinforcement_learning/reinforce.py` | Implements the REINFORCE/PPO Policy Gradient algorithms. Calculates rewards, baselines, and gradients for both Manager and Worker. |

## 2. Neural Models (logic/src/models/)
These are the deep learning agents that learn to solve routing problems.

| Agent Name | File | Architecture | Function |
| :--- | :--- | :--- | :--- |
| **HRLManager** | `models/gat_lstm_manager.py` | GAT + LSTM | **High-Level Agent**. Processes temporal states (history of bin levels) and outputs a "Gating Probability" to decide *when* to activate the routing worker. |
| **AttentionModel** | `models/attention_model.py` | Transformer (Encoder-Decoder) | **Worker Agent**. Solving the routing problem (VRP/TSP) constructively. Uses Multi-Head Attention to select the next node in the tour. |
| **TemporalAM** | `models/temporal_am.py` | Transformer | A variant of AM designed to handle time-dependent features directly within the attention mechanism. |
| **MetaRNN** | `models/meta_rnn.py` | RNN/LSTM | Meta-learning component that encodes "tasks" or "distributions" to help the AM generalize across different environments. |
| **CriticNetwork** | `models/critic_network.py` | MLP (Encoder) | Estimates the State-Value function $V(s)$ to compute the baseline for REINFORCE/PPO, reducing gradient variance. |

## 3. Classical Policies (logic/src/policies/)
Traditional OR and Heuristic solvers used as baselines or fallback mechanisms.

| Policy Name | File | Type | Description |
| :--- | :--- | :--- | :--- |
| **LookAhead** | `policies/look_ahead.py` | Rolling Horizon | A planning agent that optimizes routes over a future window ($N$ days). Can deliver simple heuristics or sub-trigger exact solvers. |
| **GurobiOptimizer** | `policies/gurobi_optimizer.py` | Exact (MIP) | Interfaces with the Gurobi solver to find the mathematically optimal route for VRPP. Uses generic/set-partitioning formulations. |
| **HexalyOptimizer** | `policies/hexaly_optimizer.py` | Local Search | Interfaces with Hexaly (LocalSolver) for high-speed near-optimal solutions on large graphs. |

## 4. Problem Environments (logic/src/problems/)
Defines the "Physics" of the simulation: State transitions, Constraints, and Rewards.

| Environment | Directory | Description |
| :--- | :--- | :--- |
| **VRPP** | `problems/vrpp/` | **Vehicle Routing Problem with Profits**. Nodes have rewards (profit) and demand. Vehicles choose which nodes to visit to maximize Profit - Cost. |
| **WCVRP** | `problems/wcvrp/` | **Waste Collection VRP**. A variant of CVRP where bin levels accumulate over time. |
| **CWCVRP** | `problems/wcvrp/` | **Capacitated Waste Collection VRP**. Combines WCVRP physics with VRPP rewards. The standard environment for the WSmart+ project. |
