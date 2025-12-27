# Logic Submodule Agents

This document maintains a registry of the intelligent agents, orchestration components, and environment physics within the `logic/src/` submodule.

## 1. Pipeline Orchestrators (logic/src/pipeline/)
These components manage the training, evaluation, and testing lifecycles.

| Agent Name | File | Responsibilities |
| :--- | :--- | :--- |
| **Trainer** | `pipeline/train.py` | The central entry point for all training operations (`train`, `mrl_train`, `hp_optim`). Manages device selection, data loading, and the outer epoch loop. |
| **Evaluator** | `pipeline/eval.py` | Handles model evaluation on test datasets. Supports `greedy`, `sampling`, and `beam_search` decoding strategies. |
| **Tester** | `pipeline/test.py` | Orchestrates high-level simulation testing across multiple seeds and policies. Supports parallel execution and checkpointing. |

## 2. Neural Models (logic/src/models/)
These are the deep learning agents that learn to solve routing problems.

| Agent Name | File | Architecture | Function |
| :--- | :--- | :--- | :--- |
| **GATLSTManager** | `models/gat_lstm_manager.py` | GAT + LSTM | **High-Level Agent**. Processes temporal states (history of bin levels) and outputs a "Gating Probability" to decide *when* to activate the routing worker. |
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
| **ALNS** | `policies/adaptive_large_neighborhood_search.py` | Metaheuristic | Implements Adaptive Large Neighborhood Search. Iteratively improves solutions by applying destroy and repair operators. |
| **BCP** | `policies/branch_cut_and_price.py` | Exact / Hybrid | A dispatcher for Branch-Cut-and-Price algorithms. Supports OR-Tools, VRPy, and Gurobi engines for solving prize-collecting VRPs. |
| **HGS** | `policies/hybrid_genetic_search.py` | Genetic Algorithm | Hybrid Genetic Search that combines evolutionary operators with local search and a Split algorithm for decoding giant tours. |
| **LastMinute** | `policies/last_minute.py` | Reactive Heuristic | Trigger-based policy that initiates collection only when bin levels exceed a pre-defined threshold. |
| **MultiVehicle** | `policies/multi_vehicle.py` | OR Solver | External solver interface (PyVRP/OR-Tools) optimized for multi-vehicle routing with capacity constraints. |
| **Regular** | `policies/regular.py` | Periodic | A baseline policy that visits every bin in the area on a fixed schedule (e.g., every $N$ days). |
| **SingleVehicle** | `policies/single_vehicle.py` | TSP Heuristic | Constructive heuristic for single-vehicle scenarios. Uses `fast_tsp` for sequencing and inserts depot trips as needed for capacity. |

## 4. Problem Environments (logic/src/problems/)
Defines the "Physics" of the simulation: State transitions, Constraints, and Rewards.

| Environment | Directory | Description |
| :--- | :--- | :--- |
| **VRPP** | `problems/vrpp/` | **Vehicle Routing Problem with Profits**. Nodes have rewards (profit) and demand. Vehicles choose which nodes to visit to maximize Profit - Cost. |
| **WCVRP** | `problems/wcvrp/` | **Waste Collection VRP**. A variant of CVRP where bin levels accumulate over time. |
| **CWCVRP** | `problems/wcvrp/` | **Capacitated Waste Collection VRP**. Combines WCVRP physics with VRPP rewards. The standard environment for the WSmart+ project. |

## 5. Neural Components (logic/src/models/modules/)
These low-level building blocks are used to construct the higher-level models.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **MultiHeadAttention** | `models/modules/multi_head_attention.py` | Standard multi-head attention mechanism used for context encoding and node selection. |
| **GraphConvolution** | `models/modules/graph_convolution.py` | Basic spatial encoder layer that aggregates features from neighboring nodes. |
| **DistanceAwareGC** | `models/modules/distance_graph_convolution.py` | Spatial encoder that scales influence by physical distance (Inverse, Exponential, or Learned). |
| **GatedGraphConv** | `models/modules/gated_graph_convolution.py` | Advanced RNN-style graph layer that updates both node and edge features through gating. |
| **EfficientGraphConv** | `models/modules/efficient_graph_convolution.py` | Lightweight multi-head convolution with multiple aggregators (mean, max, symnorm, etc.). |
| **FeedForward** | `models/modules/feed_forward.py` | Standard multi-layer perceptron (MLP) block for feature transformation. |
| **Normalization** | `models/modules/normalization.py` | Wrapper for Batch, Layer, and Instance normalization layers. |
| **ActivationFunction**| `models/modules/activation_function.py`| Wrapper for common activation functions (ReLU, Sigmoid, etc.). |
| **SkipConnection** | `models/modules/skip_connection.py` | Manages residual and dense connections to prevent vanishing gradients. |

## 6. Neural Sub-networks (logic/src/models/subnets/)
Medium-level sub-networks that combine multiple modules into functional blocks (Encoders, Decoders).

### Encoders
| Component Name | File | Description |
| :--- | :--- | :--- |
| **GATEncoder** | `models/subnets/gat_encoder.py` | Multi-head Graph Attention encoder for spatial relationship modeling. |
| **GACEncoder** | `models/subnets/gac_encoder.py` | Graph Attention Convolution encoder that incorporates edge features. |
| **TGCEncoder** | `models/subnets/tgc_encoder.py` | Transformer-style Graph Convolution encoder. |
| **GGACEncoder** | `models/subnets/ggac_encoder.py` | Gated Graph Attention Convolution encoder with edge-node interaction. |
| **GCNEncoder** | `models/subnets/gcn_encoder.py` | Standard Graph Convolutional Network encoder. |
| **PointerEncoder** | `models/subnets/ptr_encoder.py` | Simple encoder for Pointer Network architectures. |
| **MLPEncoder** | `models/subnets/mlp_encoder.py` | Layer-wise MLP encoder independent of graph structure. |

### Decoders
| Component Name | File | Description |
| :--- | :--- | :--- |
| **GATDecoder** | `models/subnets/gat_decoder.py` | Decodes graph embeddings into action log-likelihoods using attention. |
| **PointerDecoder** | `models/subnets/ptr_decoder.py` | Implements the pointing mechanism for constructive routing. |

### Predictors
| Component Name | File | Description |
| :--- | :--- | :--- |
| **GatedRecurrentFillPredictor** | `models/subnets/grf_predictor.py` | Predicts future bin fill levels using gated recurrent logic. |

## 7. Simulator Hub (logic/src/pipeline/simulator/)
The "Physics Engine" and environment management layer for the WSmart-Route simulation.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **Simulation** | `pipeline/simulator/simulation.py` | Main orchestrator for running large-scale simulation experiments (sequential or parallel). |
| **Day Runner** | `pipeline/simulator/day.py` | Logic for executing a single simulation day: state transitions, policy execution, and result logging. |
| **Bins** | `pipeline/simulator/bins.py` | Manages the state and stochastic/empirical fill logic for the bin population. |
| **Checkpointing** | `pipeline/simulator/checkpoints.py` | Handles saving and resuming simulation states to prevent data loss. |
| **Loader** | `pipeline/simulator/loader.py` | Utilities for loading area-specific coordinates, waste data, and parameters. |
| **Network** | `pipeline/simulator/network.py` | Manages distance matrices, shortest path computations (OSM/Google Maps), and graph topology. |
| **Processor** | `pipeline/simulator/processor.py` | Data normalization and transformation logic for preparing inputs for neural models. |

## 8. Reinforcement Learning (logic/src/pipeline/reinforcement_learning/)
Algorithms and training loops for optimizing neural agents via policy gradients.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **Reinforce** | `pipeline/reinforcement_learning/reinforce.py` | Implements REINFORCE and PPO algorithms. Calculates rewards, baselines, and gradients for both Manager and Worker. |
| **Epoch Manager** | `pipeline/reinforcement_learning/epoch.py` | Orchestrates the inner training loop. Manages state-to-model-to-optimizer interactions during a single epoch. |
| **HyperParam Optimizer** | `pipeline/reinforcement_learning/hpo.py` | Implements auto-tuning algorithms like Random Search, Grid Search, and DEHB to find optimal hyperparameters. |
| **Post Processor** | `pipeline/reinforcement_learning/post_processing.py` | Optional optimization layer that refines model outputs for efficiency (kg/km) post-training. |
| **Meta-Learning**| `pipeline/reinforcement_learning/meta/`| Sub-directory containing logic for across-task generalization and distribution-aware training. |
