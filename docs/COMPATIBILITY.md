# Model & Environment Compatibility Matrix

This document serves as the authoritative reference for compatibility between Neural Models, Problem Environments, and RL Algorithms in the WSmart-Route codebase.

## 1. Model-Problem Support Matrix

### Constructive Models (Autoregressive)

These models construct solutions step-by-step.

| Model Name               | Class                  | TSP | CWCVRP | VRPP | Supported Encoders      | Supported Decoders |
| ------------------------ | ---------------------- | --- | ------ | ---- | ----------------------- | ------------------ |
| **Attention Model (AM)** | `AttentionModel`       | ✅  | ✅     | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **POMO**                 | `AttentionModel` (cfg) | ✅  | ✅     | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **SymNCO**               | `AttentionModel` (cfg) | ✅  | ✅     | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **MatNet**               | `MatNet`               | ✅  | ❌     | ❌   | `MatNetEncoder`         | `MatNetDecoder`    |
| **Pointer Network**      | `PointerNetwork`       | ✅  | ❌     | ❌   | `LSTMEncoder`, `GCN`    | `PointerDecoder`   |
| **MDAM**                 | `MDAM`                 | ✅  | ✅     | ✅   | `GraphAttentionEncoder` | `MultiDecoder`     |
| **MoE-AM**               | `MoEAttentionModel`    | ✅  | ✅     | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **GFACS**                | `GFACS`                | ✅  | ❌     | ✅   | `GFACSEncoder`          | `GFACSPolicy`      |

### Improvement Models (Iterative)

These models start with a solution (or empty) and iteratively improve it.

| Model Name  | Class     | TSP | CWCVRP | VRPP | Underlying Policy |
| ----------- | --------- | --- | ------ | ---- | ----------------- |
| **NeuOpt**  | `NeuOpt`  | ✅  | ✅     | ❌   | `NeuOptPolicy`    |
| **N2S**     | `N2S`     | ✅  | ✅     | ❌   | `N2SPolicy`       |
| **DACT**    | `DACT`    | ✅  | ✅     | ❌   | `DACTPolicy`      |
| **DeepACO** | `DeepACO` | ✅  | ✅     | ✅   | `DeepACOPolicy`   |
| **GLOP**    | `GLOP`    | ✅  | ✅     | ✅   | `GLOPPolicy`      |

### Transductive / Active Search

These methods optimize the policy _during_ inference for a specific instance.

| Model Name  | Class           | TSP | CWCVRP | VRPP | Reference             |
| ----------- | --------------- | --- | ------ | ---- | --------------------- |
| **EAS**     | `EAS`           | ✅  | ✅     | ✅   | Hottung et al. (2022) |
| **EAS-Emb** | `EAS` (variant) | ✅  | ✅     | ✅   | Embedding search      |
| **EAS-Lay** | `EAS` (variant) | ✅  | ✅     | ✅   | Layer weight search   |

## 2. Encoder-Decoder Compatibility

While the codebase allows mixing via configuration, these are the validated pairs.

| Encoder Class              | Decoder Class      | Typical Use Case             |
| :------------------------- | :----------------- | :--------------------------- |
| `GraphAttentionEncoder`    | `AttentionDecoder` | Standard AM, POMO, SymNCO    |
| `GatedGraphAttConvEncoder` | `AttentionDecoder` | Deeper AM, Complex VRPs      |
| `GraphAttConvEncoder`      | `AttentionDecoder` | Lightweight AM               |
| `MatNetEncoder`            | `MatNetDecoder`    | Asymmetric TSP, SDVRP        |
| `GFACSEncoder`             | `GFACSPolicy`      | Ant Colony System (GFlowNet) |
| `LSTMEncoder`              | `PointerDecoder`   | Pointer Network (Seq2Seq)    |

## 3. RL Algorithm - Policy Compatibility

Not all RL algorithms work with all policy types.

| Algorithm        | Compatible Policies | Description                                             |
| :--------------- | :------------------ | :------------------------------------------------------ |
| **REINFORCE**    | Constructive        | Standard Policy Gradient                                |
| **PPO**          | Constructive        | Proximal Policy Optimization (Clipped)                  |
| **A2C**          | Constructive        | Advantage Actor-Critic                                  |
| **Stepwise PPO** | Constructive        | PPO applied at each decoding step (slower, more stable) |
| **DACT / N2S**   | Improvement         | Specialized learning for iterative improvement          |

## 4. Recommended Configurations

Best-practice configurations for different problem types.

| Problem        | Goal    | Recommended Model  | Config File         | Est. Training Time (1 GPU) |
| :------------- | :------ | :----------------- | :------------------ | :------------------------- |
| **TSP-50**     | Speed   | **SymNCO**         | `model/symnco.yaml` | ~4 hours                   |
| **TSP-100**    | Quality | **MatNet**         | `model/matnet.yaml` | ~12 hours                  |
| **CWCVRP-50**  | General | **POMO**           | `model/pomo.yaml`   | ~8 hours                   |
| **CWCVRP-100** | Scale   | **AttentionModel** | `model/am.yaml`     | ~24 hours                  |
| **VRPP-100**   | Payoff  | **AttentionModel** | `model/am.yaml`     | ~20 hours                  |
| **CWCVRP-100** | Dynamic | **Tam (Temporal)** | `model/tam.yaml`    | ~26 hours                  |

## 5. Classical Policy - Problem Compatibility

### 5.1 Policy-Problem Support Grid

| Policy        | Registry Key   | Approach          | TSP | CVRP | VRPP | WCVRP | SCWCVRP |
| :------------ | :------------- | :---------------- | :-: | :--: | :--: | :---: | :-----: |
| **Neural**    | `neural`       | Deep RL           |  -  |  ✅  |  ✅  |  ✅   |   ✅    |
| **TSP**       | `tsp`          | Heuristic         | ✅  | ✅¹  |  -   |   -   |    -    |
| **LKH**       | `lkh`          | Heuristic         | ✅  | ✅¹  |  -   |   -   |    -    |
| **CVRP**      | `cvrp`         | MV Solver         |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **BPC**       | `bpc`          | Exact / Heuristic |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **VRPP**      | `vrpp`         | Exact / Heuristic |  -  |  -   |  ✅  |   -   |    -    |
| **ALNS**      | `alns`         | Metaheuristic     |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **HGS**       | `hgs`          | Metaheuristic     |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **HGS+ALNS**  | `hgs_alns`     | Hybrid Meta       |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **ACO**       | `aco`          | Metaheuristic     |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **Hyper-ACO** | `hyper_aco`    | Hyper-heuristic   |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **SISR**      | `sisr`         | Metaheuristic     |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **HVPL**      | `hvpl`         | Population Meta   |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **AHVPL**     | `ahvpl`        | Population Meta   |  -  |  ✅  |  ✅  |  ✅   |    -    |
| **SANS**      | `sans` / `lac` | Metaheuristic     |  -  |  ✅  |  ✅  |  ✅   |    -    |

> ¹ TSP and LKH solve a single-vehicle TSP then split the tour by capacity for CVRP-like behavior.

### 5.2 Solver Engines

Several policies expose multiple backend engines:

| Policy   | Engine Options           | Default |
| :------- | :----------------------- | :------ |
| **CVRP** | PyVRP, OR-Tools          | PyVRP   |
| **BPC**  | Gurobi, OR-Tools, VRPy   | Gurobi  |
| **VRPP** | Gurobi, Hexaly           | Gurobi  |
| **ALNS** | custom, package, ortools | custom  |
| **HGS**  | custom, PyVRP            | custom  |
| **SANS** | "new", "og" (legacy LAC) | new     |

### 5.3 Solving Approach Classification

| Category                 | Policies                                             |
| :----------------------- | :--------------------------------------------------- |
| **Exact**                | BPC (Gurobi), VRPP (Gurobi)                          |
| **Heuristic**            | TSP (`fast_tsp`), LKH (Lin-Kernighan), VRPP (Hexaly) |
| **Multi-Vehicle Solver** | CVRP (PyVRP / OR-Tools)                              |
| **Metaheuristic**        | ALNS, HGS, ACO, SISR, SANS                           |
| **Hybrid Metaheuristic** | HGS+ALNS, HVPL (ACO+ALNS), AHVPL (HGS+ACO+ALNS)      |
| **Hyper-heuristic**      | Hyper-ACO (ACO selects operator sequences)           |
| **Neural / DRL**         | Neural (Attention Models, HRL, Meta-RL)              |

### 5.4 Selection Strategies (mandatory Determination)

In WCVRP/CVRP mode, a selection strategy runs **before** the routing policy to decide which bins must be visited:

| Strategy         | Registry Key    | Logic                                     | Use Case                       |
| :--------------- | :-------------- | :---------------------------------------- | :----------------------------- |
| **Regular**      | `regular`       | Fixed schedule: `day % (freq+1) == 1`     | Periodic collection            |
| **LastMinute**   | `last_minute`   | Collect if `fill > threshold`             | Reactive overflow prevention   |
| **Lookahead**    | `lookahead`     | Predict overflow within N days            | Proactive planning             |
| **Revenue**      | `revenue`       | Collect if `expected_revenue > threshold` | Economic optimization          |
| **ServiceLevel** | `service_level` | `fill + rate + z*std > capacity`          | Statistical overflow guarantee |

### 5.5 VRPP vs CVRP Mode

Problem mode is determined by a base-class flag in `BaseRoutingPolicy`:

```python
use_all_bins = bool(values.get("vrpp", True))
```

- **VRPP mode** (`True`): All bins are considered; the solver selects the profitable subset.
- **CVRP/WCVRP mode** (`False`): Only mandatory bins (pre-selected by a selection strategy) are routed.

SCWCVRP stochastic waste is handled by the simulator's `Bins` object, not the routing policy. Only **Neural** has explicit stochastic modeling via temporal attention (TAM) and fill-level prediction (GRFPredictor).

## 6. Vectorized Model Policies - Problem Compatibility

These are GPU-accelerated, PyTorch-native (`nn.Module`) policy implementations in `logic/src/models/policies/`. Unlike the simulation-facing policies in Section 5, these operate on batched tensors and integrate directly with the RL training pipeline.

### 6.1 Policy-Problem Support Grid

| Policy       | Registry Key | Base Class             | Approach                    | TSP | CVRP | VRPP | WCVRP | SCWCVRP |
| :----------- | :----------- | :--------------------- | :-------------------------- | :-: | :--: | :--: | :---: | :-----: |
| **HGS**      | `hgs`        | `AutoregressivePolicy` | Genetic + Local Search      | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **ALNS**     | `alns`       | `AutoregressivePolicy` | Destroy-Repair + SA         | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **ACO**      | `aco`        | `AutoregressivePolicy` | Ant Colony (Pheromone)      | ✅  |  ✅  |  -   |   -   |    -    |
| **HVPL**     | `hvpl`       | `AutoregressivePolicy` | Population (ACO+ALNS)       | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **AHVPL**    | `ahvpl`      | `AutoregressivePolicy` | Population (HGS+ACO+ALNS)   | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **HGS-ALNS** | `hgs_alns`   | `AutoregressivePolicy` | Genetic + ALNS Education    | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **ILS**      | -            | `ImprovementPolicy`    | Local Search + Perturbation | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |
| **RandomLS** | -            | `ImprovementPolicy`    | Stochastic Local Search     | ✅  |  ✅  |  ✅  |  ✅   |   ✅¹   |

> ¹ All constructive policies accept a generic `wastes` tensor. SCWCVRP stochastic waste is resolved in the environment layer; the policy receives expected wastes.

### 6.2 Policy Class Hierarchy

```
nn.Module
├── ConstructivePolicy
│   ├── AutoregressivePolicy
│   │   ├── ACO (VectorizedACOPolicy)
│   │   ├── HGS (VectorizedHGS)
│   │   │   └── HGS-ALNS (VectorizedHGSALNS)
│   │   ├── ALNS (VectorizedALNS)
│   │   └── HVPL (VectorizedHVPL)
│   │       └── AHVPL (VectorizedAHVPL)
│   └── NonAutoregressivePolicy
│
└── ImprovementPolicy
    ├── ILS (IteratedLocalSearchPolicy)
    └── RandomLS (RandomLocalSearchPolicy)
```

### 6.3 Solving Approach Details

| Policy       | Algorithm Summary                                                                                                                                                                                                                             | Key Parameters                                                     |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| **HGS**      | Evolutionary loop: tournament selection → ordered crossover (OX1) → local search education (2-opt, 3-opt, swap, relocate, 2-opt\*, swap\*) → biased fitness survivor selection with diversity (Broken Pairs Distance). Restart on stagnation. | `population_size`, `n_generations`, `elite_size`, `crossover_rate` |
| **ALNS**     | Iterative destroy-repair with adaptive operator weights. Destroy: random, worst, cluster removal. Repair: greedy insertion, regret-k. Simulated annealing acceptance.                                                                         | `max_iterations`, `start_temp`, `cooling_rate`                     |
| **ACO**      | Probabilistic construction: `P(i→j) ∝ τ^α × η^β`. ACS exploitation with probability `q0`. Pheromone update from best ants + evaporation.                                                                                                      | `n_ants`, `n_iterations`, `alpha`, `beta`, `decay`, `q0`           |
| **HVPL**     | League of teams initialized via ACO. Each team coached with ALNS. Worst teams substituted with fresh ACO constructions. Global pheromone update from best solutions.                                                                          | `n_teams`, `sub_rate`, `aco_iterations`, `alns_iterations`         |
| **AHVPL**    | Extends HVPL with HGS genetic operators: ordered crossover between teams, biased fitness with diversity, elite preservation.                                                                                                                  | + `crossover_rate`, `alpha_diversity`, `elite_size`                |
| **HGS-ALNS** | HGS with the education phase replaced by full ALNS solver instead of sequential local search. More powerful but slower refinement.                                                                                                            | HGS params + `alns_education_iterations`                           |
| **ILS**      | Local search until local optimum → perturbation (double-bridge, shuffle, random swap) → repeat. Used as expert policy for imitation learning.                                                                                                 | `ls_operator`, `perturbation_type`, `n_restarts`                   |
| **RandomLS** | Stochastic application of local search operators sampled from a probability distribution. Used as expert policy for imitation learning.                                                                                                       | `n_iterations`, `op_probs`                                         |

### 6.4 Local Search Operators

All improvement policies share a library of vectorized (GPU-batched) local search operators:

| Category                | Operators                                             | Scope                 |
| :---------------------- | :---------------------------------------------------- | :-------------------- |
| **Route (intra-route)** | 2-opt, 3-opt, swap\*, 2-opt\*, LKH                    | Within a single route |
| **Move**                | Relocate, Swap                                        | Between routes        |
| **Exchange**            | OR-opt, Cross-exchange, λ-interchange, Ejection chain | Between routes        |
| **Destroy**             | Random, Worst, Cluster, Shaw, String removal          | ALNS destroy phase    |
| **Repair**              | Greedy insertion, Regret-k insertion                  | ALNS repair phase     |
| **Unstringing**         | Type I, II, III, IV (advanced k-opt)                  | Advanced moves        |

### 6.5 Giant Tour Decomposition

All improvement policies use `vectorized_linear_split()` to convert a giant tour (single permutation of all nodes) into feasible multi-vehicle routes:

- **Algorithm**: Bellman-Ford on a DAG for optimal segmentation under capacity constraints
- **Supports**: `max_vehicles` constraint (unlimited when set to 0)
- **TSP mode**: Single vehicle, no wastes → one route covering all nodes
- **CVRP/WCVRP mode**: Multi-vehicle with waste-based splits

## 7. Environment Dependencies

- **TSP**: `TSPEnv`
- **CWCVRP**: `WCVRPEnv` (Waste Collection Vehicle Routing Problem), `CWCVRPEnv` (Capacitated Waste Collection Vehicle Routing Problem), `SCWCVRPEnv` (Stochastic Capacitated Waste Collection Vehicle Routing Problem)
- **VRPP**: `VRPPEnv` (Vehicle Routing with Profits), `CVRPPEnv` (Capacitated Vehicle Routing with Profits)
