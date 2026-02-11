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

## 5. Environment Dependencies

- **TSP**: `TSPEnv`
- **CWCVRP**: `WCVRPEnv` (Waste Collection Vehicle Routing Problem), `CWCVRPEnv` (Capacitated Waste Collection Vehicle Routing Problem), `SDWCVRPEnv` (Stochastic Demand Waste Collection Vehicle Routing Problem)
- **VRPP**: `VRPPEnv` (Vehicle Routing with Profits), `CVRPPEnv` (Capacitated Vehicle Routing with Profits)
