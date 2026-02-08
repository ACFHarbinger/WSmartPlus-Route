# Model Compatibility Matrix

This document references the compatibility between different Neural Models and Problem Environments in the WSmart-Route codebase.

## Constructive Models (Autoregressive)

| Model Name               | Class                  | TSP | CVRP | VRPP | Supported Encoders      | Supported Decoders |
| ------------------------ | ---------------------- | --- | ---- | ---- | ----------------------- | ------------------ |
| **Attention Model (AM)** | `AttentionModel`       | ✅  | ✅   | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **POMO**                 | `AttentionModel` (cfg) | ✅  | ✅   | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **SymNCO**               | `AttentionModel` (cfg) | ✅  | ✅   | ✅   | `GraphAttentionEncoder` | `AttentionDecoder` |
| **MatNet**               | `MatNet`               | ✅  | ❌   | ❌   | `MatNetEncoder`         | `MatNetDecoder`    |
| **Pointer Network**      | `PointerNetwork`       | ✅  | ❌   | ❌   | `LSTMEncoder`, `GCN`    | `PointerDecoder`   |
| **MDAM**                 | `MDAM`                 | ✅  | ✅   | ✅   | `GraphAttentionEncoder` | `MultiDecoder`     |

## Improvement Models (Iterative)

| Model Name  | Class     | TSP | CVRP | VRPP | Underlying Policy |
| ----------- | --------- | --- | ---- | ---- | ----------------- |
| **NeuOpt**  | `NeuOpt`  | ✅  | ✅   | ❌   | `NeuOptPolicy`    |
| **N2S**     | `N2S`     | ✅  | ✅   | ❌   | `N2SPolicy`       |
| **DACT**    | `DACT`    | ✅  | ✅   | ❌   | `DACTPolicy`      |
| **DeepACO** | `DeepACO` | ✅  | ✅   | ✅   | `DeepACOPolicy`   |

## Transductive / Active Search

| Model Name  | Class           | TSP | CVRP | VRPP | Reference             |
| ----------- | --------------- | --- | ---- | ---- | --------------------- |
| **EAS**     | `EAS`           | ✅  | ✅   | ✅   | Hottung et al. (2022) |
| **EAS-Emb** | `EAS` (variant) | ✅  | ✅   | ✅   | Embedding search      |
| **EAS-Lay** | `EAS` (variant) | ✅  | ✅   | ✅   | Layer weight search   |

## Environment Dependencies

- **TSP**: `TSPEnv`
- **CVRP**: `WCVRPEnv`, `CWCVRPEnv`, `SDWCVRPEnv`
- **VRPP**: `VRPPEnv` (with Prizes)

## Encoder-Decoder Compatibility

> **Note**: While the codebase allows mixing and matching via configuration, these are the tested and recommended pairs.

- **GraphAttentionEncoder** <--> **AttentionDecoder** (Standard AM)
- **GatedGraphAttConvEncoder** <--> **AttentionDecoder** (Deeper AM)
- **MatNetEncoder** <--> **MatNetDecoder** (Matrix)
