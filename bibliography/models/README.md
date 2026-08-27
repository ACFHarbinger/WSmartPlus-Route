# bibliography/models — filename map

One PDF per neural-model name used under `logic/src/models/core/`.
Filenames are **not** a reliable title: two files in this folder currently
contain a different paper than the name implies (issue #64). Do not swap
those PDFs unattended; the replacement sources below are the intended
citations for a human to fetch.

Verified by opening the first page of each file (2026-08-27, #63/#64).

| Filename | PDF actually contains | Intended reference for the code | Status |
|---|---|---|---|
| `Attention_Model.pdf` | Kool, van Hoof & Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019 | same | OK |
| `Pointer_Networks.pdf` | Vinyals, Fortunato & Jaitly, "Pointer Networks", NeurIPS 2015 | same | OK |
| `POMO.pdf` | Kwon et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning", NeurIPS 2020 | same | OK |
| `Sym-NCO.pdf` | Kim, Park & Park, "Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization", NeurIPS 2022 | same | OK |
| `DACT.pdf` | Ma et al., "Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer", NeurIPS 2021 | same | OK |
| `NeuOpt.pdf` | Ma & Cao, "Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt" | same | OK |
| `N2S.pdf` | Ma et al., "Efficient Neural Neighborhood Search for Pickup and Delivery Problems" | same | OK |
| `DeepACO.pdf` | Ye et al., "DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization", ICLR 2023 | same | OK |
| `GFACS.pdf` | Kim et al., "Ant Colony Sampling with GFlowNets for Combinatorial Optimization" (GFACS), arXiv:2403.07041 | same | OK |
| `DR-ALNS.pdf` | Reijnen et al., DR-ALNS (AAAI 2024) | same | OK |
| `GLOP.pdf` | Ye et al., "GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time" | same | OK |
| `MDAM.pdf` | Xin et al., "Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems" | same | OK |
| `PolyNet.pdf` | Hottung et al., "PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization", ICLR 2025 | same | OK |
| **`MATNet.pdf`** | **Tortora et al., "MATNet: Multi-Level Fusion Transformer-Based Model for Day-Ahead PV Generation Forecasting"** (IEEE Trans. Smart Grid) | Kwon, Choo, Yoon, Park, Park & Gwon, "Matrix Encoding Networks for Neural Combinatorial Optimization", NeurIPS 2021. [arXiv:2106.11113](https://arxiv.org/abs/2106.11113) | **WRONG PDF** — namesake |
| **`NARGNN.pdf`** | **Li, Chen & Koltun, "Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search"** (arXiv:1810.10659) | Joshi, Laurent & Bresson, "An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem", 2019. [arXiv:1906.01227](https://arxiv.org/abs/1906.01227) | **WRONG PDF** |

## Why those two replacements

**MATNet.** `logic/src/models/core/matnet/` and `MixedScoreMHA` implement Kwon mixed-score attention over a cost matrix (row/col streams, `W_mat ⊙ M`). The file in this folder is a PV-forecasting transformer that happens to share the acronym.

**NARGNN.** `logic/src/models/core/nargnn/` and `NARGNNEncoder` are an anisotropic/gated GNN that emits an \(N\times N\) edge heatmap, then a non-autoregressive decoder (greedy / sampling / beam). That is the Joshi et al. 2019 TSP-GCN heatmap line (`docs/modules/MODELS_MODULE.md` §3.3.1 even names the anisotropic gated graph conv). The file in this folder is Li/Chen/Koltun 2018 (vertex-inclusion GCN + guided tree search for SAT/MVC/MAXCUT). No tree search exists in our code.

## How to replace (human)

```
# from repo root, after confirming the arXiv PDFs
curl -L -o bibliography/models/MATNet.pdf  https://arxiv.org/pdf/2106.11113
curl -L -o bibliography/models/NARGNN.pdf https://arxiv.org/pdf/1906.01227
```

Then re-score both rows in `docs/moon/review/MODEL_IMPLEMENTATION_ANALYSIS.md` against the new files. Until that happens, the analysis scores MATNet 4/5 against Kwon (not against the PV paper) and NARGNN 2/5 against Li/Chen/Koltun.
