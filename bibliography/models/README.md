# bibliography/models — filename map

One PDF per neural-model name used under `logic/src/models/core/`.
Verified by opening the first page of each file (2026-08-27, #63/#64).
On 2026-08-27 two namesake/wrong files were replaced from arXiv and the
old copies archived under `_mismatched/`.

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
| `MATNet.pdf` | Kwon, Choo, Yoon, Park, Park & Gwon, "Matrix Encoding Networks for Neural Combinatorial Optimization", NeurIPS 2021. [arXiv:2106.11113](https://arxiv.org/abs/2106.11113) | same | OK (replaced 2026-08-27, #64). Previous file archived as `_mismatched/MATNet.Tortora-PV-forecasting.pdf` |
| `NARGNN.pdf` | Joshi, Laurent & Bresson, "An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem", 2019. [arXiv:1906.01227](https://arxiv.org/abs/1906.01227) | same | OK (replaced 2026-08-27, #64). Previous file archived as `_mismatched/NARGNN.LiChenKoltun-GCN-tree-search.pdf` |

## Why those two were replaced

**MATNet.** `logic/src/models/core/matnet/` implements Kwon mixed-score attention. The previous file was a PV-forecasting transformer that shares the acronym.

**NARGNN.** `NARGNNEncoder` is an anisotropic/gated GNN + edge heatmap (`MODELS_MODULE.md` §3.3.1). The previous file was Li/Chen/Koltun 2018 (vertex-inclusion GCN + guided tree search).

## Replacements already applied (#64)

Fetched from arXiv on 2026-08-27 after the code-side identification in `MODEL_IMPLEMENTATION_ANALYSIS.md`. First pages verified:

- `MATNet.pdf` → "Matrix Encoding Networks for Neural Combinatorial Optimization" (Kwon et al.)
- `NARGNN.pdf` → "An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem" (Joshi, Laurent, Bresson)

The previous files remain under `_mismatched/` so the namesake/wrong-paper incident is recoverable.
