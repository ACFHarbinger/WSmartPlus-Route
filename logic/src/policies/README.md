# Simulation-Facing Policy Adapters

This directory contains **simulation-facing policy adapters** that inherit from `BaseRoutingPolicy`.

## Purpose

These adapters interface classical OR solvers (HGS, ALNS, BCP, etc.) with the WSmart+ simulation engine. They handle:

- Converting simulation context into solver inputs
- Executing route optimization
- Mapping solver outputs back to simulation format

## Key Files

| File | Description |
|------|-------------|
| `base_routing_policy.py` | Abstract base class with shared utilities |
| `policy_alns.py` | Adaptive Large Neighborhood Search |
| `policy_bcp.py` | Branch-Cut-and-Price (exact) |
| `policy_hgs.py` | Hybrid Genetic Search |
| `policy_tsp.py` | TSP heuristics |
| `policy_cvrp.py` | CVRP multi-vehicle routing |

## Related Directory

See `logic/src/models/policies/` for **RL training wrappers** (e.g., `VectorizedALNS`, `VectorizedHGS`) used during neural network training.
