# Brief: Qwen (lane E: ALNS, HGS, PG-CLNS, PSOMA, SANS and their operators)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers

From `bibliography/policies/`:

- **ALNS:** `Adaptive_Large_Neighborhood_Search.pdf` (Ropke & Pisinger).
  Check the roulette weights, the σ1/σ2/σ3 scores, the reaction factor, the
  segment length, the SA acceptance, and the noise.
- **HGS:** `Hybrid_Genetic_Search.pdf` (Vidal). Check:
  - biased fitness (cost rank + diversity rank, with the elite count);
  - the OX crossover and the Split;
  - the penalty adaptation for infeasible solutions;
  - the sub-population survivor selection;
  - the local search (granular neighbourhoods, SWAP*).

  Also read `hybrid_genetic_search_with_ruin-and-recreate.pdf`, but only if
  `hgs` reaches ruin-and-recreate code.
- **PSOMA:** `Particle_Swarm_Optimization_Memetic_Algorithm.pdf`.
- **SANS:** `Simulated_Annealing_Neighborhood_Search.pdf`.
- **PG-CLNS:** there is no dedicated PDF. Find the citation in its
  docstrings. If there is none, compare it against the ALNS paper plus the
  pheromone (ACO) components, and file a P row for the missing reference.

From `bibliography/operators/`, the operator papers for the operators these
five policies actually call:

- `Greedy_Insertion-Random_Removal-Worst_Removal-Route_Removal.pdf`;
- `Shaw_Removal-Branch_and_Bound_Insertion.pdf`;
- `Cyclic_Transfer-Regret_K_Insertion-Neighbor_Removal-Cluster_Removal-Historical_Removal.pdf`;
- `Selective_Route_Exchange_Crossover-Swap_Star.pdf`;
- `Greedy_Blink_Insertion-String_Removal.pdf`.

## Scope

- `policies/route_construction/meta_heuristics/{adaptive_large_neighborhood_search,hybrid_genetic_search,pheromone_guided_cooperative_large_neighborhood_search,particle_swarm_optimization_memetic_algorithm,simulated_annealing_neighborhood_search}/**`.
- `policies/helpers/{operators,local_search}/**`: about 170 helper files in
  total. First establish which of them the five policies reach.

## Questions

1. **Paper fidelity**, per policy, following the checklist above. Regret-k:
   does it use the paper's definition (the sum of differences to the best
   insertion)? Shaw relatedness: which weights, and is it normalised?
2. **Bugs.**
   - The profit objective: are removal and insertion scored as profit, not
     just distance?
   - Mandatory bins: can a destroy operator remove a `must_go` bin and leave
     it out?
   - Capacity checks after insertion.
   - Seeds and determinism.
   - HGS: the limited split (fixed on 2026-09-25) and Split correctness when
     some bins are optional.
3. **Duplication.** This is the biggest opportunity in the codebase.
   - The five policies each have their own copies of insertion, removal,
     route-cost and feasibility helpers, and so do the `helpers/operators`
     subfolders.
   - Map every duplicated operator: name, all its locations, and whether they
     are behaviourally identical.
   - Propose one canonical home per operator. Say which callers switch, and
     which differences are real (for example, a profit-aware variant against a
     distance-only one).
4. **Dead code.** Operators under `helpers/operators`/`local_search` that no
   policy, registry or yaml on `main` reaches. Check all the policies, not
   only the five: an operator used by an out-of-scope policy is not dead.
