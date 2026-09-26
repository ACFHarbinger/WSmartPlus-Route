# Brief: Kimi (lane D: BPC, SWC-TCF, ACO-HH and the solver helpers)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers (`bibliography/policies/`)

- **BPC:** `Branch-and-Price-and-Cut.pdf`, with `Branch-and-Price.pdf`
  (Barnhart et al. 1998) and `Branch-and-Cut.pdf`. The code cites Barnhart
  et al. 1998/2000. Check:
  - the pricing (ESPPRC labelling and dominance);
  - the Ryan–Foster / edge branching;
  - the cut families (RCC, SRI/LCI) and how their duals enter pricing;
  - the node bounding.
- **SWC-TCF:** `Smart_Waste_Collection-Two-Commodity_Flow.pdf`. Check the
  two-commodity-flow constraints, the profit objective and the mandatory-bin
  constraints against the pyomo and Gurobi models.
- **ACO-HH:** `Hyper-Heuristic_Ant_Colony_Optimization.pdf`. The code cites
  "et al. (2007) §III.B". Check the pheromone over low-level heuristics, the
  visibility (operator success / time), the evaporation, and the acceptance.

## Scope

- `policies/route_construction/exact_and_decomposition_solvers/{branch_and_price_and_cut,smart_waste_collection_two_commodity_flow}/**`.
- `policies/route_construction/hyper_heuristics/ant_colony_optimization_hyper_heuristic/**`.
- `policies/helpers/solvers_and_matheuristics/**`: the parts that these three
  reach, and the duplicates of them elsewhere.
- Their `configs/policies/*` dataclasses and yaml.

## Questions

1. **BPC.** It was made exact on 2026-09-25 (375 brute-force instances,
   strong branching disabled). Now:
   - does anything still differ from the paper when strong branching is
     *enabled*?
   - are `time_limit` and `gap` honoured, and is the incumbent returned on
     timeout?
   - are the column-pool and cut-pool lifetimes correct?

   `.agent/cache/tools/bpc_bruteforce_check.py` is the harness. Take the heavy
   lock.
2. **SWC-TCF.**
   - Do the pyomo and Gurobi backends build the same model? Diff the
     constraint sets.
   - Does each map the solution back to tours the same way?
   - What happens on infeasibility or a time limit?
3. **ACO-HH.**
   - Determinism with a fixed seed.
   - Swallowed exceptions (see the 2026-09-25 `kimi_lane_d_aco_*` repros:
     are those fixed on `main`?).
   - Operator statistics that leak across days.
4. **Duplication.**
   - `helpers/solvers_and_matheuristics` against the solver code inside the
     BPC/BP/BC/MS-BPC-SP policy folders: ESPPRC labelling, the RCC
     separation, and the master-problem builders are likely copied.
   - Tour extraction from MIP solutions, repeated across the exact solvers.
5. **Dead code.** Inside the three policies and the helpers they reach. The
   BP/BC/BB policies are live, so they are out of scope for removal.
