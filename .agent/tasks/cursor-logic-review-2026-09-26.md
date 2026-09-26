# Brief: Cursor (lane F: selection, improvement, acceptance and the policy framework)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers (`bibliography/policies/`)

- `Simulated_Annealing.pdf` (and `Simulated Annealing.pdf`; keep whichever is
  the Kirkpatrick / Metropolis source): check `bmc` (the Boltzmann–Metropolis
  criterion), its cooling and its temperature units against the objective's
  scale.
- `Old_Bachelor_Acceptance.pdf`: read it only if one of the nine policies
  reaches `old_bachelor_acceptance.py`. Establish that first.
- `Travelling_Salesman_Problem.pdf`: background for `fast_tsp`. The improver
  wraps the `fast-tsp` library, so the questions are about the wrapper
  contract, not the algorithm.
- The selectors have no paper. Check them against their docstrings and the
  simulation paper's definitions of the thresholds.

## Scope

- `policies/mandatory_selection/`: `selection_{lookahead,last_minute,service_level}.py`,
  `base/`, and the factory.
- `policies/route_improvement/fast_tsp.py`, `base/`, `common/`.
- `policies/acceptance_criteria/{boltzmann_metropolis_criterion,only_improving}.py`
  and `base/`.
- `policies/route_construction/base/**`: `base_routing_policy`,
  `base_multi_period_policy`, `factory`, `registry`.
- `interfaces/**`, `envs/**` (vrpp), `data/**`, `constants/**`.
- The config dataclasses for all of the above.

## Questions

1. **Selectors.**
   - Units: `LastMinuteSelectionConfig.threshold = 70.0` is a percent. Check
     every comparison against the fill representation (percent vs fraction).
   - Lookahead horizon indexing (the off-by-one on day t+1).
   - The service-level quantile and its distribution parameters.
   - `.agent/cache/tools/selector_regression.py` exists: rerun it on `main`.
2. **fast_tsp.**
   - Is the depot kept at position 0 and the tour closed?
   - Is `must_go` preserved?
   - Is a failure handled by returning the input tour?
   - Non-repeatability: this was still open on 2026-09-25. Can a seed or time
     limit fix it?
3. **Acceptance.**
   - Maximisation vs minimisation sign in `bmc` (profit objective).
   - Temperature initialisation.
   - `oi` on ties.
4. **The policy framework.** `BaseRoutingPolicy` and
   `BaseMultiPeriodRoutingPolicy`:
   - what every adapter re-implements that the base already offers
     (config lifting, distance-matrix subsetting, the tour → global-ID
     mapping, the profit calculation);
   - whether the factory and registry duplicate each other.

   List the pattern per policy for the nine policies. Lanes D/E/C own their
   rows, but this lane owns the base-class M row.
5. **Duplication.**
   - The `envs/vrpp` reward against the simulator's profit.
   - Repeated fill/overflow calculations across `data/**`, `envs`, selectors
     and the simulator `bins/`.
6. **Dead code.**
   - `interfaces/` protocols with no implementer;
   - unused constants;
   - `data/**` generators and distributions that no yaml reaches.

   The selectors, improvers and acceptance criteria outside the retained
   slice are registered, so they are not dead.
