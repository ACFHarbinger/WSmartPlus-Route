# Diagrams

## Flowcharts (Graphviz)

| Diagram | Source | Outputs | Describes |
| --- | --- | --- | --- |
| Simulator pipeline | `simulator_pipeline_flowchart.dot` | `.pdf` `.svg` `.png` | `simulator_testing()` orchestration, parallel/sequential runners, `SimulationContext` state machine (Initializing → Running → Finishing), and the per-day command chain in `run_day()` |
| PG-CLNS meta-heuristic | `pg_clns_flowchart.dot` | `.pdf` `.svg` `.png` | `PGCLNSPolicy` adapter, `PGCLNSSolver.solve()` main loop, ACS ant construction, LNS coaching session with SA acceptance and adaptive operator weights |
| Branch-and-Price-and-Cut | `bpc_flowchart.dot` | `.pdf` `.svg` `.png` | `BPCPolicy` adapter, `run_bpc()` set-up and B&B loop, `column_generation_loop()` (Farkas Phase I, Phase II pricing, ng-expansion, cut separation), RCSPP label-correcting pricing, branching hierarchy |
| ACO-HH hyper-heuristic | `aco_hh_flowchart.dot` | `.pdf` `.svg` `.png` | `HyperACOPolicy` adapter, operator-graph pheromone/visibility set-up, main colony loop with elitism and strategic oscillation, one ant journey (sequence selection, operator application, η updates), objective, `construct()` bootstrap |
| ALNS | `alns_flowchart.dot` | `.pdf` `.svg` `.png` | `ALNSPolicy` adapter and engine dispatch (custom / package / OR-Tools), operator registries, main loop with dynamic start temperature, σ-scoring, segment weight updates, destroy/repair step |
| HGS | `hgs_flowchart.dot` | `.pdf` `.svg` `.png` | `HGSPolicy` adapter and dispatcher (custom / PyVRP), population init, main loop with restarts, survivor selection and penalty adaptation, RP-GPX offspring + education, insertion/repair, Linear Split decoding, biased fitness |
| PSOMA | `psoma_flowchart.dot` | `.pdf` `.svg` `.png` | `PSOMAPolicy` adapter, swarm init, training phase, PSO velocity/position update with ROV + Split decoding, adaptive SA intensification on the global best, stagnation stop |
| SANS | `sans_flowchart.dot` | `.pdf` `.svg` `.png` | `SANSPolicy` adapter, "new" engine (geometric SA over 18 route/add operators with re-heating and uncrossing) and legacy "og" engine (fixed-iteration annealing, refinement, rebalancing), profit function |
| SWC-TCF exact model | `swc_tcf_flowchart.dot` | `.pdf` `.svg` `.png` | `SWCTCFPolicy` adapter, backend dispatch (OR-Tools / Pyomo / native Gurobi with fallback), full two-commodity-flow MILP (variables, constraints, objectives, Gurobi params), route extraction |

Re-render after editing a `.dot` file:

```bash
cd assets/diagrams/code
for f in simulator_pipeline_flowchart pg_clns_flowchart bpc_flowchart aco_hh_flowchart alns_flowchart hgs_flowchart psoma_flowchart sans_flowchart swc_tcf_flowchart; do
  dot -Tpdf $f.dot -o $f.pdf
  dot -Tsvg $f.dot -o $f.svg
  dot -Tpng -Gdpi=150 $f.dot -o $f.png
done
```

Shape conventions: ellipse = start/end, rounded box = process, diamond = decision,
hexagon = loop, cylinder = data/file, parallelogram = context output, 3-D box = sub-process
expanded in another cluster, red fill = pruning/error path.

## Other files (in `assets/diagrams/`)

- `diagram.drawio`, `diagram_registration.drawio` — draw.io sources (registration diagram also exported as PDF).
- `wsr_simulator-*.pdf` — earlier simulator diagrams (Portuguese).
