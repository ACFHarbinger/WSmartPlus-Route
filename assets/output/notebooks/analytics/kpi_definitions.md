# KPI Definitions — Reference: `ComparacoesResultadosDashboardTestePiloto.xlsx`

Captured from the pasted EVOX/HVPL comparison table (sheets
`ComparacaoResul_491_C7_3107` and `ComparacaoEVOX_vsVRPPFixandoC7`). Each row
below is one KPI from that table, with how it maps onto the PG-CLNS batch-run
outputs in `assets/output/notebooks/<run>/` and `analytics/runs_kpi_comparison.csv`.

## Run-level parameters (table header)

| KPI (PT) | Meaning | Reference value | Our runs |
|---|---|---|---|
| R | Revenue rate (€/kg) | 0,1625 | 1.0 €/kg (`REVENUE_PER_KG` in the notebook) |
| C | Cost rate (€/km) | 1 | 1.0 €/km (`COST_PER_KM`) |
| Cap (veículo) | Vehicle capacity (kg) | 4000 (labelled "Omega" in the header) | 3500 kg (`VEHICLE_CAPACITY`) |
| Omega | Same as Cap (veículo) — header layout duplicates the label | 4000 | 3500 kg |
| No Veículos | Vehicles/trips dispatched | 2 | 3 (`n_trips`, auto multi-trip split by payload) |

## Per-route / Total KPIs

| KPI (PT) | Meaning | Our column | Notes |
|---|---|---|---|
| Algoritmo | Policy/algorithm label | `Algoritmo` | `"Lookahead + PG-CLNS"` for every row (single policy run, no multi-algorithm comparison in this batch) |
| Rota Id | Route/trip number, `Total` for the aggregate row | `Rota Id` | 1-based per-run trip index |
| No contentores recolhidos pela rota | Bins collected by this route | `No contentores recolhidos pela rota` | = `n_stops` from `routes_per_trip.csv` |
| No Contentores MustGo | Bins that overflow **today** (`fill + rate ≥ 100%`) and were forced into the route | `No Contentores MustGo` | Reconstructed from `LookaheadSelection._should_bin_be_collected` logic against the source dashboard HTML — not stored by the notebook's own outputs, since `LookaheadSelection` returns a single combined mandatory list |
| No Contentores MustGo-LookAhead | Bins added because they would overflow **before** the next forced collection day, but not today | `No Contentores MustGo-LookAhead` | Reconstructed the same way, replicating `LookaheadSelection._add_bins_to_collect` |
| No Contentores opcionais | Bins visited by the route constructor but not flagged mandatory by either rule above (PG-CLNS opportunistic stops) | `No Contentores opcionais` | = collected − (MustGo ∪ LookAhead) |
| Total No. Contentores nao visitados (sem cobertura) | Bins in the fleet not visited by this route (per-route); for the `Total` row, bins visited by **no** route at all | `Total No. Contentores nao visitados (sem cobertura)` | Per-route: `total_bins − n_stops`. Total row: `total_bins − |union of all trips|` |
| Lucro (Euros) | Net profit = revenue − cost | `Lucro (Euros)` | `REVENUE_PER_KG·kg − COST_PER_KM·km`, using our 1.0/1.0 rates (not the reference's 0.1625/1) |
| Total distância (km) | Distance travelled | `Total distancia (km)` | = `km` |
| Total peso recolhido (kg) | Waste collected | `Total peso recolhido (kg)` | = `kg` |
| Rácio (km/Ton) | Distance per tonne collected | `Racio (km/Ton)` | `km / (kg/1000)` |
| Cap_Usada% | Capacity utilisation | `Cap_Usada%` | `kg / vehicle_capacity_kg × 100`; Total row divides by `capacity × n_trips` |
| Tempo de viagem (h) | Driving time | — (`n/d`) | **Not available**: this batch's solver/notebook has no travel-time model (no average speed assumption was applied), only distance in km |
| Tempo Paragem (h) | Stop/service time | — (`n/d`) | **Not available**: no per-stop service-time model was applied in this batch run |

## Second (fixed-route) comparison block

The reference image's second table ("EVOX_Lookahead + HVPL" vs
"VRPP+ Lookahead, Mundo Rotas EVOX") re-evaluates one method's fixed routes
under the other method's cost model, to isolate routing quality from
selection quality. **Not reproduced here** — this batch run is a single
policy (Lookahead + PG-CLNS) executed twice on two different daily snapshots,
not two policies compared on the same snapshot, so there is no second
route-set to cross-evaluate against. If a like-for-like comparison against
the EVOX/HVPL reference numbers is wanted, it would need the EVOX/HVPL run to
be re-executed against the same distance matrix and economic parameters
(R = 1.0 €/kg, C = 1.0 €/km, capacity = 3500 kg) used here, or our figures
rescaled to R = 0.1625 €/kg to match the reference directly.
