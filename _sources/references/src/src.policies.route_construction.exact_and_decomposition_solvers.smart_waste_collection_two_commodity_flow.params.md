# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SWCTCFParams <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAX_ARC_DISTANCE_KM <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.MAX_ARC_DISTANCE_KM>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.MAX_ARC_DISTANCE_KM
    :summary:
    ```
````

### API

`````{py:class} SWCTCFParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams
```

````{py:attribute} framework
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.framework
```

````

````{py:attribute} engine
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.engine
```

````

````{py:attribute} gurobi_threads
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.gurobi_threads
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.gurobi_threads
```

````

````{py:attribute} gurobi_soft_mem_limit_gb
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.gurobi_soft_mem_limit_gb
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.gurobi_soft_mem_limit_gb
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.seed
```

````

````{py:attribute} formulation
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.formulation
:type: str
:value: >
   'paper'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.formulation
```

````

````{py:attribute} depot_inflow
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.depot_inflow
:type: str
:value: >
   'equal'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.depot_inflow
```

````

````{py:attribute} solver_tuning
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.solver_tuning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.solver_tuning
```

````

````{py:attribute} relax_forced_on_infeasible
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.relax_forced_on_infeasible
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.relax_forced_on_infeasible
```

````

````{py:attribute} link_depot_arcs
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.link_depot_arcs
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.link_depot_arcs
```

````

````{py:attribute} max_arc_distance_km
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.max_arc_distance_km
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.max_arc_distance_km
```

````

````{py:attribute} warm_start
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.warm_start
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.warm_start
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.SWCTCFParams.to_dict
```

````

`````

````{py:data} MAX_ARC_DISTANCE_KM
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.MAX_ARC_DISTANCE_KM
:value: >
   6000.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params.MAX_ARC_DISTANCE_KM
```

````
