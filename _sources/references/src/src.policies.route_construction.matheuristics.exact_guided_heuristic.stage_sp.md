# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_sp_stage <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.run_sp_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.run_sp_stage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.logger
```

````

````{py:function} run_sp_stage(pool: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool, n_nodes: int, vehicle_limit: int, mandatory: typing.Set[int], time_limit: float, env: typing.Optional[gurobipy.Env] = None, sp_pool_cap: int = 50000, sp_mip_gap: float = 0.0001, seed: int = 42) -> typing.Tuple[list, float]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.run_sp_stage

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp.run_sp_stage
```
````
