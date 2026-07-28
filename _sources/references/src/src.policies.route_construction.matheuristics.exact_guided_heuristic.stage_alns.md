# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_PoolHarvestingALNS <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_alns_stage <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.run_alns_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.run_alns_stage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.logger
```

````

`````{py:class} _PoolHarvestingALNS(*args, pool: src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool, **kwargs)
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS

Bases: {py:obj}`logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS.__init__
```

````{py:method} _update_weights(d_idx: int, r_idx: int, score: float) -> None
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._update_weights

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._update_weights
```

````

````{py:method} _record_routes(routes: typing.List[typing.List[int]]) -> None
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._record_routes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._record_routes
```

````

````{py:method} _select_and_apply_operators(current_routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], int, int]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._select_and_apply_operators

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns._PoolHarvestingALNS._select_and_apply_operators
```

````

`````

````{py:function} run_alns_stage(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params.ALNSParams, mandatory_nodes: typing.Optional[typing.List[int]], time_limit: float, initial_routes: typing.Optional[typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]] = None, pool: typing.Optional[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool] = None, recorder=None) -> typing.Tuple[typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute], float]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.run_alns_stage

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns.run_alns_stage
```
````
