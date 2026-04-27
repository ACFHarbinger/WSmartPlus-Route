# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bpc_stage <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.run_bpc_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.run_bpc_stage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.logger
```

````

````{py:function} run_bpc_stage(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, pipeline_params: src.policies.route_construction.matheuristics.exact_guided_heuristic.params.ExactGuidedHeuristicParams, mandatory_nodes: typing.Optional[typing.Set[int]], time_limit: float, incumbent: float = 0.0, pool: typing.Optional[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool] = None, vehicle_limit: typing.Optional[int] = None, env=None, recorder=None) -> typing.Tuple[typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute], float]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.run_bpc_stage

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc.run_bpc_stage
```
````
