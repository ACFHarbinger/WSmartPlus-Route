# {py:mod}`src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf`

```{py:module} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_route <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._build_route>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._build_route
    :summary:
    ```
* - {py:obj}`_decompose_arcs <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._decompose_arcs>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._decompose_arcs
    :summary:
    ```
* - {py:obj}`run_tcf_stage <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.run_tcf_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.run_tcf_stage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.logger
```

````

````{py:function} _build_route(nodes: typing.List[int], dist: typing.List[typing.List[float]], S: typing.Dict[int, float], R: float, C: float, source: str = 'tcf') -> src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._build_route

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._build_route
```
````

````{py:function} _decompose_arcs(arcs: typing.List[typing.Tuple[int, int]], dist: typing.List[typing.List[float]], S: typing.Dict[int, float], R: float, C: float) -> typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._decompose_arcs

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf._decompose_arcs
```
````

````{py:function} run_tcf_stage(bins: numpy.typing.NDArray[numpy.float64], dist: typing.List[typing.List[float]], env: typing.Optional[gurobipy.Env], values: typing.Dict, binsids: typing.List[int], mandatory: typing.List[int], n_vehicles: int, time_limit: float, seed: int = 42, dual_values: typing.Optional[typing.Dict[int, float]] = None, pool: typing.Optional[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool] = None) -> typing.Tuple[typing.List[src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute], float, typing.Optional[gurobipy.Model]]
:canonical: src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.run_tcf_stage

```{autodoc2-docstring} src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_tcf.run_tcf_stage
```
````
