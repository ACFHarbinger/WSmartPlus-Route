# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_select_nodes_knapsack <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._select_nodes_knapsack>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._select_nodes_knapsack
    :summary:
    ```
* - {py:obj}`run_bpc <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.run_bpc>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.run_bpc
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.logger
    :summary:
    ```
* - {py:obj}`_FARKAS_TOL <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._FARKAS_TOL>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._FARKAS_TOL
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.logger
```

````

````{py:data} _FARKAS_TOL
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._FARKAS_TOL
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._FARKAS_TOL
```

````

````{py:function} _select_nodes_knapsack(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], n_nodes: int, vehicle_limit: typing.Optional[int] = None, target_reduction: float = 0.6, time_limit: float = 10.0, env: typing.Optional[typing.Any] = None) -> typing.Set[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._select_nodes_knapsack

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine._select_nodes_knapsack
```
````

````{py:function} run_bpc(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[typing.Union[src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params.BPCParams, typing.Dict[str, typing.Any]]] = None, mandatory_indices: typing.Optional[typing.Set[int]] = None, vehicle_limit: typing.Optional[int] = None, env: typing.Optional[typing.Any] = None, node_coords: typing.Optional[numpy.ndarray] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.run_bpc

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine.run_bpc
```
````
