# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_LRNode <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_select_branch_customer <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._select_branch_customer>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._select_branch_customer
    :summary:
    ```
* - {py:obj}`run_bb_lr_uop <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop.run_bb_lr_uop>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop.run_bb_lr_uop
    :summary:
    ```
* - {py:obj}`_visited_to_routes <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._visited_to_routes>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._visited_to_routes
    :summary:
    ```
````

### API

`````{py:class} _LRNode
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode
```

````{py:attribute} bound
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.bound
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.bound
```

````

````{py:attribute} forced_in
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.forced_in
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.forced_in
```

````

````{py:attribute} forced_out
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.forced_out
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.forced_out
```

````

````{py:attribute} depth
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.depth
:type: int
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._LRNode.depth
```

````

`````

````{py:function} _select_branch_customer(op_visited: typing.Set[int], forced_in: typing.Set[int], forced_out: typing.Set[int], wastes: typing.Dict[int, float], R: float, lam: float, strategy: str) -> typing.Optional[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._select_branch_customer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._select_branch_customer
```
````

````{py:function} run_bb_lr_uop(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams] = None, mandatory_indices: typing.Optional[typing.Set[int]] = None, env: typing.Any = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop.run_bb_lr_uop

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop.run_bb_lr_uop
```
````

````{py:function} _visited_to_routes(visited: typing.Set[int], dist_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._visited_to_routes

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.lr_uop._visited_to_routes
```
````
