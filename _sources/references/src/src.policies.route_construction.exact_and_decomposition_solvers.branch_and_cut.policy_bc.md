# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndCutPolicy <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy
    :summary:
    ```
````

### API

`````{py:class} BranchAndCutPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.BCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.policy_bc.BranchAndCutPolicy._run_solver
```

````

`````
