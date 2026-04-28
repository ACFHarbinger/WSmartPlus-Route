# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MSBPCSPPolicy <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy
    :summary:
    ```
````

### API

`````{py:class} MSBPCSPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.MSBPCSPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.policy_ms_bpc_sp.MSBPCSPPolicy._run_solver
```

````

`````
