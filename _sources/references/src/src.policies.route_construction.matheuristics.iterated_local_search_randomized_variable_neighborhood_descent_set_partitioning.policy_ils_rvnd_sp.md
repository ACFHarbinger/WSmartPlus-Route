# {py:mod}`src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp`

```{py:module} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSRVNDSPPolicy <src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy
    :summary:
    ```
````

### API

`````{py:class} ILSRVNDSPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ILSRVNDSPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPPolicy._run_solver
```

````

`````
