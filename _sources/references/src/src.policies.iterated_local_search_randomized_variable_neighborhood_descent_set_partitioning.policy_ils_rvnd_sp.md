# {py:mod}`src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp`

```{py:module} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp
```

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSRVNSSPPolicy <src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy>`
  - ```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy
    :summary:
    ```
````

### API

`````{py:class} ILSRVNSSPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ILSRVNDSPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy._get_config_key

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy._run_solver

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNSSPPolicy._run_solver
```

````

`````
