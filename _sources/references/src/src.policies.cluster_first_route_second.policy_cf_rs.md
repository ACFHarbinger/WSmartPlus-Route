# {py:mod}`src.policies.cluster_first_route_second.policy_cf_rs`

```{py:module} src.policies.cluster_first_route_second.policy_cf_rs
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClusterFirstRouteSecondPolicy <src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy
    :summary:
    ```
````

### API

`````{py:class} ClusterFirstRouteSecondPolicy(config: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._get_config_key

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._run_solver

```{autodoc2-docstring} src.policies.cluster_first_route_second.policy_cf_rs.ClusterFirstRouteSecondPolicy._run_solver
```

````

`````
