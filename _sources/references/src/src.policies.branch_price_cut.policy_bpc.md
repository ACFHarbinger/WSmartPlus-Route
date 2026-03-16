# {py:mod}`src.policies.branch_price_cut.policy_bpc`

```{py:module} src.policies.branch_price_cut.policy_bpc
```

```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BCPPolicy <src.policies.branch_price_cut.policy_bpc.BCPPolicy>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc.BCPPolicy
    :summary:
    ```
````

### API

`````{py:class} BCPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.BPCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.branch_price_cut.policy_bpc.BCPPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc.BCPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc.BCPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.branch_price_cut.policy_bpc.BCPPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.branch_price_cut.policy_bpc.BCPPolicy._get_config_key

```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc.BCPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.branch_price_cut.policy_bpc.BCPPolicy._run_solver

```{autodoc2-docstring} src.policies.branch_price_cut.policy_bpc.BCPPolicy._run_solver
```

````

`````
