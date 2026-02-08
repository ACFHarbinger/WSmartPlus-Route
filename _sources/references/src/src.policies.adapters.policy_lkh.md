# {py:mod}`src.policies.adapters.policy_lkh`

```{py:module} src.policies.adapters.policy_lkh
```

```{autodoc2-docstring} src.policies.adapters.policy_lkh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LKHPolicy <src.policies.adapters.policy_lkh.LKHPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy
    :summary:
    ```
````

### API

`````{py:class} LKHPolicy(config: typing.Optional[logic.src.configs.policies.LKHConfig] = None)
:canonical: src.policies.adapters.policy_lkh.LKHPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy.__init__
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_lkh.LKHPolicy._get_config_key

```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_lkh.LKHPolicy._run_solver

```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.policy_lkh.LKHPolicy.execute

```{autodoc2-docstring} src.policies.adapters.policy_lkh.LKHPolicy.execute
```

````

`````
