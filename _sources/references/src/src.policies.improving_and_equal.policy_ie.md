# {py:mod}`src.policies.improving_and_equal.policy_ie`

```{py:module} src.policies.improving_and_equal.policy_ie
```

```{autodoc2-docstring} src.policies.improving_and_equal.policy_ie
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovingAndEqualPolicy <src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy>`
  - ```{autodoc2-docstring} src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy
    :summary:
    ```
````

### API

`````{py:class} ImprovingAndEqualPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ie.IEConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.improving_and_equal.policy_ie.ImprovingAndEqualPolicy._run_solver

````

`````
