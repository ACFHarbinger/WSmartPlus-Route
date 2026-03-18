# {py:mod}`src.policies.differential_evolution.policy_de`

```{py:module} src.policies.differential_evolution.policy_de
```

```{autodoc2-docstring} src.policies.differential_evolution.policy_de
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEPolicyAdapter <src.policies.differential_evolution.policy_de.DEPolicyAdapter>`
  - ```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter
    :summary:
    ```
````

### API

`````{py:class} DEPolicyAdapter(config: typing.Optional[typing.Union[logic.src.configs.policies.de.DEConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.differential_evolution.policy_de.DEPolicyAdapter._run_solver

```{autodoc2-docstring} src.policies.differential_evolution.policy_de.DEPolicyAdapter._run_solver
```

````

`````
