# {py:mod}`src.policies.adapters.policy_sisr`

```{py:module} src.policies.adapters.policy_sisr
```

```{autodoc2-docstring} src.policies.adapters.policy_sisr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRPolicy <src.policies.adapters.policy_sisr.SISRPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_sisr.SISRPolicy
    :summary:
    ```
````

### API

`````{py:class} SISRPolicy(config: typing.Optional[logic.src.configs.policies.SISRConfig] = None)
:canonical: src.policies.adapters.policy_sisr.SISRPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_sisr.SISRPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_sisr.SISRPolicy.__init__
```

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_sisr.SISRPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_demands: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.adapters.policy_sisr.SISRPolicy._run_solver

````

`````
