# {py:mod}`src.policies.slack_induction_by_string_removal.policy_sisr`

```{py:module} src.policies.slack_induction_by_string_removal.policy_sisr
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.policy_sisr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRPolicy <src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy
    :summary:
    ```
````

### API

`````{py:class} SISRPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.SISRConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy._run_solver

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.policy_sisr.SISRPolicy._run_solver
```

````

`````
