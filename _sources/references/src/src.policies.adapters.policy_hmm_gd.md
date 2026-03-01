# {py:mod}`src.policies.adapters.policy_hmm_gd`

```{py:module} src.policies.adapters.policy_hmm_gd
```

```{autodoc2-docstring} src.policies.adapters.policy_hmm_gd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDPolicy <src.policies.adapters.policy_hmm_gd.HMMGDPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_hmm_gd.HMMGDPolicy
    :summary:
    ```
````

### API

`````{py:class} HMMGDPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.hmm_gd.HMMGDConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.adapters.policy_hmm_gd.HMMGDPolicy

Bases: {py:obj}`logic.src.policies.adapters.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_hmm_gd.HMMGDPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adapters.policy_hmm_gd.HMMGDPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.adapters.policy_hmm_gd.HMMGDPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.adapters.policy_hmm_gd.HMMGDPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adapters.policy_hmm_gd.HMMGDPolicy._run_solver

````

`````
