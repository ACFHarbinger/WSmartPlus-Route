# {py:mod}`src.policies.ensemble_move_acceptance.policy_ema`

```{py:module} src.policies.ensemble_move_acceptance.policy_ema
```

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.policy_ema
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnsembleMoveAcceptancePolicy <src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy>`
  - ```{autodoc2-docstring} src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy
    :summary:
    ```
````

### API

`````{py:class} EnsembleMoveAcceptancePolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ema.EMAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ensemble_move_acceptance.policy_ema.EnsembleMoveAcceptancePolicy._run_solver

````

`````
