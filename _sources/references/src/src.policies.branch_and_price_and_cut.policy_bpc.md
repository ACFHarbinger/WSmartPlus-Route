# {py:mod}`src.policies.branch_and_price_and_cut.policy_bpc`

```{py:module} src.policies.branch_and_price_and_cut.policy_bpc
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPCPolicy <src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy
    :summary:
    ```
````

### API

`````{py:class} BPCPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.BPCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._get_config_key

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._get_config_key
```

````

````{py:method} _load_model(path: str, model_type: str) -> None
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._load_model

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._load_model
```

````

````{py:method} _predict_V(features: numpy.ndarray, model_type: str) -> numpy.ndarray
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._predict_V

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._predict_V
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, typing.Any]
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy.execute

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy.execute
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._run_solver

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.policy_bpc.BPCPolicy._run_solver
```

````

`````
