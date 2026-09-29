# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LASMPipelineParams <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
    :summary:
    ```
````

### API

`````{py:class} LASMPipelineParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams

Bases: {py:obj}`logic.src.configs.policies.LASMPipelineConfig`

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
```

````{py:attribute} lbbd_cut_families
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_cut_families
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_cut_families
```

````

````{py:attribute} rl_state_features
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_state_features
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_state_features
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.__post_init__
```

````

````{py:method} stage_budgets(budget_override: typing.Optional[typing.Dict[str, float]] = None) -> typing.Tuple[float, float, float, float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.stage_budgets

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.stage_budgets
```

````

````{py:method} alns_iterations() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_iterations

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_iterations
```

````

````{py:method} bpc_ng_size() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size
```

````

````{py:method} bpc_max_bb_nodes() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes
```

````

````{py:method} as_alns_values_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.as_alns_values_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.as_alns_values_dict
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.to_dict
```

````

`````
