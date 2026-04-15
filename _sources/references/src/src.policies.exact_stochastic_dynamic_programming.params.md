# {py:mod}`src.policies.exact_stochastic_dynamic_programming.params`

```{py:module} src.policies.exact_stochastic_dynamic_programming.params
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SDPParams <src.policies.exact_stochastic_dynamic_programming.params.SDPParams>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams
    :summary:
    ```
````

### API

`````{py:class} SDPParams
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams
```

````{py:attribute} time_limit
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.time_limit
:type: float
:value: >
   3600.0

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.time_limit
```

````

````{py:attribute} num_days
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.num_days
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.num_days
```

````

````{py:attribute} discrete_levels
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.discrete_levels
:type: int
:value: >
   4

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.discrete_levels
```

````

````{py:attribute} max_fill_rate
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.max_fill_rate
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.max_fill_rate
```

````

````{py:attribute} max_nodes
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.max_nodes
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.max_nodes
```

````

````{py:attribute} discount_factor
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.discount_factor
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.discount_factor
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.overflow_penalty
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.overflow_penalty
```

````

````{py:attribute} cost_weight
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.cost_weight
```

````

````{py:attribute} waste_weight
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.waste_weight
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.waste_weight
```

````

````{py:method} from_config(config: dict) -> src.policies.exact_stochastic_dynamic_programming.params.SDPParams
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.exact_stochastic_dynamic_programming.params.SDPParams.to_dict

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.params.SDPParams.to_dict
```

````

`````
