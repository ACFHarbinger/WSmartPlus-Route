# {py:mod}`src.policies.constraint_programming_with_boolean_satisfiability.params`

```{py:module} src.policies.constraint_programming_with_boolean_satisfiability.params
```

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CPSATParams <src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams>`
  - ```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams
    :summary:
    ```
````

### API

`````{py:class} CPSATParams
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams
```

````{py:attribute} num_days
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.num_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.num_days
```

````

````{py:attribute} time_limit
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.time_limit
```

````

````{py:attribute} search_workers
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.search_workers
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.search_workers
```

````

````{py:attribute} mip_gap
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.mip_gap
```

````

````{py:attribute} scaling_factor
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.scaling_factor
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.scaling_factor
```

````

````{py:attribute} waste_weight
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.waste_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.waste_weight
```

````

````{py:attribute} cost_weight
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.cost_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.cost_weight
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.overflow_penalty
```

````

````{py:attribute} mean_increment
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.mean_increment
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.mean_increment
```

````

````{py:attribute} seed
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.seed
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.to_dict

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.params.CPSATParams.to_dict
```

````

`````
