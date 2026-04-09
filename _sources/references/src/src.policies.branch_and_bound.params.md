# {py:mod}`src.policies.branch_and_bound.params`

```{py:module} src.policies.branch_and_bound.params
```

```{autodoc2-docstring} src.policies.branch_and_bound.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBParams <src.policies.branch_and_bound.params.BBParams>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams
    :summary:
    ```
````

### API

`````{py:class} BBParams
:canonical: src.policies.branch_and_bound.params.BBParams

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams
```

````{py:attribute} time_limit
:canonical: src.policies.branch_and_bound.params.BBParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.branch_and_bound.params.BBParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.mip_gap
```

````

````{py:attribute} seed
:canonical: src.policies.branch_and_bound.params.BBParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.seed
```

````

````{py:attribute} branching_strategy
:canonical: src.policies.branch_and_bound.params.BBParams.branching_strategy
:type: str
:value: >
   'strong'

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.branching_strategy
```

````

````{py:attribute} strong_branching_limit
:canonical: src.policies.branch_and_bound.params.BBParams.strong_branching_limit
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.strong_branching_limit
```

````

````{py:attribute} formulation
:canonical: src.policies.branch_and_bound.params.BBParams.formulation
:type: str
:value: >
   'dfj'

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.formulation
```

````

````{py:method} from_config(config: typing.Dict[str, typing.Any]) -> src.policies.branch_and_bound.params.BBParams
:canonical: src.policies.branch_and_bound.params.BBParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_bound.params.BBParams.from_config
```

````

`````
