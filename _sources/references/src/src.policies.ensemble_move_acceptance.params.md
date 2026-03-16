# {py:mod}`src.policies.ensemble_move_acceptance.params`

```{py:module} src.policies.ensemble_move_acceptance.params
```

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EMAParams <src.policies.ensemble_move_acceptance.params.EMAParams>`
  - ```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams
    :summary:
    ```
````

### API

`````{py:class} EMAParams
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams
```

````{py:attribute} max_iterations
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.max_iterations
```

````

````{py:attribute} rule
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.rule
:type: str
:value: >
   'G-VOT'

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.rule
```

````

````{py:attribute} criteria
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.criteria
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.criteria
```

````

````{py:attribute} sub_params
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.sub_params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.sub_params
```

````

````{py:attribute} n_removal
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.ensemble_move_acceptance.params.EMAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.params.EMAParams.time_limit
```

````

`````
