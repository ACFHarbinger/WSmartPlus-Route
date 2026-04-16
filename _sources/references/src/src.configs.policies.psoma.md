# {py:mod}`src.configs.policies.psoma`

```{py:module} src.configs.policies.psoma
```

```{autodoc2-docstring} src.configs.policies.psoma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMAConfig <src.configs.policies.psoma.PSOMAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig
    :summary:
    ```
````

### API

`````{py:class} PSOMAConfig
:canonical: src.configs.policies.psoma.PSOMAConfig

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.psoma.PSOMAConfig.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.pop_size
```

````

````{py:attribute} omega
:canonical: src.configs.policies.psoma.PSOMAConfig.omega
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.omega
```

````

````{py:attribute} c1
:canonical: src.configs.policies.psoma.PSOMAConfig.c1
:type: float
:value: >
   1.5

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.c1
```

````

````{py:attribute} c2
:canonical: src.configs.policies.psoma.PSOMAConfig.c2
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.c2
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.psoma.PSOMAConfig.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.max_iterations
```

````

````{py:attribute} local_search_freq
:canonical: src.configs.policies.psoma.PSOMAConfig.local_search_freq
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.local_search_freq
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.psoma.PSOMAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.psoma.PSOMAConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.psoma.PSOMAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.psoma.PSOMAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.psoma.PSOMAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.psoma.PSOMAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.psoma.PSOMAConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.psoma.PSOMAConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.route_improvement
```

````

`````
