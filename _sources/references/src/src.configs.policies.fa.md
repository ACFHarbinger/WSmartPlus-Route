# {py:mod}`src.configs.policies.fa`

```{py:module} src.configs.policies.fa
```

```{autodoc2-docstring} src.configs.policies.fa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FAConfig <src.configs.policies.fa.FAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.fa.FAConfig
    :summary:
    ```
````

### API

`````{py:class} FAConfig
:canonical: src.configs.policies.fa.FAConfig

```{autodoc2-docstring} src.configs.policies.fa.FAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.fa.FAConfig.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.pop_size
```

````

````{py:attribute} beta0
:canonical: src.configs.policies.fa.FAConfig.beta0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.beta0
```

````

````{py:attribute} gamma
:canonical: src.configs.policies.fa.FAConfig.gamma
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.gamma
```

````

````{py:attribute} alpha_profit
:canonical: src.configs.policies.fa.FAConfig.alpha_profit
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.alpha_profit
```

````

````{py:attribute} beta_will
:canonical: src.configs.policies.fa.FAConfig.beta_will
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.beta_will
```

````

````{py:attribute} gamma_cost
:canonical: src.configs.policies.fa.FAConfig.gamma_cost
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.gamma_cost
```

````

````{py:attribute} alpha_rnd
:canonical: src.configs.policies.fa.FAConfig.alpha_rnd
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.alpha_rnd
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.fa.FAConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.fa.FAConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.n_removal
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.fa.FAConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.fa.FAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.fa.FAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.fa.FAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.fa.FAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.fa.FAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.fa.FAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.fa.FAConfig.post_processing
```

````

`````
