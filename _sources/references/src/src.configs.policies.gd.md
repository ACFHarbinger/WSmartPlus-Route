# {py:mod}`src.configs.policies.gd`

```{py:module} src.configs.policies.gd
```

```{autodoc2-docstring} src.configs.policies.gd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GDConfig <src.configs.policies.gd.GDConfig>`
  - ```{autodoc2-docstring} src.configs.policies.gd.GDConfig
    :summary:
    ```
````

### API

`````{py:class} GDConfig
:canonical: src.configs.policies.gd.GDConfig

```{autodoc2-docstring} src.configs.policies.gd.GDConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.gd.GDConfig.engine
:type: str
:value: >
   'gd'

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.gd.GDConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.max_iterations
```

````

````{py:attribute} target_fitness_multiplier
:canonical: src.configs.policies.gd.GDConfig.target_fitness_multiplier
:type: float
:value: >
   1.1

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.target_fitness_multiplier
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.gd.GDConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.gd.GDConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.gd.GDConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.gd.GDConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.gd.GDConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.gd.GDConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.gd.GDConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.gd.GDConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.gd.GDConfig.post_processing
```

````

`````
