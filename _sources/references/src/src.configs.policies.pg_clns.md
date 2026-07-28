# {py:mod}`src.configs.policies.pg_clns`

```{py:module} src.configs.policies.pg_clns
```

```{autodoc2-docstring} src.configs.policies.pg_clns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PGCLNSConfig <src.configs.policies.pg_clns.PGCLNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig
    :summary:
    ```
````

### API

`````{py:class} PGCLNSConfig
:canonical: src.configs.policies.pg_clns.PGCLNSConfig

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig
```

````{py:attribute} population_size
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.population_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.population_size
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.max_iterations
```

````

````{py:attribute} replacement_rate
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.replacement_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.replacement_rate
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.seed
```

````

````{py:attribute} aco
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.aco
:type: src.configs.policies.aco_ks.KSparseACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.aco
```

````

````{py:attribute} lns
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.lns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.lns
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.mandatory_selection
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.pg_clns.PGCLNSConfig.route_improvement
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.pg_clns.PGCLNSConfig.route_improvement
```

````

`````
