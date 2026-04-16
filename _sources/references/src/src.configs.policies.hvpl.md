# {py:mod}`src.configs.policies.hvpl`

```{py:module} src.configs.policies.hvpl
```

```{autodoc2-docstring} src.configs.policies.hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HVPLConfig <src.configs.policies.hvpl.HVPLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig
    :summary:
    ```
````

### API

`````{py:class} HVPLConfig
:canonical: src.configs.policies.hvpl.HVPLConfig

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig
```

````{py:attribute} n_teams
:canonical: src.configs.policies.hvpl.HVPLConfig.n_teams
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.n_teams
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.hvpl.HVPLConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.max_iterations
```

````

````{py:attribute} substitution_rate
:canonical: src.configs.policies.hvpl.HVPLConfig.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hvpl.HVPLConfig.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hvpl.HVPLConfig.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.mutation_rate
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.hvpl.HVPLConfig.elite_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.elite_size
```

````

````{py:attribute} aco_init_iterations
:canonical: src.configs.policies.hvpl.HVPLConfig.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hvpl.HVPLConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hvpl.HVPLConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.seed
```

````

````{py:attribute} aco
:canonical: src.configs.policies.hvpl.HVPLConfig.aco
:type: src.configs.policies.aco_ks.KSparseACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.aco
```

````

````{py:attribute} alns
:canonical: src.configs.policies.hvpl.HVPLConfig.alns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.alns
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hvpl.HVPLConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.hvpl.HVPLConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.hvpl.HVPLConfig.mandatory_selection
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.hvpl.HVPLConfig.route_improvement
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hvpl.HVPLConfig.route_improvement
```

````

`````
