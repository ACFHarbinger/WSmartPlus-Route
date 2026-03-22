# {py:mod}`src.configs.policies.rl_hvpl`

```{py:module} src.configs.policies.rl_hvpl
```

```{autodoc2-docstring} src.configs.policies.rl_hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLHVPLConfig <src.configs.policies.rl_hvpl.RLHVPLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig
    :summary:
    ```
````

### API

`````{py:class} RLHVPLConfig
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.engine
:type: str
:value: >
   'rl_hvpl'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.engine
```

````

````{py:attribute} n_teams
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.n_teams
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.max_iterations
```

````

````{py:attribute} elite_coaching_iterations
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.elite_coaching_iterations
:type: int
:value: >
   300

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.elite_coaching_iterations
```

````

````{py:attribute} regular_coaching_iterations
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.regular_coaching_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.regular_coaching_iterations
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.elite_size
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.elite_size
```

````

````{py:attribute} sub_rate
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.sub_rate
```

````

````{py:attribute} pheromone_update_strategy
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.pheromone_update_strategy
:type: str
:value: >
   'profit'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.pheromone_update_strategy
```

````

````{py:attribute} profit_weight
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.profit_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.profit_weight
```

````

````{py:attribute} aco
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.aco
:type: src.configs.policies.aco_ks.KSparseACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.aco
```

````

````{py:attribute} alns
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.alns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.alns
```

````

````{py:attribute} rl_config
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.rl_config
:type: src.configs.policies.other.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.rl_config
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.must_go
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.rl_hvpl.RLHVPLConfig.post_processing
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_hvpl.RLHVPLConfig.post_processing
```

````

`````
