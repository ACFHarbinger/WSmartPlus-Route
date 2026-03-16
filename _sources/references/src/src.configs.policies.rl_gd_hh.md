# {py:mod}`src.configs.policies.rl_gd_hh`

```{py:module} src.configs.policies.rl_gd_hh
```

```{autodoc2-docstring} src.configs.policies.rl_gd_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLGDHHConfig <src.configs.policies.rl_gd_hh.RLGDHHConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig
    :summary:
    ```
````

### API

`````{py:class} RLGDHHConfig
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.engine
:type: str
:value: >
   'rl_gd_hh'

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.time_limit
```

````

````{py:attribute} reward_improvement
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.reward_improvement
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.reward_improvement
```

````

````{py:attribute} penalty_worsening
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.penalty_worsening
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.penalty_worsening
```

````

````{py:attribute} utility_upper_bound
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.utility_upper_bound
:type: float
:value: >
   40.0

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.utility_upper_bound
```

````

````{py:attribute} min_utility
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.min_utility
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.min_utility
```

````

````{py:attribute} target_fitness_multiplier
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.target_fitness_multiplier
:type: float
:value: >
   1.2

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.target_fitness_multiplier
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.rl_gd_hh.RLGDHHConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_gd_hh.RLGDHHConfig.post_processing
```

````

`````
