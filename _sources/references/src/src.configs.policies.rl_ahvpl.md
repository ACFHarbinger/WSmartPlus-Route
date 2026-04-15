# {py:mod}`src.configs.policies.rl_ahvpl`

```{py:module} src.configs.policies.rl_ahvpl
```

```{autodoc2-docstring} src.configs.policies.rl_ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLAHVPLConfig <src.configs.policies.rl_ahvpl.RLAHVPLConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig
    :summary:
    ```
````

### API

`````{py:class} RLAHVPLConfig
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig
```

````{py:attribute} n_teams
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.n_teams
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.n_teams
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.max_iterations
```

````

````{py:attribute} elite_coaching_max_iterations
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.elite_coaching_max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.elite_coaching_max_iterations
```

````

````{py:attribute} not_coached_max_iterations
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.not_coached_max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.not_coached_max_iterations
```

````

````{py:attribute} coaching_acceptance_threshold
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.coaching_acceptance_threshold
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.coaching_acceptance_threshold
```

````

````{py:attribute} sub_rate
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.sub_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.sub_rate
```

````

````{py:attribute} aco
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.aco
:type: src.configs.policies.aco_ks.KSparseACOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.aco
```

````

````{py:attribute} alns
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.alns
:type: src.configs.policies.alns.ALNSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.alns
```

````

````{py:attribute} hgs
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.hgs
:type: src.configs.policies.hgs.HGSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.hgs
```

````

````{py:attribute} rts
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.rts
:type: src.configs.policies.rts.RTSConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.rts
```

````

````{py:attribute} rl_config
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.rl_config
:type: src.configs.policies.other.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.rl_config
```

````

````{py:attribute} tabu_no_repeat_threshold
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.tabu_no_repeat_threshold
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.tabu_no_repeat_threshold
```

````

````{py:attribute} gls_penalty_lambda
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_lambda
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_lambda
```

````

````{py:attribute} gls_penalty_alpha
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_alpha
```

````

````{py:attribute} gls_penalty_step
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_step
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_penalty_step
```

````

````{py:attribute} gls_probability
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_probability
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.gls_probability
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.must_go
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.rl_ahvpl.RLAHVPLConfig.post_processing
:type: typing.List[typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rl_ahvpl.RLAHVPLConfig.post_processing
```

````

`````
