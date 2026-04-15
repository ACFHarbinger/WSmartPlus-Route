# {py:mod}`src.configs.policies.lkh3`

```{py:module} src.configs.policies.lkh3
```

```{autodoc2-docstring} src.configs.policies.lkh3
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LKH3Config <src.configs.policies.lkh3.LKH3Config>`
  - ```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config
    :summary:
    ```
````

### API

`````{py:class} LKH3Config
:canonical: src.configs.policies.lkh3.LKH3Config

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config
```

````{py:attribute} runs
:canonical: src.configs.policies.lkh3.LKH3Config.runs
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.runs
```

````

````{py:attribute} max_trials
:canonical: src.configs.policies.lkh3.LKH3Config.max_trials
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.max_trials
```

````

````{py:attribute} popmusic_subpath_size
:canonical: src.configs.policies.lkh3.LKH3Config.popmusic_subpath_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.popmusic_subpath_size
```

````

````{py:attribute} popmusic_trials
:canonical: src.configs.policies.lkh3.LKH3Config.popmusic_trials
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.popmusic_trials
```

````

````{py:attribute} popmusic_max_candidates
:canonical: src.configs.policies.lkh3.LKH3Config.popmusic_max_candidates
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.popmusic_max_candidates
```

````

````{py:attribute} max_k_opt
:canonical: src.configs.policies.lkh3.LKH3Config.max_k_opt
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.max_k_opt
```

````

````{py:attribute} use_ip_merging
:canonical: src.configs.policies.lkh3.LKH3Config.use_ip_merging
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.use_ip_merging
```

````

````{py:attribute} max_pool_size
:canonical: src.configs.policies.lkh3.LKH3Config.max_pool_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.max_pool_size
```

````

````{py:attribute} subgradient_iterations
:canonical: src.configs.policies.lkh3.LKH3Config.subgradient_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.subgradient_iterations
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.lkh3.LKH3Config.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.profit_aware_operators
```

````

````{py:attribute} alns_iterations
:canonical: src.configs.policies.lkh3.LKH3Config.alns_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.alns_iterations
```

````

````{py:attribute} plateau_limit
:canonical: src.configs.policies.lkh3.LKH3Config.plateau_limit
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.plateau_limit
```

````

````{py:attribute} deep_plateau_limit
:canonical: src.configs.policies.lkh3.LKH3Config.deep_plateau_limit
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.deep_plateau_limit
```

````

````{py:attribute} perturb_operator_weights
:canonical: src.configs.policies.lkh3.LKH3Config.perturb_operator_weights
:type: typing.List[float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.perturb_operator_weights
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.lkh3.LKH3Config.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.lkh3.LKH3Config.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.vrpp
```

````

````{py:attribute} dynamic_topology_discovery
:canonical: src.configs.policies.lkh3.LKH3Config.dynamic_topology_discovery
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.dynamic_topology_discovery
```

````

````{py:attribute} native_prize_collecting
:canonical: src.configs.policies.lkh3.LKH3Config.native_prize_collecting
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.native_prize_collecting
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lkh3.LKH3Config.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.lkh3.LKH3Config.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.lkh3.LKH3Config.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lkh3.LKH3Config.post_processing
```

````

`````
