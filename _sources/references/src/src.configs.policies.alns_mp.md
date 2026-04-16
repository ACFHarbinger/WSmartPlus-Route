# {py:mod}`src.configs.policies.alns_mp`

```{py:module} src.configs.policies.alns_mp
```

```{autodoc2-docstring} src.configs.policies.alns_mp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSMPConfig <src.configs.policies.alns_mp.ALNSMPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig
    :summary:
    ```
````

### API

`````{py:class} ALNSMPConfig
:canonical: src.configs.policies.alns_mp.ALNSMPConfig

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.start_temp
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.max_removal_pct
```

````

````{py:attribute} max_removal_cap
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.max_removal_cap
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.max_removal_cap
```

````

````{py:attribute} segment_size
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.segment_size
```

````

````{py:attribute} noise_factor
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.noise_factor
```

````

````{py:attribute} worst_removal_randomness
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.worst_removal_randomness
```

````

````{py:attribute} shaw_randomization
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.shaw_randomization
```

````

````{py:attribute} regret_pool
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.regret_pool
```

````

````{py:attribute} sigma_1
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.sigma_1
```

````

````{py:attribute} sigma_2
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.sigma_2
```

````

````{py:attribute} sigma_3
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.sigma_3
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.profit_aware_operators
```

````

````{py:attribute} extended_operators
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.extended_operators
```

````

````{py:attribute} seed
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.seed
```

````

````{py:attribute} engine
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.engine
```

````

````{py:attribute} horizon
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.horizon
```

````

````{py:attribute} stockout_penalty
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.stockout_penalty
```

````

````{py:attribute} forward_looking_depth
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.forward_looking_depth
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.forward_looking_depth
```

````

````{py:attribute} inter_period_operators
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.inter_period_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.inter_period_operators
```

````

````{py:attribute} shift_direction
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.shift_direction
:type: str
:value: >
   'both'

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.shift_direction
```

````

````{py:attribute} inventory_lambda
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.inventory_lambda
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.inventory_lambda
```

````

````{py:attribute} inter_period_weight
:canonical: src.configs.policies.alns_mp.ALNSMPConfig.inter_period_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.alns_mp.ALNSMPConfig.inter_period_weight
```

````

`````
