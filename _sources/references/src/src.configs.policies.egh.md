# {py:mod}`src.configs.policies.egh`

```{py:module} src.configs.policies.egh
```

```{autodoc2-docstring} src.configs.policies.egh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExactGuidedHeuristicConfig <src.configs.policies.egh.ExactGuidedHeuristicConfig>`
  - ```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig
    :summary:
    ```
````

### API

`````{py:class} ExactGuidedHeuristicConfig
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig
```

````{py:attribute} alpha
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alpha
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.seed
```

````

````{py:attribute} alns_max_iterations
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_max_iterations
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_max_iterations
```

````

````{py:attribute} alns_segment_size
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_segment_size
```

````

````{py:attribute} alns_reaction_factor
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_reaction_factor
```

````

````{py:attribute} alns_cooling_rate
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_cooling_rate
```

````

````{py:attribute} alns_start_temp_control
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_start_temp_control
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_start_temp_control
```

````

````{py:attribute} alns_sigma_1
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_1
```

````

````{py:attribute} alns_sigma_2
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_2
```

````

````{py:attribute} alns_sigma_3
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_sigma_3
```

````

````{py:attribute} alns_xi
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_xi
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_xi
```

````

````{py:attribute} alns_min_removal
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_min_removal
```

````

````{py:attribute} alns_noise_factor
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_noise_factor
```

````

````{py:attribute} alns_worst_removal_randomness
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_worst_removal_randomness
```

````

````{py:attribute} alns_shaw_randomization
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_shaw_randomization
```

````

````{py:attribute} alns_regret_pool
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_regret_pool
```

````

````{py:attribute} alns_extended_operators
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_extended_operators
```

````

````{py:attribute} alns_profit_aware_operators
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_profit_aware_operators
```

````

````{py:attribute} alns_vrpp
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_vrpp
```

````

````{py:attribute} alns_engine
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.alns_engine
```

````

````{py:attribute} bpc_ng_size_min
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_ng_size_min
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_ng_size_min
```

````

````{py:attribute} bpc_ng_size_max
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_ng_size_max
:type: int
:value: >
   16

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_ng_size_max
```

````

````{py:attribute} bpc_max_bb_nodes_min
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_max_bb_nodes_min
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_max_bb_nodes_min
```

````

````{py:attribute} bpc_max_bb_nodes_max
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_max_bb_nodes_max
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_max_bb_nodes_max
```

````

````{py:attribute} bpc_cutting_planes
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_cutting_planes
```

````

````{py:attribute} bpc_branching_strategy
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.bpc_branching_strategy
```

````

````{py:attribute} skip_bpc
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.skip_bpc
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.skip_bpc
```

````

````{py:attribute} sp_pool_cap
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.sp_pool_cap
:type: int
:value: >
   50000

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.sp_pool_cap
```

````

````{py:attribute} sp_mip_gap
:canonical: src.configs.policies.egh.ExactGuidedHeuristicConfig.sp_mip_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.egh.ExactGuidedHeuristicConfig.sp_mip_gap
```

````

`````
