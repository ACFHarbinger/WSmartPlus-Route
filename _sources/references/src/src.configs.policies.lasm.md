# {py:mod}`src.configs.policies.lasm`

```{py:module} src.configs.policies.lasm
```

```{autodoc2-docstring} src.configs.policies.lasm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LASMPipelineConfig <src.configs.policies.lasm.LASMPipelineConfig>`
  - ```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig
    :summary:
    ```
````

### API

`````{py:class} LASMPipelineConfig
:canonical: src.configs.policies.lasm.LASMPipelineConfig

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig
```

````{py:attribute} alpha
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alpha
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.lasm.LASMPipelineConfig.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.lasm.LASMPipelineConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.seed
```

````

````{py:attribute} lbbd_max_iterations
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_max_iterations
```

````

````{py:attribute} lbbd_master_time_frac
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_master_time_frac
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_master_time_frac
```

````

````{py:attribute} lbbd_sub_solver
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_sub_solver
:type: str
:value: >
   'alns'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_sub_solver
```

````

````{py:attribute} lbbd_cut_families
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_cut_families
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_cut_families
```

````

````{py:attribute} lbbd_pareto_eps
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_pareto_eps
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_pareto_eps
```

````

````{py:attribute} lbbd_sub_time_frac
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_sub_time_frac
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_sub_time_frac
```

````

````{py:attribute} lbbd_min_cover_ratio
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_min_cover_ratio
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_min_cover_ratio
```

````

````{py:attribute} lbbd_use_warm_cuts
:canonical: src.configs.policies.lasm.LASMPipelineConfig.lbbd_use_warm_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.lbbd_use_warm_cuts
```

````

````{py:attribute} alns_max_iterations
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_max_iterations
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_max_iterations
```

````

````{py:attribute} alns_segment_size
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_segment_size
```

````

````{py:attribute} alns_reaction_factor
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_reaction_factor
```

````

````{py:attribute} alns_cooling_rate
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_cooling_rate
```

````

````{py:attribute} alns_start_temp_control
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_start_temp_control
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_start_temp_control
```

````

````{py:attribute} alns_sigma_1
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_1
```

````

````{py:attribute} alns_sigma_2
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_2
```

````

````{py:attribute} alns_sigma_3
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_sigma_3
```

````

````{py:attribute} alns_xi
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_xi
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_xi
```

````

````{py:attribute} alns_min_removal
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_min_removal
```

````

````{py:attribute} alns_noise_factor
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_noise_factor
```

````

````{py:attribute} alns_worst_removal_randomness
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_worst_removal_randomness
```

````

````{py:attribute} alns_shaw_randomization
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_shaw_randomization
```

````

````{py:attribute} alns_regret_pool
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_regret_pool
```

````

````{py:attribute} alns_extended_operators
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_extended_operators
```

````

````{py:attribute} alns_profit_aware_operators
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_profit_aware_operators
```

````

````{py:attribute} alns_vrpp
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_vrpp
```

````

````{py:attribute} alns_engine
:canonical: src.configs.policies.lasm.LASMPipelineConfig.alns_engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.alns_engine
```

````

````{py:attribute} bpc_ng_size_min
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_ng_size_min
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_ng_size_min
```

````

````{py:attribute} bpc_ng_size_max
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_ng_size_max
:type: int
:value: >
   16

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_ng_size_max
```

````

````{py:attribute} bpc_max_bb_nodes_min
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_max_bb_nodes_min
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_max_bb_nodes_min
```

````

````{py:attribute} bpc_max_bb_nodes_max
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_max_bb_nodes_max
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_max_bb_nodes_max
```

````

````{py:attribute} bpc_cutting_planes
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_cutting_planes
```

````

````{py:attribute} bpc_branching_strategy
:canonical: src.configs.policies.lasm.LASMPipelineConfig.bpc_branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.bpc_branching_strategy
```

````

````{py:attribute} skip_bpc
:canonical: src.configs.policies.lasm.LASMPipelineConfig.skip_bpc
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.skip_bpc
```

````

````{py:attribute} rl_mode
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_mode
:type: str
:value: >
   'online'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_mode
```

````

````{py:attribute} rl_policy_path
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_policy_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_policy_path
```

````

````{py:attribute} rl_exploration
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_exploration
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_exploration
```

````

````{py:attribute} rl_window
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_window
```

````

````{py:attribute} rl_min_samples
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_min_samples
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_min_samples
```

````

````{py:attribute} rl_reward_shaping
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_reward_shaping
:type: str
:value: >
   'efficiency'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_reward_shaping
```

````

````{py:attribute} rl_state_features
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_state_features
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_state_features
```

````

````{py:attribute} rl_action_space
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_action_space
:type: str
:value: >
   'budgets'

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_action_space
```

````

````{py:attribute} rl_discount
:canonical: src.configs.policies.lasm.LASMPipelineConfig.rl_discount
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.rl_discount
```

````

````{py:attribute} sp_pool_cap
:canonical: src.configs.policies.lasm.LASMPipelineConfig.sp_pool_cap
:type: int
:value: >
   50000

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.sp_pool_cap
```

````

````{py:attribute} sp_mip_gap
:canonical: src.configs.policies.lasm.LASMPipelineConfig.sp_mip_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.sp_mip_gap
```

````

````{py:method} __post_init__() -> None
:canonical: src.configs.policies.lasm.LASMPipelineConfig.__post_init__

```{autodoc2-docstring} src.configs.policies.lasm.LASMPipelineConfig.__post_init__
```

````

`````
