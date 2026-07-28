# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LASMPipelineParams <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
    :summary:
    ```
````

### API

`````{py:class} LASMPipelineParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
```

````{py:attribute} alpha
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alpha
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.seed
```

````

````{py:attribute} lbbd_max_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_max_iterations
```

````

````{py:attribute} lbbd_master_time_frac
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_master_time_frac
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_master_time_frac
```

````

````{py:attribute} lbbd_sub_solver
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_sub_solver
:type: str
:value: >
   'alns'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_sub_solver
```

````

````{py:attribute} lbbd_cut_families
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_cut_families
:type: typing.List[str]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_cut_families
```

````

````{py:attribute} lbbd_pareto_eps
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_pareto_eps
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_pareto_eps
```

````

````{py:attribute} lbbd_sub_time_frac
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_sub_time_frac
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_sub_time_frac
```

````

````{py:attribute} lbbd_min_cover_ratio
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_min_cover_ratio
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_min_cover_ratio
```

````

````{py:attribute} lbbd_use_warm_cuts
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_use_warm_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.lbbd_use_warm_cuts
```

````

````{py:attribute} alns_max_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_max_iterations
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_max_iterations
```

````

````{py:attribute} alns_segment_size
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_segment_size
```

````

````{py:attribute} alns_reaction_factor
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_reaction_factor
```

````

````{py:attribute} alns_cooling_rate
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_cooling_rate
```

````

````{py:attribute} alns_start_temp_control
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_start_temp_control
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_start_temp_control
```

````

````{py:attribute} alns_sigma_1
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_1
```

````

````{py:attribute} alns_sigma_2
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_2
```

````

````{py:attribute} alns_sigma_3
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_sigma_3
```

````

````{py:attribute} alns_xi
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_xi
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_xi
```

````

````{py:attribute} alns_min_removal
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_min_removal
```

````

````{py:attribute} alns_noise_factor
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_noise_factor
```

````

````{py:attribute} alns_worst_removal_randomness
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_worst_removal_randomness
```

````

````{py:attribute} alns_shaw_randomization
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_shaw_randomization
```

````

````{py:attribute} alns_regret_pool
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_regret_pool
```

````

````{py:attribute} alns_extended_operators
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_extended_operators
```

````

````{py:attribute} alns_profit_aware_operators
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_profit_aware_operators
```

````

````{py:attribute} alns_vrpp
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_vrpp
```

````

````{py:attribute} alns_engine
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_engine
```

````

````{py:attribute} bpc_ng_size_min
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size_min
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size_min
```

````

````{py:attribute} bpc_ng_size_max
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size_max
:type: int
:value: >
   16

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size_max
```

````

````{py:attribute} bpc_max_bb_nodes_min
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes_min
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes_min
```

````

````{py:attribute} bpc_max_bb_nodes_max
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes_max
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes_max
```

````

````{py:attribute} bpc_cutting_planes
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_cutting_planes
```

````

````{py:attribute} bpc_branching_strategy
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_branching_strategy
```

````

````{py:attribute} skip_bpc
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.skip_bpc
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.skip_bpc
```

````

````{py:attribute} rl_mode
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_mode
:type: str
:value: >
   'online'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_mode
```

````

````{py:attribute} rl_policy_path
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_policy_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_policy_path
```

````

````{py:attribute} rl_exploration
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_exploration
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_exploration
```

````

````{py:attribute} rl_window
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_window
```

````

````{py:attribute} rl_min_samples
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_min_samples
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_min_samples
```

````

````{py:attribute} rl_reward_shaping
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_reward_shaping
:type: str
:value: >
   'efficiency'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_reward_shaping
```

````

````{py:attribute} rl_state_features
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_state_features
:type: typing.List[str]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_state_features
```

````

````{py:attribute} rl_action_space
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_action_space
:type: str
:value: >
   'budgets'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_action_space
```

````

````{py:attribute} rl_discount
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_discount
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.rl_discount
```

````

````{py:attribute} sp_pool_cap
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.sp_pool_cap
:type: int
:value: >
   50000

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.sp_pool_cap
```

````

````{py:attribute} sp_mip_gap
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.sp_mip_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.sp_mip_gap
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.__post_init__
```

````

````{py:method} stage_budgets(budget_override: typing.Optional[typing.Dict[str, float]] = None) -> typing.Tuple[float, float, float, float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.stage_budgets

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.stage_budgets
```

````

````{py:method} alns_iterations() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_iterations

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.alns_iterations
```

````

````{py:method} bpc_ng_size() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_ng_size
```

````

````{py:method} bpc_max_bb_nodes() -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.bpc_max_bb_nodes
```

````

````{py:method} as_alns_values_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.as_alns_values_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.as_alns_values_dict
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params.LASMPipelineParams.to_dict
```

````

`````
