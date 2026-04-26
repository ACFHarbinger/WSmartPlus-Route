# {py:mod}`src.configs.policies.abpc_hg`

```{py:module} src.configs.policies.abpc_hg
```

```{autodoc2-docstring} src.configs.policies.abpc_hg
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABPCHGConfig <src.configs.policies.abpc_hg.ABPCHGConfig>`
  - ```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig
    :summary:
    ```
````

### API

`````{py:class} ABPCHGConfig
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig
```

````{py:attribute} gamma
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.gamma
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.gamma
```

````

````{py:attribute} seed
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.seed
```

````

````{py:attribute} overflow_penalty
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.overflow_penalty
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.overflow_penalty
```

````

````{py:attribute} ph_base_rho
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.ph_base_rho
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.ph_base_rho
```

````

````{py:attribute} ph_max_iterations
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.ph_max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.ph_max_iterations
```

````

````{py:attribute} ph_convergence_tol
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.ph_convergence_tol
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.ph_convergence_tol
```

````

````{py:attribute} alns_iterations
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.alns_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.alns_iterations
```

````

````{py:attribute} alns_max_routes
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.alns_max_routes
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.alns_max_routes
```

````

````{py:attribute} alns_rc_tolerance
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.alns_rc_tolerance
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.alns_rc_tolerance
```

````

````{py:attribute} alns_remove_fraction
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.alns_remove_fraction
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.alns_remove_fraction
```

````

````{py:attribute} dive_penalty_M
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.dive_penalty_M
:type: float
:value: >
   10000.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.dive_penalty_M
```

````

````{py:attribute} fo_tabu_length
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.fo_tabu_length
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.fo_tabu_length
```

````

````{py:attribute} fo_max_unfix
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.fo_max_unfix
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.fo_max_unfix
```

````

````{py:attribute} fo_strategy
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.fo_strategy
:type: typing.Literal[overflow_urgency, scenario_divergence]
:value: >
   'overflow_urgency'

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.fo_strategy
```

````

````{py:attribute} fo_max_iterations
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.fo_max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.fo_max_iterations
```

````

````{py:attribute} ml_reliability_c
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.ml_reliability_c
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.ml_reliability_c
```

````

````{py:attribute} ml_pseudocost_ema_alpha
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.ml_pseudocost_ema_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.ml_pseudocost_ema_alpha
```

````

````{py:attribute} sc_consensus_threshold
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.sc_consensus_threshold
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.sc_consensus_threshold
```

````

````{py:attribute} benders_max_iterations
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.benders_max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.benders_max_iterations
```

````

````{py:attribute} benders_convergence_tol
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.benders_convergence_tol
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.benders_convergence_tol
```

````

````{py:attribute} benders_cut_pool_max
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.benders_cut_pool_max
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.benders_cut_pool_max
```

````

````{py:attribute} max_visits_per_bin
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.max_visits_per_bin
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.max_visits_per_bin
```

````

````{py:attribute} theta_upper_bound
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.theta_upper_bound
:type: float
:value: >
   1000000.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.theta_upper_bound
```

````

````{py:attribute} gurobi_master_time_limit
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_master_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_master_time_limit
```

````

````{py:attribute} gurobi_sub_time_limit
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_sub_time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_sub_time_limit
```

````

````{py:attribute} gurobi_mip_gap
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_mip_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_mip_gap
```

````

````{py:attribute} gurobi_output_flag
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_output_flag
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.gurobi_output_flag
```

````

````{py:attribute} subproblem_relax
:canonical: src.configs.policies.abpc_hg.ABPCHGConfig.subproblem_relax
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.abpc_hg.ABPCHGConfig.subproblem_relax
```

````

`````
