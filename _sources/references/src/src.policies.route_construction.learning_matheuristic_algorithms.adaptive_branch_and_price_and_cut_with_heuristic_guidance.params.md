# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABPCHGParams <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams
    :summary:
    ```
````

### API

`````{py:class} ABPCHGParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams
```

````{py:attribute} gamma
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gamma
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gamma
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.seed
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.overflow_penalty
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.overflow_penalty
```

````

````{py:attribute} ph_base_rho
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_base_rho
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_base_rho
```

````

````{py:attribute} ph_max_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_max_iterations
```

````

````{py:attribute} ph_convergence_tol
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_convergence_tol
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ph_convergence_tol
```

````

````{py:attribute} alns_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_iterations
```

````

````{py:attribute} alns_max_routes
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_max_routes
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_max_routes
```

````

````{py:attribute} alns_rc_tolerance
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_rc_tolerance
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_rc_tolerance
```

````

````{py:attribute} alns_remove_fraction
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_remove_fraction
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.alns_remove_fraction
```

````

````{py:attribute} dive_penalty_M
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.dive_penalty_M
:type: float
:value: >
   10000.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.dive_penalty_M
```

````

````{py:attribute} fo_tabu_length
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_tabu_length
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_tabu_length
```

````

````{py:attribute} fo_max_unfix
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_max_unfix
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_max_unfix
```

````

````{py:attribute} fo_strategy
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_strategy
:type: typing.Literal[overflow_urgency, scenario_divergence]
:value: >
   'overflow_urgency'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_strategy
```

````

````{py:attribute} fo_max_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_max_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.fo_max_iterations
```

````

````{py:attribute} ml_reliability_c
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ml_reliability_c
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ml_reliability_c
```

````

````{py:attribute} ml_pseudocost_ema_alpha
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ml_pseudocost_ema_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.ml_pseudocost_ema_alpha
```

````

````{py:attribute} sc_consensus_threshold
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.sc_consensus_threshold
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.sc_consensus_threshold
```

````

````{py:attribute} benders_max_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_max_iterations
```

````

````{py:attribute} benders_convergence_tol
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_convergence_tol
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_convergence_tol
```

````

````{py:attribute} benders_cut_pool_max
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_cut_pool_max
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.benders_cut_pool_max
```

````

````{py:attribute} max_visits_per_bin
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.max_visits_per_bin
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.max_visits_per_bin
```

````

````{py:attribute} theta_upper_bound
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.theta_upper_bound
:type: float
:value: >
   1000000.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.theta_upper_bound
```

````

````{py:attribute} gurobi_master_time_limit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_master_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_master_time_limit
```

````

````{py:attribute} gurobi_sub_time_limit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_sub_time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_sub_time_limit
```

````

````{py:attribute} gurobi_mip_gap
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_mip_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_mip_gap
```

````

````{py:attribute} gurobi_output_flag
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_output_flag
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.gurobi_output_flag
```

````

````{py:attribute} subproblem_relax
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.subproblem_relax
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.subproblem_relax
```

````

````{py:method} from_config(config: logic.src.configs.policies.abpc_hg.ABPCHGConfig) -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params.ABPCHGParams.from_config
```

````

`````
