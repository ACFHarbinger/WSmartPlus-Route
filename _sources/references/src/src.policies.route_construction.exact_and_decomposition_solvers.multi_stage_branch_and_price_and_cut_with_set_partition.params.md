# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MSBPCSPParams <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams
    :summary:
    ```
````

### API

`````{py:class} MSBPCSPParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.time_limit
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.vrpp
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.seed
```

````

````{py:attribute} search_strategy
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.search_strategy
:type: str
:value: >
   'depth_first'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.search_strategy
```

````

````{py:attribute} cutting_planes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cutting_planes
:type: str
:value: >
   'saturated_arc_lci'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cutting_planes
```

````

````{py:attribute} branching_strategy
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.branching_strategy
```

````

````{py:attribute} max_cg_iterations
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cg_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cg_iterations
```

````

````{py:attribute} max_cut_iterations
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cut_iterations
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cut_iterations
```

````

````{py:attribute} max_cuts_per_iteration
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cuts_per_iteration
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_cuts_per_iteration
```

````

````{py:attribute} max_routes_per_pricing
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_routes_per_pricing
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_routes_per_pricing
```

````

````{py:attribute} max_bb_nodes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_bb_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.max_bb_nodes
```

````

````{py:attribute} optimality_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.optimality_gap
```

````

````{py:attribute} early_termination_gap
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.early_termination_gap
```

````

````{py:attribute} use_ng_routes
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.use_ng_routes
```

````

````{py:attribute} ng_neighborhood_size
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.ng_neighborhood_size
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_comb_cuts
```

````

````{py:attribute} cut_orthogonality_threshold
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cut_orthogonality_threshold
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cut_orthogonality_threshold
```

````

````{py:attribute} use_spatial_partitioning
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.use_spatial_partitioning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.use_spatial_partitioning
```

````

````{py:attribute} enable_strong_branching_heuristic
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_strong_branching_heuristic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_strong_branching_heuristic
```

````

````{py:attribute} enable_column_pool_deduplication
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_column_pool_deduplication
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.enable_column_pool_deduplication
```

````

````{py:attribute} rc_tolerance
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rc_tolerance
:type: float
:value: >
   1e-08

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rc_tolerance
```

````

````{py:attribute} exact_mode
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.exact_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.exact_mode
```

````

````{py:attribute} strong_branching_size
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.strong_branching_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.strong_branching_size
```

````

````{py:attribute} cg_at_root_only
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cg_at_root_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.cg_at_root_only
```

````

````{py:attribute} rcspp_timeout
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rcspp_timeout
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rcspp_timeout
```

````

````{py:attribute} rcspp_max_labels
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rcspp_max_labels
:type: int
:value: >
   1000000

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.rcspp_max_labels
```

````

````{py:attribute} prefer_shorter_path_dfs
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.prefer_shorter_path_dfs
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.prefer_shorter_path_dfs
```

````

````{py:attribute} lr_pre_pruning
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_pre_pruning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_pre_pruning
```

````

````{py:attribute} lr_lambda_init
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_lambda_init
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_lambda_init
```

````

````{py:attribute} lr_max_subgradient_iters
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_max_subgradient_iters
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_max_subgradient_iters
```

````

````{py:attribute} lr_subgradient_theta
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_subgradient_theta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_subgradient_theta
```

````

````{py:attribute} lr_op_time_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_op_time_limit
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_op_time_limit
```

````

````{py:attribute} lr_pre_pruning_depth_limit
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_pre_pruning_depth_limit
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_pre_pruning_depth_limit
```

````

````{py:attribute} lr_warm_start_cg
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_warm_start_cg
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.lr_warm_start_cg
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams.to_dict
```

````

`````
