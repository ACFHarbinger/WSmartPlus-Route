# {py:mod}`src.configs.policies.ms_bpc_sp`

```{py:module} src.configs.policies.ms_bpc_sp
```

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MSBPCSPConfig <src.configs.policies.ms_bpc_sp.MSBPCSPConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig
    :summary:
    ```
````

### API

`````{py:class} MSBPCSPConfig
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.time_limit
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.route_improvement
```

````

````{py:attribute} search_strategy
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.search_strategy
:type: str
:value: >
   'depth_first'

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.search_strategy
```

````

````{py:attribute} cutting_planes
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cutting_planes
:type: str
:value: >
   'saturated_arc_lci'

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cutting_planes
```

````

````{py:attribute} branching_strategy
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.branching_strategy
```

````

````{py:attribute} max_cg_iterations
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cg_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cg_iterations
```

````

````{py:attribute} max_cut_iterations
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cut_iterations
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cut_iterations
```

````

````{py:attribute} max_cuts_per_iteration
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cuts_per_iteration
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_cuts_per_iteration
```

````

````{py:attribute} max_routes_per_pricing
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_routes_per_pricing
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_routes_per_pricing
```

````

````{py:attribute} max_bb_nodes
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_bb_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.max_bb_nodes
```

````

````{py:attribute} optimality_gap
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.optimality_gap
```

````

````{py:attribute} early_termination_gap
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.early_termination_gap
```

````

````{py:attribute} use_ng_routes
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.use_ng_routes
```

````

````{py:attribute} ng_neighborhood_size
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.ng_neighborhood_size
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_comb_cuts
```

````

````{py:attribute} cut_orthogonality_threshold
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cut_orthogonality_threshold
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cut_orthogonality_threshold
```

````

````{py:attribute} use_spatial_partitioning
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.use_spatial_partitioning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.use_spatial_partitioning
```

````

````{py:attribute} enable_strong_branching_heuristic
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_strong_branching_heuristic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_strong_branching_heuristic
```

````

````{py:attribute} enable_column_pool_deduplication
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_column_pool_deduplication
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.enable_column_pool_deduplication
```

````

````{py:attribute} rc_tolerance
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rc_tolerance
:type: float
:value: >
   1e-08

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rc_tolerance
```

````

````{py:attribute} exact_mode
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.exact_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.exact_mode
```

````

````{py:attribute} strong_branching_size
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.strong_branching_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.strong_branching_size
```

````

````{py:attribute} cg_at_root_only
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cg_at_root_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.cg_at_root_only
```

````

````{py:attribute} rcspp_timeout
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rcspp_timeout
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rcspp_timeout
```

````

````{py:attribute} rcspp_max_labels
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rcspp_max_labels
:type: int
:value: >
   1000000

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.rcspp_max_labels
```

````

````{py:attribute} prefer_shorter_path_dfs
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.prefer_shorter_path_dfs
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.prefer_shorter_path_dfs
```

````

````{py:attribute} lr_pre_pruning
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_pre_pruning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_pre_pruning
```

````

````{py:attribute} lr_lambda_init
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_lambda_init
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_lambda_init
```

````

````{py:attribute} lr_max_subgradient_iters
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_max_subgradient_iters
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_max_subgradient_iters
```

````

````{py:attribute} lr_subgradient_theta
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_subgradient_theta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_subgradient_theta
```

````

````{py:attribute} lr_op_time_limit
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_op_time_limit
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_op_time_limit
```

````

````{py:attribute} lr_pre_pruning_depth_limit
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_pre_pruning_depth_limit
:type: int
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_pre_pruning_depth_limit
```

````

````{py:attribute} lr_warm_start_cg
:canonical: src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_warm_start_cg
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.ms_bpc_sp.MSBPCSPConfig.lr_warm_start_cg
```

````

`````
