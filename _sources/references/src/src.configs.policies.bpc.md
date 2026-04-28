# {py:mod}`src.configs.policies.bpc`

```{py:module} src.configs.policies.bpc
```

```{autodoc2-docstring} src.configs.policies.bpc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPCConfig <src.configs.policies.bpc.BPCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig
    :summary:
    ```
````

### API

`````{py:class} BPCConfig
:canonical: src.configs.policies.bpc.BPCConfig

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.bpc.BPCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.time_limit
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.bpc.BPCConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bpc.BPCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bpc.BPCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.bpc.BPCConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.bpc.BPCConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.route_improvement
```

````

````{py:attribute} search_strategy
:canonical: src.configs.policies.bpc.BPCConfig.search_strategy
:type: str
:value: >
   'depth_first'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.search_strategy
```

````

````{py:attribute} cutting_planes
:canonical: src.configs.policies.bpc.BPCConfig.cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.cutting_planes
```

````

````{py:attribute} branching_strategy
:canonical: src.configs.policies.bpc.BPCConfig.branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.branching_strategy
```

````

````{py:attribute} max_cg_iterations
:canonical: src.configs.policies.bpc.BPCConfig.max_cg_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.max_cg_iterations
```

````

````{py:attribute} max_cut_iterations
:canonical: src.configs.policies.bpc.BPCConfig.max_cut_iterations
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.max_cut_iterations
```

````

````{py:attribute} max_cuts_per_iteration
:canonical: src.configs.policies.bpc.BPCConfig.max_cuts_per_iteration
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.max_cuts_per_iteration
```

````

````{py:attribute} max_routes_per_pricing
:canonical: src.configs.policies.bpc.BPCConfig.max_routes_per_pricing
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.max_routes_per_pricing
```

````

````{py:attribute} max_bb_nodes
:canonical: src.configs.policies.bpc.BPCConfig.max_bb_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.max_bb_nodes
```

````

````{py:attribute} optimality_gap
:canonical: src.configs.policies.bpc.BPCConfig.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.optimality_gap
```

````

````{py:attribute} early_termination_gap
:canonical: src.configs.policies.bpc.BPCConfig.early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.early_termination_gap
```

````

````{py:attribute} use_ng_routes
:canonical: src.configs.policies.bpc.BPCConfig.use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.use_ng_routes
```

````

````{py:attribute} ng_neighborhood_size
:canonical: src.configs.policies.bpc.BPCConfig.ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.ng_neighborhood_size
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.configs.policies.bpc.BPCConfig.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.configs.policies.bpc.BPCConfig.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_comb_cuts
```

````

````{py:attribute} cut_orthogonality_threshold
:canonical: src.configs.policies.bpc.BPCConfig.cut_orthogonality_threshold
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.cut_orthogonality_threshold
```

````

````{py:attribute} use_spatial_partitioning
:canonical: src.configs.policies.bpc.BPCConfig.use_spatial_partitioning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.use_spatial_partitioning
```

````

````{py:attribute} enable_strong_branching_heuristic
:canonical: src.configs.policies.bpc.BPCConfig.enable_strong_branching_heuristic
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_strong_branching_heuristic
```

````

````{py:attribute} enable_column_pool_deduplication
:canonical: src.configs.policies.bpc.BPCConfig.enable_column_pool_deduplication
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_column_pool_deduplication
```

````

````{py:attribute} enable_hybrid_search
:canonical: src.configs.policies.bpc.BPCConfig.enable_hybrid_search
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_hybrid_search
```

````

````{py:attribute} rc_tolerance
:canonical: src.configs.policies.bpc.BPCConfig.rc_tolerance
:type: float
:value: >
   1e-05

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.rc_tolerance
```

````

````{py:attribute} exact_mode
:canonical: src.configs.policies.bpc.BPCConfig.exact_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.exact_mode
```

````

````{py:attribute} strong_branching_size
:canonical: src.configs.policies.bpc.BPCConfig.strong_branching_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.strong_branching_size
```

````

````{py:attribute} cg_at_root_only
:canonical: src.configs.policies.bpc.BPCConfig.cg_at_root_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.cg_at_root_only
```

````

````{py:attribute} lr_pre_pruning
:canonical: src.configs.policies.bpc.BPCConfig.lr_pre_pruning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_pre_pruning
```

````

````{py:attribute} lr_lambda_init
:canonical: src.configs.policies.bpc.BPCConfig.lr_lambda_init
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_lambda_init
```

````

````{py:attribute} lr_max_subgradient_iters
:canonical: src.configs.policies.bpc.BPCConfig.lr_max_subgradient_iters
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_max_subgradient_iters
```

````

````{py:attribute} lr_subgradient_theta
:canonical: src.configs.policies.bpc.BPCConfig.lr_subgradient_theta
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_subgradient_theta
```

````

````{py:attribute} lr_op_time_limit
:canonical: src.configs.policies.bpc.BPCConfig.lr_op_time_limit
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_op_time_limit
```

````

````{py:attribute} lr_pre_pruning_depth_limit
:canonical: src.configs.policies.bpc.BPCConfig.lr_pre_pruning_depth_limit
:type: int
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_pre_pruning_depth_limit
```

````

````{py:attribute} lr_warm_start_cg
:canonical: src.configs.policies.bpc.BPCConfig.lr_warm_start_cg
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.lr_warm_start_cg
```

````

````{py:attribute} rcspp_timeout
:canonical: src.configs.policies.bpc.BPCConfig.rcspp_timeout
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.rcspp_timeout
```

````

````{py:attribute} rcspp_max_labels
:canonical: src.configs.policies.bpc.BPCConfig.rcspp_max_labels
:type: int
:value: >
   1000000

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.rcspp_max_labels
```

````

````{py:attribute} prefer_shorter_path_dfs
:canonical: src.configs.policies.bpc.BPCConfig.prefer_shorter_path_dfs
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.prefer_shorter_path_dfs
```

````

````{py:attribute} enable_node_visitation_branching
:canonical: src.configs.policies.bpc.BPCConfig.enable_node_visitation_branching
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_node_visitation_branching
```

````

````{py:attribute} enable_dssr
:canonical: src.configs.policies.bpc.BPCConfig.enable_dssr
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_dssr
```

````

````{py:attribute} dssr_max_iters
:canonical: src.configs.policies.bpc.BPCConfig.dssr_max_iters
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.dssr_max_iters
```

````

````{py:attribute} enable_reduced_cost_arc_fixing
:canonical: src.configs.policies.bpc.BPCConfig.enable_reduced_cost_arc_fixing
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.enable_reduced_cost_arc_fixing
```

````

````{py:attribute} route_budget
:canonical: src.configs.policies.bpc.BPCConfig.route_budget
:type: float
:value: >
   'float(...)'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.route_budget
```

````

`````
