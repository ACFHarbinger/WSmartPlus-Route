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

````{py:attribute} must_go
:canonical: src.configs.policies.bpc.BPCConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.bpc.BPCConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.post_processing
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

````{py:attribute} use_swc_tcf_initialization
:canonical: src.configs.policies.bpc.BPCConfig.use_swc_tcf_initialization
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.use_swc_tcf_initialization
```

````

````{py:attribute} use_swc_tcf_heuristic_pricing
:canonical: src.configs.policies.bpc.BPCConfig.use_swc_tcf_heuristic_pricing
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.use_swc_tcf_heuristic_pricing
```

````

````{py:attribute} use_swc_tcf_primal_heuristic
:canonical: src.configs.policies.bpc.BPCConfig.use_swc_tcf_primal_heuristic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.use_swc_tcf_primal_heuristic
```

````

````{py:attribute} multi_day_mode
:canonical: src.configs.policies.bpc.BPCConfig.multi_day_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.multi_day_mode
```

````

````{py:attribute} adp_model_path
:canonical: src.configs.policies.bpc.BPCConfig.adp_model_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.adp_model_path
```

````

````{py:attribute} adp_model_type
:canonical: src.configs.policies.bpc.BPCConfig.adp_model_type
:type: str
:value: >
   'sklearn'

```{autodoc2-docstring} src.configs.policies.bpc.BPCConfig.adp_model_type
```

````

`````
