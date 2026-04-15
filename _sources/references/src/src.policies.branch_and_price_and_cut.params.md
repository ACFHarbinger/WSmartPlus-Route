# {py:mod}`src.policies.branch_and_price_and_cut.params`

```{py:module} src.policies.branch_and_price_and_cut.params
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BPCParams <src.policies.branch_and_price_and_cut.params.BPCParams>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams
    :summary:
    ```
````

### API

`````{py:class} BPCParams
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams
```

````{py:attribute} time_limit
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.time_limit
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.vrpp
```

````

````{py:attribute} seed
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.seed
```

````

````{py:attribute} search_strategy
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.search_strategy
:type: str
:value: >
   'depth_first'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.search_strategy
```

````

````{py:attribute} cutting_planes
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.cutting_planes
:type: str
:value: >
   'rcc'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.cutting_planes
```

````

````{py:attribute} branching_strategy
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.branching_strategy
:type: str
:value: >
   'divergence'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.branching_strategy
```

````

````{py:attribute} max_cg_iterations
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.max_cg_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.max_cg_iterations
```

````

````{py:attribute} max_cut_iterations
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.max_cut_iterations
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.max_cut_iterations
```

````

````{py:attribute} max_cuts_per_iteration
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.max_cuts_per_iteration
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.max_cuts_per_iteration
```

````

````{py:attribute} max_routes_per_pricing
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.max_routes_per_pricing
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.max_routes_per_pricing
```

````

````{py:attribute} max_bb_nodes
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.max_bb_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.max_bb_nodes
```

````

````{py:attribute} optimality_gap
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.optimality_gap
```

````

````{py:attribute} early_termination_gap
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.early_termination_gap
```

````

````{py:attribute} use_ng_routes
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.use_ng_routes
```

````

````{py:attribute} ng_neighborhood_size
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.ng_neighborhood_size
```

````

````{py:attribute} enable_heuristic_rcc_separation
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.enable_heuristic_rcc_separation
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.enable_heuristic_rcc_separation
```

````

````{py:attribute} enable_comb_cuts
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.enable_comb_cuts
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.enable_comb_cuts
```

````

````{py:attribute} cut_orthogonality_threshold
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.cut_orthogonality_threshold
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.cut_orthogonality_threshold
```

````

````{py:attribute} use_spatial_partitioning
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.use_spatial_partitioning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.use_spatial_partitioning
```

````

````{py:attribute} enable_strong_branching_heuristic
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.enable_strong_branching_heuristic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.enable_strong_branching_heuristic
```

````

````{py:attribute} enable_column_pool_deduplication
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.enable_column_pool_deduplication
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.enable_column_pool_deduplication
```

````

````{py:attribute} rc_tolerance
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.rc_tolerance
:type: float
:value: >
   1e-08

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.rc_tolerance
```

````

````{py:attribute} exact_mode
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.exact_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.exact_mode
```

````

````{py:attribute} strong_branching_size
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.strong_branching_size
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.strong_branching_size
```

````

````{py:attribute} cg_at_root_only
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.cg_at_root_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.cg_at_root_only
```

````

````{py:attribute} use_swc_tcf_initialization
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_initialization
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_initialization
```

````

````{py:attribute} use_swc_tcf_heuristic_pricing
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_heuristic_pricing
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_heuristic_pricing
```

````

````{py:attribute} use_swc_tcf_primal_heuristic
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_primal_heuristic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.use_swc_tcf_primal_heuristic
```

````

````{py:attribute} multi_day_mode
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.multi_day_mode
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.multi_day_mode
```

````

````{py:attribute} adp_model_path
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.adp_model_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.adp_model_path
```

````

````{py:attribute} adp_model_type
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.adp_model_type
:type: str
:value: >
   'sklearn'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.adp_model_type
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.branch_and_price_and_cut.params.BPCParams
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price_and_cut.params.BPCParams.to_dict

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.params.BPCParams.to_dict
```

````

`````
