# {py:mod}`src.configs.policies.other.post_processing`

```{py:module} src.configs.policies.other.post_processing
```

```{autodoc2-docstring} src.configs.policies.other.post_processing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastTSPPostConfig <src.configs.policies.other.post_processing.FastTSPPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig
    :summary:
    ```
* - {py:obj}`LKHPostConfig <src.configs.policies.other.post_processing.LKHPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig
    :summary:
    ```
* - {py:obj}`LocalSearchPostConfig <src.configs.policies.other.post_processing.LocalSearchPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig
    :summary:
    ```
* - {py:obj}`PathPostConfig <src.configs.policies.other.post_processing.PathPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig
    :summary:
    ```
* - {py:obj}`RandomLocalSearchPostConfig <src.configs.policies.other.post_processing.RandomLocalSearchPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
    :summary:
    ```
* - {py:obj}`OrOptPostConfig <src.configs.policies.other.post_processing.OrOptPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.OrOptPostConfig
    :summary:
    ```
* - {py:obj}`CrossExchangePostConfig <src.configs.policies.other.post_processing.CrossExchangePostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.CrossExchangePostConfig
    :summary:
    ```
* - {py:obj}`GuidedLocalSearchPostConfig <src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig
    :summary:
    ```
* - {py:obj}`SimulatedAnnealingPostConfig <src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig
    :summary:
    ```
* - {py:obj}`InsertionPostConfig <src.configs.policies.other.post_processing.InsertionPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig
    :summary:
    ```
* - {py:obj}`RuinRecreatePostConfig <src.configs.policies.other.post_processing.RuinRecreatePostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig
    :summary:
    ```
* - {py:obj}`AdaptiveLNSPostConfig <src.configs.policies.other.post_processing.AdaptiveLNSPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig
    :summary:
    ```
* - {py:obj}`FixAndOptimizePostConfig <src.configs.policies.other.post_processing.FixAndOptimizePostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig
    :summary:
    ```
* - {py:obj}`SetPartitioningPostConfig <src.configs.policies.other.post_processing.SetPartitioningPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig
    :summary:
    ```
* - {py:obj}`SetPartitioningPolishPostConfig <src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig
    :summary:
    ```
* - {py:obj}`LearnedPostConfig <src.configs.policies.other.post_processing.LearnedPostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig
    :summary:
    ```
* - {py:obj}`BranchAndPricePostConfig <src.configs.policies.other.post_processing.BranchAndPricePostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig
    :summary:
    ```
* - {py:obj}`TwoPhasePostConfig <src.configs.policies.other.post_processing.TwoPhasePostConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.TwoPhasePostConfig
    :summary:
    ```
* - {py:obj}`PostProcessingConfig <src.configs.policies.other.post_processing.PostProcessingConfig>`
  - ```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig
    :summary:
    ```
````

### API

`````{py:class} FastTSPPostConfig
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig.time_limit
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.FastTSPPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.FastTSPPostConfig.seed
```

````

`````

`````{py:class} LKHPostConfig
:canonical: src.configs.policies.other.post_processing.LKHPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig
```

````{py:attribute} max_iterations
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.LKHPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.LKHPostConfig.seed
```

````

`````

`````{py:class} LocalSearchPostConfig
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig
```

````{py:attribute} ls_operator
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.ls_operator
:type: str
:value: >
   '2opt'

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.ls_operator
```

````

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.LocalSearchPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.LocalSearchPostConfig.seed
```

````

`````

`````{py:class} PathPostConfig
:canonical: src.configs.policies.other.post_processing.PathPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig
```

````{py:attribute} vehicle_capacity
:canonical: src.configs.policies.other.post_processing.PathPostConfig.vehicle_capacity
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.PathPostConfig.vehicle_capacity
```

````

`````

`````{py:class} RandomLocalSearchPostConfig
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
```

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.iterations
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.params
:type: typing.Dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.params
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.RandomLocalSearchPostConfig.seed
```

````

`````

`````{py:class} OrOptPostConfig
:canonical: src.configs.policies.other.post_processing.OrOptPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.OrOptPostConfig
```

````{py:attribute} chain_len
:canonical: src.configs.policies.other.post_processing.OrOptPostConfig.chain_len
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.other.post_processing.OrOptPostConfig.chain_len
```

````

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.OrOptPostConfig.iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.other.post_processing.OrOptPostConfig.iterations
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.OrOptPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.OrOptPostConfig.seed
```

````

`````

`````{py:class} CrossExchangePostConfig
:canonical: src.configs.policies.other.post_processing.CrossExchangePostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.CrossExchangePostConfig
```

````{py:attribute} cross_exchange_max_segment_len
:canonical: src.configs.policies.other.post_processing.CrossExchangePostConfig.cross_exchange_max_segment_len
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.other.post_processing.CrossExchangePostConfig.cross_exchange_max_segment_len
```

````

````{py:attribute} iterations
:canonical: src.configs.policies.other.post_processing.CrossExchangePostConfig.iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.other.post_processing.CrossExchangePostConfig.iterations
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.CrossExchangePostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.CrossExchangePostConfig.seed
```

````

`````

`````{py:class} GuidedLocalSearchPostConfig
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig
```

````{py:attribute} gls_iterations
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_iterations
```

````

````{py:attribute} gls_inner_iterations
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_inner_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_inner_iterations
```

````

````{py:attribute} gls_lambda_factor
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_lambda_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_lambda_factor
```

````

````{py:attribute} gls_base_operator
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_base_operator
:type: str
:value: >
   'or_opt'

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.gls_base_operator
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig.seed
```

````

`````

`````{py:class} SimulatedAnnealingPostConfig
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig
```

````{py:attribute} sa_iterations
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_iterations
:type: int
:value: >
   2000

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_iterations
```

````

````{py:attribute} sa_t_init
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_t_init
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_t_init
```

````

````{py:attribute} sa_t_min
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_t_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_t_min
```

````

````{py:attribute} sa_cooling
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_cooling
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.sa_cooling
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.params
:type: typing.Dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.params
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig.seed
```

````

`````

`````{py:class} InsertionPostConfig
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig
```

````{py:attribute} cost_per_km
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.cost_per_km
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.cost_per_km
```

````

````{py:attribute} revenue_kg
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.revenue_kg
```

````

````{py:attribute} regret_k
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.regret_k
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.regret_k
```

````

````{py:attribute} detour_epsilon
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.detour_epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.detour_epsilon
```

````

````{py:attribute} n_bins
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.n_bins
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.n_bins
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.InsertionPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.InsertionPostConfig.seed
```

````

`````

`````{py:class} RuinRecreatePostConfig
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig
```

````{py:attribute} lns_iterations
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_iterations
```

````

````{py:attribute} ruin_fraction
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.ruin_fraction
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.ruin_fraction
```

````

````{py:attribute} lns_acceptance
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_acceptance
:type: str
:value: >
   'best'

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_acceptance
```

````

````{py:attribute} lns_sa_temperature
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_sa_temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.lns_sa_temperature
```

````

````{py:attribute} repair_k
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.repair_k
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.repair_k
```

````

````{py:attribute} cost_per_km
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.cost_per_km
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.cost_per_km
```

````

````{py:attribute} revenue_kg
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.revenue_kg
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.RuinRecreatePostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.RuinRecreatePostConfig.seed
```

````

`````

`````{py:class} AdaptiveLNSPostConfig
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig
```

````{py:attribute} alns_iterations
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_iterations
```

````

````{py:attribute} ruin_fraction
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.ruin_fraction
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.ruin_fraction
```

````

````{py:attribute} alns_bandit_warm_start_path
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_bandit_warm_start_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_bandit_warm_start_path
```

````

````{py:attribute} alns_ruin_ops
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_ruin_ops
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_ruin_ops
```

````

````{py:attribute} alns_repair_ops
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_repair_ops
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.alns_repair_ops
```

````

````{py:attribute} repair_k
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.repair_k
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.repair_k
```

````

````{py:attribute} cost_per_km
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.cost_per_km
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.cost_per_km
```

````

````{py:attribute} revenue_kg
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.revenue_kg
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.revenue_kg
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.AdaptiveLNSPostConfig.seed
```

````

`````

`````{py:class} FixAndOptimizePostConfig
:canonical: src.configs.policies.other.post_processing.FixAndOptimizePostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig
```

````{py:attribute} fo_n_free
:canonical: src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_n_free
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_n_free
```

````

````{py:attribute} fo_free_fraction
:canonical: src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_free_fraction
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_free_fraction
```

````

````{py:attribute} fo_time_limit
:canonical: src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig.fo_time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.FixAndOptimizePostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.FixAndOptimizePostConfig.seed
```

````

`````

`````{py:class} SetPartitioningPostConfig
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig
```

````{py:attribute} sp_n_perturbations
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_n_perturbations
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_n_perturbations
```

````

````{py:attribute} sp_include_dp
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_include_dp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_include_dp
```

````

````{py:attribute} sp_time_limit
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig.sp_time_limit
```

````

````{py:attribute} ruin_fraction
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig.ruin_fraction
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig.ruin_fraction
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.SetPartitioningPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPostConfig.seed
```

````

`````

`````{py:class} SetPartitioningPolishPostConfig
:canonical: src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig
```

````{py:attribute} route_pool
:canonical: src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.route_pool
:type: typing.Optional[typing.List[typing.List[int]]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.route_pool
```

````

````{py:attribute} sp_time_limit
:canonical: src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.sp_time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.sp_time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig.seed
```

````

`````

`````{py:class} LearnedPostConfig
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig
```

````{py:attribute} learned_weights_path
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig.learned_weights_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig.learned_weights_path
```

````

````{py:attribute} learned_max_iter
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig.learned_max_iter
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig.learned_max_iter
```

````

````{py:attribute} learned_min_improvement
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig.learned_min_improvement
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig.learned_min_improvement
```

````

````{py:attribute} learned_neighborhood_size
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig.learned_neighborhood_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig.learned_neighborhood_size
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.LearnedPostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.LearnedPostConfig.seed
```

````

`````

`````{py:class} BranchAndPricePostConfig
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig
```

````{py:attribute} bp_max_iterations
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_iterations
```

````

````{py:attribute} bp_max_routes_per_iteration
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_routes_per_iteration
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_routes_per_iteration
```

````

````{py:attribute} bp_optimality_gap
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_optimality_gap
:type: float
:value: >
   0.0001

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_optimality_gap
```

````

````{py:attribute} bp_branching_strategy
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_branching_strategy
:type: str
:value: >
   'edge'

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_branching_strategy
```

````

````{py:attribute} bp_max_branch_nodes
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_branch_nodes
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_max_branch_nodes
```

````

````{py:attribute} bp_use_exact_pricing
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_exact_pricing
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_exact_pricing
```

````

````{py:attribute} bp_use_ng_routes
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_ng_routes
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_ng_routes
```

````

````{py:attribute} bp_ng_neighborhood_size
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_ng_neighborhood_size
:type: int
:value: >
   8

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_ng_neighborhood_size
```

````

````{py:attribute} bp_tree_search_strategy
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_tree_search_strategy
:type: str
:value: >
   'best_first'

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_tree_search_strategy
```

````

````{py:attribute} bp_vehicle_limit
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_vehicle_limit
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_vehicle_limit
```

````

````{py:attribute} bp_cleanup_frequency
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_cleanup_frequency
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_cleanup_frequency
```

````

````{py:attribute} bp_cleanup_threshold
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_cleanup_threshold
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_cleanup_threshold
```

````

````{py:attribute} bp_early_termination_gap
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_early_termination_gap
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_early_termination_gap
```

````

````{py:attribute} bp_allow_heuristic_ryan_foster
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_allow_heuristic_ryan_foster
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_allow_heuristic_ryan_foster
```

````

````{py:attribute} bp_time_limit
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_time_limit
```

````

````{py:attribute} bp_use_cspy
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_cspy
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.bp_use_cspy
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.BranchAndPricePostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.BranchAndPricePostConfig.seed
```

````

`````

`````{py:class} TwoPhasePostConfig
:canonical: src.configs.policies.other.post_processing.TwoPhasePostConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.TwoPhasePostConfig
```

````{py:attribute} phase_one
:canonical: src.configs.policies.other.post_processing.TwoPhasePostConfig.phase_one
:type: str
:value: >
   'cheapest_insertion'

```{autodoc2-docstring} src.configs.policies.other.post_processing.TwoPhasePostConfig.phase_one
```

````

````{py:attribute} phase_two
:canonical: src.configs.policies.other.post_processing.TwoPhasePostConfig.phase_two
:type: str
:value: >
   'lkh'

```{autodoc2-docstring} src.configs.policies.other.post_processing.TwoPhasePostConfig.phase_two
```

````

````{py:attribute} seed
:canonical: src.configs.policies.other.post_processing.TwoPhasePostConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.other.post_processing.TwoPhasePostConfig.seed
```

````

`````

`````{py:class} PostProcessingConfig
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig
```

````{py:attribute} methods
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.methods
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.methods
```

````

````{py:attribute} fast_tsp
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.fast_tsp
:type: src.configs.policies.other.post_processing.FastTSPPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.fast_tsp
```

````

````{py:attribute} lkh
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.lkh
:type: src.configs.policies.other.post_processing.LKHPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.lkh
```

````

````{py:attribute} local_search
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.local_search
:type: src.configs.policies.other.post_processing.LocalSearchPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.local_search
```

````

````{py:attribute} random_local_search
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.random_local_search
:type: src.configs.policies.other.post_processing.RandomLocalSearchPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.random_local_search
```

````

````{py:attribute} path
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.path
:type: src.configs.policies.other.post_processing.PathPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.path
```

````

````{py:attribute} or_opt
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.or_opt
:type: src.configs.policies.other.post_processing.OrOptPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.or_opt
```

````

````{py:attribute} cross_exchange
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.cross_exchange
:type: src.configs.policies.other.post_processing.CrossExchangePostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.cross_exchange
```

````

````{py:attribute} guided_local_search
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.guided_local_search
:type: src.configs.policies.other.post_processing.GuidedLocalSearchPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.guided_local_search
```

````

````{py:attribute} simulated_annealing
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.simulated_annealing
:type: src.configs.policies.other.post_processing.SimulatedAnnealingPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.simulated_annealing
```

````

````{py:attribute} insertion
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.insertion
:type: src.configs.policies.other.post_processing.InsertionPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.insertion
```

````

````{py:attribute} ruin_recreate
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.ruin_recreate
:type: src.configs.policies.other.post_processing.RuinRecreatePostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.ruin_recreate
```

````

````{py:attribute} adaptive_lns
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.adaptive_lns
:type: src.configs.policies.other.post_processing.AdaptiveLNSPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.adaptive_lns
```

````

````{py:attribute} two_phase
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.two_phase
:type: src.configs.policies.other.post_processing.TwoPhasePostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.two_phase
```

````

````{py:attribute} fix_and_optimize
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.fix_and_optimize
:type: src.configs.policies.other.post_processing.FixAndOptimizePostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.fix_and_optimize
```

````

````{py:attribute} set_partitioning
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.set_partitioning
:type: src.configs.policies.other.post_processing.SetPartitioningPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.set_partitioning
```

````

````{py:attribute} set_partitioning_polish
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.set_partitioning_polish
:type: src.configs.policies.other.post_processing.SetPartitioningPolishPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.set_partitioning_polish
```

````

````{py:attribute} branch_and_price
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.branch_and_price
:type: src.configs.policies.other.post_processing.BranchAndPricePostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.branch_and_price
```

````

````{py:attribute} learned
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.learned
:type: src.configs.policies.other.post_processing.LearnedPostConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.learned
```

````

````{py:attribute} max_iter
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.max_iter
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.max_iter
```

````

````{py:attribute} dp_max_nodes
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.dp_max_nodes
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.dp_max_nodes
```

````

````{py:attribute} chain_lengths
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.chain_lengths
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.chain_lengths
```

````

````{py:attribute} params
:canonical: src.configs.policies.other.post_processing.PostProcessingConfig.params
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.other.post_processing.PostProcessingConfig.params
```

````

`````
