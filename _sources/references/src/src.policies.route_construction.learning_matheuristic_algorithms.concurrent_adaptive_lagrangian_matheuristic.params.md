# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookaheadParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams
    :summary:
    ```
* - {py:obj}`LagrangianParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams
    :summary:
    ```
* - {py:obj}`DualBoundParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams
    :summary:
    ```
* - {py:obj}`BanditParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams
    :summary:
    ```
* - {py:obj}`RegretParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams
    :summary:
    ```
* - {py:obj}`CALMParams <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams
    :summary:
    ```
````

### API

`````{py:class} LookaheadParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams
```

````{py:attribute} horizon
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.horizon
```

````

````{py:attribute} scenario_method
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_method
:type: str
:value: >
   'stochastic'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_method
```

````

````{py:attribute} scenario_distribution
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_distribution
:type: str
:value: >
   'gamma'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_distribution
```

````

````{py:attribute} scenario_dist_kwargs
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_dist_kwargs
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.scenario_dist_kwargs
```

````

````{py:attribute} n_scenarios
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.n_scenarios
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.n_scenarios
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.seed
```

````

````{py:attribute} capacity_cap
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.capacity_cap
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.capacity_cap
```

````

````{py:attribute} volume
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.volume
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.volume
```

````

````{py:attribute} density
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.density
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.density
```

````

````{py:attribute} revenue_per_kg
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.revenue_per_kg
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams.revenue_per_kg
```

````

`````

`````{py:class} LagrangianParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams
```

````{py:attribute} max_outer_iterations
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.max_outer_iterations
:type: int
:value: >
   12

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.max_outer_iterations
```

````

````{py:attribute} stagnation_patience
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.stagnation_patience
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.stagnation_patience
```

````

````{py:attribute} dual_bound_tolerance
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.dual_bound_tolerance
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.dual_bound_tolerance
```

````

````{py:attribute} asynchronous_updates
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.asynchronous_updates
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.asynchronous_updates
```

````

````{py:attribute} polyak_mu_default
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_default
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_default
```

````

````{py:attribute} polyak_mu_floor
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_floor
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_floor
```

````

````{py:attribute} polyak_mu_ceil
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_ceil
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.polyak_mu_ceil
```

````

````{py:attribute} lambda_min
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.lambda_min
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.lambda_min
```

````

````{py:attribute} lambda_max
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.lambda_max
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.lambda_max
```

````

````{py:attribute} gamma_init
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.gamma_init
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.gamma_init
```

````

````{py:attribute} gamma_decay
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.gamma_decay
:type: float
:value: >
   0.85

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams.gamma_decay
```

````

`````

`````{py:class} DualBoundParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams
```

````{py:attribute} strategy
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.strategy
:type: str
:value: >
   'ema'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.strategy
```

````

````{py:attribute} ema_alpha
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.ema_alpha
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.ema_alpha
```

````

````{py:attribute} ema_quality_threshold
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.ema_quality_threshold
:type: float
:value: >
   1.05

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.ema_quality_threshold
```

````

````{py:attribute} bundle_size
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_size
```

````

````{py:attribute} bundle_proximal_weight
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_proximal_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_proximal_weight
```

````

````{py:attribute} bundle_descent_threshold
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_descent_threshold
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_descent_threshold
```

````

````{py:attribute} bundle_weight_increase
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_weight_increase
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_weight_increase
```

````

````{py:attribute} bundle_weight_decrease
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_weight_decrease
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams.bundle_weight_decrease
```

````

`````

`````{py:class} BanditParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams
```

````{py:attribute} enabled
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.enabled
```

````

````{py:attribute} alpha
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.alpha
```

````

````{py:attribute} ridge_lambda
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.ridge_lambda
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.ridge_lambda
```

````

````{py:attribute} feature_dim
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.feature_dim
:type: int
:value: >
   8

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.feature_dim
```

````

````{py:attribute} engines
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.engines
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.engines
```

````

````{py:attribute} cut_strategies
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.cut_strategies
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.cut_strategies
```

````

````{py:attribute} reward_scale
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.reward_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.reward_scale
```

````

````{py:attribute} reward_clip
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.reward_clip
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams.reward_clip
```

````

`````

`````{py:class} RegretParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams
```

````{py:attribute} enabled
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.enabled
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.enabled
```

````

````{py:attribute} soft_bias_coefficient
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.soft_bias_coefficient
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.soft_bias_coefficient
```

````

````{py:attribute} escalation_patience
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.escalation_patience
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.escalation_patience
```

````

````{py:attribute} hard_fix_top_fraction
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.hard_fix_top_fraction
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.hard_fix_top_fraction
```

````

````{py:attribute} hard_fix_max_periods
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.hard_fix_max_periods
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams.hard_fix_max_periods
```

````

`````

`````{py:class} CALMParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams
```

````{py:attribute} lookahead
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.lookahead
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.lookahead
```

````

````{py:attribute} lagrangian
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.lagrangian
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.lagrangian
```

````

````{py:attribute} dual_bound
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.dual_bound
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.dual_bound
```

````

````{py:attribute} bandit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.bandit
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.bandit
```

````

````{py:attribute} regret
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.regret
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.RegretParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.regret
```

````

````{py:attribute} tpks
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.tpks
:type: logic.src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.tpks
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.time_limit
:type: float
:value: >
   600.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.seed
```

````

````{py:attribute} verbose
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.verbose
```

````

````{py:attribute} stockout_penalty
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.stockout_penalty
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.from_config
```

````

````{py:method} to_dict() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.to_dict

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.CALMParams.to_dict
```

````

`````
