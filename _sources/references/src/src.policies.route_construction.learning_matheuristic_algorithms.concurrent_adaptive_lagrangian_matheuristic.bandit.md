# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BanditArm <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm
    :summary:
    ```
* - {py:obj}`LinUCBBandit <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_context <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.build_context>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.build_context
    :summary:
    ```
````

### API

`````{py:class} BanditArm
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm
```

````{py:attribute} engine
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.engine
:type: str
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.engine
```

````

````{py:attribute} cut_strategy
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.cut_strategy
:type: str
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.cut_strategy
```

````

````{py:attribute} A
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.A
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.A
```

````

````{py:attribute} A_inv
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.A_inv
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.A_inv
```

````

````{py:attribute} b
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.b
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.b
```

````

````{py:attribute} n_pulls
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.n_pulls
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.n_pulls
```

````

````{py:attribute} cumulative_reward
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.cumulative_reward
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.cumulative_reward
```

````

````{py:property} theta
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.theta
:type: numpy.ndarray

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm.theta
```

````

`````

`````{py:class} LinUCBBandit(params: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.BanditParams, rng: typing.Optional[numpy.random.Generator] = None)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.__init__
```

````{py:method} _build_arms() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit._build_arms

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit._build_arms
```

````

````{py:property} arms
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.arms
:type: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.BanditArm]

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.arms
```

````

````{py:method} select_arm(context: numpy.ndarray) -> typing.Tuple[int, str, str, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.select_arm

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.select_arm
```

````

````{py:method} update(arm_index: int, context: numpy.ndarray, reward: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.update

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.update
```

````

````{py:method} summary() -> typing.List[typing.Dict[str, typing.Union[float, str]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.summary

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit.summary
```

````

`````

````{py:function} build_context(*, outer_iter: int, max_outer: int, primal_gap_frac: float, dual_progress_frac: float, iters_since_improvement: int, stagnation_patience: int, fraction_bins_selected: float, fraction_periods_saturated: float, lambda_norm: float, lambda_norm_scale: float, feature_dim: int) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.build_context

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.build_context
```
````
