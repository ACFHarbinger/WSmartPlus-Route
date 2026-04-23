# {py:mod}`src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual`

```{py:module} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DualBoundTracker <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker
    :summary:
    ```
* - {py:obj}`EMADualBoundTracker <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker
    :summary:
    ```
* - {py:obj}`BundleEntry <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry
    :summary:
    ```
* - {py:obj}`ProximalBundleDualBoundTracker <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_dual_bound_tracker <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.build_dual_bound_tracker>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.build_dual_bound_tracker
    :summary:
    ```
````

### API

`````{py:class} DualBoundTracker
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker
```

````{py:method} submit(period: int, lambdas: numpy.ndarray, subgrad: numpy.ndarray, lagrangian_value_contrib: float, tour_quality_ratio: float) -> bool
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.submit
:abstractmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.submit
```

````

````{py:method} current_dual_bound() -> float
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.current_dual_bound
:abstractmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.current_dual_bound
```

````

````{py:method} suggest_step(current_lambdas: numpy.ndarray, aggregate_subgrad: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.suggest_step
:abstractmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker.suggest_step
```

````

`````

`````{py:class} EMADualBoundTracker(dual_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams, lag_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams, n_bins: int, horizon: int)
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker

Bases: {py:obj}`src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.__init__
```

````{py:method} submit(period: int, lambdas: numpy.ndarray, subgrad: numpy.ndarray, lagrangian_value_contrib: float, tour_quality_ratio: float) -> bool
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.submit

````

````{py:method} current_dual_bound() -> float
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.current_dual_bound

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.current_dual_bound
```

````

````{py:method} suggest_step(current_lambdas: numpy.ndarray, aggregate_subgrad: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.suggest_step

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.suggest_step
```

````

````{py:method} polyak_step(current_lambdas: numpy.ndarray, aggregate_subgrad: numpy.ndarray, upper_bound: float, mu: typing.Optional[float] = None) -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.polyak_step

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.polyak_step
```

````

````{py:method} ema_snapshot() -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.ema_snapshot

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.EMADualBoundTracker.ema_snapshot
```

````

`````

`````{py:class} BundleEntry
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry
```

````{py:attribute} lambdas
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.lambdas
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.lambdas
```

````

````{py:attribute} subgrad
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.subgrad
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.subgrad
```

````

````{py:attribute} value
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.value
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.BundleEntry.value
```

````

`````

`````{py:class} ProximalBundleDualBoundTracker(dual_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams, lag_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams, n_bins: int, horizon: int)
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker

Bases: {py:obj}`src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.__init__
```

````{py:method} submit(period: int, lambdas: numpy.ndarray, subgrad: numpy.ndarray, lagrangian_value_contrib: float, tour_quality_ratio: float) -> bool
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.submit

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.submit
```

````

````{py:method} commit_outer_iteration(lambdas: numpy.ndarray, full_lagrangian_value: float) -> bool
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.commit_outer_iteration

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.commit_outer_iteration
```

````

````{py:method} current_dual_bound() -> float
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.current_dual_bound

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.current_dual_bound
```

````

````{py:method} suggest_step(current_lambdas: numpy.ndarray, aggregate_subgrad: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.suggest_step

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker.suggest_step
```

````

````{py:method} _solve_bundle_qp() -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker._solve_bundle_qp

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.ProximalBundleDualBoundTracker._solve_bundle_qp
```

````

`````

````{py:function} build_dual_bound_tracker(dual_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.DualBoundParams, lag_params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams, n_bins: int, horizon: int) -> src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.build_dual_bound_tracker

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.dual.build_dual_bound_tracker
```
````
