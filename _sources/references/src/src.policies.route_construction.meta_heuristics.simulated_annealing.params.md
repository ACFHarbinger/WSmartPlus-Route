# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing.params`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing.params
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAParams <src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams
    :summary:
    ```
````

### API

`````{py:class} SAParams
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams
```

````{py:attribute} initial_temperature
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.initial_temperature
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.initial_temperature
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.cooling_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.cooling_rate
```

````

````{py:attribute} min_temperature
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.min_temperature
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.min_temperature
```

````

````{py:attribute} target_acceptances_per_node
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.target_acceptances_per_node
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.target_acceptances_per_node
```

````

````{py:attribute} max_attempts_multiplier
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.max_attempts_multiplier
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.max_attempts_multiplier
```

````

````{py:attribute} frozen_streak_limit
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.frozen_streak_limit
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.frozen_streak_limit
```

````

````{py:attribute} auto_calibrate_temperature
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.auto_calibrate_temperature
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.auto_calibrate_temperature
```

````

````{py:attribute} target_initial_acceptance
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.target_initial_acceptance
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.target_initial_acceptance
```

````

````{py:attribute} calibration_samples
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.calibration_samples
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.calibration_samples
```

````

````{py:attribute} n_restarts
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.n_restarts
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.n_restarts
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.profit_aware_operators
```

````

````{py:attribute} nb_granular
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.nb_granular
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.nb_granular
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.acceptance_criterion
:type: typing.Optional[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.acceptance_criterion
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams.from_config
```

````

`````
