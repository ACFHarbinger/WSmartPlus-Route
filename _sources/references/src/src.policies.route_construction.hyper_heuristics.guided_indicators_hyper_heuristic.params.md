# {py:mod}`src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params`

```{py:module} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHParams <src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams
    :summary:
    ```
````

### API

`````{py:class} GIHHParams
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams
```

````{py:attribute} time_limit
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.max_iterations
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.seed
```

````

````{py:attribute} seg
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.seg
:type: int
:value: >
   80

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.seg
```

````

````{py:attribute} alpha
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.beta
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.beta
```

````

````{py:attribute} gamma
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.gamma
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.gamma
```

````

````{py:attribute} min_prob
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.min_prob
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.min_prob
```

````

````{py:attribute} nonimp_threshold
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.nonimp_threshold
:type: int
:value: >
   150

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.nonimp_threshold
```

````

````{py:attribute} vrpp
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.profit_aware_operators
```

````

````{py:attribute} acceptance_criterion
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.acceptance_criterion
:type: typing.Optional[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.acceptance_criterion
```

````

````{py:method} __post_init__()
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.__post_init__

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams.from_config
```

````

`````
