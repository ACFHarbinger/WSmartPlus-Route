# {py:mod}`src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params`

```{py:module} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ARCOParams <src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams
    :summary:
    ```
````

### API

`````{py:class} ARCOParams
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams
```

````{py:attribute} constructors
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.constructors
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.constructors
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.time_limit
:type: float
:value: >
   120.0

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.time_limit
```

````

````{py:attribute} selection_strategy
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.selection_strategy
:type: str
:value: >
   'epsilon_greedy'

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.selection_strategy
```

````

````{py:attribute} epsilon
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.epsilon
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.epsilon
```

````

````{py:attribute} temperature
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.temperature
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.temperature
```

````

````{py:attribute} alpha_ema
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.alpha_ema
:type: float
:value: >
   0.15

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.alpha_ema
```

````

````{py:attribute} weight_init
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.weight_init
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.weight_init
```

````

````{py:attribute} weight_floor
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.weight_floor
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.weight_floor
```

````

````{py:attribute} decay
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.decay
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.decay
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params.ARCOParams.from_config
```

````

`````
