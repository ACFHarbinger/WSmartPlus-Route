# {py:mod}`src.interfaces.context.problem_context`

```{py:module} src.interfaces.context.problem_context
```

```{autodoc2-docstring} src.interfaces.context.problem_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProblemContext <src.interfaces.context.problem_context.ProblemContext>`
  - ```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext
    :summary:
    ```
````

### API

`````{py:class} ProblemContext
:canonical: src.interfaces.context.problem_context.ProblemContext

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext
```

````{py:attribute} distance_matrix
:canonical: src.interfaces.context.problem_context.ProblemContext.distance_matrix
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.distance_matrix
```

````

````{py:attribute} wastes
:canonical: src.interfaces.context.problem_context.ProblemContext.wastes
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.wastes
```

````

````{py:attribute} fill_rate_means
:canonical: src.interfaces.context.problem_context.ProblemContext.fill_rate_means
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.fill_rate_means
```

````

````{py:attribute} fill_rate_stds
:canonical: src.interfaces.context.problem_context.ProblemContext.fill_rate_stds
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.fill_rate_stds
```

````

````{py:attribute} capacity
:canonical: src.interfaces.context.problem_context.ProblemContext.capacity
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.capacity
```

````

````{py:attribute} max_fill
:canonical: src.interfaces.context.problem_context.ProblemContext.max_fill
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.max_fill
```

````

````{py:attribute} revenue_per_kg
:canonical: src.interfaces.context.problem_context.ProblemContext.revenue_per_kg
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.revenue_per_kg
```

````

````{py:attribute} cost_per_km
:canonical: src.interfaces.context.problem_context.ProblemContext.cost_per_km
:type: float
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.cost_per_km
```

````

````{py:attribute} horizon
:canonical: src.interfaces.context.problem_context.ProblemContext.horizon
:type: int
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.horizon
```

````

````{py:attribute} mandatory
:canonical: src.interfaces.context.problem_context.ProblemContext.mandatory
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.mandatory
```

````

````{py:attribute} locations
:canonical: src.interfaces.context.problem_context.ProblemContext.locations
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.locations
```

````

````{py:attribute} scenario_tree
:canonical: src.interfaces.context.problem_context.ProblemContext.scenario_tree
:type: typing.Optional[logic.src.pipeline.simulations.bins.prediction.ScenarioTree]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.scenario_tree
```

````

````{py:attribute} area
:canonical: src.interfaces.context.problem_context.ProblemContext.area
:type: str
:value: >
   'Rio Maior'

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.area
```

````

````{py:attribute} waste_type
:canonical: src.interfaces.context.problem_context.ProblemContext.waste_type
:type: str
:value: >
   'plastic'

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.waste_type
```

````

````{py:attribute} n_vehicles
:canonical: src.interfaces.context.problem_context.ProblemContext.n_vehicles
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.n_vehicles
```

````

````{py:attribute} seed
:canonical: src.interfaces.context.problem_context.ProblemContext.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.seed
```

````

````{py:attribute} day_index
:canonical: src.interfaces.context.problem_context.ProblemContext.day_index
:type: int
:value: >
   0

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.day_index
```

````

````{py:attribute} extra
:canonical: src.interfaces.context.problem_context.ProblemContext.extra
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.extra
```

````

````{py:property} current_day
:canonical: src.interfaces.context.problem_context.ProblemContext.current_day
:type: int

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.current_day
```

````

````{py:property} fill_rates
:canonical: src.interfaces.context.problem_context.ProblemContext.fill_rates
:type: numpy.ndarray

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.fill_rates
```

````

````{py:method} from_kwargs(kwargs: typing.Dict[str, typing.Any], capacity: float, revenue_per_kg: float, cost_per_km: float) -> src.interfaces.context.problem_context.ProblemContext
:canonical: src.interfaces.context.problem_context.ProblemContext.from_kwargs
:classmethod:

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.from_kwargs
```

````

````{py:method} advance(route: typing.List[int], delta: typing.Optional[numpy.ndarray] = None) -> src.interfaces.context.problem_context.ProblemContext
:canonical: src.interfaces.context.problem_context.ProblemContext.advance

```{autodoc2-docstring} src.interfaces.context.problem_context.ProblemContext.advance
```

````

`````
