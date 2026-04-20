# {py:mod}`src.policies.route_improvement.simulated_annealing`

```{py:module} src.policies.route_improvement.simulated_annealing
```

```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulatedAnnealingRouteImprover <src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover
    :summary:
    ```
````

### API

`````{py:class} SimulatedAnnealingRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover.process
```

````

````{py:method} _random_move(routes: typing.List[typing.List[int]], dm: numpy.ndarray, rng: numpy.random.Generator, wastes: typing.Dict[int, float], capacity: float) -> typing.Tuple[typing.Any, float]
:canonical: src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover._random_move

```{autodoc2-docstring} src.policies.route_improvement.simulated_annealing.SimulatedAnnealingRouteImprover._random_move
```

````

`````
