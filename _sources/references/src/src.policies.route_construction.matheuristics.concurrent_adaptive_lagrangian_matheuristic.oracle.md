# {py:mod}`src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle`

```{py:module} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PeriodIncumbent <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent
    :summary:
    ```
* - {py:obj}`InsertionCostOracle <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle
    :summary:
    ```
````

### API

`````{py:class} PeriodIncumbent
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent
```

````{py:attribute} tour
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.tour
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.tour
```

````

````{py:attribute} cost
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.cost
:type: float
:value: >
   'float(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.cost
```

````

````{py:attribute} selection
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.selection
:type: frozenset
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent.selection
```

````

`````

`````{py:class} InsertionCostOracle(n_bins: int, horizon: int, alpha: float = 0.3, quality_threshold: float = 1.05, default_delta: float = 0.0)
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.__init__
```

````{py:method} snapshot() -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.snapshot

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.snapshot
```

````

````{py:method} get_incumbent(period: int) -> src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.PeriodIncumbent
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.get_incumbent

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.get_incumbent
```

````

````{py:method} update_from_routing(period: int, tour: typing.List[int], tour_cost: float, selection: typing.List[int], insertion_costs_for_unselected: typing.Dict[int, float]) -> str
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.update_from_routing

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.update_from_routing
```

````

````{py:method} cheapest_insertion_cost(dist_matrix: numpy.ndarray, tour: typing.List[int], bin_id: int) -> float
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.cheapest_insertion_cost
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.cheapest_insertion_cost
```

````

````{py:method} batch_insertion_costs(dist_matrix: numpy.ndarray, tour: typing.List[int], candidates: typing.List[int]) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.batch_insertion_costs
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle.batch_insertion_costs
```

````

`````
