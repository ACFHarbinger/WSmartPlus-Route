# {py:mod}`src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy`

```{py:module} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvolutionStrategy <src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy
    :summary:
    ```
* - {py:obj}`BaldwinianStrategy <src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy
    :summary:
    ```
* - {py:obj}`LamarckianStrategy <src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_evolution_strategy <src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.create_evolution_strategy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.create_evolution_strategy
    :summary:
    ```
````

### API

`````{py:class} EvolutionStrategy
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy
```

````{py:method} get_surviving_vector(original_vector: numpy.ndarray, optimized_routes: typing.List[typing.List[int]], encoder_func: typing.Callable[[typing.List[typing.List[int]]], numpy.ndarray]) -> numpy.ndarray
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy.get_surviving_vector
:abstractmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy.get_surviving_vector
```

````

`````

`````{py:class} BaldwinianStrategy
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy

Bases: {py:obj}`src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy
```

````{py:method} get_surviving_vector(original_vector: numpy.ndarray, optimized_routes: typing.List[typing.List[int]], encoder_func: typing.Callable[[typing.List[typing.List[int]]], numpy.ndarray]) -> numpy.ndarray
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy.get_surviving_vector

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.BaldwinianStrategy.get_surviving_vector
```

````

`````

`````{py:class} LamarckianStrategy
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy

Bases: {py:obj}`src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy
```

````{py:method} get_surviving_vector(original_vector: numpy.ndarray, optimized_routes: typing.List[typing.List[int]], encoder_func: typing.Callable[[typing.List[typing.List[int]]], numpy.ndarray]) -> numpy.ndarray
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy.get_surviving_vector

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.LamarckianStrategy.get_surviving_vector
```

````

`````

````{py:function} create_evolution_strategy(strategy_name: str) -> src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.EvolutionStrategy
:canonical: src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.create_evolution_strategy

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.differential_evolution.evolution_strategy.create_evolution_strategy
```
````
