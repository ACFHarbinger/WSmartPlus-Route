# {py:mod}`src.policies.helpers.reinforcement_learning.alns_perturbation_context`

```{py:module} src.policies.helpers.reinforcement_learning.alns_perturbation_context
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSPerturbationContext <src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext>`
  - ```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext
    :summary:
    ```
````

### API

`````{py:class} ALNSPerturbationContext(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float)
:canonical: src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext.__init__
```

````{py:method} _build_structures() -> None
:canonical: src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._build_structures

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._build_structures
```

````

````{py:method} _update_map(changed_routes: set) -> None
:canonical: src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._update_map

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._update_map
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._get_load_cached

```{autodoc2-docstring} src.policies.helpers.reinforcement_learning.alns_perturbation_context.ALNSPerturbationContext._get_load_cached
```

````

`````
