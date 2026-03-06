# {py:mod}`src.models.policies.adaptive_large_neighborhood_search`

```{py:module} src.models.policies.adaptive_large_neighborhood_search
```

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedALNS <src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS>`
  - ```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedALNS(dist_matrix, wastes, vehicle_capacity, time_limit=1.0, device: str = 'cuda', generator: typing.Optional[torch.Generator] = None)
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__init__
```

````{py:method} solve(initial_solutions: torch.Tensor, n_iterations: int = 2000, time_limit: typing.Optional[float] = None, max_vehicles: int = 0, start_temp: float = 0.5, cooling_rate: float = 0.9995) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.solve

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.solve
```

````

`````
