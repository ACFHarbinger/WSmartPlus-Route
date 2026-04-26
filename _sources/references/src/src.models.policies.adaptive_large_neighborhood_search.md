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

`````{py:class} VectorizedALNS(dist_matrix: torch.Tensor, wastes: torch.Tensor, vehicle_capacity: float, time_limit: float = 1.0, device: str = 'cuda', seed: int = 42, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__init__
```

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__getstate__

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__setstate__

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.__setstate__
```

````

````{py:method} solve(initial_solutions: torch.Tensor, n_iterations: int = 2000, time_limit: typing.Optional[float] = None, max_vehicles: int = 0, start_temp: float = 0.5, cooling_rate: float = 0.9995) -> typing.Tuple[typing.List[typing.List[int]], torch.Tensor]
:canonical: src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.solve

```{autodoc2-docstring} src.models.policies.adaptive_large_neighborhood_search.VectorizedALNS.solve
```

````

`````
