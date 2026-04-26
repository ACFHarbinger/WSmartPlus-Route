# {py:mod}`src.models.policies.hybrid_genetic_search`

```{py:module} src.models.policies.hybrid_genetic_search
```

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHGS <src.models.policies.hybrid_genetic_search.VectorizedHGS>`
  - ```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS
    :summary:
    ```
````

### API

`````{py:class} VectorizedHGS(dist_matrix: torch.Tensor, wastes: torch.Tensor, vehicle_capacity: typing.Union[float, torch.Tensor], max_iterations: int = 50, time_limit: float = 1.0, device: str = 'cuda', seed: int = 42, generator: typing.Optional[torch.Generator] = None, rng: typing.Optional[random.Random] = None, **kwargs: typing.Any)
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.__init__
```

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS.__getstate__

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS.__setstate__

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.__setstate__
```

````

````{py:method} solve(initial_solutions: torch.Tensor, n_generations: int = 50, population_size: int = 10, elite_size: int = 5, time_limit: typing.Optional[float] = None, max_vehicles: int = 0, crossover_rate: float = 0.7) -> typing.Tuple[typing.Union[torch.Tensor, typing.List[typing.List[int]]], torch.Tensor]
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS.solve

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.solve
```

````

````{py:method} educate(routes_list: typing.List[typing.List[int]], split_costs: torch.Tensor, max_vehicles: int = 0) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS.educate

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.educate
```

````

`````
