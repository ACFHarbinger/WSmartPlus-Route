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

`````{py:class} VectorizedHGS(dist_matrix: torch.Tensor, demands: torch.Tensor, vehicle_capacity: typing.Any, time_limit: float = 1.0, device: str = 'cuda')
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.__init__
```

````{py:method} solve(initial_solutions: torch.Tensor, n_generations: int = 50, population_size: int = 10, elite_size: int = 5, time_limit: typing.Optional[float] = None, max_vehicles: int = 0) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hybrid_genetic_search.VectorizedHGS.solve

```{autodoc2-docstring} src.models.policies.hybrid_genetic_search.VectorizedHGS.solve
```

````

`````
