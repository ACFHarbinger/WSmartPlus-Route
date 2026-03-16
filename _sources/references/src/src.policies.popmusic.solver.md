# {py:mod}`src.policies.popmusic.solver`

```{py:module} src.policies.popmusic.solver
```

```{autodoc2-docstring} src.policies.popmusic.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_popmusic <src.policies.popmusic.solver.run_popmusic>`
  - ```{autodoc2-docstring} src.policies.popmusic.solver.run_popmusic
    :summary:
    ```
* - {py:obj}`find_route_neighbors <src.policies.popmusic.solver.find_route_neighbors>`
  - ```{autodoc2-docstring} src.policies.popmusic.solver.find_route_neighbors
    :summary:
    ```
* - {py:obj}`split_tour <src.policies.popmusic.solver.split_tour>`
  - ```{autodoc2-docstring} src.policies.popmusic.solver.split_tour
    :summary:
    ```
````

### API

````{py:function} run_popmusic(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, n_vehicles: int, subproblem_size: int = 3, max_iterations: int = 100, base_solver: str = 'fast_tsp', seed: int = 42, wastes: typing.Optional[typing.Dict[int, float]] = None, capacity: float = 1000000000.0, R: float = 1.0, C: float = 0.0) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.popmusic.solver.run_popmusic

```{autodoc2-docstring} src.policies.popmusic.solver.run_popmusic
```
````

````{py:function} find_route_neighbors(seed_idx: int, centroids: typing.List[numpy.ndarray], k: int) -> typing.List[int]
:canonical: src.policies.popmusic.solver.find_route_neighbors

```{autodoc2-docstring} src.policies.popmusic.solver.find_route_neighbors
```
````

````{py:function} split_tour(tour: typing.List[int], k: int, distance_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.popmusic.solver.split_tour

```{autodoc2-docstring} src.policies.popmusic.solver.split_tour
```
````
