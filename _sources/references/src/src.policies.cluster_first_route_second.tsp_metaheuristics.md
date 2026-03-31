# {py:mod}`src.policies.cluster_first_route_second.tsp_metaheuristics`

```{py:module} src.policies.cluster_first_route_second.tsp_metaheuristics
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_calculate_tour_distance <src.policies.cluster_first_route_second.tsp_metaheuristics._calculate_tour_distance>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._calculate_tour_distance
    :summary:
    ```
* - {py:obj}`_two_opt_swap <src.policies.cluster_first_route_second.tsp_metaheuristics._two_opt_swap>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._two_opt_swap
    :summary:
    ```
* - {py:obj}`_apply_swap_sequence <src.policies.cluster_first_route_second.tsp_metaheuristics._apply_swap_sequence>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._apply_swap_sequence
    :summary:
    ```
* - {py:obj}`_get_swap_sequence <src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_sequence>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_sequence
    :summary:
    ```
* - {py:obj}`_get_swap_operators <src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_operators>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_operators
    :summary:
    ```
* - {py:obj}`_apply_2opt_local_search <src.policies.cluster_first_route_second.tsp_metaheuristics._apply_2opt_local_search>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._apply_2opt_local_search
    :summary:
    ```
* - {py:obj}`find_route_pso <src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_pso>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_pso
    :summary:
    ```
* - {py:obj}`find_route_aco <src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_aco>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_aco
    :summary:
    ```
* - {py:obj}`_eer_crossover <src.policies.cluster_first_route_second.tsp_metaheuristics._eer_crossover>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._eer_crossover
    :summary:
    ```
* - {py:obj}`find_route_ga <src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_ga>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_ga
    :summary:
    ```
````

### API

````{py:function} _calculate_tour_distance(tour: typing.List[int], distance_matrix: numpy.ndarray) -> float
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._calculate_tour_distance

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._calculate_tour_distance
```
````

````{py:function} _two_opt_swap(tour: typing.List[int], i: int, k: int) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._two_opt_swap

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._two_opt_swap
```
````

````{py:function} _apply_swap_sequence(tour: typing.List[int], swap_ops: typing.List[typing.Tuple[int, int]], max_swaps: int = 3) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._apply_swap_sequence

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._apply_swap_sequence
```
````

````{py:function} _get_swap_sequence(source: typing.List[int], target: typing.List[int]) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_sequence

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_sequence
```
````

````{py:function} _get_swap_operators(tour1: typing.List[int], tour2: typing.List[int]) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_operators

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._get_swap_operators
```
````

````{py:function} _apply_2opt_local_search(tour: typing.List[int], distance_matrix: numpy.ndarray, n_customers: int, max_swaps: int = 10) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._apply_2opt_local_search

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._apply_2opt_local_search
```
````

````{py:function} find_route_pso(distance_matrix: numpy.ndarray, cluster: typing.List[int], time_limit: float = 60.0, seed: int = 42, n_particles: int = 50, c1: float = 1.5, c2: float = 1.5) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_pso

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_pso
```
````

````{py:function} find_route_aco(distance_matrix: numpy.ndarray, cluster: typing.List[int], time_limit: float = 60.0, seed: int = 42, n_ants: typing.Optional[int] = None, alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, q: float = 100.0) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_aco

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_aco
```
````

````{py:function} _eer_crossover(p1: typing.List[int], p2: typing.List[int], customers: typing.List[int]) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics._eer_crossover

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics._eer_crossover
```
````

````{py:function} find_route_ga(distance_matrix: numpy.ndarray, cluster: typing.List[int], time_limit: float = 60.0, seed: int = 42, pop_size: int = 50, mutation_rate: float = 0.2) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_ga

```{autodoc2-docstring} src.policies.cluster_first_route_second.tsp_metaheuristics.find_route_ga
```
````
