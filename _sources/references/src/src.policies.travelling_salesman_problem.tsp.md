# {py:mod}`src.policies.travelling_salesman_problem.tsp`

```{py:module} src.policies.travelling_salesman_problem.tsp
```

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`find_route <src.policies.travelling_salesman_problem.tsp.find_route>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.find_route
    :summary:
    ```
* - {py:obj}`get_route_cost <src.policies.travelling_salesman_problem.tsp.get_route_cost>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_route_cost
    :summary:
    ```
* - {py:obj}`get_path_cost <src.policies.travelling_salesman_problem.tsp.get_path_cost>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_path_cost
    :summary:
    ```
* - {py:obj}`get_multi_tour <src.policies.travelling_salesman_problem.tsp.get_multi_tour>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_multi_tour
    :summary:
    ```
* - {py:obj}`get_partial_tour <src.policies.travelling_salesman_problem.tsp.get_partial_tour>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_partial_tour
    :summary:
    ```
* - {py:obj}`dist_matrix_from_graph <src.policies.travelling_salesman_problem.tsp.dist_matrix_from_graph>`
  - ```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.dist_matrix_from_graph
    :summary:
    ```
````

### API

````{py:function} find_route(C, to_collect, time_limit=2.0, seed=42)
:canonical: src.policies.travelling_salesman_problem.tsp.find_route

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.find_route
```
````

````{py:function} get_route_cost(distancesC, tour)
:canonical: src.policies.travelling_salesman_problem.tsp.get_route_cost

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_route_cost
```
````

````{py:function} get_path_cost(G, p)
:canonical: src.policies.travelling_salesman_problem.tsp.get_path_cost

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_path_cost
```
````

````{py:function} get_multi_tour(tour, bins_waste, max_capacity, distance_matrix)
:canonical: src.policies.travelling_salesman_problem.tsp.get_multi_tour

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_multi_tour
```
````

````{py:function} get_partial_tour(tour: typing.List[int], bins: numpy.ndarray, max_capacity: float, distance_matrix: numpy.ndarray, cost: float) -> typing.Tuple[numpy.ndarray, float]
:canonical: src.policies.travelling_salesman_problem.tsp.get_partial_tour

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.get_partial_tour
```
````

````{py:function} dist_matrix_from_graph(G: networkx.Graph) -> typing.Tuple[numpy.ndarray, typing.List[typing.List[typing.List[int]]]]
:canonical: src.policies.travelling_salesman_problem.tsp.dist_matrix_from_graph

```{autodoc2-docstring} src.policies.travelling_salesman_problem.tsp.dist_matrix_from_graph
```
````
