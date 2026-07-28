# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_2_routes <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_2_routes>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_2_routes
    :summary:
    ```
* - {py:obj}`move_n_2_routes_random <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_random>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_random
    :summary:
    ```
* - {py:obj}`move_n_2_routes_consecutive <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_consecutive>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_consecutive
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.__all__>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.__all__
:value: >
   ['move_2_routes', 'move_n_2_routes_random', 'move_n_2_routes_consecutive']

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.__all__
```

````

````{py:function} move_2_routes(routes_list: typing.List[typing.List[int]], rng: random.Random) -> None
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_2_routes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_2_routes
```
````

````{py:function} move_n_2_routes_random(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_random

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_random
```
````

````{py:function} move_n_2_routes_consecutive(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_consecutive

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_move.move_n_2_routes_consecutive
```
````
