# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`swap_2_routes <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_2_routes>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_2_routes
    :summary:
    ```
* - {py:obj}`swap_n_2_routes_random <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_random>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_random
    :summary:
    ```
* - {py:obj}`swap_n_2_routes_consecutive <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_consecutive>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_consecutive
    :summary:
    ```
````

### API

````{py:function} swap_2_routes(routes_list: typing.List[typing.List[int]], rng: random.Random) -> None
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_2_routes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_2_routes
```
````

````{py:function} swap_n_2_routes_random(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_random

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_random
```
````

````{py:function} swap_n_2_routes_consecutive(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_consecutive

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.operators.inter_swap.swap_n_2_routes_consecutive
```
````
