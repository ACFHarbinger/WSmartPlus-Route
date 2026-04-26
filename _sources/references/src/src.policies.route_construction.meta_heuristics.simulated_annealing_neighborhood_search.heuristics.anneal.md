# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_annealing_loop <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop
    :summary:
    ```
* - {py:obj}`_update_removed_bins <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins
    :summary:
    ```
* - {py:obj}`_rollback_removed_bins <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins
    :summary:
    ```
````

### API

````{py:function} run_annealing_loop(initial_solution: typing.List[typing.List[int]], data: pandas.DataFrame, distance_matrix: numpy.ndarray, mandatory_bins: typing.List[int], values: typing.Dict, n_bins: int, chosen_combination: typing.Tuple, time_limit: float, rng: random.Random, np_rng: numpy.random.Generator) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop
```
````

````{py:function} _update_removed_bins(proc: str, removed: typing.List[int], b_rem: typing.Optional[int], b_add: typing.Optional[int], bs_rem_rnd: typing.List[int], bs_rem_con: typing.List[int], bs_add_rnd: typing.List[int], bs_add_con: typing.List[int], bs_rnd: typing.List[int], bs_con: typing.List[int])
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins
```
````

````{py:function} _rollback_removed_bins(proc: str, removed: typing.List[int], b_rem: typing.Optional[int], b_add: typing.Optional[int], bs_rem_rnd: typing.List[int], bs_rem_con: typing.List[int], bs_add_rnd: typing.List[int], bs_add_con: typing.List[int], bs_rnd: typing.List[int], bs_con: typing.List[int])
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins
```
````
