# {py:mod}`src.policies.simulated_annealing_neighborhood_search.heuristics.anneal`

```{py:module} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_annealing_loop <src.policies.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop
    :summary:
    ```
* - {py:obj}`_update_removed_bins <src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins
    :summary:
    ```
* - {py:obj}`_rollback_removed_bins <src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins
    :summary:
    ```
````

### API

````{py:function} run_annealing_loop(initial_solution: typing.List[typing.List[int]], data: pandas.DataFrame, distance_matrix: numpy.ndarray, must_go_bins: typing.List[int], values: typing.Dict, n_bins: int, chosen_combination: typing.Tuple, time_limit: float) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal.run_annealing_loop
```
````

````{py:function} _update_removed_bins(proc, removed, b_rem, b_add, bs_rem_rnd, bs_rem_con, bs_add_rnd, bs_add_con, bs_rnd, bs_con)
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._update_removed_bins
```
````

````{py:function} _rollback_removed_bins(proc, removed, b_rem, b_add, bs_rem_rnd, bs_rem_con, bs_add_rnd, bs_add_con, bs_rnd, bs_con)
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.anneal._rollback_removed_bins
```
````
