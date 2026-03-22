# {py:mod}`src.policies.tabu_search.solver`

```{py:module} src.policies.tabu_search.solver
```

```{autodoc2-docstring} src.policies.tabu_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSSolver <src.policies.tabu_search.solver.TSSolver>`
  - ```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver
    :summary:
    ```
* - {py:obj}`TSLSAdapter <src.policies.tabu_search.solver.TSLSAdapter>`
  - ```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter
    :summary:
    ```
````

### API

`````{py:class} TSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.tabu_search.params.TSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.tabu_search.solver.TSSolver

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.tabu_search.solver.TSSolver.solve

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver.solve
```

````

````{py:method} _select_best_candidate(candidates: typing.List[typing.Tuple[typing.List[typing.List[int]], typing.Tuple[str, typing.Tuple[int, ...]]]], best_profit: float) -> typing.Tuple[typing.Optional[typing.List[typing.List[int]]], float, typing.Optional[typing.Tuple[str, typing.Tuple[int, ...]]]]
:canonical: src.policies.tabu_search.solver.TSSolver._select_best_candidate

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._select_best_candidate
```

````

````{py:method} _is_tabu(move_desc: typing.Tuple[str, typing.Tuple[int, ...]]) -> bool
:canonical: src.policies.tabu_search.solver.TSSolver._is_tabu

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._is_tabu
```

````

````{py:method} _add_to_tabu_list(move_desc: typing.Tuple[str, typing.Tuple[int, ...]])
:canonical: src.policies.tabu_search.solver.TSSolver._add_to_tabu_list

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._add_to_tabu_list
```

````

````{py:method} _clean_tabu_list()
:canonical: src.policies.tabu_search.solver.TSSolver._clean_tabu_list

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._clean_tabu_list
```

````

````{py:method} _compute_dynamic_tenure() -> int
:canonical: src.policies.tabu_search.solver.TSSolver._compute_dynamic_tenure

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._compute_dynamic_tenure
```

````

````{py:method} _update_frequency_memory(routes: typing.List[typing.List[int]])
:canonical: src.policies.tabu_search.solver.TSSolver._update_frequency_memory

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._update_frequency_memory
```

````

````{py:method} _update_move_frequency(move_desc: typing.Tuple[str, typing.Tuple[int, ...]])
:canonical: src.policies.tabu_search.solver.TSSolver._update_move_frequency

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._update_move_frequency
```

````

````{py:method} _compute_frequency_penalty(move_desc: typing.Tuple[str, typing.Tuple[int, ...]]) -> float
:canonical: src.policies.tabu_search.solver.TSSolver._compute_frequency_penalty

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._compute_frequency_penalty
```

````

````{py:method} _update_elite_pool(routes: typing.List[typing.List[int]], profit: float)
:canonical: src.policies.tabu_search.solver.TSSolver._update_elite_pool

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._update_elite_pool
```

````

````{py:method} _intensification_phase(best_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._intensification_phase

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._intensification_phase
```

````

````{py:method} _diversification_phase(current_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._diversification_phase

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._diversification_phase
```

````

````{py:method} _diversification_restart() -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._diversification_restart

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._diversification_restart
```

````

````{py:method} _build_diversified_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._build_diversified_solution

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._build_diversified_solution
```

````

````{py:method} _path_relink(routes1: typing.List[typing.List[int]], routes2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._path_relink

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._path_relink
```

````

````{py:method} _generate_candidates(routes: typing.List[typing.List[int]]) -> typing.List[typing.Tuple[typing.List[typing.List[int]], typing.Tuple[str, typing.Tuple[int, ...]]]]
:canonical: src.policies.tabu_search.solver.TSSolver._generate_candidates

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._generate_candidates
```

````

````{py:method} _generate_swap_moves(routes: typing.List[typing.List[int]], max_new: int = 5) -> typing.List[typing.Tuple[typing.List[typing.List[int]], typing.Tuple[str, typing.Tuple[int, ...]]]]
:canonical: src.policies.tabu_search.solver.TSSolver._generate_swap_moves

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._generate_swap_moves
```

````

````{py:method} _generate_relocate_moves(routes: typing.List[typing.List[int]], max_new: int = 5) -> typing.List[typing.Tuple[typing.List[typing.List[int]], typing.Tuple[str, typing.Tuple[int, ...]]]]
:canonical: src.policies.tabu_search.solver.TSSolver._generate_relocate_moves

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._generate_relocate_moves
```

````

````{py:method} _generate_2opt_moves(routes: typing.List[typing.List[int]], max_new: int = 3) -> typing.List[typing.Tuple[typing.List[typing.List[int]], typing.Tuple[str, typing.Tuple[int, ...]]]]
:canonical: src.policies.tabu_search.solver.TSSolver._generate_2opt_moves

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._generate_2opt_moves
```

````

````{py:method} _llh_random_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._llh_random_greedy

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._llh_random_greedy
```

````

````{py:method} _llh_worst_regret(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._llh_worst_regret

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._llh_worst_regret
```

````

````{py:method} _llh_cluster_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._llh_cluster_greedy

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._llh_cluster_greedy
```

````

````{py:method} _llh_worst_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._llh_worst_greedy

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._llh_worst_greedy
```

````

````{py:method} _llh_random_regret(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._llh_random_regret

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._llh_random_regret
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.tabu_search.solver.TSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.tabu_search.solver.TSSolver._evaluate

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.tabu_search.solver.TSSolver._cost

```{autodoc2-docstring} src.policies.tabu_search.solver.TSSolver._cost
```

````

`````

`````{py:class} TSLSAdapter(routes, dist_matrix, wastes, capacity, cost_unit)
:canonical: src.policies.tabu_search.solver.TSLSAdapter

```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter.__init__
```

````{py:method} _get_load_cached(r_idx: int) -> float
:canonical: src.policies.tabu_search.solver.TSLSAdapter._get_load_cached

```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter._get_load_cached
```

````

````{py:method} _update_map(affected_indices: typing.Set[int])
:canonical: src.policies.tabu_search.solver.TSLSAdapter._update_map

```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter._update_map
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.tabu_search.solver.TSLSAdapter._calc_load_fresh

```{autodoc2-docstring} src.policies.tabu_search.solver.TSLSAdapter._calc_load_fresh
```

````

`````
