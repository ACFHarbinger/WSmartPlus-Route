# {py:mod}`src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun`

```{py:module} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun
```

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_alpha_measures <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_alpha_measures>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_alpha_measures
    :summary:
    ```
* - {py:obj}`get_candidate_set <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.get_candidate_set>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.get_candidate_set
    :summary:
    ```
* - {py:obj}`_improve_tour <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun._improve_tour>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun._improve_tour
    :summary:
    ```
* - {py:obj}`compute_1tree <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_1tree>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_1tree
    :summary:
    ```
* - {py:obj}`run_subgradient <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.run_subgradient>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.run_subgradient
    :summary:
    ```
* - {py:obj}`solve_lkh1 <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh1>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh1
    :summary:
    ```
* - {py:obj}`solve_lkh <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh
    :summary:
    ```
````

### API

````{py:function} compute_alpha_measures(distance_matrix: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_alpha_measures

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_alpha_measures
```
````

````{py:function} get_candidate_set(distance_matrix: numpy.ndarray, alpha_measures: numpy.ndarray, max_candidates: int = 5) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.get_candidate_set

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.get_candidate_set
```
````

````{py:function} _improve_tour(curr_tour: typing.List[int], curr_cost: float, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, rng: random.Random, dont_look_bits: typing.Optional[numpy.ndarray] = None, max_k: int = 5, fixed_edges: typing.Optional[typing.Union[typing.Set[typing.Tuple[int, int]], typing.FrozenSet[typing.Tuple[int, int]]]] = None) -> typing.Tuple[typing.List[int], float, bool, typing.Optional[numpy.ndarray]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun._improve_tour

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun._improve_tour
```
````

````{py:function} compute_1tree(penalized_distances: numpy.ndarray, root: int = 0) -> typing.Tuple[float, numpy.ndarray]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_1tree

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.compute_1tree
```
````

````{py:function} run_subgradient(distance_matrix: numpy.ndarray, max_iter: int = 100, mu_init: float = 1.0, patience: int = 20, root: int = 0, verbose: bool = False) -> typing.Tuple[numpy.ndarray, float, numpy.ndarray]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.run_subgradient

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.run_subgradient
```
````

````{py:function} solve_lkh1(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 100, max_k: int = 5, n_candidates: int = 5, sg_max_iter: int = 100, sg_mu_init: float = 1.0, sg_patience: int = 20, pool_size: int = 5, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh1

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh1
```
````

````{py:function} solve_lkh(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 100, max_k: int = 5, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun.solve_lkh
```
````
