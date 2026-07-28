# {py:mod}`src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic`

```{py:module} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`decompose_tour <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.decompose_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.decompose_tour
    :summary:
    ```
* - {py:obj}`optimize_subpath <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.optimize_subpath>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.optimize_subpath
    :summary:
    ```
* - {py:obj}`_collect_edges <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._collect_edges>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._collect_edges
    :summary:
    ```
* - {py:obj}`_generate_randomized_tour <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._generate_randomized_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._generate_randomized_tour
    :summary:
    ```
* - {py:obj}`generate_optimized_tour <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.generate_optimized_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.generate_optimized_tour
    :summary:
    ```
* - {py:obj}`popmusic_candidates <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.popmusic_candidates>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.popmusic_candidates
    :summary:
    ```
````

### API

````{py:function} decompose_tour(tour: typing.List[int], subpath_size: int, overlap: int = 5) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.decompose_tour

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.decompose_tour
```
````

````{py:function} optimize_subpath(subpath: typing.List[int], distance_matrix: numpy.ndarray, max_trials: int = 50) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.optimize_subpath

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.optimize_subpath
```
````

````{py:function} _collect_edges(path: typing.List[int]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._collect_edges

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._collect_edges
```
````

````{py:function} _generate_randomized_tour(initial_tour: typing.List[int], np_rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._generate_randomized_tour

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic._generate_randomized_tour
```
````

````{py:function} generate_optimized_tour(initial_tour: typing.List[int], distance_matrix: numpy.ndarray, subpath_size: int, max_trials: int, np_rng: numpy.random.Generator) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.generate_optimized_tour

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.generate_optimized_tour
```
````

````{py:function} popmusic_candidates(distance_matrix: numpy.ndarray, initial_tour: typing.List[int], coords: typing.Optional[numpy.ndarray] = None, subpath_size: int = 50, n_runs: int = 5, max_trials: int = 50, max_candidates: int = 5, np_rng: typing.Optional[numpy.random.Generator] = None) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.popmusic_candidates

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic.popmusic_candidates
```
````
