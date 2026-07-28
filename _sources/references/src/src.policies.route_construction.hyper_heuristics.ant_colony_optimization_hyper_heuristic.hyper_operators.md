# {py:mod}`src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators`

```{py:module} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperOperatorContext <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_2opt_intra <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_intra>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_intra
    :summary:
    ```
* - {py:obj}`apply_3opt_intra <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_3opt_intra>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_3opt_intra
    :summary:
    ```
* - {py:obj}`apply_2opt_star <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_star>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_star
    :summary:
    ```
* - {py:obj}`apply_swap <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap
    :summary:
    ```
* - {py:obj}`apply_swap_star <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap_star>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap_star
    :summary:
    ```
* - {py:obj}`apply_relocate <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_relocate>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_relocate
    :summary:
    ```
* - {py:obj}`apply_perturb <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_perturb>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_perturb
    :summary:
    ```
* - {py:obj}`apply_kick <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_kick>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_kick
    :summary:
    ```
* - {py:obj}`apply_shaw_removal <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_shaw_removal>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_shaw_removal
    :summary:
    ```
* - {py:obj}`apply_string_removal <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_string_removal>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_string_removal
    :summary:
    ```
* - {py:obj}`apply_random_removal <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_random_removal>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_random_removal
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HYPER_OPERATORS <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HYPER_OPERATORS>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HYPER_OPERATORS
    :summary:
    ```
* - {py:obj}`OPERATOR_NAMES <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.OPERATOR_NAMES>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.OPERATOR_NAMES
    :summary:
    ```
````

### API

`````{py:class} HyperOperatorContext(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, profit_aware_operators: bool = False, vrpp: bool = True, use_cache: bool = False)
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext.__init__
```

````{py:method} _build_structures() -> None
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._build_structures

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._build_structures
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._calc_load_fresh

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._calc_load_fresh
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._get_load_cached

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._get_load_cached
```

````

````{py:method} get_dist(i: int, j: int) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext.get_dist

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext.get_dist
```

````

````{py:method} _update_map(affected_indices: set)
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._update_map

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._update_map
```

````

````{py:method} _compute_top_insertions(route_idx: typing.Optional[int] = None)
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._compute_top_insertions

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext._compute_top_insertions
```

````

`````

````{py:function} apply_2opt_intra(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_intra

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_intra
```
````

````{py:function} apply_3opt_intra(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_3opt_intra

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_3opt_intra
```
````

````{py:function} apply_2opt_star(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_star

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_2opt_star
```
````

````{py:function} apply_swap(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap
```
````

````{py:function} apply_swap_star(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap_star

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_swap_star
```
````

````{py:function} apply_relocate(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_relocate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_relocate
```
````

````{py:function} apply_perturb(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, k: int = 3) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_perturb

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_perturb
```
````

````{py:function} apply_kick(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, destroy_ratio: float = 0.2) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_kick

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_kick
```
````

````{py:function} apply_shaw_removal(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, n: typing.Optional[int] = None) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_shaw_removal

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_shaw_removal
```
````

````{py:function} apply_string_removal(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, n: typing.Optional[int] = None) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_string_removal

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_string_removal
```
````

````{py:function} apply_random_removal(ctx: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext, n: typing.Optional[int] = None) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_random_removal

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.apply_random_removal
```
````

````{py:data} HYPER_OPERATORS
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HYPER_OPERATORS
:type: typing.Dict[str, typing.Callable[[src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext], bool]]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HYPER_OPERATORS
```

````

````{py:data} OPERATOR_NAMES
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.OPERATOR_NAMES
:value: >
   'list(...)'

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.OPERATOR_NAMES
```

````
