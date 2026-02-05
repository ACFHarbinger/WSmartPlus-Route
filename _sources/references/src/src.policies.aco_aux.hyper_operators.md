# {py:mod}`src.policies.aco_aux.hyper_operators`

```{py:module} src.policies.aco_aux.hyper_operators
```

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperOperatorContext <src.policies.aco_aux.hyper_operators.HyperOperatorContext>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_2opt_intra <src.policies.aco_aux.hyper_operators.apply_2opt_intra>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_2opt_intra
    :summary:
    ```
* - {py:obj}`apply_3opt_intra <src.policies.aco_aux.hyper_operators.apply_3opt_intra>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_3opt_intra
    :summary:
    ```
* - {py:obj}`apply_2opt_star <src.policies.aco_aux.hyper_operators.apply_2opt_star>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_2opt_star
    :summary:
    ```
* - {py:obj}`apply_swap <src.policies.aco_aux.hyper_operators.apply_swap>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_swap
    :summary:
    ```
* - {py:obj}`apply_swap_star <src.policies.aco_aux.hyper_operators.apply_swap_star>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_swap_star
    :summary:
    ```
* - {py:obj}`apply_relocate <src.policies.aco_aux.hyper_operators.apply_relocate>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_relocate
    :summary:
    ```
* - {py:obj}`apply_perturb <src.policies.aco_aux.hyper_operators.apply_perturb>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_perturb
    :summary:
    ```
* - {py:obj}`apply_kick <src.policies.aco_aux.hyper_operators.apply_kick>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_kick
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HYPER_OPERATORS <src.policies.aco_aux.hyper_operators.HYPER_OPERATORS>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HYPER_OPERATORS
    :summary:
    ```
* - {py:obj}`OPERATOR_NAMES <src.policies.aco_aux.hyper_operators.OPERATOR_NAMES>`
  - ```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.OPERATOR_NAMES
    :summary:
    ```
````

### API

`````{py:class} HyperOperatorContext(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, C: float)
:canonical: src.policies.aco_aux.hyper_operators.HyperOperatorContext

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext.__init__
```

````{py:method} _build_structures()
:canonical: src.policies.aco_aux.hyper_operators.HyperOperatorContext._build_structures

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext._build_structures
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.aco_aux.hyper_operators.HyperOperatorContext._calc_load_fresh

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext._calc_load_fresh
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.aco_aux.hyper_operators.HyperOperatorContext._get_load_cached

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext._get_load_cached
```

````

````{py:method} _update_map(affected_indices: set)
:canonical: src.policies.aco_aux.hyper_operators.HyperOperatorContext._update_map

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HyperOperatorContext._update_map
```

````

`````

````{py:function} apply_2opt_intra(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_2opt_intra

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_2opt_intra
```
````

````{py:function} apply_3opt_intra(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_3opt_intra

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_3opt_intra
```
````

````{py:function} apply_2opt_star(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_2opt_star

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_2opt_star
```
````

````{py:function} apply_swap(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_swap

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_swap
```
````

````{py:function} apply_swap_star(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_swap_star

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_swap_star
```
````

````{py:function} apply_relocate(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, max_attempts: int = 50) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_relocate

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_relocate
```
````

````{py:function} apply_perturb(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, k: int = 3) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_perturb

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_perturb
```
````

````{py:function} apply_kick(ctx: src.policies.aco_aux.hyper_operators.HyperOperatorContext, destroy_ratio: float = 0.2) -> bool
:canonical: src.policies.aco_aux.hyper_operators.apply_kick

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.apply_kick
```
````

````{py:data} HYPER_OPERATORS
:canonical: src.policies.aco_aux.hyper_operators.HYPER_OPERATORS
:type: typing.Dict[str, typing.Callable[[src.policies.aco_aux.hyper_operators.HyperOperatorContext], bool]]
:value: >
   None

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.HYPER_OPERATORS
```

````

````{py:data} OPERATOR_NAMES
:canonical: src.policies.aco_aux.hyper_operators.OPERATOR_NAMES
:value: >
   'list(...)'

```{autodoc2-docstring} src.policies.aco_aux.hyper_operators.OPERATOR_NAMES
```

````
