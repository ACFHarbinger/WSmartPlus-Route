# {py:mod}`src.policies.helpers.operators.perturbation_shaking.sans.intra_swap`

```{py:module} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap
```

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`swap_1_route <src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_1_route>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_1_route
    :summary:
    ```
* - {py:obj}`swap_n_route_random <src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_random>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_random
    :summary:
    ```
* - {py:obj}`swap_n_route_consecutive <src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_consecutive
    :summary:
    ```
````

### API

````{py:function} swap_1_route(routes_list: typing.List[typing.List[int]], rng: random.Random) -> None
:canonical: src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_1_route

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_1_route
```
````

````{py:function} swap_n_route_random(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_random

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_random
```
````

````{py:function} swap_n_route_consecutive(routes_list: typing.List[typing.List[int]], rng: random.Random, n: typing.Optional[int] = None) -> typing.Optional[int]
:canonical: src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_consecutive

```{autodoc2-docstring} src.policies.helpers.operators.perturbation_shaking.sans.intra_swap.swap_n_route_consecutive
```
````
