# {py:mod}`src.policies.other.operators.perturbation.evolutionary`

```{py:module} src.policies.other.operators.perturbation.evolutionary
```

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`evolutionary_perturbation <src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation
    :summary:
    ```
* - {py:obj}`evolutionary_perturbation_profit <src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation_profit
    :summary:
    ```
* - {py:obj}`_cluster_profit <src.policies.other.operators.perturbation.evolutionary._cluster_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._cluster_profit
    :summary:
    ```
* - {py:obj}`_sequence_profit <src.policies.other.operators.perturbation.evolutionary._sequence_profit>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._sequence_profit
    :summary:
    ```
* - {py:obj}`_select_target_routes <src.policies.other.operators.perturbation.evolutionary._select_target_routes>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._select_target_routes
    :summary:
    ```
* - {py:obj}`_cluster_cost <src.policies.other.operators.perturbation.evolutionary._cluster_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._cluster_cost
    :summary:
    ```
* - {py:obj}`_sequence_cost <src.policies.other.operators.perturbation.evolutionary._sequence_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._sequence_cost
    :summary:
    ```
* - {py:obj}`_order_crossover <src.policies.other.operators.perturbation.evolutionary._order_crossover>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._order_crossover
    :summary:
    ```
* - {py:obj}`_mutate_swap <src.policies.other.operators.perturbation.evolutionary._mutate_swap>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._mutate_swap
    :summary:
    ```
* - {py:obj}`_apply_cluster <src.policies.other.operators.perturbation.evolutionary._apply_cluster>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._apply_cluster
    :summary:
    ```
````

### API

````{py:function} evolutionary_perturbation(ls: typing.Any, target_routes: typing.Optional[typing.List[int]] = None, pop_size: int = 10, n_generations: int = 5, rng: typing.Optional[random.Random] = None) -> bool
:canonical: src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation
```
````

````{py:function} evolutionary_perturbation_profit(ls: typing.Any, target_routes: typing.Optional[typing.List[int]] = None, pop_size: int = 10, n_generations: int = 5, rng: typing.Optional[random.Random] = None) -> bool
:canonical: src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation_profit

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary.evolutionary_perturbation_profit
```
````

````{py:function} _cluster_profit(ls: typing.Any, route_indices: typing.List[int]) -> float
:canonical: src.policies.other.operators.perturbation.evolutionary._cluster_profit

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._cluster_profit
```
````

````{py:function} _sequence_profit(ls: typing.Any, seq: typing.List[int]) -> float
:canonical: src.policies.other.operators.perturbation.evolutionary._sequence_profit

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._sequence_profit
```
````

````{py:function} _select_target_routes(ls: typing.Any, n: int = 2) -> typing.List[int]
:canonical: src.policies.other.operators.perturbation.evolutionary._select_target_routes

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._select_target_routes
```
````

````{py:function} _cluster_cost(ls: typing.Any, route_indices: typing.List[int]) -> float
:canonical: src.policies.other.operators.perturbation.evolutionary._cluster_cost

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._cluster_cost
```
````

````{py:function} _sequence_cost(d, seq: typing.List[int]) -> float
:canonical: src.policies.other.operators.perturbation.evolutionary._sequence_cost

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._sequence_cost
```
````

````{py:function} _order_crossover(p1: typing.List[int], p2: typing.List[int], rng: random.Random) -> typing.List[int]
:canonical: src.policies.other.operators.perturbation.evolutionary._order_crossover

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._order_crossover
```
````

````{py:function} _mutate_swap(seq: typing.List[int], rng: random.Random, prob: float = 0.3) -> None
:canonical: src.policies.other.operators.perturbation.evolutionary._mutate_swap

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._mutate_swap
```
````

````{py:function} _apply_cluster(ls: typing.Any, route_indices: typing.List[int], best_seq: typing.List[int]) -> None
:canonical: src.policies.other.operators.perturbation.evolutionary._apply_cluster

```{autodoc2-docstring} src.policies.other.operators.perturbation.evolutionary._apply_cluster
```
````
