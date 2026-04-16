# {py:mod}`src.policies.helpers.operators.perturbation.evolutionary`

```{py:module} src.policies.helpers.operators.perturbation.evolutionary
```

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`evolutionary_perturbation <src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation
    :summary:
    ```
* - {py:obj}`evolutionary_perturbation_profit <src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation_profit
    :summary:
    ```
* - {py:obj}`_decode_chromosome <src.policies.helpers.operators.perturbation.evolutionary._decode_chromosome>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._decode_chromosome
    :summary:
    ```
* - {py:obj}`_eval_cvrp_chromosome <src.policies.helpers.operators.perturbation.evolutionary._eval_cvrp_chromosome>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._eval_cvrp_chromosome
    :summary:
    ```
* - {py:obj}`_eval_vrpp_chromosome <src.policies.helpers.operators.perturbation.evolutionary._eval_vrpp_chromosome>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._eval_vrpp_chromosome
    :summary:
    ```
* - {py:obj}`_map_target_to_global <src.policies.helpers.operators.perturbation.evolutionary._map_target_to_global>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._map_target_to_global
    :summary:
    ```
* - {py:obj}`_apply_to_solution <src.policies.helpers.operators.perturbation.evolutionary._apply_to_solution>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._apply_to_solution
    :summary:
    ```
* - {py:obj}`_order_crossover <src.policies.helpers.operators.perturbation.evolutionary._order_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._order_crossover
    :summary:
    ```
* - {py:obj}`_mutate_swap <src.policies.helpers.operators.perturbation.evolutionary._mutate_swap>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._mutate_swap
    :summary:
    ```
* - {py:obj}`_get_waste_val <src.policies.helpers.operators.perturbation.evolutionary._get_waste_val>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._get_waste_val
    :summary:
    ```
* - {py:obj}`_sequence_cost <src.policies.helpers.operators.perturbation.evolutionary._sequence_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._sequence_cost
    :summary:
    ```
* - {py:obj}`_sequence_profit <src.policies.helpers.operators.perturbation.evolutionary._sequence_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._sequence_profit
    :summary:
    ```
````

### API

````{py:function} evolutionary_perturbation(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], target_routes: typing.List[typing.List[int]], pop_size: int = 10, n_generations: int = 5, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation
```
````

````{py:function} evolutionary_perturbation_profit(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, target_routes: typing.List[typing.List[int]], pop_size: int = 10, n_generations: int = 5, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation_profit

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary.evolutionary_perturbation_profit
```
````

````{py:function} _decode_chromosome(seq: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation.evolutionary._decode_chromosome

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._decode_chromosome
```
````

````{py:function} _eval_cvrp_chromosome(seq: typing.List[int], d: numpy.ndarray, cap: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray]) -> float
:canonical: src.policies.helpers.operators.perturbation.evolutionary._eval_cvrp_chromosome

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._eval_cvrp_chromosome
```
````

````{py:function} _eval_vrpp_chromosome(seq: typing.List[int], d: numpy.ndarray, cap: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.perturbation.evolutionary._eval_vrpp_chromosome

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._eval_vrpp_chromosome
```
````

````{py:function} _map_target_to_global(routes: typing.List[typing.List[int]], targets: typing.List[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.helpers.operators.perturbation.evolutionary._map_target_to_global

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._map_target_to_global
```
````

````{py:function} _apply_to_solution(routes: typing.List[typing.List[int]], indices: typing.List[int], best_seq: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.perturbation.evolutionary._apply_to_solution

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._apply_to_solution
```
````

````{py:function} _order_crossover(p1: typing.List[int], p2: typing.List[int], rng: random.Random) -> typing.List[int]
:canonical: src.policies.helpers.operators.perturbation.evolutionary._order_crossover

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._order_crossover
```
````

````{py:function} _mutate_swap(seq: typing.List[int], rng: random.Random, prob: float) -> None
:canonical: src.policies.helpers.operators.perturbation.evolutionary._mutate_swap

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._mutate_swap
```
````

````{py:function} _get_waste_val(wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], node: int) -> float
:canonical: src.policies.helpers.operators.perturbation.evolutionary._get_waste_val

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._get_waste_val
```
````

````{py:function} _sequence_cost(d: numpy.ndarray, seq: typing.List[int]) -> float
:canonical: src.policies.helpers.operators.perturbation.evolutionary._sequence_cost

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._sequence_cost
```
````

````{py:function} _sequence_profit(seq: typing.List[int], d: numpy.ndarray, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.perturbation.evolutionary._sequence_profit

```{autodoc2-docstring} src.policies.helpers.operators.perturbation.evolutionary._sequence_profit
```
````
