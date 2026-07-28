# {py:mod}`src.policies.helpers.operators.evolutionary_mutation.swap`

```{py:module} src.policies.helpers.operators.evolutionary_mutation.swap
```

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`swap_mutation <src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation
    :summary:
    ```
* - {py:obj}`swap_mutation_profit <src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation_profit
    :summary:
    ```
* - {py:obj}`_encode <src.policies.helpers.operators.evolutionary_mutation.swap._encode>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._encode
    :summary:
    ```
* - {py:obj}`_decode <src.policies.helpers.operators.evolutionary_mutation.swap._decode>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._decode
    :summary:
    ```
* - {py:obj}`_get_demand <src.policies.helpers.operators.evolutionary_mutation.swap._get_demand>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._get_demand
    :summary:
    ```
* - {py:obj}`_repair_capacity_profit <src.policies.helpers.operators.evolutionary_mutation.swap._repair_capacity_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._repair_capacity_profit
    :summary:
    ```
````

### API

````{py:function} swap_mutation(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], n_swaps: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation
```
````

````{py:function} swap_mutation_profit(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, n_swaps: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation_profit

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap.swap_mutation_profit
```
````

````{py:function} _encode(routes: typing.List[typing.List[int]]) -> tuple
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap._encode

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._encode
```
````

````{py:function} _decode(chromosome: typing.List[int], n_vehicles: int) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap._decode

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._decode
```
````

````{py:function} _get_demand(wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], node: int) -> float
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap._get_demand

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._get_demand
```
````

````{py:function} _repair_capacity_profit(routes: typing.List[typing.List[int]], capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, dist: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.swap._repair_capacity_profit

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.swap._repair_capacity_profit
```
````
