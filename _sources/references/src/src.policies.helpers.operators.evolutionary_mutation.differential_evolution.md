# {py:mod}`src.policies.helpers.operators.evolutionary_mutation.differential_evolution`

```{py:module} src.policies.helpers.operators.evolutionary_mutation.differential_evolution
```

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`de_rand_1_mutation <src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_rand_1_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_rand_1_mutation
    :summary:
    ```
* - {py:obj}`de_best_1_mutation <src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_best_1_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_best_1_mutation
    :summary:
    ```
* - {py:obj}`_de_mutate <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._de_mutate>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._de_mutate
    :summary:
    ```
* - {py:obj}`_encode <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._encode>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._encode
    :summary:
    ```
* - {py:obj}`_copy_solution <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._copy_solution>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._copy_solution
    :summary:
    ```
* - {py:obj}`_get_demand <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._get_demand>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._get_demand
    :summary:
    ```
* - {py:obj}`_total_cost <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._total_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._total_cost
    :summary:
    ```
* - {py:obj}`_split_into_routes <src.policies.helpers.operators.evolutionary_mutation.differential_evolution._split_into_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._split_into_routes
    :summary:
    ```
````

### API

````{py:function} de_rand_1_mutation(population: typing.List[typing.List[typing.List[int]]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], F: float = 0.5, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_rand_1_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_rand_1_mutation
```
````

````{py:function} de_best_1_mutation(population: typing.List[typing.List[typing.List[int]]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], F: float = 0.5, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_best_1_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution.de_best_1_mutation
```
````

````{py:function} _de_mutate(target: typing.List[typing.List[int]], base: typing.List[typing.List[int]], donor2: typing.List[typing.List[int]], donor3: typing.List[typing.List[int]], F: float, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._de_mutate

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._de_mutate
```
````

````{py:function} _encode(routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[int], int]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._encode

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._encode
```
````

````{py:function} _copy_solution(sol: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._copy_solution

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._copy_solution
```
````

````{py:function} _get_demand(wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], node: int) -> float
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._get_demand

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._get_demand
```
````

````{py:function} _total_cost(routes: typing.List[typing.List[int]], dist: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._total_cost

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._total_cost
```
````

````{py:function} _split_into_routes(nodes: typing.List[int], n_vehicles: int, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.differential_evolution._split_into_routes

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.differential_evolution._split_into_routes
```
````
