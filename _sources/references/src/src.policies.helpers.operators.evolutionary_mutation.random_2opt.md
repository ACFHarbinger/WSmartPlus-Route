# {py:mod}`src.policies.helpers.operators.evolutionary_mutation.random_2opt`

```{py:module} src.policies.helpers.operators.evolutionary_mutation.random_2opt
```

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.random_2opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`random_2opt_mutation <src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation
    :summary:
    ```
* - {py:obj}`random_2opt_mutation_profit <src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation_profit
    :summary:
    ```
````

### API

````{py:function} random_2opt_mutation(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], n_moves: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation
```
````

````{py:function} random_2opt_mutation_profit(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, n_moves: int = 1, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation_profit

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.random_2opt.random_2opt_mutation_profit
```
````
