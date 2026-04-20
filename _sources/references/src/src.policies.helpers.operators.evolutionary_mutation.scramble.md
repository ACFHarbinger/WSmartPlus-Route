# {py:mod}`src.policies.helpers.operators.evolutionary_mutation.scramble`

```{py:module} src.policies.helpers.operators.evolutionary_mutation.scramble
```

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.scramble
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`scramble_mutation <src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation
    :summary:
    ```
* - {py:obj}`scramble_mutation_profit <src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation_profit
    :summary:
    ```
````

### API

````{py:function} scramble_mutation(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], min_segment: int = 2, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation
```
````

````{py:function} scramble_mutation_profit(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, min_segment: int = 2, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation_profit

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.scramble.scramble_mutation_profit
```
````
