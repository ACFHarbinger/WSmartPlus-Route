# {py:mod}`src.policies.helpers.operators.evolutionary_mutation.inversion`

```{py:module} src.policies.helpers.operators.evolutionary_mutation.inversion
```

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`inversion_mutation <src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation
    :summary:
    ```
* - {py:obj}`inversion_mutation_profit <src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation_profit
    :summary:
    ```
* - {py:obj}`_get_demand <src.policies.helpers.operators.evolutionary_mutation.inversion._get_demand>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion._get_demand
    :summary:
    ```
````

### API

````{py:function} inversion_mutation(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation
```
````

````{py:function} inversion_mutation_profit(routes: typing.List[typing.List[int]], distance_matrix: numpy.ndarray, capacity: float, wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], revenue: float, cost_unit: float, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation_profit

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion.inversion_mutation_profit
```
````

````{py:function} _get_demand(wastes: typing.Union[typing.Dict[int, float], numpy.ndarray], node: int) -> float
:canonical: src.policies.helpers.operators.evolutionary_mutation.inversion._get_demand

```{autodoc2-docstring} src.policies.helpers.operators.evolutionary_mutation.inversion._get_demand
```
````
