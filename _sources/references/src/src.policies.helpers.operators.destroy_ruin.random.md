# {py:mod}`src.policies.helpers.operators.destroy_ruin.random`

```{py:module} src.policies.helpers.operators.destroy_ruin.random
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.random
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`random_removal <src.policies.helpers.operators.destroy_ruin.random.random_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.random.random_removal
    :summary:
    ```
* - {py:obj}`random_profit_removal <src.policies.helpers.operators.destroy_ruin.random.random_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.random.random_profit_removal
    :summary:
    ```
````

### API

````{py:function} random_removal(routes: typing.List[typing.List[int]], n_remove: int, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.random.random_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.random.random_removal
```
````

````{py:function} random_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, bias_strength: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.random.random_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.random.random_profit_removal
```
````
