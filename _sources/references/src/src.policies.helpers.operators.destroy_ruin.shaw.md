# {py:mod}`src.policies.helpers.operators.destroy_ruin.shaw`

```{py:module} src.policies.helpers.operators.destroy_ruin.shaw
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.shaw
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`shaw_removal <src.policies.helpers.operators.destroy_ruin.shaw.shaw_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.shaw.shaw_removal
    :summary:
    ```
* - {py:obj}`shaw_profit_removal <src.policies.helpers.operators.destroy_ruin.shaw.shaw_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.shaw.shaw_profit_removal
    :summary:
    ```
````

### API

````{py:function} shaw_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Optional[typing.Dict[int, float]] = None, time_windows: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None, randomization_factor: float = 2.0, phi: float = 9.0, chi: float = 3.0, psi: float = 2.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.shaw.shaw_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.shaw.shaw_removal
```
````

````{py:function} shaw_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, randomization_factor: float = 2.0, phi: float = 9.0, psi: float = 5.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.shaw.shaw_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.shaw.shaw_profit_removal
```
````
