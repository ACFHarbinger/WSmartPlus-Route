# {py:mod}`src.policies.other.operators.destroy.shaw`

```{py:module} src.policies.other.operators.destroy.shaw
```

```{autodoc2-docstring} src.policies.other.operators.destroy.shaw
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`shaw_removal <src.policies.other.operators.destroy.shaw.shaw_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.shaw.shaw_removal
    :summary:
    ```
* - {py:obj}`shaw_profit_removal <src.policies.other.operators.destroy.shaw.shaw_profit_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.shaw.shaw_profit_removal
    :summary:
    ```
````

### API

````{py:function} shaw_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, nodes: typing.List[int], wastes: typing.Optional[typing.List[int]] = None, waste_dict: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None, time_windows: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None, relatedness_weights: typing.Tuple[float, float, float] = (0.5, 0.3, 0.2), randomization_factor: float = 2.0, phi: float = 9.0, chi: float = 3.0, psi: float = 2.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.shaw.shaw_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.shaw.shaw_removal
```
````

````{py:function} shaw_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, relatedness_weights: typing.Tuple[float, float] = (0.6, 0.4), randomization_factor: float = 2.0, phi: float = 9.0, psi: float = 5.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.shaw.shaw_profit_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.shaw.shaw_profit_removal
```
````
