# {py:mod}`src.policies.helpers.operators.destroy.sector`

```{py:module} src.policies.helpers.operators.destroy.sector
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.sector
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sector_removal <src.policies.helpers.operators.destroy.sector.sector_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.sector.sector_removal
    :summary:
    ```
* - {py:obj}`sector_profit_removal <src.policies.helpers.operators.destroy.sector.sector_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.sector.sector_profit_removal
    :summary:
    ```
````

### API

````{py:function} sector_removal(routes: typing.List[typing.List[int]], n_remove: int, coords: numpy.ndarray, depot: typing.Tuple[float, float] = (0.0, 0.0), rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.sector.sector_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.sector.sector_removal
```
````

````{py:function} sector_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, coords: numpy.ndarray, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, depot: typing.Tuple[float, float] = (0.0, 0.0), bias_low_profit: bool = True, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.sector.sector_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.sector.sector_profit_removal
```
````
