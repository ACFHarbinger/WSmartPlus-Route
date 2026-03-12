# {py:mod}`src.policies.other.operators.destroy.historical`

```{py:module} src.policies.other.operators.destroy.historical
```

```{autodoc2-docstring} src.policies.other.operators.destroy.historical
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`historical_removal <src.policies.other.operators.destroy.historical.historical_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.historical.historical_removal
    :summary:
    ```
* - {py:obj}`historical_profit_removal <src.policies.other.operators.destroy.historical.historical_profit_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.historical.historical_profit_removal
    :summary:
    ```
````

### API

````{py:function} historical_removal(routes: typing.List[typing.List[int]], n_remove: int, history: typing.Dict[int, float], rng: typing.Optional[random.Random] = None, noise: float = 0.1) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.historical.historical_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.historical.historical_removal
```
````

````{py:function} historical_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, history: typing.Dict[int, float], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, alpha: float = 0.5, rng: typing.Optional[random.Random] = None, noise: float = 0.1) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.historical.historical_profit_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.historical.historical_profit_removal
```
````
