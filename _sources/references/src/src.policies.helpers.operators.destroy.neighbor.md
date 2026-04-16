# {py:mod}`src.policies.helpers.operators.destroy.neighbor`

```{py:module} src.policies.helpers.operators.destroy.neighbor
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.neighbor
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`neighbor_removal <src.policies.helpers.operators.destroy.neighbor.neighbor_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.neighbor.neighbor_removal
    :summary:
    ```
* - {py:obj}`neighbor_profit_removal <src.policies.helpers.operators.destroy.neighbor.neighbor_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.neighbor.neighbor_profit_removal
    :summary:
    ```
````

### API

````{py:function} neighbor_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.neighbor.neighbor_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.neighbor.neighbor_removal
```
````

````{py:function} neighbor_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.neighbor.neighbor_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.neighbor.neighbor_profit_removal
```
````
