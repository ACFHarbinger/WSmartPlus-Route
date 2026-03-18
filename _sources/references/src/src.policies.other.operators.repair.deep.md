# {py:mod}`src.policies.other.operators.repair.deep`

```{py:module} src.policies.other.operators.repair.deep
```

```{autodoc2-docstring} src.policies.other.operators.repair.deep
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`deep_insertion <src.policies.other.operators.repair.deep.deep_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.deep.deep_insertion
    :summary:
    ```
* - {py:obj}`deep_profit_insertion <src.policies.other.operators.repair.deep.deep_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.deep.deep_profit_insertion
    :summary:
    ```
````

### API

````{py:function} deep_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, alpha: float = 0.3, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.deep.deep_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.deep.deep_insertion
```
````

````{py:function} deep_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, alpha: float = 0.3, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.deep.deep_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.deep.deep_profit_insertion
```
````
