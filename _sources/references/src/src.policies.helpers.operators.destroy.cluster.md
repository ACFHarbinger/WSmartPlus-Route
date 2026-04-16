# {py:mod}`src.policies.helpers.operators.destroy.cluster`

```{py:module} src.policies.helpers.operators.destroy.cluster
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy.cluster
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cluster_removal <src.policies.helpers.operators.destroy.cluster.cluster_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.cluster.cluster_removal
    :summary:
    ```
* - {py:obj}`cluster_profit_removal <src.policies.helpers.operators.destroy.cluster.cluster_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy.cluster.cluster_profit_removal
    :summary:
    ```
````

### API

````{py:function} cluster_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, nodes: typing.List[int], rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.cluster.cluster_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.cluster.cluster_removal
```
````

````{py:function} cluster_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy.cluster.cluster_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy.cluster.cluster_profit_removal
```
````
