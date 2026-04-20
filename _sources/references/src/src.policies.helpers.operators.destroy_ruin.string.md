# {py:mod}`src.policies.helpers.operators.destroy_ruin.string`

```{py:module} src.policies.helpers.operators.destroy_ruin.string
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_sisr_ruin_pass <src.policies.helpers.operators.destroy_ruin.string._sisr_ruin_pass>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string._sisr_ruin_pass
    :summary:
    ```
* - {py:obj}`string_removal <src.policies.helpers.operators.destroy_ruin.string.string_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string.string_removal
    :summary:
    ```
* - {py:obj}`string_profit_removal <src.policies.helpers.operators.destroy_ruin.string.string_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string.string_profit_removal
    :summary:
    ```
````

### API

````{py:function} _sisr_ruin_pass(routes: typing.List[typing.List[int]], adjacency: typing.List[typing.Tuple[float, int]], ks: int, ls_max: float, rng: random.Random) -> typing.Set[int]
:canonical: src.policies.helpers.operators.destroy_ruin.string._sisr_ruin_pass

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string._sisr_ruin_pass
```
````

````{py:function} string_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, max_string_len: int = 4, avg_string_len: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.string.string_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string.string_removal
```
````

````{py:function} string_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, max_string_len: int = 4, avg_string_len: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.string.string_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.string.string_profit_removal
```
````
