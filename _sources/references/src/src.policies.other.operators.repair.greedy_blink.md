# {py:mod}`src.policies.other.operators.repair.greedy_blink`

```{py:module} src.policies.other.operators.repair.greedy_blink
```

```{autodoc2-docstring} src.policies.other.operators.repair.greedy_blink
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_insertion_with_blinks <src.policies.other.operators.repair.greedy_blink.greedy_insertion_with_blinks>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.greedy_blink.greedy_insertion_with_blinks
    :summary:
    ```
* - {py:obj}`greedy_profit_insertion_with_blinks <src.policies.other.operators.repair.greedy_blink.greedy_profit_insertion_with_blinks>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.greedy_blink.greedy_profit_insertion_with_blinks
    :summary:
    ```
````

### API

````{py:function} greedy_insertion_with_blinks(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, blink_rate: float = 0.1, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.greedy_blink.greedy_insertion_with_blinks

```{autodoc2-docstring} src.policies.other.operators.repair.greedy_blink.greedy_insertion_with_blinks
```
````

````{py:function} greedy_profit_insertion_with_blinks(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, blink_rate: float = 0.1, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.greedy_blink.greedy_profit_insertion_with_blinks

```{autodoc2-docstring} src.policies.other.operators.repair.greedy_blink.greedy_profit_insertion_with_blinks
```
````
