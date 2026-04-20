# {py:mod}`src.policies.helpers.operators.inter_route_local_search.cross_exchange`

```{py:module} src.policies.helpers.operators.inter_route_local_search.cross_exchange
```

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cross_exchange <src.policies.helpers.operators.inter_route_local_search.cross_exchange.cross_exchange>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.cross_exchange
    :summary:
    ```
* - {py:obj}`lambda_interchange <src.policies.helpers.operators.inter_route_local_search.cross_exchange.lambda_interchange>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.lambda_interchange
    :summary:
    ```
* - {py:obj}`_seg_boundary_cost <src.policies.helpers.operators.inter_route_local_search.cross_exchange._seg_boundary_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange._seg_boundary_cost
    :summary:
    ```
* - {py:obj}`improved_cross_exchange <src.policies.helpers.operators.inter_route_local_search.cross_exchange.improved_cross_exchange>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.improved_cross_exchange
    :summary:
    ```
````

### API

````{py:function} cross_exchange(ls: typing.Any, r_a: int, seg_a_start: int, seg_a_len: int, r_b: int, seg_b_start: int, seg_b_len: int) -> bool
:canonical: src.policies.helpers.operators.inter_route_local_search.cross_exchange.cross_exchange

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.cross_exchange
```
````

````{py:function} lambda_interchange(ls: typing.Any, lambda_max: int = 2) -> bool
:canonical: src.policies.helpers.operators.inter_route_local_search.cross_exchange.lambda_interchange

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.lambda_interchange
```
````

````{py:function} _seg_boundary_cost(d, prev_node: int, seg: typing.List[int], next_node: int) -> float
:canonical: src.policies.helpers.operators.inter_route_local_search.cross_exchange._seg_boundary_cost

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange._seg_boundary_cost
```
````

````{py:function} improved_cross_exchange(ls: typing.Any, r_a: int, seg_a_start: int, seg_a_len: int, r_b: int, seg_b_start: int, seg_b_len: int) -> bool
:canonical: src.policies.helpers.operators.inter_route_local_search.cross_exchange.improved_cross_exchange

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.cross_exchange.improved_cross_exchange
```
````
