# {py:mod}`src.policies.other.operators.inter_route.cyclic_transfer`

```{py:module} src.policies.other.operators.inter_route.cyclic_transfer
```

```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cyclic_transfer <src.policies.other.operators.inter_route.cyclic_transfer.cyclic_transfer>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer.cyclic_transfer
    :summary:
    ```
* - {py:obj}`_evaluate_shift <src.policies.other.operators.inter_route.cyclic_transfer._evaluate_shift>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer._evaluate_shift
    :summary:
    ```
* - {py:obj}`_apply_shift <src.policies.other.operators.inter_route.cyclic_transfer._apply_shift>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer._apply_shift
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RouteCut <src.policies.other.operators.inter_route.cyclic_transfer.RouteCut>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer.RouteCut
    :summary:
    ```
````

### API

````{py:data} RouteCut
:canonical: src.policies.other.operators.inter_route.cyclic_transfer.RouteCut
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer.RouteCut
```

````

````{py:function} cyclic_transfer(ls: typing.Any, participants: typing.List[src.policies.other.operators.inter_route.cyclic_transfer.RouteCut]) -> bool
:canonical: src.policies.other.operators.inter_route.cyclic_transfer.cyclic_transfer

```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer.cyclic_transfer
```
````

````{py:function} _evaluate_shift(ls: typing.Any, participants: typing.List[src.policies.other.operators.inter_route.cyclic_transfer.RouteCut], nodes: typing.List[int], demands: typing.List[float], direction: int) -> float
:canonical: src.policies.other.operators.inter_route.cyclic_transfer._evaluate_shift

```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer._evaluate_shift
```
````

````{py:function} _apply_shift(ls: typing.Any, participants: typing.List[src.policies.other.operators.inter_route.cyclic_transfer.RouteCut], nodes: typing.List[int], direction: int) -> None
:canonical: src.policies.other.operators.inter_route.cyclic_transfer._apply_shift

```{autodoc2-docstring} src.policies.other.operators.inter_route.cyclic_transfer._apply_shift
```
````
