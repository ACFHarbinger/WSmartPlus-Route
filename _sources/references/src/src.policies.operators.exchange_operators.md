# {py:mod}`src.policies.operators.exchange_operators`

```{py:module} src.policies.operators.exchange_operators
```

```{autodoc2-docstring} src.policies.operators.exchange_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_or_opt <src.policies.operators.exchange_operators.move_or_opt>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators.move_or_opt
    :summary:
    ```
* - {py:obj}`cross_exchange <src.policies.operators.exchange_operators.cross_exchange>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators.cross_exchange
    :summary:
    ```
* - {py:obj}`ejection_chain <src.policies.operators.exchange_operators.ejection_chain>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators.ejection_chain
    :summary:
    ```
* - {py:obj}`_try_insert_with_chain <src.policies.operators.exchange_operators._try_insert_with_chain>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators._try_insert_with_chain
    :summary:
    ```
* - {py:obj}`_rollback_ejections <src.policies.operators.exchange_operators._rollback_ejections>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators._rollback_ejections
    :summary:
    ```
* - {py:obj}`lambda_interchange <src.policies.operators.exchange_operators.lambda_interchange>`
  - ```{autodoc2-docstring} src.policies.operators.exchange_operators.lambda_interchange
    :summary:
    ```
````

### API

````{py:function} move_or_opt(ls: typing.Any, node: int, chain_len: int, r_idx: int, pos: int) -> bool
:canonical: src.policies.operators.exchange_operators.move_or_opt

```{autodoc2-docstring} src.policies.operators.exchange_operators.move_or_opt
```
````

````{py:function} cross_exchange(ls: typing.Any, r_a: int, seg_a_start: int, seg_a_len: int, r_b: int, seg_b_start: int, seg_b_len: int) -> bool
:canonical: src.policies.operators.exchange_operators.cross_exchange

```{autodoc2-docstring} src.policies.operators.exchange_operators.cross_exchange
```
````

````{py:function} ejection_chain(ls: typing.Any, source_route: int, max_depth: int = 5) -> bool
:canonical: src.policies.operators.exchange_operators.ejection_chain

```{autodoc2-docstring} src.policies.operators.exchange_operators.ejection_chain
```
````

````{py:function} _try_insert_with_chain(ls: typing.Any, node: int, excluded_route: int, depth: int, log: typing.List[typing.Tuple[int, int, int]]) -> bool
:canonical: src.policies.operators.exchange_operators._try_insert_with_chain

```{autodoc2-docstring} src.policies.operators.exchange_operators._try_insert_with_chain
```
````

````{py:function} _rollback_ejections(ls: typing.Any, log: typing.List[typing.Tuple[int, int, int]], source_route: int) -> None
:canonical: src.policies.operators.exchange_operators._rollback_ejections

```{autodoc2-docstring} src.policies.operators.exchange_operators._rollback_ejections
```
````

````{py:function} lambda_interchange(ls: typing.Any, lambda_max: int = 2) -> bool
:canonical: src.policies.operators.exchange_operators.lambda_interchange

```{autodoc2-docstring} src.policies.operators.exchange_operators.lambda_interchange
```
````
