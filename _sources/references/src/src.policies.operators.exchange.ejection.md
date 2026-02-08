# {py:mod}`src.policies.operators.exchange.ejection`

```{py:module} src.policies.operators.exchange.ejection
```

```{autodoc2-docstring} src.policies.operators.exchange.ejection
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ejection_chain <src.policies.operators.exchange.ejection.ejection_chain>`
  - ```{autodoc2-docstring} src.policies.operators.exchange.ejection.ejection_chain
    :summary:
    ```
* - {py:obj}`_try_insert_with_chain <src.policies.operators.exchange.ejection._try_insert_with_chain>`
  - ```{autodoc2-docstring} src.policies.operators.exchange.ejection._try_insert_with_chain
    :summary:
    ```
* - {py:obj}`_rollback_ejections <src.policies.operators.exchange.ejection._rollback_ejections>`
  - ```{autodoc2-docstring} src.policies.operators.exchange.ejection._rollback_ejections
    :summary:
    ```
````

### API

````{py:function} ejection_chain(ls: typing.Any, source_route: int, max_depth: int = 5) -> bool
:canonical: src.policies.operators.exchange.ejection.ejection_chain

```{autodoc2-docstring} src.policies.operators.exchange.ejection.ejection_chain
```
````

````{py:function} _try_insert_with_chain(ls: typing.Any, node: int, excluded_route: int, depth: int, log: typing.List[typing.Tuple[int, int, int]]) -> bool
:canonical: src.policies.operators.exchange.ejection._try_insert_with_chain

```{autodoc2-docstring} src.policies.operators.exchange.ejection._try_insert_with_chain
```
````

````{py:function} _rollback_ejections(ls: typing.Any, log: typing.List[typing.Tuple[int, int, int]], source_route: int) -> None
:canonical: src.policies.operators.exchange.ejection._rollback_ejections

```{autodoc2-docstring} src.policies.operators.exchange.ejection._rollback_ejections
```
````
