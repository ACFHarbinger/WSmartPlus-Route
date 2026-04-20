# {py:mod}`src.policies.helpers.operators.inter_route_local_search.ejection_chain`

```{py:module} src.policies.helpers.operators.inter_route_local_search.ejection_chain
```

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ejection_chain <src.policies.helpers.operators.inter_route_local_search.ejection_chain.ejection_chain>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain.ejection_chain
    :summary:
    ```
* - {py:obj}`_try_insert_with_chain <src.policies.helpers.operators.inter_route_local_search.ejection_chain._try_insert_with_chain>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain._try_insert_with_chain
    :summary:
    ```
* - {py:obj}`_rollback_ejections <src.policies.helpers.operators.inter_route_local_search.ejection_chain._rollback_ejections>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain._rollback_ejections
    :summary:
    ```
````

### API

````{py:function} ejection_chain(ls: typing.Any, source_route: int, max_depth: int = 5) -> bool
:canonical: src.policies.helpers.operators.inter_route_local_search.ejection_chain.ejection_chain

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain.ejection_chain
```
````

````{py:function} _try_insert_with_chain(ls: typing.Any, node: int, excluded_route: int, depth: int, log: typing.List[typing.Tuple[int, int, int]]) -> bool
:canonical: src.policies.helpers.operators.inter_route_local_search.ejection_chain._try_insert_with_chain

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain._try_insert_with_chain
```
````

````{py:function} _rollback_ejections(ls: typing.Any, log: typing.List[typing.Tuple[int, int, int]], source_route: int) -> None
:canonical: src.policies.helpers.operators.inter_route_local_search.ejection_chain._rollback_ejections

```{autodoc2-docstring} src.policies.helpers.operators.inter_route_local_search.ejection_chain._rollback_ejections
```
````
