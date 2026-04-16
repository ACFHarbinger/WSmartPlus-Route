# {py:mod}`src.policies.helpers.operators.intra_route.relocate`

```{py:module} src.policies.helpers.operators.intra_route.relocate
```

```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_relocate <src.policies.helpers.operators.intra_route.relocate.move_relocate>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.move_relocate
    :summary:
    ```
* - {py:obj}`relocate_chain <src.policies.helpers.operators.intra_route.relocate.relocate_chain>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.relocate_chain
    :summary:
    ```
* - {py:obj}`move_or_opt <src.policies.helpers.operators.intra_route.relocate.move_or_opt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.move_or_opt
    :summary:
    ```
* - {py:obj}`_chain_edge_cost <src.policies.helpers.operators.intra_route.relocate._chain_edge_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate._chain_edge_cost
    :summary:
    ```
````

### API

````{py:function} move_relocate(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.operators.intra_route.relocate.move_relocate

```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.move_relocate
```
````

````{py:function} relocate_chain(ls: typing.Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int, chain_len: int = 1) -> bool
:canonical: src.policies.helpers.operators.intra_route.relocate.relocate_chain

```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.relocate_chain
```
````

````{py:function} move_or_opt(ls: typing.Any, r_idx: int, pos: int, chain_len: int) -> bool
:canonical: src.policies.helpers.operators.intra_route.relocate.move_or_opt

```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate.move_or_opt
```
````

````{py:function} _chain_edge_cost(d, prev_node: int, chain: typing.List[int], next_node: int) -> float
:canonical: src.policies.helpers.operators.intra_route.relocate._chain_edge_cost

```{autodoc2-docstring} src.policies.helpers.operators.intra_route.relocate._chain_edge_cost
```
````
