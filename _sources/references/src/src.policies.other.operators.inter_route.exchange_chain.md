# {py:mod}`src.policies.other.operators.inter_route.exchange_chain`

```{py:module} src.policies.other.operators.inter_route.exchange_chain
```

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_chain_edge_cost <src.policies.other.operators.inter_route.exchange_chain._chain_edge_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain._chain_edge_cost
    :summary:
    ```
* - {py:obj}`exchange_2_0 <src.policies.other.operators.inter_route.exchange_chain.exchange_2_0>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_2_0
    :summary:
    ```
* - {py:obj}`exchange_2_1 <src.policies.other.operators.inter_route.exchange_chain.exchange_2_1>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_2_1
    :summary:
    ```
* - {py:obj}`exchange_k_0 <src.policies.other.operators.inter_route.exchange_chain.exchange_k_0>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_k_0
    :summary:
    ```
* - {py:obj}`exchange_k_h <src.policies.other.operators.inter_route.exchange_chain.exchange_k_h>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_k_h
    :summary:
    ```
````

### API

````{py:function} _chain_edge_cost(d, prev_node: int, chain: typing.List[int], next_node: int) -> float
:canonical: src.policies.other.operators.inter_route.exchange_chain._chain_edge_cost

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain._chain_edge_cost
```
````

````{py:function} exchange_2_0(ls: typing.Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool
:canonical: src.policies.other.operators.inter_route.exchange_chain.exchange_2_0

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_2_0
```
````

````{py:function} exchange_2_1(ls: typing.Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool
:canonical: src.policies.other.operators.inter_route.exchange_chain.exchange_2_1

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_2_1
```
````

````{py:function} exchange_k_0(ls: typing.Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int, k: int = 2) -> bool
:canonical: src.policies.other.operators.inter_route.exchange_chain.exchange_k_0

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_k_0
```
````

````{py:function} exchange_k_h(ls: typing.Any, r_src: int, pos_src: int, k: int, r_dst: int, pos_dst: int, h: int) -> bool
:canonical: src.policies.other.operators.inter_route.exchange_chain.exchange_k_h

```{autodoc2-docstring} src.policies.other.operators.inter_route.exchange_chain.exchange_k_h
```
````
