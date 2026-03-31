# {py:mod}`src.policies.branch_and_price_and_cut.search_strategy`

```{py:module} src.policies.branch_and_price_and_cut.search_strategy
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NodeSelectionStrategy <src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy
    :summary:
    ```
* - {py:obj}`BestFirstSearch <src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch
    :summary:
    ```
* - {py:obj}`DepthFirstSearch <src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_search_strategy <src.policies.branch_and_price_and_cut.search_strategy.create_search_strategy>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.create_search_strategy
    :summary:
    ```
````

### API

`````{py:class} NodeSelectionStrategy
:canonical: src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy
```

````{py:method} select_node(open_nodes: typing.List[src.policies.branch_and_price.branching.BranchNode]) -> typing.Optional[src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy.select_node
:abstractmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy.select_node
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy.get_name
:abstractmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy.get_name
```

````

`````

`````{py:class} BestFirstSearch
:canonical: src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch

Bases: {py:obj}`src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch
```

````{py:method} select_node(open_nodes: typing.List[src.policies.branch_and_price.branching.BranchNode]) -> typing.Optional[src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch.select_node

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch.select_node
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.search_strategy.BestFirstSearch.get_name

````

`````

`````{py:class} DepthFirstSearch
:canonical: src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch

Bases: {py:obj}`src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch
```

````{py:method} select_node(open_nodes: typing.List[src.policies.branch_and_price.branching.BranchNode]) -> typing.Optional[src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch.select_node

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch.select_node
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.search_strategy.DepthFirstSearch.get_name

````

`````

````{py:function} create_search_strategy(strategy_name: str) -> src.policies.branch_and_price_and_cut.search_strategy.NodeSelectionStrategy
:canonical: src.policies.branch_and_price_and_cut.search_strategy.create_search_strategy

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.search_strategy.create_search_strategy
```
````
