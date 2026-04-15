# {py:mod}`src.policies.other.branching_solvers.branching.tree`

```{py:module} src.policies.other.branching_solvers.branching.tree
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndBoundTree <src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FrontierItem <src.policies.other.branching_solvers.branching.tree.FrontierItem>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.FrontierItem
    :summary:
    ```
````

### API

````{py:data} FrontierItem
:canonical: src.policies.other.branching_solvers.branching.tree.FrontierItem
:value: >
   None

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.FrontierItem
```

````

`````{py:class} BranchAndBoundTree(v_model: typing.Optional[src.policies.other.branching_solvers.vrpp_model.VRPPModel] = None, params: typing.Optional[logic.src.policies.branch_and_price_and_cut.params.BPCParams] = None, max_nodes: int = 1000, strategy: str = 'edge', search_strategy: str = 'best_first')
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.__init__
```

````{py:method} add_node(node: logic.src.policies.other.branching_solvers.common.node.BranchNode) -> None
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.add_node

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.add_node
```

````

````{py:method} get_next_node() -> typing.Optional[logic.src.policies.other.branching_solvers.common.node.BranchNode]
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.get_next_node

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.get_next_node
```

````

````{py:method} is_empty() -> bool
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.is_empty

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.is_empty
```

````

````{py:method} update_incumbent(node: logic.src.policies.other.branching_solvers.common.node.BranchNode, value: float) -> bool
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.update_incumbent

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.update_incumbent
```

````

````{py:method} prune_by_bound() -> int
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.prune_by_bound

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.prune_by_bound
```

````

````{py:method} record_explored() -> None
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.record_explored

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.record_explored
```

````

````{py:method} find_strong_branching_candidates(routes: typing.List[src.policies.other.branching_solvers.common.route.Route], route_values: typing.Dict[int, float], max_candidates: int = 5) -> typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.find_strong_branching_candidates

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.find_strong_branching_candidates
```

````

````{py:method} branch(node: logic.src.policies.other.branching_solvers.common.node.BranchNode, routes: typing.List[src.policies.other.branching_solvers.common.route.Route], route_values: typing.Dict[int, float], mandatory_nodes: typing.Set[int], strong_candidate: typing.Optional[typing.Any] = None) -> typing.Optional[typing.Tuple[logic.src.policies.other.branching_solvers.common.node.BranchNode, logic.src.policies.other.branching_solvers.common.node.BranchNode]]
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.branch

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.branch
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.get_statistics

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.tree.BranchAndBoundTree.get_statistics
```

````

`````
