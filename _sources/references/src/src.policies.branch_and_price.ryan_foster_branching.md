# {py:mod}`src.policies.branch_and_price.ryan_foster_branching`

```{py:module} src.policies.branch_and_price.ryan_foster_branching
```

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchingConstraint <src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint
    :summary:
    ```
* - {py:obj}`BranchNode <src.policies.branch_and_price.ryan_foster_branching.BranchNode>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchNode
    :summary:
    ```
* - {py:obj}`RyanFosterBranching <src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching
    :summary:
    ```
* - {py:obj}`BranchAndBoundTree <src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree
    :summary:
    ```
````

### API

`````{py:class} BranchingConstraint(node_r: int, node_s: int, together: bool)
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: src.policies.branch_and_price.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint.__repr__

````

`````

`````{py:class} BranchNode(constraints: typing.Optional[typing.List[src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint]] = None, parent: typing.Optional[src.policies.branch_and_price.ryan_foster_branching.BranchNode] = None, depth: int = 0)
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchNode

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchNode.__init__
```

````{py:method} get_all_constraints() -> typing.List[src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint]
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchNode.get_all_constraints

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchNode.get_all_constraints
```

````

````{py:method} is_route_feasible(route: src.policies.branch_and_price.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchNode.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchNode.is_route_feasible
```

````

`````

`````{py:class} RyanFosterBranching
:canonical: src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching
```

````{py:method} find_branching_pair(routes: typing.List[src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-06) -> typing.Optional[typing.Tuple[int, int]]
:canonical: src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.find_branching_pair
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.find_branching_pair
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price.ryan_foster_branching.BranchNode, node_r: int, node_s: int) -> typing.Tuple[src.policies.branch_and_price.ryan_foster_branching.BranchNode, src.policies.branch_and_price.ryan_foster_branching.BranchNode]
:canonical: src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.create_child_nodes
```

````

````{py:method} modify_pricing_for_constraint(constraint: src.policies.branch_and_price.ryan_foster_branching.BranchingConstraint, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float]) -> typing.Tuple[numpy.ndarray, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.modify_pricing_for_constraint
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.RyanFosterBranching.modify_pricing_for_constraint
```

````

`````

`````{py:class} BranchAndBoundTree()
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.__init__
```

````{py:method} get_next_node() -> typing.Optional[src.policies.branch_and_price.ryan_foster_branching.BranchNode]
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.get_next_node

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.get_next_node
```

````

````{py:method} add_node(node: src.policies.branch_and_price.ryan_foster_branching.BranchNode) -> None
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.add_node

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.add_node
```

````

````{py:method} prune_by_bound() -> int
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.prune_by_bound

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.prune_by_bound
```

````

````{py:method} update_incumbent(node: src.policies.branch_and_price.ryan_foster_branching.BranchNode, solution_value: float) -> bool
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.update_incumbent

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.update_incumbent
```

````

````{py:method} is_empty() -> bool
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.is_empty

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.is_empty
```

````

````{py:method} get_statistics() -> typing.Dict
:canonical: src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.ryan_foster_branching.BranchAndBoundTree.get_statistics
```

````

`````
