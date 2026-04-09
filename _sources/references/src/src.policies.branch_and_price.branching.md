# {py:mod}`src.policies.branch_and_price.branching`

```{py:module} src.policies.branch_and_price.branching
```

```{autodoc2-docstring} src.policies.branch_and_price.branching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeBranchingConstraint <src.policies.branch_and_price.branching.EdgeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranchingConstraint
    :summary:
    ```
* - {py:obj}`RyanFosterBranchingConstraint <src.policies.branch_and_price.branching.RyanFosterBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranchingConstraint
    :summary:
    ```
* - {py:obj}`BranchNode <src.policies.branch_and_price.branching.BranchNode>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchNode
    :summary:
    ```
* - {py:obj}`EdgeBranching <src.policies.branch_and_price.branching.EdgeBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranching
    :summary:
    ```
* - {py:obj}`MultiEdgePartitionBranching <src.policies.branch_and_price.branching.MultiEdgePartitionBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.MultiEdgePartitionBranching
    :summary:
    ```
* - {py:obj}`RyanFosterBranching <src.policies.branch_and_price.branching.RyanFosterBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranching
    :summary:
    ```
* - {py:obj}`BranchAndBoundTree <src.policies.branch_and_price.branching.BranchAndBoundTree>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchingConstraint <src.policies.branch_and_price.branching.BranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchingConstraint
    :summary:
    ```
* - {py:obj}`AnyBranchingConstraint <src.policies.branch_and_price.branching.AnyBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.branching.AnyBranchingConstraint
    :summary:
    ```
````

### API

`````{py:class} EdgeBranchingConstraint(u: int, v: int, must_use: bool)
:canonical: src.policies.branch_and_price.branching.EdgeBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.branch_and_price.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price.branching.EdgeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranchingConstraint.is_route_feasible
```

````

````{py:method} _edge_in_route(nodes: typing.List[int]) -> bool
:canonical: src.policies.branch_and_price.branching.EdgeBranchingConstraint._edge_in_route

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranchingConstraint._edge_in_route
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price.branching.EdgeBranchingConstraint.__repr__

````

`````

`````{py:class} RyanFosterBranchingConstraint(node_r: int, node_s: int, together: bool)
:canonical: src.policies.branch_and_price.branching.RyanFosterBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.branch_and_price.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price.branching.RyanFosterBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price.branching.RyanFosterBranchingConstraint.__repr__

````

`````

````{py:data} BranchingConstraint
:canonical: src.policies.branch_and_price.branching.BranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchingConstraint
```

````

````{py:data} AnyBranchingConstraint
:canonical: src.policies.branch_and_price.branching.AnyBranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price.branching.AnyBranchingConstraint
```

````

`````{py:class} BranchNode(constraints: typing.Optional[typing.List[src.policies.branch_and_price.branching.AnyBranchingConstraint]] = None, parent: typing.Optional[src.policies.branch_and_price.branching.BranchNode] = None, depth: int = 0, lp_bound_hint: typing.Optional[float] = None)
:canonical: src.policies.branch_and_price.branching.BranchNode

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchNode.__init__
```

````{py:method} get_all_constraints() -> typing.List[src.policies.branch_and_price.branching.AnyBranchingConstraint]
:canonical: src.policies.branch_and_price.branching.BranchNode.get_all_constraints

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchNode.get_all_constraints
```

````

````{py:method} is_route_feasible(route: logic.src.policies.branch_and_price.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price.branching.BranchNode.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchNode.is_route_feasible
```

````

`````

`````{py:class} EdgeBranching
:canonical: src.policies.branch_and_price.branching.EdgeBranching

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranching
```

````{py:method} compute_arc_flow(routes: typing.List[logic.src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float]) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price.branching.EdgeBranching.compute_arc_flow
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranching.compute_arc_flow
```

````

````{py:method} find_branching_arc(routes: typing.List[logic.src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-05) -> typing.Optional[typing.Tuple[int, int]]
:canonical: src.policies.branch_and_price.branching.EdgeBranching.find_branching_arc
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranching.find_branching_arc
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price.branching.BranchNode, u: int, v: int) -> typing.Tuple[src.policies.branch_and_price.branching.BranchNode, src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price.branching.EdgeBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.EdgeBranching.create_child_nodes
```

````

`````

`````{py:class} MultiEdgePartitionBranching
:canonical: src.policies.branch_and_price.branching.MultiEdgePartitionBranching

```{autodoc2-docstring} src.policies.branch_and_price.branching.MultiEdgePartitionBranching
```

````{py:method} find_divergence_node(routes: typing.List[logic.src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-05, node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]]]
:canonical: src.policies.branch_and_price.branching.MultiEdgePartitionBranching.find_divergence_node
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.MultiEdgePartitionBranching.find_divergence_node
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price.branching.BranchNode, divergence_node: int, arc_set_1: typing.List[typing.Tuple[int, int]], arc_set_2: typing.List[typing.Tuple[int, int]]) -> typing.Tuple[src.policies.branch_and_price.branching.BranchNode, src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price.branching.MultiEdgePartitionBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.MultiEdgePartitionBranching.create_child_nodes
```

````

`````

`````{py:class} RyanFosterBranching
:canonical: src.policies.branch_and_price.branching.RyanFosterBranching

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranching
```

````{py:method} find_branching_pair(routes: typing.List[logic.src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-05) -> typing.Optional[typing.Tuple[int, int]]
:canonical: src.policies.branch_and_price.branching.RyanFosterBranching.find_branching_pair
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranching.find_branching_pair
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price.branching.BranchNode, node_r: int, node_s: int) -> typing.Tuple[src.policies.branch_and_price.branching.BranchNode, src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price.branching.RyanFosterBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.branching.RyanFosterBranching.create_child_nodes
```

````

`````

`````{py:class} BranchAndBoundTree(v_model: typing.Optional[logic.src.policies.branch_and_cut.vrpp_model.VRPPModel] = None, node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None, max_nodes: int = 1000, strategy: str = 'edge', search_strategy: str = 'best_first')
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.__init__
```

````{py:attribute} node_coords
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.node_coords
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.node_coords
```

````

````{py:method} add_node(node: src.policies.branch_and_price.branching.BranchNode) -> None
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.add_node

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.add_node
```

````

````{py:method} get_next_node() -> typing.Optional[src.policies.branch_and_price.branching.BranchNode]
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.get_next_node

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.get_next_node
```

````

````{py:method} branch(node: src.policies.branch_and_price.branching.BranchNode, routes: typing.List[logic.src.policies.branch_and_price.master_problem.Route], route_values: typing.Dict[int, float]) -> typing.Optional[typing.Tuple[src.policies.branch_and_price.branching.BranchNode, src.policies.branch_and_price.branching.BranchNode]]
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.branch

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.branch
```

````

````{py:method} prune_by_bound() -> int
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.prune_by_bound

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.prune_by_bound
```

````

````{py:method} update_incumbent(node: src.policies.branch_and_price.branching.BranchNode, value: float) -> bool
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.update_incumbent

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.update_incumbent
```

````

````{py:method} is_empty() -> bool
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.is_empty

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.is_empty
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.branching.BranchAndBoundTree.get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.branching.BranchAndBoundTree.get_statistics
```

````

`````
