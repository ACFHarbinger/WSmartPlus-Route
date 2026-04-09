# {py:mod}`src.policies.branch_and_price_and_cut.branching`

```{py:module} src.policies.branch_and_price_and_cut.branching
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeBranchingConstraint <src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint
    :summary:
    ```
* - {py:obj}`RyanFosterBranchingConstraint <src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint
    :summary:
    ```
* - {py:obj}`BranchNode <src.policies.branch_and_price_and_cut.branching.BranchNode>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchNode
    :summary:
    ```
* - {py:obj}`EdgeBranching <src.policies.branch_and_price_and_cut.branching.EdgeBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranching
    :summary:
    ```
* - {py:obj}`MultiEdgePartitionBranching <src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching
    :summary:
    ```
* - {py:obj}`RyanFosterBranching <src.policies.branch_and_price_and_cut.branching.RyanFosterBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranching
    :summary:
    ```
* - {py:obj}`FleetSizeBranchingConstraint <src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint
    :summary:
    ```
* - {py:obj}`NodeVisitationBranchingConstraint <src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint
    :summary:
    ```
* - {py:obj}`FleetSizeBranching <src.policies.branch_and_price_and_cut.branching.FleetSizeBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranching
    :summary:
    ```
* - {py:obj}`NodeVisitationBranching <src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching
    :summary:
    ```
* - {py:obj}`BranchAndBoundTree <src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchingConstraint <src.policies.branch_and_price_and_cut.branching.BranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchingConstraint
    :summary:
    ```
* - {py:obj}`AnyBranchingConstraint <src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint
    :summary:
    ```
````

### API

`````{py:class} EdgeBranchingConstraint(u: int, v: int, must_use: bool)
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint.is_route_feasible
```

````

````{py:method} _edge_in_route(nodes: typing.List[int]) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint._edge_in_route

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint._edge_in_route
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranchingConstraint.__repr__

````

`````

`````{py:class} RyanFosterBranchingConstraint(node_r: int, node_s: int, together: bool, mandatory_nodes: typing.Optional[typing.Set[int]] = None)
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranchingConstraint.__repr__

````

`````

````{py:data} BranchingConstraint
:canonical: src.policies.branch_and_price_and_cut.branching.BranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchingConstraint
```

````

`````{py:class} BranchNode(constraints: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]] = None, parent: typing.Optional[src.policies.branch_and_price_and_cut.branching.BranchNode] = None, depth: int = 0, lp_bound_hint: typing.Optional[float] = None)
:canonical: src.policies.branch_and_price_and_cut.branching.BranchNode

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchNode.__init__
```

````{py:method} get_all_constraints() -> typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]
:canonical: src.policies.branch_and_price_and_cut.branching.BranchNode.get_all_constraints

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchNode.get_all_constraints
```

````

````{py:method} is_route_feasible(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.BranchNode.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchNode.is_route_feasible
```

````

`````

`````{py:class} EdgeBranching
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranching
```

````{py:method} compute_arc_flow(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float]) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranching.compute_arc_flow
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranching.compute_arc_flow
```

````

````{py:method} find_branching_arc(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-05) -> typing.Optional[typing.Tuple[typing.Tuple[int, int], float]]
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranching.find_branching_arc
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranching.find_branching_arc
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price_and_cut.branching.BranchNode, u: int, v: int, arc_flow: float = 0.5) -> typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.EdgeBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.EdgeBranching.create_child_nodes
```

````

`````

`````{py:class} MultiEdgePartitionBranching
:canonical: src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching
```

````{py:method} find_divergence_node(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], tol: float = 1e-05, node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.find_divergence_node
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.find_divergence_node
```

````

````{py:method} find_multiple_divergence_nodes(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None, limit: int = 5, tol: float = 1e-05) -> typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.find_multiple_divergence_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.find_multiple_divergence_nodes
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price_and_cut.branching.BranchNode, divergence_node: int, arc_set_1: typing.List[typing.Tuple[int, int]], arc_set_2: typing.List[typing.Tuple[int, int]], strength: float = 0.5) -> typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.MultiEdgePartitionBranching.create_child_nodes
```

````

`````

`````{py:class} RyanFosterBranching
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranching
```

````{py:method} find_branching_pair(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], mandatory_nodes: typing.Set[int], tol: float = 1e-05) -> typing.Optional[typing.Tuple[typing.Tuple[int, int], float]]
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranching.find_branching_pair
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranching.find_branching_pair
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price_and_cut.branching.BranchNode, node_r: int, node_s: int, together_sum: float = 0.5, mandatory_nodes: typing.Optional[typing.Set[int]] = None) -> typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.RyanFosterBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.RyanFosterBranching.create_child_nodes
```

````

`````

`````{py:class} FleetSizeBranchingConstraint(limit: int, is_upper: bool)
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranchingConstraint.__repr__

````

`````

`````{py:class} NodeVisitationBranchingConstraint(node: int, forced: bool)
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranchingConstraint.__repr__

````

`````

`````{py:class} FleetSizeBranching
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranching
```

````{py:method} find_fleet_branching(route_values: typing.Dict[int, float], tol: float = 0.0001) -> typing.Optional[float]
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranching.find_fleet_branching
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranching.find_fleet_branching
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price_and_cut.branching.BranchNode, fleet_usage: float) -> typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.FleetSizeBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.FleetSizeBranching.create_child_nodes
```

````

`````

`````{py:class} NodeVisitationBranching
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching
```

````{py:method} find_node_branching(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], optional_nodes: typing.Set[int], tol: float = 0.0001) -> typing.Optional[typing.Tuple[int, float]]
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching.find_node_branching
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching.find_node_branching
```

````

````{py:method} create_child_nodes(parent: src.policies.branch_and_price_and_cut.branching.BranchNode, node: int, visitation: float) -> typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.NodeVisitationBranching.create_child_nodes
```

````

`````

`````{py:class} BranchAndBoundTree(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, params: typing.Optional[src.policies.branch_and_price_and_cut.params.BPCParams] = None, max_nodes: int = 1000, strategy: str = 'edge', search_strategy: str = 'best_first')
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.__init__
```

````{py:attribute} node_coords
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.node_coords
:type: typing.Optional[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.node_coords
```

````

````{py:method} add_node(node: src.policies.branch_and_price_and_cut.branching.BranchNode) -> None
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.add_node

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.add_node
```

````

````{py:method} get_next_node() -> typing.Optional[src.policies.branch_and_price_and_cut.branching.BranchNode]
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.get_next_node
:abstractmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.get_next_node
```

````

````{py:method} branch(node: src.policies.branch_and_price_and_cut.branching.BranchNode, routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], mandatory_nodes: typing.Optional[typing.Set[int]] = None, strong_candidate: typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]] = None) -> typing.Optional[typing.Tuple[src.policies.branch_and_price_and_cut.branching.BranchNode, src.policies.branch_and_price_and_cut.branching.BranchNode]]
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.branch

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.branch
```

````

````{py:method} find_strong_branching_candidates(routes: typing.List[src.policies.branch_and_price_and_cut.master_problem.Route], route_values: typing.Dict[int, float], max_candidates: int = 5) -> typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.find_strong_branching_candidates

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.find_strong_branching_candidates
```

````

````{py:method} prune_by_bound() -> int
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.prune_by_bound

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.prune_by_bound
```

````

````{py:method} record_explored() -> None
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.record_explored

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.record_explored
```

````

````{py:method} update_incumbent(node: src.policies.branch_and_price_and_cut.branching.BranchNode, value: float) -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.update_incumbent

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.update_incumbent
```

````

````{py:method} is_empty() -> bool
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.is_empty

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.is_empty
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.get_statistics

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.BranchAndBoundTree.get_statistics
```

````

`````

````{py:data} AnyBranchingConstraint
:canonical: src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint
```

````
