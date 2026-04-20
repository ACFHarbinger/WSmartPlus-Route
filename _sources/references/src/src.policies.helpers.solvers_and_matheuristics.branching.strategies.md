# {py:mod}`src.policies.helpers.solvers_and_matheuristics.branching.strategies`

```{py:module} src.policies.helpers.solvers_and_matheuristics.branching.strategies
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeBranching <src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching
    :summary:
    ```
* - {py:obj}`MultiEdgePartitionBranching <src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching
    :summary:
    ```
* - {py:obj}`RyanFosterBranching <src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching
    :summary:
    ```
* - {py:obj}`FleetSizeBranching <src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching
    :summary:
    ```
* - {py:obj}`NodeVisitationBranching <src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching
    :summary:
    ```
````

### API

`````{py:class} EdgeBranching
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching
```

````{py:method} compute_arc_flow(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float]) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.compute_arc_flow
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.compute_arc_flow
```

````

````{py:method} find_branching_arc(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float], tol: float = 1e-05) -> typing.Optional[typing.Tuple[typing.Tuple[int, int], float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.find_branching_arc
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.find_branching_arc
```

````

````{py:method} create_child_nodes(parent: logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, u: int, v: int, flow: float = 0.5) -> typing.Tuple[logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.EdgeBranching.create_child_nodes
```

````

`````

`````{py:class} MultiEdgePartitionBranching
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching
```

````{py:method} find_divergence_node(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float], tol: float = 1e-05, node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None, n_nodes: int = 0) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.find_divergence_node
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.find_divergence_node
```

````

````{py:method} find_multiple_divergence_nodes(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float], node_coords: typing.Optional[typing.Union[numpy.ndarray, typing.Dict[int, typing.Tuple[float, float]]]] = None, limit: int = 5, tol: float = 1e-05, n_nodes: int = 0) -> typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.find_multiple_divergence_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.find_multiple_divergence_nodes
```

````

````{py:method} create_child_nodes(parent: logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, divergence_node: int, arc_set_1: typing.List[typing.Tuple[int, int]], arc_set_2: typing.List[typing.Tuple[int, int]], strength: float = 0.5) -> typing.Tuple[logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.MultiEdgePartitionBranching.create_child_nodes
```

````

`````

`````{py:class} RyanFosterBranching
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching
```

````{py:method} find_branching_pair(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float], mandatory_nodes: typing.Set[int], tol: float = 1e-05) -> typing.Optional[typing.Tuple[typing.Tuple[int, int], float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching.find_branching_pair
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching.find_branching_pair
```

````

````{py:method} create_child_nodes(parent: logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, node_r: int, node_s: int, together_sum: float = 0.5) -> typing.Tuple[logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.RyanFosterBranching.create_child_nodes
```

````

`````

`````{py:class} FleetSizeBranching
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching
```

````{py:method} find_fleet_branching(route_values: typing.Dict[int, float], tol: float = 0.0001) -> typing.Optional[float]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching.find_fleet_branching
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching.find_fleet_branching
```

````

````{py:method} create_child_nodes(parent: logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, fleet_usage: float) -> typing.Tuple[logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.FleetSizeBranching.create_child_nodes
```

````

`````

`````{py:class} NodeVisitationBranching
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching
```

````{py:method} find_node_branching(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], route_values: typing.Dict[int, float], optional_nodes: typing.Set[int], tol: float = 0.0001) -> typing.Optional[typing.Tuple[int, float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching.find_node_branching
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching.find_node_branching
```

````

````{py:method} create_child_nodes(parent: logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, node: int, visitation: float) -> typing.Tuple[logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode, logic.src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode]
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching.create_child_nodes
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.strategies.NodeVisitationBranching.create_child_nodes
```

````

`````
