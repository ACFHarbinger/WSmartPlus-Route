# {py:mod}`src.policies.other.branching_solvers.branching.constraints`

```{py:module} src.policies.other.branching_solvers.branching.constraints
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeBranchingConstraint <src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint
    :summary:
    ```
* - {py:obj}`RyanFosterBranchingConstraint <src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint
    :summary:
    ```
* - {py:obj}`FleetSizeBranchingConstraint <src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint
    :summary:
    ```
* - {py:obj}`NodeVisitationBranchingConstraint <src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AnyBranchingConstraint <src.policies.other.branching_solvers.branching.constraints.AnyBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.AnyBranchingConstraint
    :summary:
    ```
* - {py:obj}`BranchingConstraint <src.policies.other.branching_solvers.branching.constraints.BranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.BranchingConstraint
    :summary:
    ```
````

### API

`````{py:class} EdgeBranchingConstraint(u: int, v: int, must_use: bool)
:canonical: src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.other.branching_solvers.common.route.Route) -> bool
:canonical: src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint.is_route_feasible
```

````

````{py:method} _edge_in_route(nodes: typing.List[int]) -> bool
:canonical: src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint._edge_in_route

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint._edge_in_route
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.EdgeBranchingConstraint.__repr__
```

````

`````

`````{py:class} RyanFosterBranchingConstraint(node_r: int, node_s: int, together: bool)
:canonical: src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.other.branching_solvers.common.route.Route) -> bool
:canonical: src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.RyanFosterBranchingConstraint.__repr__
```

````

`````

`````{py:class} FleetSizeBranchingConstraint(limit: int, is_upper: bool)
:canonical: src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.other.branching_solvers.common.route.Route) -> bool
:canonical: src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.other.branching_solvers.branching.constraints.FleetSizeBranchingConstraint.__repr__

````

`````

`````{py:class} NodeVisitationBranchingConstraint(node: int, forced: bool)
:canonical: src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.other.branching_solvers.common.route.Route) -> bool
:canonical: src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.other.branching_solvers.branching.constraints.NodeVisitationBranchingConstraint.__repr__

````

`````

````{py:data} AnyBranchingConstraint
:canonical: src.policies.other.branching_solvers.branching.constraints.AnyBranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.AnyBranchingConstraint
```

````

````{py:data} BranchingConstraint
:canonical: src.policies.other.branching_solvers.branching.constraints.BranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.other.branching_solvers.branching.constraints.BranchingConstraint
```

````
