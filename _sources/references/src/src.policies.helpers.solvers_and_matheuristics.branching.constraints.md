# {py:mod}`src.policies.helpers.solvers_and_matheuristics.branching.constraints`

```{py:module} src.policies.helpers.solvers_and_matheuristics.branching.constraints
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeBranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint
    :summary:
    ```
* - {py:obj}`RyanFosterBranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint
    :summary:
    ```
* - {py:obj}`FleetSizeBranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint
    :summary:
    ```
* - {py:obj}`NodeVisitationBranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AnyBranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint
    :summary:
    ```
* - {py:obj}`BranchingConstraint <src.policies.helpers.solvers_and_matheuristics.branching.constraints.BranchingConstraint>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.BranchingConstraint
    :summary:
    ```
````

### API

`````{py:class} EdgeBranchingConstraint(u: int, v: int, must_use: bool)
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint.is_route_feasible
```

````

````{py:method} _edge_in_route(nodes: typing.List[int]) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint._edge_in_route

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint._edge_in_route
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.EdgeBranchingConstraint.__repr__
```

````

`````

`````{py:class} RyanFosterBranchingConstraint(node_r: int, node_s: int, together: bool)
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.RyanFosterBranchingConstraint.__repr__
```

````

`````

`````{py:class} FleetSizeBranchingConstraint(limit: int, is_upper: bool)
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.FleetSizeBranchingConstraint.__repr__
```

````

`````

`````{py:class} NodeVisitationBranchingConstraint(node: int, forced: bool)
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint.__init__
```

````{py:method} is_route_feasible(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint.is_route_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint.is_route_feasible
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint.__repr__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.NodeVisitationBranchingConstraint.__repr__
```

````

`````

````{py:data} AnyBranchingConstraint
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint
```

````

````{py:data} BranchingConstraint
:canonical: src.policies.helpers.solvers_and_matheuristics.branching.constraints.BranchingConstraint
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.branching.constraints.BranchingConstraint
```

````
