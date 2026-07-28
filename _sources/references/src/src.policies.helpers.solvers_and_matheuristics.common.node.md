# {py:mod}`src.policies.helpers.solvers_and_matheuristics.common.node`

```{py:module} src.policies.helpers.solvers_and_matheuristics.common.node
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Node <src.policies.helpers.solvers_and_matheuristics.common.node.Node>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node
    :summary:
    ```
* - {py:obj}`BranchNode <src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode
    :summary:
    ```
````

### API

`````{py:class} Node
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.Node

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node
```

````{py:attribute} bound
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.Node.bound
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node.bound
```

````

````{py:attribute} fixed_x
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.Node.fixed_x
:type: typing.Dict[typing.Tuple[int, int], int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node.fixed_x
```

````

````{py:attribute} fixed_y
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.Node.fixed_y
:type: typing.Dict[int, int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node.fixed_y
```

````

````{py:attribute} depth
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.Node.depth
:type: int
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.Node.depth
```

````

`````

`````{py:class} BranchNode(constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint]] = None, parent: typing.Optional[src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode] = None, depth: int = 0, lp_bound_hint: typing.Optional[float] = None, branching_rule: str = 'none')
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.__init__
```

````{py:method} get_all_constraints() -> typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint]
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.get_all_constraints

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.get_all_constraints
```

````

````{py:method} is_route_feasible(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.is_route_feasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.is_route_feasible
```

````

````{py:method} __lt__(other: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.__lt__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.__lt__
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.__repr__

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.common.node.BranchNode.__repr__
```

````

`````
