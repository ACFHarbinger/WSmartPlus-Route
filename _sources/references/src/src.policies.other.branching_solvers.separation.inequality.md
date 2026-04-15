# {py:mod}`src.policies.other.branching_solvers.separation.inequality`

```{py:module} src.policies.other.branching_solvers.separation.inequality
```

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Inequality <src.policies.other.branching_solvers.separation.inequality.Inequality>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.Inequality
    :summary:
    ```
* - {py:obj}`PCSubtourEliminationCut <src.policies.other.branching_solvers.separation.inequality.PCSubtourEliminationCut>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.PCSubtourEliminationCut
    :summary:
    ```
* - {py:obj}`CapacityCut <src.policies.other.branching_solvers.separation.inequality.CapacityCut>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CapacityCut
    :summary:
    ```
* - {py:obj}`CombInequality <src.policies.other.branching_solvers.separation.inequality.CombInequality>`
  - ```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CombInequality
    :summary:
    ```
````

### API

`````{py:class} Inequality(inequality_type: str, degree_of_violation: float)
:canonical: src.policies.other.branching_solvers.separation.inequality.Inequality

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.Inequality
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.Inequality.__init__
```

````{py:method} __lt__(other)
:canonical: src.policies.other.branching_solvers.separation.inequality.Inequality.__lt__

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.Inequality.__lt__
```

````

`````

````{py:class} PCSubtourEliminationCut(node_set: typing.Set[int], violation: float, facet_form: str = '2.1', node_i: int = -1, node_j: int = -1)
:canonical: src.policies.other.branching_solvers.separation.inequality.PCSubtourEliminationCut

Bases: {py:obj}`src.policies.other.branching_solvers.separation.inequality.Inequality`

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.PCSubtourEliminationCut
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.PCSubtourEliminationCut.__init__
```

````

````{py:class} CapacityCut(node_set: typing.Set[int], total_demand: float, capacity: float, violation: float)
:canonical: src.policies.other.branching_solvers.separation.inequality.CapacityCut

Bases: {py:obj}`src.policies.other.branching_solvers.separation.inequality.Inequality`

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CapacityCut
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CapacityCut.__init__
```

````

````{py:class} CombInequality(handle: typing.Set[int], teeth: typing.List[typing.Set[int]], violation: float)
:canonical: src.policies.other.branching_solvers.separation.inequality.CombInequality

Bases: {py:obj}`src.policies.other.branching_solvers.separation.inequality.Inequality`

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CombInequality
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.branching_solvers.separation.inequality.CombInequality.__init__
```

````
