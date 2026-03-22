# {py:mod}`src.policies.branch_and_cut.separation`

```{py:module} src.policies.branch_and_cut.separation
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Inequality <src.policies.branch_and_cut.separation.Inequality>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.separation.Inequality
    :summary:
    ```
* - {py:obj}`SubtourEliminationCut <src.policies.branch_and_cut.separation.SubtourEliminationCut>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.separation.SubtourEliminationCut
    :summary:
    ```
* - {py:obj}`CapacityCut <src.policies.branch_and_cut.separation.CapacityCut>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.separation.CapacityCut
    :summary:
    ```
* - {py:obj}`CombInequality <src.policies.branch_and_cut.separation.CombInequality>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.separation.CombInequality
    :summary:
    ```
* - {py:obj}`SeparationEngine <src.policies.branch_and_cut.separation.SeparationEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine
    :summary:
    ```
````

### API

`````{py:class} Inequality(inequality_type: str, degree_of_violation: float)
:canonical: src.policies.branch_and_cut.separation.Inequality

```{autodoc2-docstring} src.policies.branch_and_cut.separation.Inequality
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation.Inequality.__init__
```

````{py:method} __lt__(other)
:canonical: src.policies.branch_and_cut.separation.Inequality.__lt__

```{autodoc2-docstring} src.policies.branch_and_cut.separation.Inequality.__lt__
```

````

`````

````{py:class} SubtourEliminationCut(node_set: typing.Set[int], violation: float)
:canonical: src.policies.branch_and_cut.separation.SubtourEliminationCut

Bases: {py:obj}`src.policies.branch_and_cut.separation.Inequality`

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SubtourEliminationCut
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SubtourEliminationCut.__init__
```

````

````{py:class} CapacityCut(node_set: typing.Set[int], total_demand: float, capacity: float, violation: float)
:canonical: src.policies.branch_and_cut.separation.CapacityCut

Bases: {py:obj}`src.policies.branch_and_cut.separation.Inequality`

```{autodoc2-docstring} src.policies.branch_and_cut.separation.CapacityCut
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation.CapacityCut.__init__
```

````

````{py:class} CombInequality(handle: typing.Set[int], teeth: typing.List[typing.Set[int]], violation: float)
:canonical: src.policies.branch_and_cut.separation.CombInequality

Bases: {py:obj}`src.policies.branch_and_cut.separation.Inequality`

```{autodoc2-docstring} src.policies.branch_and_cut.separation.CombInequality
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation.CombInequality.__init__
```

````

`````{py:class} SeparationEngine(model)
:canonical: src.policies.branch_and_cut.separation.SeparationEngine

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine.__init__
```

````{py:method} separate(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray] = None, max_cuts: int = 100, iteration: int = 0) -> typing.List[src.policies.branch_and_cut.separation.Inequality]
:canonical: src.policies.branch_and_cut.separation.SeparationEngine.separate

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine.separate
```

````

````{py:method} _separate_subtours_heuristic(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_subtours_heuristic

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_subtours_heuristic
```

````

````{py:method} _separate_disconnected_components(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_disconnected_components

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_disconnected_components
```

````

````{py:method} _separate_weak_subtours(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_weak_subtours

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_weak_subtours
```

````

````{py:method} _get_cut_value(node_set: typing.Set[int], x_vals: numpy.ndarray) -> float
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._get_cut_value

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._get_cut_value
```

````

````{py:method} _separate_capacity_cuts(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_capacity_cuts

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_capacity_cuts
```

````

````{py:method} _separate_capacity_cuts_exact(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_capacity_cuts_exact

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_capacity_cuts_exact
```

````

````{py:method} _separate_subtours_exact(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_subtours_exact

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_subtours_exact
```

````

````{py:method} _extract_min_cut(capacity: numpy.ndarray, flow: numpy.ndarray, source: int, sink: int) -> typing.Set[int]
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._extract_min_cut

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._extract_min_cut
```

````

````{py:method} _separate_comb_heuristic(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._separate_comb_heuristic

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._separate_comb_heuristic
```

````

````{py:method} _grow_handle(seed: int, adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float], max_size: int = 15) -> typing.Set[int]
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._grow_handle

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._grow_handle
```

````

````{py:method} _find_teeth_for_handle(handle: typing.Set[int], adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float]) -> typing.List[typing.Set[int]]
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._find_teeth_for_handle

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._find_teeth_for_handle
```

````

````{py:method} _grow_tooth(anchor: int, handle: typing.Set[int], adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float], max_size: int = 7) -> typing.Optional[typing.Set[int]]
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._grow_tooth

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._grow_tooth
```

````

````{py:method} _compute_comb_violation(handle: typing.Set[int], teeth: typing.List[typing.Set[int]], x_vals: numpy.ndarray) -> float
:canonical: src.policies.branch_and_cut.separation.SeparationEngine._compute_comb_violation

```{autodoc2-docstring} src.policies.branch_and_cut.separation.SeparationEngine._compute_comb_violation
```

````

`````
