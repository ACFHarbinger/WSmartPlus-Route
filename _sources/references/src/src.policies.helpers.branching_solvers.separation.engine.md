# {py:mod}`src.policies.helpers.branching_solvers.separation.engine`

```{py:module} src.policies.helpers.branching_solvers.separation.engine
```

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SeparationEngine <src.policies.helpers.branching_solvers.separation.engine.SeparationEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine
    :summary:
    ```
````

### API

`````{py:class} SeparationEngine(model, enable_heuristic_rcc_separation: bool = True, enable_comb_cuts: bool = False)
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.__init__
```

````{py:attribute} USE_COMB_CUTS
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.USE_COMB_CUTS
:value: >
   False

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.USE_COMB_CUTS
```

````

````{py:attribute} _EXACT_SEP_PERIOD
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._EXACT_SEP_PERIOD
:value: >
   3

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._EXACT_SEP_PERIOD
```

````

````{py:method} separate_integer(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray] = None, max_cuts: int = 100, iteration: int = 0, sec_only: bool = False) -> typing.List[logic.src.policies.helpers.branching_solvers.separation.inequality.Inequality]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate_integer

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate_integer
```

````

````{py:method} separate_fractional(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray] = None, max_cuts: int = 50, iteration: int = 0, node_count: int = 0) -> typing.List[logic.src.policies.helpers.branching_solvers.separation.inequality.Inequality]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate_fractional

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate_fractional
```

````

````{py:method} separate(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray] = None, max_cuts: int = 100, iteration: int = 0) -> typing.List[logic.src.policies.helpers.branching_solvers.separation.inequality.Inequality]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine.separate
```

````

````{py:method} _separate_subtours_heuristic(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_subtours_heuristic

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_subtours_heuristic
```

````

````{py:method} _separate_disconnected_components(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_disconnected_components

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_disconnected_components
```

````

````{py:method} _separate_weak_subtours(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_weak_subtours

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_weak_subtours
```

````

````{py:method} _get_cut_value(node_set: typing.Set[int], x_vals: numpy.ndarray) -> float
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._get_cut_value

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._get_cut_value
```

````

````{py:method} _separate_capacity_cuts(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_capacity_cuts

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_capacity_cuts
```

````

````{py:method} _separate_capacity_cuts_maxflow_heuristic(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray], root_node: bool = False, max_cuts: int = 50)
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_capacity_cuts_maxflow_heuristic

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_capacity_cuts_maxflow_heuristic
```

````

````{py:method} _separate_pcsec_exact(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray], root_node: bool = False)
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_pcsec_exact

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_pcsec_exact
```

````

````{py:method} _extract_min_cut(capacity: numpy.ndarray, flow: numpy.ndarray, source: int, sink: int) -> typing.Set[int]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._extract_min_cut

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._extract_min_cut
```

````

````{py:method} _separate_comb_heuristic(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_comb_heuristic

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_comb_heuristic
```

````

````{py:method} _grow_handle(seed: int, adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float], max_size: int = 15) -> typing.Set[int]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._grow_handle

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._grow_handle
```

````

````{py:method} _find_teeth_for_handle(handle: typing.Set[int], adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float]) -> typing.List[typing.Set[int]]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._find_teeth_for_handle

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._find_teeth_for_handle
```

````

````{py:method} _grow_tooth(anchor: int, handle: typing.Set[int], adjacency: typing.Dict[int, typing.List[int]], edge_weights: typing.Dict[typing.Tuple[int, int], float], max_size: int = 7) -> typing.Optional[typing.Set[int]]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._grow_tooth

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._grow_tooth
```

````

````{py:method} _compute_comb_violation(handle: typing.Set[int], teeth: typing.List[typing.Set[int]], x_vals: numpy.ndarray) -> float
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._compute_comb_violation

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._compute_comb_violation
```

````

````{py:method} _separate_gsec_h2(x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_gsec_h2

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._separate_gsec_h2
```

````

````{py:method} _strengthen_pool(ineq_list: typing.List[logic.src.policies.helpers.branching_solvers.separation.inequality.Inequality], x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray])
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._strengthen_pool

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._strengthen_pool
```

````

````{py:method} _refine_pcsec_build(s_set: typing.Set[int], x_vals: numpy.ndarray, y_vals: typing.Optional[numpy.ndarray]) -> typing.Set[int]
:canonical: src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._refine_pcsec_build

```{autodoc2-docstring} src.policies.helpers.branching_solvers.separation.engine.SeparationEngine._refine_pcsec_build
```

````

`````
