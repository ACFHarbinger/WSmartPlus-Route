# {py:mod}`src.policies.branch_and_price_and_cut.cutting_planes`

```{py:module} src.policies.branch_and_price_and_cut.cutting_planes
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CuttingPlaneEngine <src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine
    :summary:
    ```
* - {py:obj}`RoundedCapacityCutEngine <src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine
    :summary:
    ```
* - {py:obj}`SubsetRowCutEngine <src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine
    :summary:
    ```
* - {py:obj}`EdgeCliqueCutEngine <src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine
    :summary:
    ```
* - {py:obj}`KnapsackCoverEngine <src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine
    :summary:
    ```
* - {py:obj}`CompositeCuttingPlaneEngine <src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine
    :summary:
    ```
* - {py:obj}`LiftedCoverCutEngine <src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_cutting_plane_engine <src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine
    :summary:
    ```
````

### API

`````{py:class} CuttingPlaneEngine
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine.separate_and_add_cuts
:abstractmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine.get_name
:abstractmethod:

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine.get_name
```

````

````{py:method} _is_orthogonal(candidate_vec: numpy.ndarray, active_vecs: typing.List[numpy.ndarray], threshold: float = 0.8) -> bool
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine._is_orthogonal

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine._is_orthogonal
```

````

`````

`````{py:class} RoundedCapacityCutEngine(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, sep_engine: src.policies.branch_and_price_and_cut.separation.SeparationEngine)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.get_name

````

`````

`````{py:class} SubsetRowCutEngine(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine.separate_and_add_cuts

````

````{py:method} _evaluate_and_add_sri(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, node_set: typing.Set[int]) -> bool
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine._evaluate_and_add_sri

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine._evaluate_and_add_sri
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.SubsetRowCutEngine.get_name

````

`````

`````{py:class} EdgeCliqueCutEngine(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, capacity: float = 1.0, epsilon: float = 0.01)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.EdgeCliqueCutEngine.get_name

````

`````

`````{py:class} KnapsackCoverEngine(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, sep_engine: src.policies.branch_and_price_and_cut.separation.SeparationEngine)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.KnapsackCoverEngine.get_name

````

`````

`````{py:class} CompositeCuttingPlaneEngine(engines: typing.List[src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine])
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine.separate_and_add_cuts

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.CompositeCuttingPlaneEngine.get_name

````

`````

`````{py:class} LiftedCoverCutEngine(v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, epsilon: float = 0.01)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine.separate_and_add_cuts

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.LiftedCoverCutEngine.get_name

````

`````

````{py:function} create_cutting_plane_engine(engine_name: str, v_model: src.policies.branch_and_price_and_cut.vrpp_model.VRPPModel, sep_engine: typing.Optional[src.policies.branch_and_price_and_cut.separation.SeparationEngine] = None) -> src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine
```
````
