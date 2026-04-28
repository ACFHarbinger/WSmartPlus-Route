# {py:mod}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes`

```{py:module} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CuttingPlaneEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine
    :summary:
    ```
* - {py:obj}`RoundedCapacityCutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine
    :summary:
    ```
* - {py:obj}`SubsetRowCutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine
    :summary:
    ```
* - {py:obj}`EdgeCliqueCutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine
    :summary:
    ```
* - {py:obj}`KnapsackCoverEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine
    :summary:
    ```
* - {py:obj}`CompositeCuttingPlaneEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine
    :summary:
    ```
* - {py:obj}`BasicFleetCoverEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine
    :summary:
    ```
* - {py:obj}`PhysicalCapacityLCIEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine
    :summary:
    ```
* - {py:obj}`SaturatedArcLCIEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine
    :summary:
    ```
* - {py:obj}`RoundedMultistarCutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine
    :summary:
    ```
* - {py:obj}`MinCutInequalityEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine
    :summary:
    ```
* - {py:obj}`TriangleCliqueCutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine
    :summary:
    ```
* - {py:obj}`LimitedMemoryRank1CutEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine
    :summary:
    ```
* - {py:obj}`NodeProfitBoundEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine
    :summary:
    ```
* - {py:obj}`PathEliminationEngine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_cutting_plane_engine <src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.create_cutting_plane_engine>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.create_cutting_plane_engine
    :summary:
    ```
````

### API

`````{py:class} CuttingPlaneEngine
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.separate_and_add_cuts
:abstractmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.get_name
:abstractmethod:

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.get_name
```

````

````{py:property} engines
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.engines
:type: typing.List[src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine]

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine.engines
```

````

````{py:method} _is_orthogonal(candidate_vec: numpy.ndarray, active_vecs: typing.List[numpy.ndarray], threshold: float = 0.8) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine._is_orthogonal

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine._is_orthogonal
```

````

`````

`````{py:class} RoundedCapacityCutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, sep_engine: logic.src.policies.helpers.solvers_and_matheuristics.separation.SeparationEngine)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedCapacityCutEngine.get_name
```

````

`````

`````{py:class} SubsetRowCutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine.separate_and_add_cuts
```

````

````{py:method} _evaluate_and_add_sri(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, node_set: typing.Set[int], **kwargs) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine._evaluate_and_add_sri

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine._evaluate_and_add_sri
```

````

````{py:method} _is_orthogonal_content(candidate_dict: typing.Dict[str, float], active_vecs: typing.List[typing.Dict[str, float]], threshold: float = 0.8) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine._is_orthogonal_content

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine._is_orthogonal_content
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SubsetRowCutEngine.get_name
```

````

`````

`````{py:class} EdgeCliqueCutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, capacity: float = 1.0, epsilon: float = 0.01)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.EdgeCliqueCutEngine.get_name
```

````

`````

`````{py:class} KnapsackCoverEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, sep_engine: logic.src.policies.helpers.solvers_and_matheuristics.separation.SeparationEngine)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.KnapsackCoverEngine.get_name
```

````

`````

`````{py:class} CompositeCuttingPlaneEngine(engines: typing.List[src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine])
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.__init__
```

````{py:property} engines
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.engines
:type: typing.List[src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine]

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.engines
```

````

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CompositeCuttingPlaneEngine.get_name
```

````

`````

`````{py:class} BasicFleetCoverEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, epsilon: float = 0.01)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.BasicFleetCoverEngine.get_name
```

````

`````

`````{py:class} PhysicalCapacityLCIEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, epsilon: float = 0.01)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PhysicalCapacityLCIEngine.get_name
```

````

`````

`````{py:class} SaturatedArcLCIEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, epsilon: float = 0.01)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.SaturatedArcLCIEngine.get_name
```

````

`````

`````{py:class} RoundedMultistarCutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, sep_engine: logic.src.policies.helpers.solvers_and_matheuristics.separation.SeparationEngine, epsilon: float = 0.01)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine.__init__
```

````{py:method} _add_inequalities(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, edge_vars: typing.Dict[typing.Tuple[int, int], float], node_visits: typing.Dict[int, float], candidate_ineqs: typing.List[typing.Any], max_cuts: int) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine._add_inequalities

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine._add_inequalities
```

````

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.RoundedMultistarCutEngine.get_name
```

````

`````

````{py:function} create_cutting_plane_engine(engine_name: str, v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, sep_engine: typing.Optional[logic.src.policies.helpers.solvers_and_matheuristics.separation.SeparationEngine] = None, dist_matrix: typing.Optional[numpy.ndarray] = None, route_budget: float = float('inf'), max_subset: int = 50, **kwargs: dict) -> src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.create_cutting_plane_engine

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.create_cutting_plane_engine
```
````

`````{py:class} MinCutInequalityEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.MinCutInequalityEngine.get_name
```

````

`````

`````{py:class} TriangleCliqueCutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, dist_matrix: typing.Optional[numpy.ndarray] = None, route_budget: float = float('inf'))
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine.__init__
```

````{py:method} _build_conflict_pairs(capacity: float) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine._build_conflict_pairs

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine._build_conflict_pairs
```

````

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.TriangleCliqueCutEngine.get_name
```

````

`````

`````{py:class} LimitedMemoryRank1CutEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, max_subset_size: int = 5)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.LimitedMemoryRank1CutEngine.get_name
```

````

`````

`````{py:class} NodeProfitBoundEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel)
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine.__init__
```

````{py:method} _fractional_knapsack(items: typing.List[typing.Tuple[float, float]], capacity: float) -> float
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine._fractional_knapsack

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine._fractional_knapsack
```

````

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.NodeProfitBoundEngine.get_name
```

````

`````

`````{py:class} PathEliminationEngine(v_model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, dist_matrix: typing.Optional[numpy.ndarray] = None, route_budget: float = float('inf'))
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine

Bases: {py:obj}`src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine.__init__
```

````{py:method} _is_path_infeasible(path: typing.List[int], capacity: float) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine._is_path_infeasible

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine._is_path_infeasible
```

````

````{py:method} separate_and_add_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine.get_name

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.PathEliminationEngine.get_name
```

````

`````
