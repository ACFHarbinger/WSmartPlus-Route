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

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
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

`````

`````{py:class} RoundedCapacityCutEngine(v_model: src.policies.branch_and_cut.vrpp_model.VRPPModel, sep_engine: src.policies.branch_and_cut.separation.SeparationEngine)
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine

Bases: {py:obj}`src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.__init__
```

````{py:method} separate_and_add_cuts(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, max_cuts: int, **kwargs) -> int
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.separate_and_add_cuts
```

````

````{py:method} get_name() -> str
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.RoundedCapacityCutEngine.get_name

````

`````

````{py:function} create_cutting_plane_engine(engine_name: str, v_model: src.policies.branch_and_cut.vrpp_model.VRPPModel, sep_engine: typing.Optional[src.policies.branch_and_cut.separation.SeparationEngine] = None) -> src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine
:canonical: src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.cutting_planes.create_cutting_plane_engine
```
````
