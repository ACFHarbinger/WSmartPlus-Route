# {py:mod}`src.configs.policies.swc_tcf`

```{py:module} src.configs.policies.swc_tcf
```

```{autodoc2-docstring} src.configs.policies.swc_tcf
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SWCTCFConfig <src.configs.policies.swc_tcf.SWCTCFConfig>`
  - ```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig
    :summary:
    ```
````

### API

`````{py:class} SWCTCFConfig
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig
```

````{py:attribute} Omega
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.Omega
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.Omega
```

````

````{py:attribute} psi
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.psi
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.psi
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.time_limit
:type: float
:value: >
   600.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.seed
```

````

````{py:attribute} engine
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.engine
:type: str
:value: >
   'gurobi'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.engine
```

````

````{py:attribute} gurobi_threads
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.gurobi_threads
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.gurobi_threads
```

````

````{py:attribute} gurobi_soft_mem_limit_gb
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.gurobi_soft_mem_limit_gb
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.gurobi_soft_mem_limit_gb
```

````

````{py:attribute} framework
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.framework
```

````

````{py:attribute} formulation
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.formulation
:type: str
:value: >
   'paper'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.formulation
```

````

````{py:attribute} depot_inflow
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.depot_inflow
:type: str
:value: >
   'equal'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.depot_inflow
```

````

````{py:attribute} solver_tuning
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.solver_tuning
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.solver_tuning
```

````

````{py:attribute} relax_forced_on_infeasible
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.relax_forced_on_infeasible
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.relax_forced_on_infeasible
```

````

````{py:attribute} link_depot_arcs
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.link_depot_arcs
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.link_depot_arcs
```

````

````{py:attribute} max_arc_distance_km
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.max_arc_distance_km
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.max_arc_distance_km
```

````

````{py:attribute} warm_start
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.warm_start
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.warm_start
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.route_improvement
```

````

`````
