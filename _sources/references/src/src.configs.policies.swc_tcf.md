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

````{py:attribute} delta
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.delta
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.delta
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

````{py:attribute} framework
:canonical: src.configs.policies.swc_tcf.SWCTCFConfig.framework
:type: str
:value: >
   'ortools'

```{autodoc2-docstring} src.configs.policies.swc_tcf.SWCTCFConfig.framework
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
